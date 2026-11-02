"""HTTP surface: management API (/api/v1, client tokens) and inference (/v1, lease keys).

Nothing here logs request bodies, prompts or credentials.
"""
from __future__ import annotations

import json
import logging

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route

from . import params as P
from .manager import ApiError, Manager
from .routing import ant_error, ant_stream, classify, oai_error, oai_stream, replica_of

log = logging.getLogger("sserved")


def _bearer(req: Request) -> str:
    a = req.headers.get("authorization", "")
    if a.lower().startswith("bearer "):
        return a[7:].strip()
    return req.headers.get("x-api-key", "").strip()


def _mgmt_err(e: ApiError) -> JSONResponse:
    h = {"Retry-After": str(e.retry_after)} if e.retry_after else None
    return JSONResponse({"error": {"code": e.code, "message": e.msg}}, status_code=e.status, headers=h)


def build_app(mgr: Manager) -> Starlette:
    hub = mgr.hub

    def principal(req: Request) -> dict:
        p = mgr.db.principal(_bearer(req))
        if p is None:
            raise ApiError(401, "invalid_token", "missing or invalid client token")
        return p

    def mgmt(fn):
        async def h(req: Request) -> Response:
            try:
                return await fn(req, principal(req))
            except ApiError as e:
                return _mgmt_err(e)
        return h

    async def body_json(req: Request) -> dict:
        try:
            b = await req.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ApiError(400, "invalid_json", "request body is not JSON")
        if not isinstance(b, dict):
            raise ApiError(400, "invalid_json", "request body must be a JSON object")
        return b

    # ------------------------------------------------------------ management
    @mgmt
    async def leases_post(req, p):
        return JSONResponse(mgr.acquire(p, await body_json(req)), status_code=201)

    @mgmt
    async def leases_get(req, p):
        return JSONResponse({"leases": mgr.list_leases(p)})

    @mgmt
    async def lease_get(req, p):
        return JSONResponse(mgr.lease_view(mgr._own(p, req.path_params["id"])))

    @mgmt
    async def lease_delete(req, p):
        return JSONResponse(mgr.release(p, req.path_params["id"]))

    @mgmt
    async def lease_renew(req, p):
        b = await body_json(req) if (await req.body()) else {}
        return JSONResponse(mgr.renew(p, req.path_params["id"], b.get("ttl_s")))

    @mgmt
    async def lease_rotate(req, p):
        return JSONResponse(mgr.rotate_key(p, req.path_params["id"]))

    @mgmt
    async def status(req, p):
        return JSONResponse(mgr.status(p))

    @mgmt
    async def events(req, p):
        since = float(req.query_params.get("since", 0))
        if p.get("admin"):
            rows = mgr.db.q("SELECT * FROM events WHERE ts>? ORDER BY id LIMIT 500", since)
        else:
            rows = mgr.db.q("SELECT * FROM events WHERE ts>? AND deployment IN (SELECT deployment FROM leases "
                            "WHERE client=?) ORDER BY id LIMIT 500", since, p["name"])
        return JSONResponse({"events": rows})

    async def healthz(req):
        return JSONResponse({"ok": True})

    # ------------------------------------------------------------- inference
    def inference_err(e: ApiError, anthropic: bool) -> JSONResponse:
        h = {"Retry-After": str(e.retry_after)} if e.retry_after else None
        if anthropic:
            etype = {401: "authentication_error", 404: "not_found_error", 503: "overloaded_error",
                     429: "rate_limit_error", 400: "invalid_request_error"}.get(e.status, "api_error")
            body = ant_error(etype, f"{e.code}: {e.msg}")
        else:
            body = oai_error(e.status, "invalid_request_error" if e.status < 500 else "service_unavailable", e.msg, e.code)
        return JSONResponse(body, status_code=e.status, headers=h)

    def upstream_err(exc: Exception, d: dict, anthropic: bool) -> JSONResponse:
        status, etype, cap = classify(exc)
        if cap:
            hub.capacity_error(d["id"])
        msg = f"upstream error ({type(exc).__name__})"
        h = {"Retry-After": "15"} if status in (429, 503) else None
        if anthropic:
            atype = {429: "rate_limit_error", 503: "overloaded_error", 400: "invalid_request_error",
                     404: "not_found_error"}.get(status, "api_error")
            return JSONResponse(ant_error(atype, msg), status_code=status, headers=h)
        return JSONResponse(oai_error(status, etype, msg), status_code=status, headers=h)

    async def infer(req: Request, anthropic: bool) -> Response:
        try:
            body = await body_json(req)
            lease, d = mgr.authorize(_bearer(req), body.get("model"))
            try:   # rebuilt from an allowlist: clients never reach LiteLLM's routing/transport kwargs
                kwargs = P.anthropic_messages(body) if anthropic else P.openai_chat(body)
                if anthropic and (h := P.beta_header(req.headers.get("anthropic-beta"))):
                    kwargs["extra_headers"] = h
            except P.BadParams as e:
                raise ApiError(400, "unsupported_parameter", str(e))
            mgr.admission(d)
        except ApiError as e:
            return inference_err(e, anthropic)
        flight = hub.begin(lease["id"], d["id"])
        mgr.touch(lease["id"])

        def done(f=flight):
            hub.end(f)
            mgr.touch(lease["id"])

        stream = bool(kwargs.get("stream"))
        if stream:
            # No Router retries/fallbacks for streams (server-enforced; clients cannot set these): LiteLLM
            # may otherwise re-issue a stream upstream after a role/lifecycle chunk already reached us.
            kwargs.update(num_retries=0, fallbacks=[])
        try:
            if anthropic:
                resp = await hub.router.aanthropic_messages(model=d["id"], **kwargs)
            else:
                resp = await hub.router.acompletion(model=d["id"], **kwargs)
            flight.replica = replica_of(resp)
            if not stream:
                done()
                if anthropic:
                    out = dict(resp) if isinstance(resp, dict) else resp.model_dump()
                    out.pop("_hidden_params", None)
                    return JSONResponse(out)
                return Response(resp.model_dump_json(exclude_none=True), media_type="application/json")
            it = resp.__aiter__()
            first = await it.__anext__()          # errors before the first byte become HTTP errors
        except StopAsyncIteration:
            done()
            return Response(b"", media_type="text/event-stream")
        except Exception as e:  # noqa: BLE001
            done()
            return upstream_err(e, d, anthropic)
        if anthropic:
            if isinstance(first, str):
                first = first.encode()
            gen = ant_stream(first, it, flight, hub)
        else:
            gen = oai_stream(first, it, flight, hub)

        async def body_iter():
            try:
                async for b in gen:
                    yield b
            finally:
                done()
        return StreamingResponse(body_iter(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def chat(req):
        return await infer(req, anthropic=False)

    async def messages(req):
        return await infer(req, anthropic=True)

    async def models(req):
        try:
            lease, d = mgr.authorize(_bearer(req), None)
        except ApiError as e:
            return inference_err(e, False)
        m = mgr.cfg.models[d["model"]]
        mml = d["max_model_len"] or m.max_model_len or None
        card = {"id": m.served, "object": "model", "owned_by": "sserved", "root": m.name}
        if mml:
            card["max_model_len"] = mml
        return JSONResponse({"object": "list", "data": [card]})

    async def count_tokens(req):
        return inference_err(ApiError(404, "not_supported", "/v1/messages/count_tokens is not supported"), True)

    routes = [
        Route("/healthz", healthz),
        Route("/api/v1/leases", leases_post, methods=["POST"]),
        Route("/api/v1/leases", leases_get, methods=["GET"]),
        Route("/api/v1/leases/{id}", lease_get, methods=["GET"]),
        Route("/api/v1/leases/{id}", lease_delete, methods=["DELETE"]),
        Route("/api/v1/leases/{id}/renew", lease_renew, methods=["POST"]),
        Route("/api/v1/leases/{id}/rotate-key", lease_rotate, methods=["POST"]),
        Route("/api/v1/status", status),
        Route("/api/v1/events", events),
        Route("/v1/chat/completions", chat, methods=["POST"]),
        Route("/v1/messages", messages, methods=["POST"]),
        Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]),
        Route("/v1/models", models),
    ]
    rp = mgr.cfg.server.root_path.rstrip("/")
    if rp:   # serve both: behind a prefix-stripping proxy, and when the prefix reaches us
        routes = [Mount(rp, routes=list(routes))] + routes
    return Starlette(routes=routes)
