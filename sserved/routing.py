"""The one routing layer: an in-process LiteLLM Router (library mode, pinned).

Membership is the admission control. A replica is added with
Router.add_deployment only after the manager verified it (health, model
identity) and removed with Router.delete_deployment when it fails, drains or
changes generation. LiteLLM's own health-check routing is not used: its filter
returns *all* deployments when every one is unhealthy. When a deployment group
has no admitted replica, the HTTP layer answers 503 without calling the Router.

Streams are never replayed after the first byte reached the client. LiteLLM
1.106.2 only retries before content (chat completions: before the stream is
returned; /v1/messages: lifecycle frames are buffered while a retry is
possible). The guard below aborts a stream that would restart after bytes were
sent anyway (a second message_start, or a changed chat completion id).
"""
from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field

os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
os.environ.setdefault("LITELLM_TELEMETRY", "False")

import litellm  # noqa: E402
from litellm import Router  # noqa: E402

litellm.telemetry = False
litellm.turn_off_message_logging = True
litellm.suppress_debug_info = True
litellm.drop_params = False

ENDPOINTS = ["/v1/chat/completions", "/v1/messages"]
_ids = itertools.count(1)


@dataclass
class Flight:
    id: int
    lease: str
    group: str
    start: float
    replica: str | None = None      # known once LiteLLM picked a deployment
    done: bool = False


@dataclass
class Hub:
    # Cooldowns off: replica health is the manager's job (probes + membership). LiteLLM cooldowns would
    # let a few client-induced 500s take the only replica out for everyone. Retries (pre-first-byte)
    # still move a failed request to another admitted replica.
    router: Router = field(default_factory=lambda: Router(
        model_list=[], routing_strategy="simple-shuffle", num_retries=2, retry_after=0,
        disable_cooldowns=True, timeout=3600))
    members: dict[str, set[str]] = field(default_factory=dict)   # group -> admitted replica ids
    flights: dict[int, Flight] = field(default_factory=dict)
    removed_at: dict[str, float] = field(default_factory=dict)   # replica -> time it left the router
    capacity_errors: dict[str, list[float]] = field(default_factory=dict)   # group -> timestamps
    ready_since: dict[str, float] = field(default_factory=dict)  # group -> since when >= 1 replica is admitted

    # -- membership --------------------------------------------------------------
    def admit(self, group: str, replica: str, served: str, lport: int, key: str) -> None:
        if replica in self.members.get(group, set()):
            return
        self.router.add_deployment(litellm.types.router.Deployment(
            model_name=group,
            litellm_params={"model": f"hosted_vllm/{served}", "api_base": f"http://127.0.0.1:{lport}/v1",
                            "api_key": key, "timeout": 3600, "stream_timeout": 600, "max_retries": 0},
            model_info={"id": replica, "supported_endpoints": ENDPOINTS}))
        if not self.members.get(group):
            self.ready_since[group] = time.time()
        self.members.setdefault(group, set()).add(replica)
        self.removed_at.pop(replica, None)

    def evict(self, group: str, replica: str) -> None:
        if replica in self.members.get(group, set()):
            self.router.delete_deployment(replica)
            self.members[group].discard(replica)
            self.removed_at[replica] = time.time()
            if not self.members[group]:
                self.ready_since.pop(group, None)

    def admitted(self, group: str) -> set[str]:
        return set(self.members.get(group, set()))

    # -- in-flight accounting ------------------------------------------------------
    def begin(self, lease: str, group: str) -> Flight:
        f = Flight(next(_ids), lease, group, time.time())
        self.flights[f.id] = f
        return f

    def end(self, f: Flight) -> None:
        f.done = True
        self.flights.pop(f.id, None)

    def inflight_lease(self, lease: str) -> int:
        return sum(1 for f in self.flights.values() if f.lease == lease)

    def inflight_group(self, group: str) -> int:
        return sum(1 for f in self.flights.values() if f.group == group)

    def drained(self, group: str, replica: str) -> bool:
        """True once no request can still be using `replica`: every request that started before it
        left the router has finished or is known to be on another replica. Requests whose replica
        LiteLLM has not reported yet count as possibly on it (conservative)."""
        t = self.removed_at.get(replica)
        if t is None:
            return False
        for f in self.flights.values():
            if f.group == group and f.start <= t and f.replica in (None, replica):
                return False
        return True

    def capacity_error(self, group: str) -> None:
        self.capacity_errors.setdefault(group, []).append(time.time())

    def capacity_errors_since(self, group: str, t: float) -> int:
        lst = [x for x in self.capacity_errors.get(group, []) if x >= t]
        self.capacity_errors[group] = lst
        return len(lst)


# -- error mapping -----------------------------------------------------------------
# Scale pressure only from overload the backend itself reports (429, 503). Connection failures and timeouts
# usually mean a broken tunnel or login node: probes and tunnel supervision repair those, and a new GPU
# allocation would not help. A 500 can be a parser/model failure; 400/401/404 are client/config errors.


def classify(e: Exception) -> tuple[int, str, bool]:
    """-> (http status, error type, counts as capacity pressure)."""
    if isinstance(e, litellm.RateLimitError):
        return 429, "rate_limited", True
    if isinstance(e, litellm.ServiceUnavailableError):
        return 503, "service_unavailable", True
    if isinstance(e, litellm.BadRequestError):
        return 400, "invalid_request_error", False
    if isinstance(e, (litellm.AuthenticationError, litellm.PermissionDeniedError)):
        # the backend rejected the manager's key: a manager/backend problem, not the client's
        return 502, "backend_auth_error", False
    if isinstance(e, litellm.NotFoundError):
        return 404, "not_found_error", False
    if isinstance(e, litellm.Timeout):
        return 504, "timeout", False
    if isinstance(e, litellm.InternalServerError):
        return 502, "upstream_error", False
    if isinstance(e, litellm.APIConnectionError) or "No deployments available" in str(e):
        return 503, "service_unavailable", False
    status = getattr(e, "status_code", 502)
    return (status if isinstance(status, int) and 400 <= status < 600 else 502), "upstream_error", False


def oai_error(status: int, etype: str, msg: str, code: str | None = None) -> dict:
    return {"error": {"message": msg, "type": etype, "code": code or etype}}


def ant_error(etype: str, msg: str) -> dict:
    return {"type": "error", "error": {"type": etype, "message": msg}}


def replica_of(obj) -> str | None:
    hp = getattr(obj, "_hidden_params", None)
    if hp is None and isinstance(obj, dict):
        hp = obj.get("_hidden_params")
    return (hp or {}).get("model_id")


async def oai_stream(first, rest, flight: Flight, hub: Hub):
    """SSE for chat completions; `first` was already fetched (so errors before any byte become
    proper HTTP statuses). After the first byte, an error ends the stream with an error event."""
    cid = getattr(first, "id", None)
    try:
        yield f"data: {first.model_dump_json(exclude_none=True)}\n\n".encode()
        async for ch in rest:
            if cid and getattr(ch, "id", cid) != cid:
                yield _sse_err("stream_restarted", "upstream restarted the response; aborting to avoid replay")
                return
            yield f"data: {ch.model_dump_json(exclude_none=True)}\n\n".encode()
        yield b"data: [DONE]\n\n"
    except Exception as e:  # noqa: BLE001
        status, etype, cap = classify(e)
        if cap:
            hub.capacity_error(flight.group)
        yield _sse_err(etype, f"stream interrupted: {type(e).__name__}")
    finally:
        hub.end(flight)


def _sse_err(etype: str, msg: str) -> bytes:
    return f"data: {json.dumps(oai_error(0, etype, msg))}\n\n".encode()


async def ant_stream(first: bytes, rest, flight: Flight, hub: Hub):
    starts = first.count(b"event: message_start")
    try:
        yield first
        async for ch in rest:
            if isinstance(ch, str):
                ch = ch.encode()
            starts += ch.count(b"event: message_start")
            if starts > 1:
                yield _ant_sse_err("api_error", "upstream restarted the response; aborting to avoid replay")
                return
            yield ch
    except Exception as e:  # noqa: BLE001
        status, etype, cap = classify(e)
        if cap:
            hub.capacity_error(flight.group)
        yield _ant_sse_err("overloaded_error" if cap else "api_error", f"stream interrupted: {type(e).__name__}")
    finally:
        hub.end(flight)


def _ant_sse_err(etype: str, msg: str) -> bytes:
    return f"event: error\ndata: {json.dumps(ant_error(etype, msg))}\n\n".encode()
