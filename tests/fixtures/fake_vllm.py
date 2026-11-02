#!/usr/bin/env python3
"""Stand-in for `vllm serve`: OpenAI + Anthropic routes, SSE, tool calls, /metrics.

Behaviour is driven by a JSON control file that is re-read on every request, so a
test can flip modes on a running server:

    {"mode": "ok" | "http500" | "http429" | "hang" | "break_stream" | "down",
     "break_after": 2,          # content chunks emitted before break_stream drops the connection
     "delay": 0.0,              # seconds slept before answering
     "chunk_delay": 0.0,        # seconds between stream chunks
     "metrics": {"vllm:num_requests_waiting": 3, ...},   # gauges to report
     "no_metrics": false}

Every request is appended to <ctl>.log as one JSON line (path, auth ok, stream,
model) without bodies, so tests can count backend calls.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse
from starlette.routing import Route

A = argparse.Namespace()
STARTED = time.monotonic()
INFLIGHT = 0


def ctl() -> dict:
    try:
        return json.loads(Path(A.ctl).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def record(req: Request, **kv) -> None:
    with open(A.ctl + ".log", "a") as f:
        f.write(json.dumps({"t": time.time(), "path": req.url.path, "port": A.port, **kv}) + "\n")


def authed(req: Request) -> bool:
    if not A.key:
        return True
    got = req.headers.get("authorization", "")
    return got == f"Bearer {A.key}" or req.headers.get("x-api-key") == A.key


def loading() -> bool:
    return time.monotonic() - STARTED < A.load_s


async def health(req: Request) -> Response:
    if loading() or ctl().get("mode") == "down":
        return Response(status_code=503)
    return Response(status_code=200)


async def models(req: Request) -> Response:
    if not authed(req):
        return JSONResponse({"error": "unauthorized"}, status_code=401)
    return JSONResponse({"object": "list", "data": [{
        "id": A.model, "object": "model", "owned_by": "vllm", "root": A.model,
        "max_model_len": A.max_model_len}]})


async def metrics(req: Request) -> Response:
    c = ctl()
    if c.get("no_metrics"):
        return Response(status_code=404)
    g = {"vllm:num_requests_running": INFLIGHT, "vllm:num_requests_waiting": 0,
         "vllm:kv_cache_usage_perc": 0.1, "vllm:num_preemptions_total": 0}
    g.update(c.get("metrics", {}))
    lines = [f'{k}{{model_name="{A.model}"}} {v}' for k, v in g.items()]
    return PlainTextResponse("\n".join(lines) + "\n")


async def gate(req: Request, body: dict) -> Response | None:
    """Common failure modes; returns a response to send instead, or None."""
    c = ctl()
    if not authed(req):
        record(req, auth=False)
        return JSONResponse({"error": {"message": "unauthorized"}}, status_code=401)
    record(req, auth=True, stream=bool(body.get("stream")), model=body.get("model"),
           beta=req.headers.get("anthropic-beta"), keys=sorted(body))
    if body.get("model") != A.model:
        return JSONResponse({"error": {"message": f"model {body.get('model')} not served"}}, status_code=404)
    if c.get("delay"):
        await asyncio.sleep(c["delay"])
    mode = c.get("mode", "ok")
    if mode == "http500":
        return JSONResponse({"error": {"message": "stub 500"}}, status_code=500)
    if mode == "http429":
        return JSONResponse({"error": {"message": "stub overloaded"}}, status_code=429)
    if mode == "hang":
        await asyncio.sleep(3600)
    return None


def chunked(text: str, n: int = 4) -> list[str]:
    k = max(1, len(text) // n)
    return [text[i:i + k] for i in range(0, len(text), k)]


class Break(Exception):
    pass


async def stream_body(events, c: dict):
    global INFLIGHT
    INFLIGHT += 1
    try:
        for i, ev in enumerate(events):
            if c.get("mode") == "break_stream" and i == c.get("break_after", 2):
                raise Break("stub stream break")   # uvicorn drops the connection mid-body
            yield ev.encode()
            if c.get("chunk_delay"):
                await asyncio.sleep(c["chunk_delay"])
    finally:
        INFLIGHT -= 1


async def chat(req: Request) -> Response:
    body = await req.json()
    if (r := await gate(req, body)) is not None:
        return r
    c = ctl()
    cid, created = "chatcmpl-" + uuid.uuid4().hex[:12], int(time.time())
    tools = body.get("tools")
    text = f"hello from {A.name}"
    call = {"id": "call_" + uuid.uuid4().hex[:8], "type": "function",
            "function": {"name": tools[0]["function"]["name"], "arguments": '{"city": "Thuwal"}'}} if tools else None

    def frame(delta, finish=None):
        return "data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created, "model": A.model,
                                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}) + "\n\n"

    if body.get("stream"):
        ev = [frame({"role": "assistant", "content": ""}), frame({"reasoning_content": "thinking"})]
        ev += [frame({"content": p}) for p in chunked(text)]
        if call:
            ev.append(frame({"tool_calls": [{"index": 0, "id": call["id"], "type": "function",
                                             "function": {"name": call["function"]["name"], "arguments": ""}}]}))
            ev.append(frame({"tool_calls": [{"index": 0, "function": {"arguments": call["function"]["arguments"]}}]}))
        ev.append(frame({}, "tool_calls" if call else "stop"))
        if (body.get("stream_options") or {}).get("include_usage"):
            ev.append("data: " + json.dumps({"id": cid, "object": "chat.completion.chunk", "created": created,
                                             "model": A.model, "choices": [],
                                             "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}}) + "\n\n")
        ev.append("data: [DONE]\n\n")
        return StreamingResponse(stream_body(ev, c), media_type="text/event-stream")
    msg = {"role": "assistant", "content": text, "reasoning_content": "thinking"}
    if call:
        msg["tool_calls"] = [call]
    return JSONResponse({"id": cid, "object": "chat.completion", "created": created, "model": A.model,
                         "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if call else "stop"}],
                         "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12}})


async def messages(req: Request) -> Response:
    body = await req.json()
    if (r := await gate(req, body)) is not None:
        return r
    c = ctl()
    mid = "msg_" + uuid.uuid4().hex[:12]
    tools = body.get("tools")
    text = f"hello from {A.name}"
    tool = {"type": "tool_use", "id": "toolu_" + uuid.uuid4().hex[:8], "name": tools[0]["name"],
            "input": {"city": "Thuwal"}} if tools else None
    stop = "tool_use" if tool else "end_turn"
    if body.get("stream"):
        def sse(event, data):
            return f"event: {event}\ndata: {json.dumps(data)}\n\n"
        ev = [sse("message_start", {"type": "message_start", "message": {
                  "id": mid, "type": "message", "role": "assistant", "model": A.model, "content": [],
                  "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 5, "output_tokens": 0}}}),
              sse("content_block_start", {"type": "content_block_start", "index": 0,
                                          "content_block": {"type": "text", "text": ""}})]
        ev += [sse("content_block_delta", {"type": "content_block_delta", "index": 0,
                                           "delta": {"type": "text_delta", "text": p}}) for p in chunked(text)]
        ev.append(sse("content_block_stop", {"type": "content_block_stop", "index": 0}))
        if tool:
            ev.append(sse("content_block_start", {"type": "content_block_start", "index": 1, "content_block": {
                "type": "tool_use", "id": tool["id"], "name": tool["name"], "input": {}}}))
            ev.append(sse("content_block_delta", {"type": "content_block_delta", "index": 1, "delta": {
                "type": "input_json_delta", "partial_json": json.dumps(tool["input"])}}))
            ev.append(sse("content_block_stop", {"type": "content_block_stop", "index": 1}))
        ev.append(sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                                        "usage": {"output_tokens": 7}}))
        ev.append(sse("message_stop", {"type": "message_stop"}))
        return StreamingResponse(stream_body(ev, c), media_type="text/event-stream")
    content = [{"type": "text", "text": text}] + ([tool] if tool else [])
    return JSONResponse({"id": mid, "type": "message", "role": "assistant", "model": A.model, "content": content,
                         "stop_reason": stop, "stop_sequence": None,
                         "usage": {"input_tokens": 5, "output_tokens": 7}})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--key-file", default=os.environ.get("SSERVE_KEY_FILE", ""))
    ap.add_argument("--ctl", required=True)
    ap.add_argument("--name", default="")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--load-s", type=float, default=0.0, help="seconds /health answers 503 (model loading)")
    ap.parse_args(namespace=A)
    A.key = Path(A.key_file).read_text().strip() if A.key_file else ""
    A.name = A.name or f"port{A.port}"
    app = Starlette(routes=[
        Route("/health", health), Route("/metrics", metrics), Route("/v1/models", models),
        Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/messages", messages, methods=["POST"]),
    ])
    uvicorn.run(app, host=A.host, port=A.port, log_level="warning")


if __name__ == "__main__":
    main()
