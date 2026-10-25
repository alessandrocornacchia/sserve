#!/usr/bin/env python3
"""A stand-in for `vllm serve` on the test cluster: the OpenAI endpoints sserve
and examples/client.py use, answering every chat with an echo of the prompt.

    /health                 200, no key needed (as vLLM)
    /v1/models              the served model
    /v1/chat/completions    echo of the last user message, streamed or not

Requires `Authorization: Bearer $VLLM_API_KEY` on /v1/* when VLLM_API_KEY is set.
"""

import argparse
import json
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "sserve/mock-llm"
KEY = os.environ.get("VLLM_API_KEY", "")


class Handler(BaseHTTPRequestHandler):
    def send_json(self, code: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def authorized(self) -> bool:
        if KEY and self.headers.get("Authorization") != f"Bearer {KEY}":
            self.send_json(401, {"error": {"message": "Unauthorized", "code": 401}})
            return False
        return True

    def do_GET(self) -> None:
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/v1/models":
            if self.authorized():
                self.send_json(200, {"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "sserve"}]})
        else:
            self.send_json(404, {"error": {"message": f"no route {self.path}", "code": 404}})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            return self.send_json(404, {"error": {"message": f"no route {self.path}", "code": 404}})
        if not self.authorized():
            return
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        prompt = next((m.get("content", "") for m in reversed(req.get("messages", [])) if m.get("role") == "user"), "")
        answer = f"echo: {prompt}"
        base = {"id": f"chatcmpl-{time.time_ns()}", "created": int(time.time()), "model": MODEL}
        if not req.get("stream"):
            n = len(answer.split())
            return self.send_json(200, {**base, "object": "chat.completion",
                                        "choices": [{"index": 0, "finish_reason": "stop",
                                                     "message": {"role": "assistant", "content": answer}}],
                                        "usage": {"prompt_tokens": len(prompt.split()), "completion_tokens": n}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for word in answer.split(" "):
            chunk = {**base, "object": "chat.completion.chunk",
                     "choices": [{"index": 0, "delta": {"content": word + " "}, "finish_reason": None}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    server = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"mock vLLM serving {MODEL} on {a.host}:{a.port} (key {'required' if KEY else 'not set'})", flush=True)
    server.serve_forever()
