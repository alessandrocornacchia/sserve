"""Clients cannot steer LiteLLM's routing, transport, credentials or retries through the body."""
from __future__ import annotations

import http.server
import threading

import httpx
import pytest

from conftest import MODEL, free_port


@pytest.fixture
def trap():
    hits = []

    class H(http.server.BaseHTTPRequestHandler):
        def _any(self):
            hits.append(self.path)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"choices":[{"message":{"role":"assistant","content":"TRAPPED"}}]}')
        do_GET = do_POST = _any

        def log_message(self, *a):
            pass
    port = free_port()
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", hits
    srv.shutdown()
    srv.server_close()


def test_control_kwargs_rejected_and_nothing_reaches_client_url(stack, trap):
    url, hits = trap
    lease, jid = stack.ready_one()
    key = lease["inference_key"]
    msgs = [{"role": "user", "content": "hi"}]
    evil = [
        {"api_base": url + "/v1"}, {"base_url": url + "/v1"}, {"api_key": "x"},
        {"custom_llm_provider": "openai"}, {"model_list": [{"model_name": MODEL, "litellm_params": {
            "model": "openai/x", "api_base": url}}]},
        {"fallbacks": [{"model": "openai/x", "api_base": url}]}, {"num_retries": 50}, {"specific_deployment": True},
        {"extra_headers": {"x": "y"}}, {"metadata": {"model_group": "x"}}, {"mock_response": "fake"},
        {"extra_body": {"api_base": url}}, {"litellm_params": {"api_base": url}}, {"timeout": 0.001},
        {"tools": [{"type": "mcp", "server_url": url, "server_label": "x"}]},
        {"stream_options": {"include_usage": True, "api_base": url}},
    ]
    n0 = len(stack.job_log(jid))
    for extra in evil:
        r = stack.chat_body(key, {"model": MODEL, "messages": msgs, **extra})
        assert r.status_code == 400, (extra, r.text)
        assert r.json()["error"]["code"] == "unsupported_parameter", extra
    ant_evil = [{"mcp_servers": [{"type": "url", "url": url, "name": "x"}]},
                {"tools": [{"type": "web_search_20250305", "name": "web_search"}]},
                {"api_base": url}, {"metadata": {"user_id": "u", "model_group": "x"}}, {"metadata": "x"}]
    for extra in ant_evil:
        r = httpx.post(stack.url + "/v1/messages", headers={"x-api-key": key},
                       json={"model": MODEL, "max_tokens": 8, "messages": msgs, **extra}, timeout=30)
        assert r.status_code == 400, (extra, r.text)
    assert hits == []
    assert len(stack.job_log(jid)) == n0                    # rejected before any upstream call
    # Claude Code's metadata.user_id is accepted (and not forwarded)
    r = httpx.post(stack.url + "/v1/messages", headers={"x-api-key": key},
                   json={"model": MODEL, "max_tokens": 8, "messages": msgs, "metadata": {"user_id": "u-1"}}, timeout=30)
    assert r.status_code == 200, r.text
    assert "metadata" not in stack.job_log(jid)[-1]["keys"]
    # allowed content that names a URL is passed to vLLM, never fetched by the manager
    r = stack.chat_body(key, {"model": MODEL, "messages": [{"role": "user", "content": [
        {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": url + "/img.png"}}]}]})
    assert r.status_code == 200, r.text
    assert hits == []


def test_vllm_sampling_extras_forwarded(stack):
    lease, jid = stack.ready_one()
    r = stack.chat_body(lease["inference_key"], {"model": MODEL, "messages": [{"role": "user", "content": "hi"}],
                                                 "top_k": 20, "chat_template_kwargs": {"enable_thinking": False}})
    assert r.status_code == 200, r.text
    keys = stack.job_log(jid)[-1]["keys"]
    assert "top_k" in keys and "chat_template_kwargs" in keys
