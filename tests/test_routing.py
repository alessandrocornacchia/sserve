"""Inference through the manager: real LiteLLM 1.106.2 Router -> tunnel stand-in -> fake vLLM."""
from __future__ import annotations

import json
import os
import signal
import threading
import time

import httpx

from conftest import MODEL, wait_for

TOOLS_ANT = [{"name": "weather", "description": "w", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}]


def sse_data(text: str) -> list:
    out = []
    for line in text.splitlines():
        if line.startswith("data: "):
            d = line[6:]
            out.append(d if d == "[DONE]" else json.loads(d))
    return out


def ant(stack, key, stream=False, tools=True, headers=None, **kw):
    body = {"model": MODEL, "max_tokens": 64, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
    if tools:
        body["tools"] = TOOLS_ANT
    return httpx.post(stack.url + "/v1/messages", json=body, timeout=60,
                      headers={"x-api-key": key, "anthropic-version": "2023-06-01", **(headers or {})}, **kw)


def test_openai_nonstream_and_stream_tool_calls(stack):
    lease, jid = stack.ready_one()
    key = lease["inference_key"]
    r = stack.chat(key, tools=True)
    assert r.status_code == 200, r.text
    msg = r.json()["choices"][0]["message"]
    assert msg["tool_calls"][0]["function"] == {"name": "weather", "arguments": '{"city": "Thuwal"}'}
    assert msg["reasoning_content"] == "thinking"
    assert r.json()["choices"][0]["finish_reason"] == "tool_calls"
    r = stack.chat(key, stream=True, tools=True)
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = sse_data(r.text)
    assert ev[-1] == "[DONE]"
    chunks = ev[:-1]
    assert len({c["id"] for c in chunks}) == 1
    text = "".join(c["choices"][0]["delta"].get("content") or "" for c in chunks if c["choices"])
    assert text == f"hello from job{jid}"
    calls = [tc for c in chunks if c["choices"] for tc in c["choices"][0]["delta"].get("tool_calls") or []]
    assert calls[0]["function"]["name"] == "weather"
    assert "".join(tc["function"].get("arguments") or "" for tc in calls) == '{"city": "Thuwal"}'
    assert any(c["choices"][0]["delta"].get("reasoning_content") == "thinking" for c in chunks if c["choices"])
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"


def test_openai_sdk_unmodified_client(stack):
    import openai
    lease, _ = stack.ready_one()
    c = openai.OpenAI(base_url=lease["openai_base_url"], api_key=lease["inference_key"])
    out = c.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": "hi"}],
                                    tools=[{"type": "function", "function": {"name": "weather", "parameters": {"type": "object", "properties": {}}}}])
    assert out.choices[0].message.tool_calls[0].function.name == "weather"
    parts = [ch.choices[0].delta.content for ch in c.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": "hi"}], stream=True,
        stream_options={"include_usage": True}) if ch.choices]
    assert "".join(p or "" for p in parts).startswith("hello from")
    assert [m.id for m in c.models.list()] == [MODEL]


def test_anthropic_nonstream_and_stream_tool_use(stack):
    lease, jid = stack.ready_one()
    key = lease["inference_key"]
    r = ant(stack, key)
    assert r.status_code == 200, r.text
    b = r.json()
    assert "_hidden_params" not in b
    assert b["stop_reason"] == "tool_use"
    assert b["content"][1] == {"type": "tool_use", "id": b["content"][1]["id"], "name": "weather", "input": {"city": "Thuwal"}}
    r = ant(stack, key, stream=True, headers={"anthropic-beta": "fine-grained-tool-streaming-2025-05-14"})
    assert r.status_code == 200
    events = [l[7:] for l in r.text.splitlines() if l.startswith("event: ")]
    assert events[0] == "message_start" and events[-1] == "message_stop"
    assert events.count("message_start") == 1
    datas = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    deltas = [d["delta"] for d in datas if d["type"] == "content_block_delta"]
    assert "".join(x.get("text", "") for x in deltas) == f"hello from job{jid}"
    assert json.loads("".join(x.get("partial_json", "") for x in deltas)) == {"city": "Thuwal"}
    # native passthrough, not translated to chat completions
    log = stack.job_log(jid)
    paths = [e["path"] for e in log]
    assert paths.count("/v1/messages") == 2 and "/v1/chat/completions" not in paths
    assert log[-1]["beta"] == "fine-grained-tool-streaming-2025-05-14"   # anthropic-beta forwarded


def test_anthropic_sdk_unmodified_client(stack):
    import anthropic
    lease, _ = stack.ready_one()
    c = anthropic.Anthropic(base_url=lease["anthropic_base_url"], auth_token=lease["inference_key"])
    m = c.messages.create(model=MODEL, max_tokens=32, messages=[{"role": "user", "content": "hi"}], tools=TOOLS_ANT)
    assert m.stop_reason == "tool_use" and m.content[1].input == {"city": "Thuwal"}
    with c.messages.stream(model=MODEL, max_tokens=32, messages=[{"role": "user", "content": "hi"}]) as s:
        assert "".join(s.text_stream).startswith("hello from")


def test_no_upstream_retry_after_role_chunk(stack):
    """Stream breaks right after the first (role / message_start) chunk: exactly one upstream request
    per protocol, even though LiteLLM could retry internally."""
    lease, jid = stack.ready_one()
    for route in ("oai", "ant"):
        stack.job_ctl(jid, mode="break_stream", break_after=1)
        n0 = len(stack.job_log(jid))
        r = stack.chat(lease["inference_key"], stream=True) if route == "oai" else \
            ant(stack, lease["inference_key"], stream=True, tools=False)
        assert len(stack.job_log(jid)) - n0 == 1, route
        assert "error" in r.text
    stack.job_ctl(jid)


def test_no_replay_after_first_byte(stack):
    lease, jid = stack.ready_one()
    stack.acquire("bob")
    key = lease["inference_key"]
    stack.job_ctl(jid, mode="break_stream", break_after=3)
    n0 = len(stack.job_log(jid))
    r = stack.chat(key, stream=True)
    ev = sse_data(r.text)
    assert r.status_code == 200
    assert "[DONE]" not in ev and "error" in ev[-1]                      # explicit, not a fake success
    assert len(stack.job_log(jid)) - n0 == 1                              # exactly one upstream request
    n0 = len(stack.job_log(jid))
    r = ant(stack, key, stream=True, tools=False)
    assert "event: error" in r.text
    assert r.text.count("event: message_start") <= 1
    # LiteLLM may retry *before* any content (lifecycle frames are buffered); never after
    datas = [json.loads(l[6:]) for l in r.text.splitlines() if l.startswith("data: ")]
    starts = sum(1 for d in datas if d.get("type") == "message_start")
    assert starts <= 1


def test_membership_auth_and_model_isolation(stack):
    lease, jid = stack.ready_one()
    r = stack.chat(lease["inference_key"], model="other/model")
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_found"
    r = stack.chat("ssk_nope")
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"
    r = stack.chat(stack.tok["alice"])                    # a client token is not an inference key
    assert r.status_code == 401
    r = httpx.get(stack.url + "/v1/models", headers={"Authorization": f"Bearer {lease['inference_key']}"})
    assert r.json()["data"][0]["max_model_len"] == 4096   # from the backend card (NIKA reads this)
    assert stack.api("GET", "/api/v1/status", "bogus").status_code == 401


def test_dead_backend_is_503_without_delivery(stack):
    lease, jid = stack.ready_one()
    key = lease["inference_key"]
    stack.job_ctl(jid, mode="down")                       # /health 503: unrouted after 2 probes
    wait_for(lambda: stack.dep()["ready_replicas"] == 0, what="unrouted")
    n0 = len(stack.job_log(jid))
    for _ in range(5):
        r = stack.chat(key)
        assert r.status_code == 503 and r.json()["error"]["code"] == "no_ready_replica"
    assert len(stack.job_log(jid)) == n0                  # nothing reached the known-dead backend
    stack.job_ctl(jid)                                    # recovers: re-admitted, same job
    wait_for(lambda: stack.dep()["ready_replicas"] == 1, what="re-admitted")
    assert stack.chat(key).status_code == 200
    assert stack.jobs()[jid]["state"] == "RUNNING"


def test_tunnel_loss_recovers_tunnel_not_job(stack):
    lease, jid = stack.ready_one()
    fwd = [p for p in _pids_with("fwd.py") if str(stack.tmp) in _environ(p)]
    assert len(fwd) == 1
    os.kill(fwd[0], signal.SIGKILL)
    wait_for(lambda: [p for p in _pids_with("fwd.py") if str(stack.tmp) in _environ(p) and p != fwd[0]],
             what="tunnel restarted")
    wait_for(lambda: stack.chat(lease["inference_key"]).status_code == 200, what="served again")
    assert stack.jobs()[jid]["state"] == "RUNNING"
    assert len(stack.jobs()) == 2                          # no replacement allocation (only the cancelled sibling)


def _pids_with(s: str) -> list[int]:
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if s in open(f"/proc/{d}/cmdline").read():
                    out.append(int(d))
            except OSError:
                pass
    return out


def _environ(pid: int) -> str:
    try:
        return open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
    except OSError:
        return ""
