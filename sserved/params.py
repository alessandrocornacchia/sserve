"""What a client request may carry into LiteLLM.

Request bodies are passed to LiteLLM as keyword arguments, and LiteLLM treats
many keywords as *its own* controls: api_base/base_url/api_key (where the
request goes and with which credential), custom_llm_provider, model_list,
fallbacks, num_retries, specific_deployment, metadata (router internals),
mock_response, extra_headers, MCP tools/servers, ... A client must never be
able to steer routing, transport, retries or credentials, so the body is
rebuilt from an explicit allowlist of API parameters. Anything else is
rejected with 400 (not silently dropped), so the client knows.

vLLM-specific sampling fields (sent top-level by the OpenAI SDK's extra_body)
are forwarded to vLLM inside `extra_body`, from a separate allowlist.
"""
from __future__ import annotations

from typing import Any


class BadParams(Exception):
    pass


OPENAI_CHAT = {
    "messages", "stream", "stream_options", "temperature", "top_p", "n", "stop", "max_tokens",
    "max_completion_tokens", "presence_penalty", "frequency_penalty", "logit_bias", "logprobs", "top_logprobs",
    "user", "seed", "tools", "tool_choice", "parallel_tool_calls", "response_format", "reasoning_effort",
}
# forwarded to vLLM via extra_body (sampling / chat-template knobs only)
VLLM_EXTRA = {
    "top_k", "min_p", "repetition_penalty", "length_penalty", "min_tokens", "stop_token_ids", "ignore_eos",
    "skip_special_tokens", "spaces_between_special_tokens", "include_stop_str_in_output",
    "chat_template_kwargs", "add_generation_prompt", "continue_final_message", "echo",
    "guided_json", "guided_regex", "guided_choice", "guided_grammar", "structured_outputs",
}
ANTHROPIC_MESSAGES = {
    "messages", "system", "max_tokens", "stop_sequences", "stream", "temperature", "top_p", "top_k",
    "tools", "tool_choice", "thinking", "context_management", "output_format", "service_tier",
}
STREAM_OPTIONS = {"include_usage", "continuous_usage_stats"}
ANTHROPIC_TOOL_TYPES = {None, "custom"}     # server-side tools (web search, MCP, ...) are not offered


def _check_unknown(body: dict, allowed: set[str], route: str) -> None:
    bad = sorted(k for k in body if k not in allowed and k != "model")
    if bad:
        raise BadParams(f"unsupported parameter(s) for {route}: {', '.join(bad)}")


def openai_chat(body: dict) -> dict[str, Any]:
    _check_unknown(body, OPENAI_CHAT | VLLM_EXTRA, "/v1/chat/completions")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise BadParams("messages must be a non-empty list")
    out = {k: body[k] for k in OPENAI_CHAT if k in body}
    so = out.get("stream_options")
    if so is not None:
        if not isinstance(so, dict) or set(so) - STREAM_OPTIONS:
            raise BadParams(f"stream_options may only contain {sorted(STREAM_OPTIONS)}")
    for t in out.get("tools") or []:
        if not isinstance(t, dict) or t.get("type") != "function" or not isinstance(t.get("function"), dict):
            raise BadParams("only tools of type 'function' are supported")
    if "tools" in out and not isinstance(out["tools"], list):
        raise BadParams("tools must be a list")
    extra = {k: body[k] for k in VLLM_EXTRA if k in body}
    if extra:
        out["extra_body"] = extra
    return out


def anthropic_messages(body: dict) -> dict[str, Any]:
    md = body.get("metadata")
    if md is not None:
        # Claude Code sends metadata.user_id. LiteLLM uses `metadata` for router internals, so it is
        # accepted only in that exact shape and not forwarded (vLLM does not use it).
        if not (isinstance(md, dict) and set(md) <= {"user_id"} and isinstance(md.get("user_id", ""), str)):
            raise BadParams("metadata may only contain user_id (string)")
        body = {k: v for k, v in body.items() if k != "metadata"}
    _check_unknown(body, ANTHROPIC_MESSAGES, "/v1/messages")
    if not isinstance(body.get("messages"), list) or not body["messages"]:
        raise BadParams("messages must be a non-empty list")
    if not isinstance(body.get("max_tokens"), int):
        raise BadParams("max_tokens (integer) is required")
    out = {k: body[k] for k in ANTHROPIC_MESSAGES if k in body}
    if "tools" in out and not isinstance(out["tools"], list):
        raise BadParams("tools must be a list")
    for t in out.get("tools") or []:
        if not isinstance(t, dict) or t.get("type") not in ANTHROPIC_TOOL_TYPES or "name" not in t:
            raise BadParams("only custom (client-executed) tools are supported")
    return out


def beta_header(v: str | None) -> dict[str, str]:
    """anthropic-beta is forwarded to vLLM as a plain header value; nothing else from the client is."""
    if not v:
        return {}
    if len(v) > 512 or any(c in v for c in "\r\n\0"):
        raise BadParams("invalid anthropic-beta header")
    return {"anthropic-beta": v}
