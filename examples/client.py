#!/usr/bin/env python3
"""Smoke-test an sserve endpoint: list the models, then send one chat request.

Reads the variables that `eval "$(sserve env <model>)"` exports:
OPENAI_BASE_URL, OPENAI_API_KEY and SSERVE_MODEL. Standard library only.

    python3 examples/client.py
    python3 examples/client.py "Write a haiku about GPUs" --stream
"""

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE = KEY = ""


def env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        sys.exit(f'{name} is not set: run  eval "$(sserve env <model>)"  first')
    return val


def call(path: str, body: dict | None = None):
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    )
    try:
        return urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as e:
        sys.exit(f"{path}: HTTP {e.code}: {e.read().decode(errors='replace')[:500]}")
    except OSError as e:
        sys.exit(f"{path}: {e} (is the tunnel up? `sserve status`)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("prompt", nargs="?", default="In one sentence, what is SLURM?")
    ap.add_argument("--stream", action="store_true", help="print tokens as they arrive")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--reasoning", action="store_true", help="print the reasoning trace, if the server returns one")
    a = ap.parse_args()
    global BASE, KEY
    BASE, KEY = env("OPENAI_BASE_URL").rstrip("/"), env("OPENAI_API_KEY")

    with call("/models") as r:
        models = [m["id"] for m in json.load(r)["data"]]
    print(f"models: {', '.join(models)}")
    model = os.environ.get("SSERVE_MODEL") or models[0]

    body = {"model": model, "max_tokens": a.max_tokens, "stream": a.stream,
            "messages": [{"role": "user", "content": a.prompt}]}
    t0 = time.monotonic()
    if a.stream:
        with call("/chat/completions", body) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0]["delta"]
                print(delta.get("content") or "", end="", flush=True)
        print(f"\n\n({time.monotonic() - t0:.1f}s)")
        return

    with call("/chat/completions", body) as r:
        out = json.load(r)
    msg, usage = out["choices"][0]["message"], out.get("usage", {})
    reasoning = msg.get("reasoning_content") or msg.get("reasoning")  # with vLLM's --reasoning-parser
    if reasoning and a.reasoning:
        print(f"\n[reasoning]\n{reasoning.strip()}")
    elif reasoning:
        print(f"[reasoning: {len(reasoning)} chars, --reasoning to show]")
    print(f"\n[answer]\n{(msg.get('content') or '').strip()}")
    print(f"\n({time.monotonic() - t0:.1f}s, {usage.get('prompt_tokens', '?')} prompt + "
          f"{usage.get('completion_tokens', '?')} completion tokens)")


if __name__ == "__main__":
    main()
