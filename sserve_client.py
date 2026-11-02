"""Client for an sserved manager: stdlib only, usable from any machine.

Needs the manager URL and a client token (from the operator), via flags, the
environment (SSERVE_MANAGER_URL, SSERVE_MANAGER_TOKEN or SSERVE_MANAGER_TOKEN_FILE)
or the [manager] table of ~/.config/sserve/config.toml (url, token_file).
No cluster credentials are involved.

Python:
    from sserve_client import Client, lease
    with lease("ornith-ai/Ornith-1.5-122B", wait=3600) as l:
        subprocess.run(cmd, env={**os.environ, **l.env})

CLI: `sserve lease ...` (see `sserve lease -h`).
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    import tomllib
except ImportError:  # Python < 3.11
    tomllib = None

CONFIG_PATH = Path(os.environ.get("SSERVE_CONFIG", "~/.config/sserve/config.toml")).expanduser()
LEASE_DIR = Path(os.environ.get("SSERVE_LEASE_DIR", "~/.config/sserve/leases")).expanduser()

EX_UNAVAILABLE, EX_TEMPFAIL, EX_NOPERM = 69, 75, 77


class ManagerError(Exception):
    def __init__(self, status: int, code: str, message: str):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.message = status, code, message


def settings(url: str | None = None, token: str | None = None, token_file: str | None = None) -> tuple[str, str]:
    cfg = {}
    if CONFIG_PATH.exists() and tomllib is not None:
        with CONFIG_PATH.open("rb") as f:
            cfg = tomllib.load(f).get("manager", {})
    url = url or os.environ.get("SSERVE_MANAGER_URL") or cfg.get("url")
    token = token or os.environ.get("SSERVE_MANAGER_TOKEN")
    tf = token_file or os.environ.get("SSERVE_MANAGER_TOKEN_FILE") or cfg.get("token_file")
    if not token and tf:
        p = Path(tf).expanduser()
        if p.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
            print(f"sserve: warning: {p} is readable by others; chmod 600 it", file=sys.stderr)
        token = p.read_text().strip()
    if not url:
        raise ManagerError(0, "no_manager", "no manager URL: set SSERVE_MANAGER_URL or [manager] url in "
                                            f"{CONFIG_PATH}")
    if not token:
        raise ManagerError(0, "no_token", "no client token: set SSERVE_MANAGER_TOKEN(_FILE) or [manager] token_file")
    return url.rstrip("/"), token


class Client:
    def __init__(self, url: str | None = None, token: str | None = None, token_file: str | None = None,
                 timeout: float = 30):
        self.url, self.token = settings(url, token, token_file)
        self.timeout = timeout

    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.url + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self.token}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read() or b"{}")
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read()).get("error", {})
            except (json.JSONDecodeError, AttributeError):
                err = {}
            raise ManagerError(e.code, err.get("code", "http_error"), err.get("message", e.reason)) from None
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ManagerError(0, "unreachable", f"manager {self.url} unreachable: {e}") from None

    def acquire(self, model: str, cluster: str | None = None, gpus: int | None = None,
                overrides: dict | None = None, ttl_s: float | None = None, idle_timeout_s: float | None = None,
                prep_grace_s: float | None = None, label: str | None = None,
                acquire_timeout_s: float | None = None) -> dict:
        body = {"model": model, "cluster": cluster, "gpus": gpus, "overrides": overrides or {}, "ttl_s": ttl_s,
                "idle_timeout_s": idle_timeout_s, "prep_grace_s": prep_grace_s, "label": label,
                "acquire_timeout_s": acquire_timeout_s}
        return self._req("POST", "/api/v1/leases", {k: v for k, v in body.items() if v is not None})

    def renew(self, lease_id: str, ttl_s: float | None = None) -> dict:
        return self._req("POST", f"/api/v1/leases/{lease_id}/renew", {"ttl_s": ttl_s} if ttl_s else {})

    def release(self, lease_id: str) -> dict:
        return self._req("DELETE", f"/api/v1/leases/{lease_id}")

    def get(self, lease_id: str) -> dict:
        return self._req("GET", f"/api/v1/leases/{lease_id}")

    def leases(self) -> list[dict]:
        return self._req("GET", "/api/v1/leases")["leases"]

    def rotate_key(self, lease_id: str) -> dict:
        return self._req("POST", f"/api/v1/leases/{lease_id}/rotate-key")

    def status(self) -> dict:
        return self._req("GET", "/api/v1/status")


def lease_env(info: dict, key: str) -> dict[str, str]:
    """What an unmodified OpenAI- or Anthropic-SDK client needs."""
    return {"OPENAI_BASE_URL": info["openai_base_url"], "OPENAI_API_KEY": key,
            "ANTHROPIC_BASE_URL": info["anthropic_base_url"], "ANTHROPIC_AUTH_TOKEN": key,
            "SSERVE_MODEL": info["model"], "SSERVE_LEASE_ID": info["lease_id"]}


class Lease:
    """An acquired lease with background renewal. Use as a context manager (releases on exit)."""

    def __init__(self, client: Client, info: dict, key: str, ttl_s: float, renew: bool = True):
        self.client, self.info, self.key, self.ttl = client, info, key, ttl_s
        self.lost: str | None = None
        self._stop = threading.Event()
        if renew:
            threading.Thread(target=self._renew_loop, daemon=True).start()

    @property
    def id(self) -> str:
        return self.info["lease_id"]

    @property
    def env(self) -> dict[str, str]:
        return lease_env(self.info, self.key)

    def _renew_loop(self) -> None:
        period = max(10.0, self.ttl / 3)
        while not self._stop.wait(period):
            try:
                self.info = {**self.info, **self.client.renew(self.id, self.ttl)}
            except ManagerError as e:
                if e.status in (404, 410):
                    self.lost = f"{e.code}: {e.message}"
                    print(f"sserve: lease {self.id} ended: {e.message}", file=sys.stderr)
                    return
                print(f"sserve: lease renewal failed ({e}); retrying", file=sys.stderr)
                period = 10.0
            else:
                period = max(10.0, self.ttl / 3)

    def wait_ready(self, timeout: float, poll: float = 10, quiet: bool = False) -> dict:
        """Block until the model has a ready replica. Raises ManagerError on a blocked deployment
        or when the lease ends; TimeoutError after `timeout` (the shared job is not touched)."""
        deadline, last = time.monotonic() + timeout, None
        while True:
            info = self.client.get(self.id)
            svc = info["service"]
            if info["state"] != "active":
                raise ManagerError(410, f"lease_{info['state']}", info.get("end_reason") or "lease ended")
            if svc["state"] == "blocked":
                raise ManagerError(503, "deployment_blocked", svc["blocked"]["reason"])
            if svc["ready_replicas"] > 0:
                return info
            line = describe(svc)
            if not quiet and line != last:
                print(f"sserve: {line}", file=sys.stderr)
                last = line
            if time.monotonic() > deadline:
                raise TimeoutError(f"not ready after {timeout:.0f}s: {line}")
            time.sleep(poll)

    def close(self, release: bool = True) -> None:
        self._stop.set()
        if release and not self.lost:
            try:
                self.client.release(self.id)
            except ManagerError as e:
                print(f"sserve: release of {self.id} failed: {e}", file=sys.stderr)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def lease(model: str, *, wait: float | None = None, client: Client | None = None, ttl_s: float = 300,
          **kw) -> Lease:
    c = client or Client()
    info = c.acquire(model, ttl_s=ttl_s, **kw)
    lz = Lease(c, info, info.pop("inference_key"), ttl_s)
    if wait:
        try:
            lz.wait_ready(wait)
        except BaseException:
            lz.close()
            raise
    return lz


def describe(svc: dict) -> str:
    jobs = svc.get("jobs") or []
    parts = [f"{svc['model']}: {svc['state']}"]
    for j in jobs:
        bit = f"{j['profile']} job {j['job_id'] or '?'} {j['slurm_state'] or j['state']}"
        if j.get("reason") and j["slurm_state"] == "PENDING":
            bit += f" ({j['reason']})"
        elif j.get("node"):
            bit += f" on {j['node']}"
        parts.append(bit)
    if svc.get("blocked"):
        parts.append("BLOCKED: " + svc["blocked"]["reason"])
    return "; ".join(parts)


# ------------------------------------------------------------------- saved leases
def _save(name: str, data: dict) -> Path:
    LEASE_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(LEASE_DIR, 0o700)
    p = LEASE_DIR / f"{name}.json"
    tmp = p.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, p)
    return p


def _load(ref: str) -> dict | None:
    p = LEASE_DIR / f"{ref}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _ref(ref: str) -> tuple[str, dict | None]:
    saved = _load(ref)
    return (saved["lease_id"] if saved else ref), saved


def _export(env: dict[str, str]) -> str:
    import shlex
    return "\n".join(f"export {k}={shlex.quote(v)}" for k, v in env.items())


# ---------------------------------------------------------------------------- CLI
def cli(argv: list[str]) -> int:
    cmd_argv: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, cmd_argv = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(prog="sserve lease", description="Reserve a model on an sserved manager.")
    ap.add_argument("--manager", help="manager URL (default: $SSERVE_MANAGER_URL or [manager] url)")
    ap.add_argument("--token-file", help="client token file (default: $SSERVE_MANAGER_TOKEN(_FILE) or [manager] token_file)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def lease_opts(p):
        p.add_argument("model", help="model name as configured by the operator (HF id)")
        p.add_argument("--cluster")
        p.add_argument("--gpus", type=int)
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="allowed launch override")
        p.add_argument("--ttl", type=float, default=300, help="lease TTL in seconds (renewed automatically by run)")
        p.add_argument("--idle-timeout", type=float)
        p.add_argument("--prep-grace", type=float, help="seconds after the model first becomes ready before idle counts")
        p.add_argument("--acquire-timeout", type=float, help="end the lease if the model is not ready by then")
        p.add_argument("--label")

    aq = sub.add_parser("acquire", help="acquire a lease; prints exports, or saves it with --save")
    lease_opts(aq)
    aq.add_argument("--save", metavar="NAME", help=f"store lease id and key in {LEASE_DIR}/NAME.json (mode 600)")
    aq.add_argument("--wait", type=float, metavar="SECONDS", help="wait until ready")
    rn = sub.add_parser("renew")
    rn.add_argument("ref", help="saved name or lease id")
    rn.add_argument("--ttl", type=float)
    for name in ("release", "env", "rotate-key"):
        p = sub.add_parser(name)
        p.add_argument("ref")
        if name == "rotate-key":
            p.add_argument("--save", metavar="NAME")
    wt = sub.add_parser("wait")
    wt.add_argument("ref")
    wt.add_argument("--timeout", type=float, default=7200)
    stp = sub.add_parser("status")
    stp.add_argument("ref", nargs="?")
    stp.add_argument("--json", action="store_true")
    rnp = sub.add_parser("run", help="acquire, wait, run a command with the endpoint env, release on exit")
    lease_opts(rnp)
    rnp.add_argument("--wait-timeout", type=float, default=7200,
                     help="give up waiting after this long (releases this lease only; never cancels shared jobs)")
    rnp.add_argument("--no-wait", action="store_true")
    a = ap.parse_args(argv)
    if cmd_argv and a.cmd != "run":
        ap.error("`--` is only used by `run`")

    try:
        c = Client(a.manager, token_file=a.token_file)
        if a.cmd in ("acquire", "run"):
            ov = {}
            for kv in a.set:
                k, _, v = kv.partition("=")
                ov[k] = v
            info = c.acquire(a.model, cluster=a.cluster, gpus=a.gpus, overrides=ov, ttl_s=a.ttl,
                             idle_timeout_s=a.idle_timeout, prep_grace_s=a.prep_grace, label=a.label,
                             acquire_timeout_s=a.acquire_timeout)
            key = info.pop("inference_key")
            if a.cmd == "acquire":
                if a.save:
                    p = _save(a.save, {"lease_id": info["lease_id"], "key": key, "info": info, "manager": c.url})
                    print(f"lease {info['lease_id']} saved to {p}; renew with `sserve lease renew {a.save}` "
                          f"within {a.ttl:.0f}s", file=sys.stderr)
                if a.wait:
                    lz = Lease(c, info, key, a.ttl)      # renews while waiting
                    try:
                        lz.wait_ready(a.wait)
                    finally:
                        lz.close(release=False)
                if not a.save:
                    print(_export(lease_env(info, key)))
                    print(f"lease {info['lease_id']}: renew within {a.ttl:.0f}s "
                          f"(`sserve lease renew {info['lease_id']}`)", file=sys.stderr)
                return 0
            return _run(c, info, key, a, cmd_argv)
        lid, saved = _ref(a.ref) if getattr(a, "ref", None) else (None, None)
        if a.cmd == "renew":
            info = c.renew(lid, a.ttl)
            print(f"lease {lid} renewed until {time.strftime('%F %T', time.localtime(info['expires']))}", file=sys.stderr)
        elif a.cmd == "release":
            c.release(lid)
            if saved:
                (LEASE_DIR / f"{a.ref}.json").unlink(missing_ok=True)
            print(f"lease {lid} released", file=sys.stderr)
        elif a.cmd == "env":
            if not saved:
                print("sserve: the key is only known to whoever acquired the lease; use a name saved with "
                      "--save, or `sserve lease rotate-key <id>` to get a new key", file=sys.stderr)
                return 1
            info = c.get(lid)
            print(_export(lease_env(info, saved["key"])))
        elif a.cmd == "rotate-key":
            info = c.rotate_key(lid)
            key = info.pop("inference_key")
            if a.save or saved:
                _save(a.save or a.ref, {"lease_id": lid, "key": key, "info": info, "manager": c.url})
                print(f"new key saved; the previous key no longer works", file=sys.stderr)
            else:
                print(_export(lease_env(info, key)))
        elif a.cmd == "wait":
            Lease(c, c.get(lid), "", 0, renew=False).wait_ready(a.timeout)
        elif a.cmd == "status":
            if lid:
                info = c.get(lid)
                print(json.dumps(info, indent=2) if a.json else f"{lid} {info['state']}: {describe(info['service'])}")
            else:
                ls = c.leases()
                if a.json:
                    print(json.dumps(ls, indent=2))
                for l in [] if a.json else ls:
                    print(f"{l['lease_id']} {l['state']} expires {time.strftime('%T', time.localtime(l['expires']))} "
                          f"inflight={l['inflight']}: {describe(l['service'])}")
        return 0
    except ManagerError as e:
        print(f"sserve: {e.message} ({e.code})", file=sys.stderr)
        return EX_NOPERM if e.status in (401, 403) else EX_UNAVAILABLE
    except TimeoutError as e:
        print(f"sserve: {e}", file=sys.stderr)
        return EX_TEMPFAIL


def _run(c: Client, info: dict, key: str, a, cmd: list[str]) -> int:
    if not cmd:
        print("sserve: nothing to run; use `sserve lease run MODEL -- COMMAND ...`", file=sys.stderr)
        c.release(info["lease_id"])
        return 2
    lz = Lease(c, info, key, a.ttl)
    try:
        if not a.no_wait:
            lz.wait_ready(a.wait_timeout)
        print(f"sserve: {info['model']} ready at {info['openai_base_url']} (lease {lz.id})", file=sys.stderr)
        p = subprocess.Popen(cmd, env={**os.environ, **lz.env})
        def fwd(sig, _):
            p.send_signal(sig)
        for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(s, fwd)
        rc = p.wait()
        if lz.lost:
            print(f"sserve: note: the lease ended during the command ({lz.lost})", file=sys.stderr)
        return rc
    except ManagerError as e:
        print(f"sserve: {e.message} ({e.code})", file=sys.stderr)
        return EX_UNAVAILABLE
    except TimeoutError as e:
        print(f"sserve: {e}; releasing lease {lz.id} (shared jobs are not cancelled by this)", file=sys.stderr)
        return EX_TEMPFAIL
    except KeyboardInterrupt:
        return 130
    finally:
        lz.close()


if __name__ == "__main__":
    sys.exit(cli(sys.argv[1:]))
