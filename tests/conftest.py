"""Test stack: a real sserved process + fake SLURM + fake vLLM jobs + TCP forwarders as tunnels.

Every process a test starts is tracked and stopped in teardown; the teardown
asserts that nothing the stack started is left running.
"""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"
PY = sys.executable
MODEL = "org/fake-7b"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_for(fn, timeout=20.0, every=0.1, what="condition"):
    deadline = time.monotonic() + timeout
    last = None
    while time.monotonic() < deadline:
        try:
            last = fn()
            if last:
                return last
        except Exception as e:  # noqa: BLE001
            last = e
        time.sleep(every)
    raise AssertionError(f"timed out waiting for {what}; last={last!r}")


def toml_value(v):
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{k} = {toml_value(x)}" for k, x in v.items()) + "}"
    return json.dumps(v)


def to_toml(d: dict) -> str:
    out = []
    def section(prefix, tbl):
        scalars = {k: v for k, v in tbl.items() if not isinstance(v, dict) or k == "env"}
        subs = {k: v for k, v in tbl.items() if isinstance(v, dict) and k != "env"}
        if prefix and scalars:
            out.append(f"[{prefix}]")
        for k, v in scalars.items():
            out.append(f"{k} = {toml_value(v)}")
        for k, v in subs.items():
            section(f"{prefix}.{json.dumps(k)}" if prefix else k, v)
    section("", d)
    return "\n".join(out) + "\n"


class Stack:
    def __init__(self, tmp: Path, **over):
        self.tmp = tmp
        self.bin = tmp / "bin"
        self.bin.mkdir()
        for name in ("sbatch", "squeue", "sacct", "scancel", "scontrol"):
            os.symlink(FIX / "fakeslurm.py", self.bin / name)
        os.chmod(FIX / "fakeslurm.py", 0o755)
        self.db = tmp / "slurm.json"
        self.port = free_port()
        self.state = tmp / "state"
        self.cfg_path = tmp / "sserved.toml"
        env = {"PATH": f"{self.bin}:{os.environ['PATH']}", "FAKESLURM_DB": str(self.db),
               "FAKE_PY": PY, "FAKE_VLLM": str(FIX / "fake_vllm.py"), "USER": os.environ.get("USER", "ubuntu")}
        env.update(over.pop("job_env", {}))
        self.cluster_env = env
        cfg = {
            "server": {"listen": f"127.0.0.1:{self.port}", "state_dir": str(self.state), "reconcile_interval_s": 0.3},
            "leases": {"default_ttl_s": 60, "prep_grace_s": 30, "idle_timeout_s": 30, "linger_s": 0},
            "scaling": {"window_s": 1.5, "cooldown_s": 2, "submit_backoff_s": 0.5, "drain_deadline_s": 10,
                        "health_fail_unroute": 2, "health_fail_replace_s": 30, "unknown_terminal_grace_s": 2,
                        "orphan_sweep_s": 1,
                        "max_total_gpus": 16, "replace_before_s": 0, "replace_requires_activity_s": 3600},
            "clusters": {"fake": {"host": "fake-login", "runner": "local", "runs_dir": str(tmp / "runs"),
                                  "deploy_dir": str(FIX / "recipes"), "env": env,
                                  "tunnel_cmd": [PY, "-I", str(FIX / "fwd.py"), "{lport}", "127.0.0.1", "{port}"]}},
            "profiles": {
                "batch": {"cluster": "fake", "sbatch_args": ["--partition=batch", "--qos=batch"]},
                "free": {"cluster": "fake", "sbatch_args": ["--partition=freecycle", "--qos=freecycle"]},
                "pi": {"cluster": "fake", "sbatch_args": ["--partition=pi", "--account=pi-x"], "opt_in": True},
            },
            "models": {MODEL: {"recipe": "fake-model", "gpus": 1, "profiles": ["batch", "free", "pi"],
                               "max_model_len": 4096, "allowed_overrides": ["max_model_len"],
                               "startup_timeout_s": 20, "max_replicas": 3}},
        }
        for k, v in over.items():
            sec, key = k.split("__", 1)
            if sec == "models":
                cfg["models"][MODEL][key] = v
            elif sec == "profiles":
                name, key2 = key.split("__")
                cfg["profiles"][name][key2] = v
            else:
                cfg[sec][key] = v
        self.cfg = cfg
        self.write_cfg()
        self.proc: subprocess.Popen | None = None
        self.url = f"http://127.0.0.1:{self.port}"
        self.admin = self.cli("token", "create", "admin", "--admin").strip()
        self.tok = {n: self.cli("token", "create", n).strip() for n in ("alice", "bob")}

    def write_cfg(self):
        self.cfg_path.write_text(to_toml(self.cfg))

    def cli(self, *args) -> str:
        return subprocess.run([PY, "-m", "sserved", "-c", str(self.cfg_path), *args], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout

    def start(self):
        self.log = open(self.tmp / "sserved.log", "ab")
        self.proc = subprocess.Popen([PY, "-m", "sserved", "-c", str(self.cfg_path), "serve"], cwd=ROOT,
                                     stdout=self.log, stderr=self.log, preexec_fn=_session_dying_with_parent)
        wait_for(lambda: httpx.get(self.url + "/healthz", timeout=1).status_code == 200, what="sserved up")

    def stop(self, sig=signal.SIGTERM):
        if self.proc and self.proc.poll() is None:
            os.killpg(self.proc.pid, sig)
            try:
                self.proc.wait(20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.proc = None

    # -- fake slurm ------------------------------------------------------------------
    def slurm(self, *args) -> str:
        return subprocess.run([PY, str(FIX / "fakeslurm.py"), "ctl", *args], capture_output=True, text=True,
                              check=True, env={**os.environ, **self.cluster_env}).stdout

    def jobs(self) -> dict:
        return json.loads(self.slurm("list"))

    def job_ctl(self, jid: str, **ctl):
        run = self.jobs()[jid]["chdir"]
        Path(run, "ctl.json").write_text(json.dumps(ctl))

    def job_log(self, jid: str) -> list[dict]:
        p = Path(self.jobs()[jid]["chdir"], "ctl.json.log")
        return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []

    # -- API --------------------------------------------------------------------------
    def api(self, method, path, token, **kw) -> httpx.Response:
        return httpx.request(method, self.url + path, headers={"Authorization": f"Bearer {token}"}, timeout=30, **kw)

    def acquire(self, who="alice", **body) -> dict:
        r = self.api("POST", "/api/v1/leases", self.tok[who], json={"model": MODEL, **body})
        assert r.status_code == 201, r.text
        return r.json()

    def status(self) -> dict:
        return self.api("GET", "/api/v1/status", self.admin).json()

    def dep(self, i=0) -> dict:
        return self.status()["deployments"][i]

    def pending_jobs(self, profile=None):
        """PENDING jobs submitted by this stack's manager (tag carries its instance id)."""
        inst = self.instance()
        return [j for j in self.jobs().values() if j["state"] == "PENDING"
                and j.get("comment", "").startswith(f"sserved:{inst}:") and not j["name"].startswith("sservem-d-lost")
                and (profile is None or f"--partition={profile}" in j["text"])]

    def instance(self) -> str:
        import sqlite3
        if not getattr(self, "_inst", None):
            self._inst = sqlite3.connect(self.state / "state.db").execute(
                "SELECT v FROM meta WHERE k='instance'").fetchone()[0]
        return self._inst

    def chat(self, key, stream=False, tools=False, model=MODEL, **kw) -> httpx.Response:
        body = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": stream}
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": "weather", "parameters": {
                "type": "object", "properties": {"city": {"type": "string"}}}}}]
        return httpx.post(self.url + "/v1/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {key}"}, timeout=60, **kw)

    def chat_body(self, key, body) -> httpx.Response:
        return httpx.post(self.url + "/v1/chat/completions", json=body,
                          headers={"Authorization": f"Bearer {key}"}, timeout=60)

    def ready_one(self, who="alice", **body) -> tuple[dict, str]:
        """Acquire, start the first submitted job, wait for routing."""
        lease = self.acquire(who, **body)
        jid = wait_for(lambda: next(iter(j["id"] for j in self.pending_jobs()), None), what="a submitted job")
        self.slurm("start", jid, "--node", "node-a")
        wait_for(lambda: self.dep()["ready_replicas"] == 1, what="replica ready")
        return lease, jid

    def events(self) -> list[str]:
        return [e["msg"] for e in self.api("GET", "/api/v1/events", self.admin).json()["events"]]

    def close(self):
        self.stop()
        try:
            self.slurm("killall")
        except Exception:  # noqa: BLE001
            pass


def leftover_pids(tmp: Path) -> list[int]:
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit():
            continue
        try:
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            env = (d / "environ").read_bytes().decode(errors="replace")
            st = (d / "stat").read_text().rsplit(")", 1)[1].split()[0]
        except OSError:
            continue
        if (str(tmp) in cmd or f"FAKESLURM_DB={tmp}" in env) and st not in ("Z", "X") and int(d.name) != os.getpid():
            out.append(int(d.name))
    return out


@pytest.fixture
def stack_factory(tmp_path):
    made = []

    def make(**over):
        d = tmp_path / f"s{len(made)}"
        d.mkdir()
        s = Stack(d, **over)
        made.append(s)
        return s
    yield make
    for s in made:
        s.close()
    # nothing started by the stack may survive (sserved, fake vLLM jobs, tunnel forwarders)
    wait_for(lambda: not leftover_pids(tmp_path), timeout=10, what=f"no leftover processes {leftover_pids(tmp_path)}")


@pytest.fixture
def stack(stack_factory):
    s = stack_factory()
    s.start()
    return s


def _session_dying_with_parent():
    """New session (so the stack can signal the group) that also dies if pytest is killed."""
    import ctypes
    os.setsid()
    ctypes.CDLL(None).prctl(1, signal.SIGTERM)   # PR_SET_PDEATHSIG


def stale_task_processes() -> list[int]:
    """Orphaned processes provably started by this suite: fake vLLM jobs / tunnel forwarders from this
    fixtures directory, or test sserved instances, whose config/state lives in a pytest tmp dir."""
    out = []
    for d in Path("/proc").iterdir():
        if not d.name.isdigit() or int(d.name) == os.getpid():
            continue
        try:
            if d.stat().st_uid != os.getuid():
                continue
            cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            env = (d / "environ").read_bytes().decode(errors="replace")
            stat = (d / "stat").read_text().rsplit(")", 1)[1].split()
            cwd = (d / "cwd").resolve().as_posix()
        except OSError:
            continue
        if stat[0] in ("Z", "X") or stat[1] != "1":          # only orphans (parent gone)
            continue
        ours_fixture = str(FIX) in cmd and ("/pytest-of-" in cmd or "FAKESLURM_DB=/tmp/pytest-of-" in env)
        ours_daemon = "-m sserved" in cmd and "/pytest-of-" in cmd and cwd == str(ROOT)
        if ours_fixture or ours_daemon:
            out.append(int(d.name))
    return out


@pytest.fixture(scope="session", autouse=True)
def reap_stale_task_processes():
    def reap():
        for pid in stale_task_processes():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    reap()                      # leftovers of an interrupted earlier run
    yield
    reap()
    time.sleep(1)
    left = stale_task_processes()
    assert not left, f"task-created processes left behind: {left}"
