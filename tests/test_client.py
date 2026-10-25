"""The stdlib client (`sserve lease ...`) as a remote machine would use it, and from a container."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
import time

import pytest

from conftest import MODEL, ROOT, free_port, wait_for

CHILD = r'''
import json, os, sys, urllib.request
b = os.environ["OPENAI_BASE_URL"]; k = os.environ["OPENAI_API_KEY"]
req = urllib.request.Request(b + "/chat/completions", method="POST",
    data=json.dumps({"model": os.environ["SSERVE_MODEL"], "messages": [{"role": "user", "content": "hi"}]}).encode(),
    headers={"Authorization": "Bearer " + k, "Content-Type": "application/json"})
print(json.dumps({"base": b, "anthropic": os.environ["ANTHROPIC_BASE_URL"],
                  "answer": json.loads(urllib.request.urlopen(req, timeout=30).read())["choices"][0]["message"]["content"]}))
sys.exit(7)
'''


def client_env(stack, who, home):
    """A separate 'machine': own HOME, token file, lease dir; no cluster access, no shared files."""
    home.mkdir(parents=True, exist_ok=True)
    tf = home / "token"
    tf.write_text(stack.tok[who] + "\n")
    tf.chmod(0o600)
    return {"PATH": os.environ["PATH"], "HOME": str(home), "SSERVE_MANAGER_URL": stack.url,
            "SSERVE_MANAGER_TOKEN_FILE": str(tf), "SSERVE_CONFIG": str(home / "none.toml"),
            "SSERVE_LEASE_DIR": str(home / "leases")}


def sserve(env, *args, timeout=90, **kw):
    return subprocess.run([sys.executable, str(ROOT / "sserve"), "lease", *args], env=env, capture_output=True,
                          text=True, timeout=timeout, **kw)


def starter(stack):
    def go():
        jid = wait_for(lambda: next(iter(j["id"] for j in stack.pending_jobs()), None), timeout=30, what="queued")
        stack.slurm("start", jid)
    t = threading.Thread(target=go, daemon=True)
    t.start()
    return t


def test_lease_run_unmodified_command(stack, tmp_path):
    env = client_env(stack, "alice", tmp_path / "alice")
    starter(stack)
    p = sserve(env, "run", MODEL, "--wait-timeout", "60", "--", sys.executable, "-c", CHILD)
    assert p.returncode == 7, p.stderr                       # the command's exit code
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["base"] == stack.url + "/v1" and out["anthropic"] == stack.url
    assert out["answer"].startswith("hello from job")
    ls = stack.api("GET", "/api/v1/leases", stack.tok["alice"]).json()["leases"]
    assert ls and all(l["state"] == "released" for l in ls)  # released on exit


def test_saved_lease_env_rotate_release(stack, tmp_path):
    env = client_env(stack, "bob", tmp_path / "bob")
    p = sserve(env, "acquire", MODEL, "--save", "bench1")
    assert p.returncode == 0, p.stderr
    f = tmp_path / "bob" / "leases" / "bench1.json"
    assert oct(f.stat().st_mode & 0o777) == "0o600"
    key1 = json.loads(f.read_text())["key"]
    p = sserve(env, "env", "bench1")
    assert f"OPENAI_API_KEY={key1}" in p.stdout and f"OPENAI_BASE_URL={stack.url}/v1" in p.stdout
    assert sserve(env, "renew", "bench1").returncode == 0
    assert sserve(env, "rotate-key", "bench1").returncode == 0
    key2 = json.loads(f.read_text())["key"]
    assert key2 != key1
    p = sserve(env, "status", "bench1")
    assert "active" in p.stdout and "queued" in p.stdout or "waiting" in p.stdout or "starting" in p.stdout
    assert sserve(env, "release", "bench1").returncode == 0
    assert not f.exists()
    # another client cannot act on bob's lease id
    lid = stack.api("GET", "/api/v1/leases", stack.tok["bob"]).json()["leases"][0]["lease_id"]
    other = client_env(stack, "alice", tmp_path / "alice")
    p = sserve(other, "release", lid)
    assert p.returncode == 69 and "no such lease" in p.stderr


def test_wait_timeout_releases_only_own_lease(stack, tmp_path):
    held = stack.acquire("bob", ttl_s=600)
    env = client_env(stack, "alice", tmp_path / "alice")
    p = sserve(env, "run", MODEL, "--wait-timeout", "3", "--", "true")
    assert p.returncode == 75 and "shared jobs are not cancelled" in p.stderr
    time.sleep(1.5)
    assert len(stack.pending_jobs()) == 2                    # bob's demand keeps the queued jobs
    assert stack.api("GET", f"/api/v1/leases/{held['lease_id']}", stack.tok["bob"]).json()["state"] == "active"


def test_concurrent_clients_from_separate_homes(stack, tmp_path):
    starter(stack)
    envs = [client_env(stack, w, tmp_path / w) for w in ("alice", "bob")]
    procs = [subprocess.Popen([sys.executable, str(ROOT / "sserve"), "lease", "run", MODEL, "--wait-timeout", "60",
                               "--", sys.executable, "-c", CHILD], env=e, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for e in envs]
    outs = [p.communicate(timeout=120) for p in procs]
    assert [p.returncode for p in procs] == [7, 7], outs
    assert len(stack.status()["deployments"]) == 1
    started = [j for j in stack.jobs().values() if j.get("started")]
    assert len(started) == 1                                 # both clients shared one allocation
    # both leases released (linger 0): the shared job is then released too
    wait_for(lambda: stack.jobs()[started[0]["id"]]["state"] == "CANCELLED", what="released after last client")


def test_advertised_url_is_what_clients_get(stack_factory, tmp_path):
    s = stack_factory(server__advertised_url="https://inference.example.org/sserve")
    s.start()
    lease = s.acquire()
    assert lease["openai_base_url"] == "https://inference.example.org/sserve/v1"
    assert lease["anthropic_base_url"] == "https://inference.example.org/sserve"


def test_root_path_prefix_served(stack_factory):
    import httpx
    s = stack_factory(server__root_path="/sserve")
    s.start()
    assert httpx.get(s.url + "/sserve/healthz").status_code == 200
    r = httpx.post(s.url + "/sserve/api/v1/leases", headers={"Authorization": f"Bearer {s.tok['alice']}"},
                   json={"model": MODEL})
    assert r.status_code == 201 and r.json()["openai_base_url"].endswith("/sserve/v1")


def _docker_ok():
    if not shutil.which("docker"):
        return False
    p = subprocess.run(["docker", "image", "inspect", "python:3.12-slim"], capture_output=True)
    return p.returncode == 0


def _docker0():
    p = subprocess.run(["ip", "-4", "-o", "addr", "show", "docker0"], capture_output=True, text=True)
    return p.stdout.split()[3].split("/")[0] if p.returncode == 0 and p.stdout else None


@pytest.mark.skipif(not _docker_ok() or not _docker0(), reason="docker or python:3.12-slim image not available")
def test_client_in_separate_container(stack_factory, tmp_path):
    ip = _docker0()
    port = free_port()
    s = stack_factory(server__listen=f"{ip}:{port}", server__advertised_url=f"http://{ip}:{port}")
    s.url = f"http://{ip}:{port}"
    s.start()
    ctx = tmp_path / "ctr"
    ctx.mkdir()
    shutil.copy(ROOT / "sserve", ctx / "sserve")
    shutil.copy(ROOT / "sserve_client.py", ctx / "sserve_client.py")
    (ctx / "child.py").write_text(CHILD)
    (ctx / "token").write_text(s.tok["alice"])
    os.chmod(ctx / "token", 0o644)              # readable by the container user (warning expected)
    name = f"sserved-test-{os.getpid()}"
    starter(s)
    try:
        p = subprocess.run(["docker", "run", "--rm", "--pull", "never", "--name", name, "-v", f"{ctx}:/c:ro",
                            "-e", f"SSERVE_MANAGER_URL=http://{ip}:{port}", "-e", "SSERVE_MANAGER_TOKEN_FILE=/c/token",
                            "-e", "SSERVE_LEASE_DIR=/tmp/leases", "python:3.12-slim",
                            "python", "/c/sserve", "lease", "run", MODEL, "--wait-timeout", "60", "--",
                            "python", "/c/child.py"], capture_output=True, text=True, timeout=180)
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    assert p.returncode == 7, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["base"] == f"http://{ip}:{port}/v1" and out["answer"].startswith("hello from job")
