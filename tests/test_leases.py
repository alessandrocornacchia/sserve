"""Leases: TTL, idle retirement, long requests, isolation between clients, key rotation, secrets."""
from __future__ import annotations

import concurrent.futures as cf
import os
import stat
import threading
import time

import httpx
import pytest

from conftest import MODEL, wait_for


@pytest.fixture
def fast(stack_factory):
    s = stack_factory(leases__min_ttl_s=1, leases__idle_timeout_s=3, leases__prep_grace_s=0)
    s.start()
    return s


def test_ttl_expiry_ends_lease_and_job(fast):
    lease = fast.acquire(ttl_s=2)
    jid = wait_for(lambda: next(iter(j["id"] for j in fast.pending_jobs()), None), what="queued")
    wait_for(lambda: fast.api("GET", f"/api/v1/leases/{lease['lease_id']}", fast.tok["alice"]).json()["state"] == "expired",
             what="expired")
    r = fast.chat(lease["inference_key"])
    assert r.status_code == 401 and r.json()["error"]["code"] == "lease_expired"
    r = fast.api("POST", f"/api/v1/leases/{lease['lease_id']}/renew", fast.tok["alice"])
    assert r.status_code == 410 and r.json()["error"]["code"] == "lease_expired"
    # no demand: queued jobs are not held
    wait_for(lambda: all(j["state"] == "CANCELLED" for j in fast.jobs().values()), what="pending cancelled")


def test_idle_retires_even_a_renewing_client(fast):
    lease, jid = fast.ready_one()
    lid, tok = lease["lease_id"], fast.tok["alice"]
    assert fast.chat(lease["inference_key"]).status_code == 200
    t0 = time.time()
    while time.time() - t0 < 8:                       # keeps renewing, sends no inference
        r = fast.api("POST", f"/api/v1/leases/{lid}/renew", tok, json={"ttl_s": 30})
        if r.status_code == 410:
            break
        time.sleep(0.5)
    assert r.status_code == 410 and r.json()["error"]["code"] == "lease_idle"
    r = fast.chat(lease["inference_key"])
    assert r.status_code == 401 and r.json()["error"]["code"] == "lease_idle"
    wait_for(lambda: fast.jobs()[jid]["state"] == "CANCELLED", what="idle capacity released")


def test_health_probes_are_not_activity(fast):
    lease, jid = fast.ready_one()
    # replica is probed every 0.3 s; with no client traffic the lease still goes idle
    wait_for(lambda: fast.api("GET", f"/api/v1/leases/{lease['lease_id']}", fast.tok["alice"]).json()["state"] == "idle",
             timeout=15, what="idle despite probes")


def test_long_request_is_activity(fast):
    lease, jid = fast.ready_one()
    fast.job_ctl(jid, chunk_delay=1.0)                # ~11 s stream, idle timeout 3 s
    out = {}

    def run():
        t = time.time()
        out["r"] = fast.chat(lease["inference_key"], stream=True)
        out["dt"] = time.time() - t
    t = threading.Thread(target=run)
    t.start()
    time.sleep(1)
    for _ in range(5):                                # 1..6 s: well past the 3 s idle timeout
        v = fast.api("GET", f"/api/v1/leases/{lease['lease_id']}", fast.tok["alice"]).json()
        assert v["state"] == "active" and v["inflight"] == 1, (v, out, fast.job_log(jid)[-1:])
        fast.api("POST", f"/api/v1/leases/{lease['lease_id']}/renew", fast.tok["alice"], json={"ttl_s": 30})
        time.sleep(1)
    t.join(30)
    assert out["r"].status_code == 200 and out["r"].text.rstrip().endswith("data: [DONE]")
    wait_for(lambda: fast.api("GET", f"/api/v1/leases/{lease['lease_id']}", fast.tok["alice"]).json()["state"] == "idle",
             what="idle after the request ended")


def test_clients_cannot_touch_each_others_leases(stack):
    a = stack.acquire("alice")
    b_tok = stack.tok["bob"]
    for method, path in (("GET", ""), ("POST", "/renew"), ("DELETE", ""), ("POST", "/rotate-key")):
        r = stack.api(method, f"/api/v1/leases/{a['lease_id']}{path}", b_tok)
        assert r.status_code == 404, (method, path, r.text)
    assert stack.api("GET", "/api/v1/leases", b_tok).json()["leases"] == []
    assert stack.api("GET", "/api/v1/status", b_tok).json()["deployments"] == []
    assert stack.api("GET", f"/api/v1/leases/{a['lease_id']}", stack.tok["alice"]).json()["state"] == "active"


def test_concurrent_independent_clients(stack):
    for n in ("c1", "c2", "c3", "c4"):
        stack.tok[n] = stack.cli("token", "create", n).strip()
    names = ["alice", "bob", "c1", "c2", "c3", "c4"]

    def life(n):
        l = stack.acquire(n)
        for _ in range(10):
            r = stack.api("POST", f"/api/v1/leases/{l['lease_id']}/renew", stack.tok[n])
            assert r.status_code == 200, r.text
        return l

    with cf.ThreadPoolExecutor(6) as ex:
        leases = list(ex.map(life, names))
    assert len({l["deployment"] for l in leases}) == 1           # one shared deployment
    assert len({l["inference_key"] for l in leases}) == 6
    with cf.ThreadPoolExecutor(6) as ex:
        list(ex.map(lambda nl: stack.api("DELETE", f"/api/v1/leases/{nl[1]['lease_id']}", stack.tok[nl[0]]),
                    list(zip(names, leases))[:5]))
    st = stack.dep()
    assert st["active_leases"] == 1
    assert stack.api("GET", f"/api/v1/leases/{leases[5]['lease_id']}", stack.tok["c4"]).json()["state"] == "active"


def test_rotate_key(stack):
    lease, _ = stack.ready_one()
    r = stack.api("POST", f"/api/v1/leases/{lease['lease_id']}/rotate-key", stack.tok["alice"])
    new = r.json()["inference_key"]
    assert new != lease["inference_key"]
    assert stack.chat(lease["inference_key"]).status_code == 401
    assert stack.chat(new).status_code == 200


def test_incompatible_overrides_never_share(stack):
    a = stack.acquire("alice")
    b = stack.acquire("bob", overrides={"max_model_len": 2048})
    c = stack.acquire("bob", overrides={"max_model_len": "4096"})
    assert a["deployment"] != b["deployment"] and a["deployment"] == c["deployment"]
    r = stack.api("POST", "/api/v1/leases", stack.tok["alice"], json={"model": MODEL, "gpus": 4})
    assert r.status_code == 400 and r.json()["error"]["code"] == "override_not_allowed"
    r = stack.api("POST", "/api/v1/leases", stack.tok["alice"], json={"model": "meta/unknown"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "model_not_in_catalog"
    jobs = wait_for(lambda: [j for j in stack.pending_jobs() if "MAX_MODEL_LEN=2048" in j["text"]], what="2048 job")
    assert jobs


def test_no_secrets_in_status_logs_or_world_readable_state(stack):
    lease, jid = stack.ready_one()
    stack.chat(lease["inference_key"])
    secrets_ = [lease["inference_key"], stack.tok["alice"], stack.tok["bob"], stack.admin]
    blob = stack.api("GET", "/api/v1/status", stack.admin).text + stack.api("GET", "/api/v1/events", stack.admin).text
    blob += stack.api("GET", f"/api/v1/leases/{lease['lease_id']}", stack.tok["alice"]).text
    backend_key = open(os.path.join(stack.jobs()[jid]["chdir"], "api_key")).read().strip()
    secrets_.append(backend_key)
    blob += (stack.tmp / "sserved.log").read_text()
    for f in (stack.state / "logs").glob("*"):
        blob += f.read_text()
    for s in secrets_:
        assert s not in blob
    assert stat.S_IMODE(os.stat(stack.state).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(stack.state / "state.db").st_mode) == 0o600
    run = stack.jobs()[jid]["chdir"]
    assert stat.S_IMODE(os.stat(run).st_mode) == 0o700
    assert stat.S_IMODE(os.stat(os.path.join(run, "api_key")).st_mode) == 0o600
    # backend keys never appear in any job's argv
    assert backend_key not in stack.jobs()[jid]["text"]


def test_slow_queue_does_not_idle_a_renewing_lease(fast):
    lease = fast.acquire(ttl_s=5, prep_grace_s=1)
    lid, tok = lease["lease_id"], fast.tok["alice"]
    jid = wait_for(lambda: next(iter(j["id"] for j in fast.pending_jobs()), None), what="queued")
    t0 = time.time()
    while time.time() - t0 < 8:                      # queue wait >> idle timeout (3 s) + prep grace (1 s)
        r = fast.api("POST", f"/api/v1/leases/{lid}/renew", tok, json={"ttl_s": 5})
        assert r.status_code == 200 and r.json()["state"] == "active", r.text
        assert r.json()["available_since"] is None
        time.sleep(0.5)
    fast.slurm("start", jid, "--node", "node-a")
    wait_for(lambda: fast.api("GET", f"/api/v1/leases/{lid}", tok).json()["available_since"], what="available")
    v = fast.api("GET", f"/api/v1/leases/{lid}", tok).json()
    assert v["state"] == "active"
    # now the grace (1 s) and idle timeout (3 s) run from first availability
    t1 = time.time()
    while True:
        r = fast.api("POST", f"/api/v1/leases/{lid}/renew", tok, json={"ttl_s": 5})
        if r.status_code == 410:
            break
        assert time.time() - t1 < 10
        time.sleep(0.3)
    assert r.json()["error"]["code"] == "lease_idle"
    assert time.time() - t1 >= 3.5


def test_acquire_timeout_ends_never_ready_demand(fast):
    lease = fast.acquire(ttl_s=5, acquire_timeout_s=3)
    lid, tok = lease["lease_id"], fast.tok["alice"]
    wait_for(lambda: fast.pending_jobs(), what="queued")
    t0 = time.time()
    while time.time() - t0 < 10:                     # client keeps renewing; the model never becomes ready
        r = fast.api("POST", f"/api/v1/leases/{lid}/renew", tok, json={"ttl_s": 5})
        if r.status_code == 410:
            break
        time.sleep(0.3)
    assert r.status_code == 410 and r.json()["error"]["code"] == "lease_unfulfilled"
    wait_for(lambda: all(j["state"] == "CANCELLED" for j in fast.jobs().values()), what="queued jobs released")
