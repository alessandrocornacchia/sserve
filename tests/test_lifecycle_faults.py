"""Regressions for scheduler-command failures, cancellation outcomes, ownership, breakers,
config drift and repeated walltime handovers."""
from __future__ import annotations

import json
import os
import signal
import sqlite3
import subprocess
import threading
import time

import httpx

from conftest import MODEL, PY, ROOT, wait_for


def owned(s, state=None):
    inst = s.instance()
    return {k: v for k, v in s.jobs().items() if v.get("comment", "").startswith(f"sserved:{inst}:")
            and (state is None or v["state"] == state)}


def db(s):
    return sqlite3.connect(s.state / "state.db")


def test_failed_squeue_and_sacct_change_nothing(stack_factory):
    s = stack_factory(scaling__submit_resolve_age_s=1)
    s.start()
    lease, jid = s.ready_one(ttl_s=600)
    n = len(s.jobs())
    s.slurm("fail", "squeue", "on")
    s.slurm("fail", "sacct", "on")
    time.sleep(6)                                   # >> unknown_terminal_grace_s (2 s)
    assert s.chat(lease["inference_key"]).status_code == 200      # still served
    st = db(s).execute("SELECT state FROM candidates WHERE job_id=?", (jid,)).fetchone()[0]
    assert st == "ready"
    assert len(s.jobs()) == n                       # no resubmission
    assert any("squeue failed" in m for m in s.events())
    s.slurm("fail", "squeue", "off")
    time.sleep(3)                                   # squeue fine, sacct still failing: job is listed, all good
    assert db(s).execute("SELECT state FROM candidates WHERE job_id=?", (jid,)).fetchone()[0] == "ready"
    s.slurm("fail", "sacct", "off")


def test_unresolvable_submission_stays_unknown(stack_factory):
    s = stack_factory(scaling__submit_resolve_age_s=1)
    s.slurm("sbatch-timeout", "on")
    s.slurm("fail", "squeue", "on")
    s.slurm("fail", "sacct", "on")
    s.start()
    s.acquire(ttl_s=600)
    time.sleep(6)
    states = [r[0] for r in db(s).execute("SELECT state FROM candidates")]
    assert states and all(x == "unknown" for x in states), states
    assert len(s.jobs()) == 2                       # one per profile: no duplicates while unknown
    s.slurm("fail", "squeue", "off")
    s.slurm("fail", "sacct", "off")
    s.slurm("sbatch-timeout", "off")
    wait_for(lambda: all(r[0] == "queued" for r in db(s).execute("SELECT state FROM candidates")), what="resolved")
    assert len(s.jobs()) == 2 and not owned(s, "CANCELLED")
    assert not s.dep()["blocked"]


def test_failed_sibling_cancel_is_retried_and_policy_cancel_still_detected(stack):
    stack.acquire(ttl_s=600)
    wait_for(lambda: len(stack.pending_jobs()) == 2, what="two queued")
    free = stack.pending_jobs("freecycle")[0]["id"]
    batch = stack.pending_jobs("batch")[0]["id"]
    stack.slurm("fail", "scancel", "on")
    stack.slurm("start", free)
    wait_for(lambda: stack.dep()["ready_replicas"] == 1, what="ready")
    wait_for(lambda: any("cancel not confirmed" in m for m in stack.events()), what="failed cancel reported")
    assert stack.jobs()[batch]["state"] == "PENDING"            # not reported as cancelled
    acked = db(stack).execute("SELECT cancel_acked FROM candidates WHERE job_id=?", (batch,)).fetchone()[0]
    assert acked is None
    stack.slurm("end", batch, "CANCELLED", "--by", "0")         # the site cancels it meanwhile
    wait_for(lambda: stack.dep()["blocked"], what="policy cancellation detected despite our intent")
    assert "cancelled by uid 0" in stack.dep()["blocked"]["reason"]
    stack.slurm("fail", "scancel", "off")


def test_failed_cancel_eventually_succeeds(stack):
    stack.acquire(ttl_s=600)
    wait_for(lambda: len(stack.pending_jobs()) == 2, what="two queued")
    free = stack.pending_jobs("freecycle")[0]["id"]
    batch = stack.pending_jobs("batch")[0]["id"]
    stack.slurm("fail", "scancel", "on")
    stack.slurm("start", free)
    wait_for(lambda: any("cancel not confirmed" in m for m in stack.events()), what="failed cancel")
    time.sleep(1)
    assert stack.jobs()[batch]["state"] == "PENDING"
    stack.slurm("fail", "scancel", "off")
    wait_for(lambda: stack.jobs()[batch]["state"] == "CANCELLED", timeout=30, what="retried cancel")
    time.sleep(1)
    assert not stack.dep()["blocked"]                           # our own cancellation: no breaker


def test_job_considered_lost_but_alive_is_tracked_again(stack):
    lease, jid = stack.ready_one(ttl_s=600)
    stack.slurm("set", jid, "hide=1", "hide_acct=1")
    wait_for(lambda: db(stack).execute("SELECT state FROM candidates WHERE job_id=?", (jid,)).fetchone()[0] == "ended",
             what="considered lost")
    stack.slurm("set", jid, "hide=0", "hide_acct=0")
    wait_for(lambda: any("tracking again" in m for m in stack.events()), what="revived")
    # never left running untracked: it is either serving again or cancelled as surplus
    def settled():
        st = db(stack).execute("SELECT state FROM candidates WHERE job_id=?", (jid,)).fetchone()[0]
        return st in ("ready", "ended") and (st == "ready" or stack.jobs()[jid]["state"] == "CANCELLED")
    wait_for(settled, timeout=30, what="revived job settled")
    running = owned(stack, "RUNNING")
    tracked = {r[0] for r in db(stack).execute("SELECT job_id FROM candidates WHERE state NOT IN ('ended')")}
    assert set(running) <= tracked


def test_policy_breaker_is_model_wide_and_survives_restart(stack):
    lease, jid = stack.ready_one(ttl_s=600)
    stack.slurm("end", jid, "CANCELLED", "--by", "0")
    wait_for(lambda: stack.dep()["state"] == "blocked", what="blocked")
    n = len(stack.jobs())
    other = stack.acquire("bob", overrides={"max_model_len": 2048}, ttl_s=600)     # different spec
    assert other["deployment"] != lease["deployment"]
    r = stack.api("POST", "/api/v1/leases", stack.tok["bob"], json={"model": MODEL, "cluster": "fake"})
    assert r.status_code == 201
    time.sleep(2)
    assert len(stack.jobs()) == n                   # no work via another spec, cluster or override
    stack.stop()
    stack.start()
    time.sleep(2)
    assert len(stack.jobs()) == n and all(d["state"] == "blocked" for d in stack.status()["deployments"])
    assert "reset" in stack.cli("reset", MODEL)
    wait_for(lambda: len(stack.jobs()) > n, what="submissions after operator reset")


def test_breaker_cancels_queued_candidates(stack_factory):
    s = stack_factory(job_env={"FAKE_CRASH": "1"}, scaling__max_startup_failures=1, scaling__max_candidates_per_slot=2)
    s.start()
    s.acquire(ttl_s=600)
    wait_for(lambda: len(s.pending_jobs()) == 2, what="two queued")
    first, second = [j["id"] for j in s.pending_jobs()]
    s.slurm("start", first)                         # crashes -> breaker
    wait_for(lambda: s.dep()["state"] == "blocked", what="blocked")
    wait_for(lambda: s.jobs()[second]["state"] == "CANCELLED", what="queued sibling released")


def test_config_drift_with_live_jobs_refuses_start(stack):
    lease, jid = stack.ready_one(ttl_s=600)
    stack.stop()
    good = dict(stack.cfg["models"]["org/fake-7b"])
    stack.cfg["models"]["org/fake-7b"]["env"] = {"VLLM_DTYPE": "float16"}      # would change what jobs run
    stack.write_cfg()
    p = subprocess.run([PY, "-m", "sserved", "-c", str(stack.cfg_path), "serve"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 4 and "launch settings" in p.stderr
    del stack.cfg["clusters"]["fake"]
    stack.write_cfg()
    p = subprocess.run([PY, "-m", "sserved", "-c", str(stack.cfg_path), "serve"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode in (2, 4)                  # unknown cluster in profiles, or live job on it
    stack.cfg["models"]["org/fake-7b"] = good
    stack.cfg["clusters"]["fake"] = Stackcfg_cluster(stack)
    stack.write_cfg()
    stack.start()
    wait_for(lambda: stack.dep()["ready_replicas"] == 1, what="adopted with the original config")


def Stackcfg_cluster(stack):
    from conftest import FIX, PY as py
    return {"host": "fake-login", "runner": "local", "runs_dir": str(stack.tmp / "runs"),
            "deploy_dir": str(FIX / "recipes"), "env": stack.cluster_env,
            "tunnel_cmd": [py, "-I", str(FIX / "fwd.py"), "{lport}", "127.0.0.1", "{port}"]}


def test_two_consecutive_walltime_handovers_keep_one_replica(stack_factory):
    s = stack_factory(scaling__replace_before_s=100, scaling__drain_deadline_s=5)
    s.start()
    lease, a = s.ready_one(ttl_s=600)
    current = a
    for round_ in range(2):
        s.slurm("set", current, "time_left=50")
        assert s.chat(lease["inference_key"]).status_code == 200       # recent real traffic
        wait_for(lambda: s.pending_jobs(), what=f"replacement racing #{round_}")
        new = s.pending_jobs()[0]["id"]
        s.slurm("start", new, "--node", f"node-{round_}")
        wait_for(lambda: s.jobs()[current]["state"] == "CANCELLED", timeout=30, what=f"old replica retired #{round_}")
        wait_for(lambda: not s.pending_jobs(), what="siblings cleaned")
        st = s.dep()
        assert st["target_replicas"] == 1 and st["ready_replicas"] == 1
        primaries = [x for x in st["slots"] if x["replaces"] is None and x["state"] in ("racing", "serving")]
        assert len(primaries) == 1, st["slots"]
        assert list(owned(s, "RUNNING")) == [new]
        current = new
    time.sleep(2)
    assert not s.pending_jobs() and len(owned(s, "RUNNING")) == 1       # no accumulation


def test_invalid_numbers_are_400(stack):
    for body in ({"ttl_s": "abc"}, {"idle_timeout_s": [1]}, {"ttl_s": -5}, {"prep_grace_s": "nan"}):
        r = stack.api("POST", "/api/v1/leases", stack.tok["alice"], json={"model": MODEL, **body})
        assert r.status_code == 400, (body, r.text)
    lease = stack.acquire()
    r = stack.api("POST", f"/api/v1/leases/{lease['lease_id']}/renew", stack.tok["alice"], json={"ttl_s": "x"})
    assert r.status_code == 400


def test_scheduler_stderr_warnings_never_become_job_ids(stack_factory):
    s = stack_factory()
    for cmd in ("sbatch", "squeue", "sacct", "scontrol"):
        s.slurm("warn", cmd, "on")
    s.start()
    s.acquire(ttl_s=600)
    wait_for(lambda: len(s.pending_jobs()) == 2, what="two queued")
    real = set(owned(s))
    wait_for(lambda: {r[0] for r in db(s).execute("SELECT job_id FROM candidates")} == real, what="ids recorded")
    # ambiguous submission resolved by name while squeue/sacct print warnings on stderr
    s.slurm("sbatch-timeout", "on")
    s.cli("profile", "enable", "pi")
    wait_for(lambda: len(owned(s)) == 3, what="pi candidate submitted")
    wait_for(lambda: all(r[0] and r[0].isdigit() for r in db(s).execute("SELECT job_id FROM candidates")),
             what="all ids resolved")
    assert {r[0] for r in db(s).execute("SELECT job_id FROM candidates")} == set(owned(s))
    for r in db(s).execute("SELECT run_dir FROM candidates"):
        jf = os.path.join(r[0], "jobid")
        if os.path.exists(jf):
            assert open(jf).read().strip().isdigit()


def test_tunnel_breaks_under_load_repair_not_scale_out(stack_factory):
    s = stack_factory(scaling__window_s=60, scaling__up_capacity_errors=3)
    s.start()
    lease, jid = s.ready_one(ttl_s=600)
    stop, codes = threading.Event(), []

    def load():
        while not stop.is_set():
            try:
                codes.append(s.chat(lease["inference_key"]).status_code)
            except httpx.HTTPError:
                codes.append(-1)
    threads = [threading.Thread(target=load) for _ in range(4)]
    for t in threads:
        t.start()
    try:
        for _ in range(3):
            fwd = [p for p in _pids("fwd.py") if str(s.tmp) in _env(p)]
            for p in fwd:
                os.kill(p, signal.SIGKILL)
            time.sleep(1.5)
        wait_for(lambda: s.chat(lease["inference_key"]).status_code == 200, timeout=30, what="served again")
    finally:
        stop.set()
        for t in threads:
            t.join(30)
    assert any(c != 200 for c in codes) and codes.count(200) > 0       # the outage was real and visible
    time.sleep(2)
    st = s.dep()
    assert st["target_replicas"] == 1 and st["ready_replicas"] == 1
    assert s.pending_jobs() == [] and list(owned(s, "RUNNING")) == [jid]   # same allocation, no new job
    assert not any(m.startswith("target") for m in s.events())


def _pids(needle):
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                if needle in open(f"/proc/{d}/cmdline").read():
                    out.append(int(d))
            except OSError:
                pass
    return out


def _env(pid):
    try:
        return open(f"/proc/{pid}/environ", "rb").read().decode(errors="replace")
    except OSError:
        return ""


def test_sigterm_runs_graceful_shutdown(stack):
    lease, jid = stack.ready_one(ttl_s=600)
    pid = stack.proc.pid
    stack.stop()                                     # SIGTERM
    kinds = [r[0] for r in db(stack).execute("SELECT kind FROM events ORDER BY id")]
    assert kinds[-1] == "stop"                       # shutdown path ran (tunnels closed, state recorded)
    assert not [p for p in _pids("fwd.py") if str(stack.tmp) in _env(p)]
    assert stack.jobs()[jid]["state"] == "RUNNING"   # jobs are left for adoption on restart
    assert not os.path.exists(f"/proc/{pid}")
