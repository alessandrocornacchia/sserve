"""Reconciliation against the fake scheduler: ambiguity, requeue, terminal states, policy, restart, ownership."""
from __future__ import annotations

import os
import signal
import sqlite3
import subprocess
import time

from conftest import MODEL, PY, ROOT, wait_for


def submitted(stack) -> int:
    return len(stack.jobs())


def foreign(stack, name: str, comment: str = "") -> str:
    script = stack.tmp / f"{name}.sbatch"
    script.write_text("#!/bin/bash\nsleep 1000\n")
    args = ["sbatch", "--parsable", f"--job-name={name}", f"--chdir={stack.tmp}"]
    if comment:
        args.append(f"--comment={comment}")
    return subprocess.run(args + [str(script)], env={**os.environ, **stack.cluster_env}, capture_output=True,
                          text=True, check=True).stdout.strip()


def test_unknown_submission_is_resolved_not_duplicated(stack):
    stack.slurm("sbatch-timeout", "on")         # sbatch submits, but reports a socket timeout
    lease = stack.acquire()
    wait_for(lambda: any("outcome unknown" in m for m in stack.events()), what="unknown outcome")
    wait_for(lambda: sum("unresolved submission found as job" in m for m in stack.events()) == 2, what="resolved")
    stack.slurm("sbatch-timeout", "off")
    time.sleep(2)
    assert submitted(stack) == 2                 # one per profile, never resubmitted
    st = stack.dep()
    assert sorted(j["state"] for j in st["jobs"]) == ["queued", "queued"]


def test_sbatch_rejection_is_bounded_and_blocks(stack_factory):
    s = stack_factory(scaling__max_startup_failures=2)
    s.slurm("reject", "--partition=freecycle", "Invalid qos specification")
    s.slurm("reject", "--partition=batch", "Invalid account or account/partition combination specified")
    s.start()
    s.acquire()
    wait_for(lambda: s.dep()["state"] == "blocked", what="blocked after rejections")
    assert "sbatch rejected 2 times" in s.dep()["blocked"]["reason"]
    time.sleep(2)
    assert submitted(s) == 0
    ev = s.events()
    assert sum("sbatch rejected on" in m for m in ev) == 2


def test_requeue_relocation_new_generation(stack):
    lease, jid = stack.ready_one()
    key = lease["inference_key"]
    stack.slurm("requeue", jid)
    wait_for(lambda: stack.dep()["ready_replicas"] == 0, what="unrouted on requeue")
    assert stack.chat(key).status_code == 503
    wait_for(lambda: any("back in queue" in m for m in stack.events()), what="requeue event")
    stack.slurm("start", jid, "--node", "node-b")
    wait_for(lambda: stack.dep()["ready_replicas"] == 1, what="ready again")
    j = [x for x in stack.dep()["jobs"] if x["job_id"] == jid][0]
    assert j["node"] == "node-b" and j["restarts"] == 1
    assert stack.chat(key).status_code == 200
    # any sibling raced during the requeue is cancelled once the slot is served again
    wait_for(lambda: all(v["state"] in ("CANCELLED", "RUNNING") for v in stack.jobs().values()), what="siblings cancelled")
    assert [k for k, v in stack.jobs().items() if v["state"] == "RUNNING"] == [jid]


def test_policy_cancellation_blocks_until_operator_reset(stack):
    lease, jid = stack.ready_one()
    stack.slurm("end", jid, "CANCELLED", "--by", "0")
    wait_for(lambda: stack.dep()["state"] == "blocked", what="blocked")
    assert "cancelled by uid 0" in stack.dep()["blocked"]["reason"]
    r = stack.chat(lease["inference_key"])
    assert r.status_code == 503 and r.json()["error"]["code"] == "deployment_blocked"
    n = submitted(stack)
    stack.acquire("bob")                         # a new acquire does not reset the breaker
    time.sleep(2)
    assert submitted(stack) == n and stack.dep()["state"] == "blocked"
    dep = stack.dep()["deployment"]
    assert stack.cli("reset", dep).strip() == "reset"
    wait_for(lambda: submitted(stack) > n, what="resubmitted after operator reset")


def test_timeout_with_demand_resubmits_within_budget(stack_factory):
    s = stack_factory(scaling__max_submissions_per_day=3)
    s.start()
    lease, jid = s.ready_one()
    s.slurm("end", jid, "TIMEOUT")
    wait_for(lambda: submitted(s) == 3, what="resubmitted after TIMEOUT")
    assert s.dep()["state"] != "blocked"
    for j in s.pending_jobs():
        s.slurm("end", j["id"], "PREEMPTED")
    time.sleep(2)
    assert submitted(s) == 3                     # daily budget reached: no more submissions
    assert any("daily submission budget" in m for m in s.events())


def test_startup_crashes_open_breaker(stack_factory):
    s = stack_factory(job_env={"FAKE_CRASH": "1"})
    s.start()
    s.acquire()
    deadline = time.time() + 40
    while s.dep()["state"] != "blocked" and time.time() < deadline:
        for j in s.pending_jobs():
            s.slurm("start", j["id"])
        time.sleep(0.5)
    assert s.dep()["state"] == "blocked"
    assert "consecutive startup failures" in s.dep()["blocked"]["reason"]
    assert all(v["state"] in ("FAILED", "CANCELLED", "PENDING") for v in s.jobs().values())


def test_startup_deadline_cancels_owned_job(stack_factory):
    s = stack_factory(job_env={"FAKE_LOAD_S": "100"}, models__startup_timeout_s=2)
    s.start()
    s.acquire()
    jid = wait_for(lambda: next(iter(j["id"] for j in s.pending_jobs()), None), what="queued")
    s.slurm("start", jid)
    wait_for(lambda: s.jobs()[jid]["state"] == "CANCELLED", what="cancelled at startup deadline")
    assert any("not healthy 2s after starting" in m for m in s.events())


def test_short_startup_errors_do_not_count(stack_factory):
    s = stack_factory(job_env={"FAKE_LOAD_S": "2"})
    s.start()
    lease = s.acquire()
    jid = wait_for(lambda: next(iter(j["id"] for j in s.pending_jobs()), None), what="queued")
    s.slurm("start", jid)
    wait_for(lambda: s.dep()["ready_replicas"] == 1, what="ready after a slow /health")
    db = sqlite3.connect(s.state / "state.db")
    assert db.execute("SELECT startup_failures FROM deployments").fetchone()[0] == 0


def test_restart_adopts_jobs_and_leaves_no_tunnels(stack):
    lease, jid = stack.ready_one()
    n = submitted(stack)
    stack.stop(signal.SIGKILL)                   # crash: no graceful cleanup
    time.sleep(1)
    from conftest import leftover_pids
    alive = [p for p in leftover_pids(stack.tmp) if b"fwd.py" in open(f"/proc/{p}/cmdline", "rb").read()]
    assert alive == []                           # tunnels died with the controller (PDEATHSIG)
    stack.start()
    wait_for(lambda: stack.dep()["ready_replicas"] == 1, what="re-admitted after restart")
    assert stack.chat(lease["inference_key"]).status_code == 200
    assert submitted(stack) == n and stack.jobs()[jid]["state"] == "RUNNING"


def test_second_controller_refused(stack):
    p = subprocess.run([PY, "-m", "sserved", "-c", str(stack.cfg_path), "serve"], cwd=ROOT,
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 3 and "another sserved" in p.stderr


def test_only_owned_jobs_are_ever_cancelled(stack):
    a = foreign(stack, "sserve-ornith")                               # plain sserve job
    b = foreign(stack, "sservem-d-xyz-abc", "sserved:otherinst:c-1")  # another manager's job
    c = foreign(stack, "benchmark")                                   # anything else
    lease, jid = stack.ready_one()
    stack.api("DELETE", f"/api/v1/leases/{lease['lease_id']}", stack.tok["alice"])
    wait_for(lambda: stack.jobs()[jid]["state"] == "CANCELLED", what="own job cancelled")
    time.sleep(1)
    for x in (a, b, c):
        assert stack.jobs()[x]["state"] == "PENDING"


def test_orphan_with_our_tag_is_cancelled(stack):
    inst = sqlite3.connect(stack.state / "state.db").execute("SELECT v FROM meta WHERE k='instance'").fetchone()[0]
    o = foreign(stack, "sservem-d-lost-c00000", f"sserved:{inst}:c-lost")
    wait_for(lambda: stack.jobs()[o]["state"] == "CANCELLED", what="orphan cancelled")
    assert any("no known candidate" in m for m in stack.events())


def test_pi_opt_in_persists_across_restart(stack):
    assert "pi\tdisabled" in stack.cli("profile", "list")
    stack.cli("profile", "enable", "pi")
    stack.stop()
    stack.start()
    assert "pi\tenabled" in stack.cli("profile", "list")
    stack.acquire()
    wait_for(lambda: len(stack.pending_jobs()) == 3, what="pi raced too")
    assert stack.pending_jobs("pi")


def test_duplicate_startup_capacity_is_bounded(stack_factory):
    s = stack_factory(job_env={"FAKE_LOAD_S": "30"}, scaling__max_startup_gpus=0)
    s.start()
    s.acquire()
    wait_for(lambda: len(s.pending_jobs()) == 2, what="two queued")
    first, second = [j["id"] for j in s.pending_jobs()]
    s.slurm("start", first)
    wait_for(lambda: any(j["state"] == "starting" for j in s.dep()["jobs"]), what="first starting")
    s.slurm("start", second)
    wait_for(lambda: s.jobs()[second]["state"] == "CANCELLED", what="duplicate start-up cancelled")
    assert s.jobs()[first]["state"] == "RUNNING"
    assert any("duplicate start-up capacity cap" in m for m in s.events())
