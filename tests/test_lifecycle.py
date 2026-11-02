"""End-to-end: leases drive owned SLURM jobs through a real sserved process."""
from __future__ import annotations

import json
import time

import httpx

from conftest import MODEL, wait_for


def test_race_first_healthy_wins_and_sibling_cancelled(stack):
    lease = stack.acquire()
    # opt-in PI profile is not raced; batch and freecycle both queued
    wait_for(lambda: len(stack.pending_jobs()) == 2, what="two candidates queued")
    parts = sorted("batch" if "--partition=batch" in j["text"] else "free" for j in stack.pending_jobs())
    assert parts == ["batch", "free"]
    def reasons():
        d = stack.dep()
        return d if d["jobs"] and all(j["reason"] == "Priority" for j in d["jobs"]) else None
    st = wait_for(reasons, what="real queue reason reported")
    assert st["state"] == "queued" and len(st["jobs"]) == 2
    free = stack.pending_jobs("freecycle")[0]["id"]
    batch = stack.pending_jobs("batch")[0]["id"]
    # ownership tag and job name
    j = stack.jobs()[free]
    assert j["comment"].startswith("sserved:") and j["name"].startswith("sservem-")
    assert "#SBATCH --requeue" in j["text"] and "--time=3:00:00" in j["text"]
    stack.slurm("start", free, "--node", "node-f")
    wait_for(lambda: stack.dep()["ready_replicas"] == 1, what="ready")
    wait_for(lambda: stack.jobs()[batch]["state"] == "CANCELLED", what="sibling cancelled")
    assert stack.jobs()[free]["state"] == "RUNNING"
    r = stack.chat(lease["inference_key"])
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] == f"hello from job{free}"


def test_no_ready_replica_is_503_and_backend_untouched(stack):
    lease = stack.acquire()
    wait_for(lambda: stack.pending_jobs(), what="queued")
    r = stack.chat(lease["inference_key"])
    assert r.status_code == 503 and r.headers.get("retry-after")
    assert r.json()["error"]["code"] == "no_ready_replica"


def test_release_last_lease_cancels_jobs(stack):
    lease, jid = stack.ready_one()
    other = stack.acquire("bob")
    assert other["deployment"] == lease["deployment"]          # compatible: shared
    assert len(stack.jobs()) == 2                              # no extra replica for the joiner
    stack.api("DELETE", f"/api/v1/leases/{lease['lease_id']}", stack.tok["alice"])
    time.sleep(1.5)
    assert stack.jobs()[jid]["state"] == "RUNNING"             # bob still holds it
    r = stack.chat(other["inference_key"])
    assert r.status_code == 200
    stack.api("DELETE", f"/api/v1/leases/{other['lease_id']}", stack.tok["bob"])
    wait_for(lambda: stack.jobs()[jid]["state"] == "CANCELLED", what="job cancelled after last release")
