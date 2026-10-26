"""Autoscaling from vLLM metrics and capacity errors; drain; walltime replacement."""
from __future__ import annotations

import threading
import time

from conftest import wait_for


def owned_running(s):
    inst = s.instance()
    return [k for k, v in s.jobs().items() if v["state"] == "RUNNING" and v["comment"].startswith(f"sserved:{inst}:")]


def start_pending(s, node="node-b"):
    for j in s.pending_jobs():
        s.slurm("start", j["id"], "--node", node)
        return j["id"]


def test_scale_up_on_sustained_waiting_bounded_by_max_replicas(stack_factory):
    s = stack_factory(models__max_replicas=2)
    s.start()
    lease, a = s.ready_one()
    s.job_ctl(a, metrics={"vllm:num_requests_waiting": 6})
    wait_for(lambda: s.dep()["target_replicas"] == 2, what="scale up")
    wait_for(lambda: len(s.pending_jobs()) == 2, what="new slot racing")
    time.sleep(2)
    assert len(s.pending_jobs()) == 2            # racing capacity counts: no further slot meanwhile
    b = start_pending(s)
    s.job_ctl(b, metrics={"vllm:num_requests_waiting": 6})
    wait_for(lambda: s.dep()["ready_replicas"] == 2, what="two replicas")
    time.sleep(3)
    assert s.dep()["target_replicas"] == 2 and s.pending_jobs() == []   # max_replicas holds
    ev = [m for m in s.events() if m.startswith("target")]
    assert ev and "pressure" in ev[0]


def test_non_capacity_errors_never_scale(stack):
    lease, a = stack.ready_one(ttl_s=600)
    stack.job_ctl(a, mode="http500")
    for _ in range(12):
        assert stack.chat(lease["inference_key"]).status_code == 502
    for _ in range(6):
        assert stack.chat(lease["inference_key"], model="nope").status_code == 404
        assert stack.chat_body(lease["inference_key"], {"model": "org/fake-7b", "messages": []}).status_code == 400
    time.sleep(3)
    assert stack.dep()["target_replicas"] == 1 and not stack.pending_jobs()


def test_capacity_errors_scale_up(stack_factory):
    stack = stack_factory(scaling__window_s=60)
    stack.start()
    lease, a = stack.ready_one(ttl_s=600)
    stack.job_ctl(a, mode="http429")
    for _ in range(8):
        assert stack.chat(lease["inference_key"]).status_code in (429, 503)
    wait_for(lambda: stack.dep()["target_replicas"] == 2, what="scale up on 429s")


def _two_replicas(s):
    lease, a = s.ready_one()
    s.job_ctl(a, metrics={"vllm:num_requests_waiting": 6})
    wait_for(lambda: len(s.pending_jobs()) == 2, what="second slot racing")
    b = start_pending(s)
    wait_for(lambda: s.dep()["ready_replicas"] == 2, what="two replicas")
    s.job_ctl(a)
    return lease, a, b


def test_missing_metrics_block_scale_down_then_redundant_replica_drained(stack_factory):
    s = stack_factory(models__max_replicas=2)
    s.start()
    lease, a, b = _two_replicas(s)
    s.job_ctl(b, no_metrics=True)
    time.sleep(5)                                # zero load on a, unknown on b: no scale-down
    assert s.dep()["ready_replicas"] == 2 and s.dep()["target_replicas"] == 2
    s.job_ctl(b)
    wait_for(lambda: s.dep()["target_replicas"] == 1, what="scale down")
    wait_for(lambda: len(owned_running(s)) == 1, what="one replica cancelled")
    time.sleep(3)
    assert s.dep()["ready_replicas"] == 1        # the final replica stays while the lease is active
    assert s.chat(lease["inference_key"]).status_code == 200


def test_metric_series_are_joined_by_scrape_time():
    from sserved.manager import align
    a = [(100.0, {"w": 1}), (115.0, {"w": 2}), (130.0, {"w": 3})]
    b = [(101.0, {"w": 10}), (131.5, {"w": 30})]           # offset clock; missed the 115 s scrape
    assert align([a, b], 7.5) == [({"w": 1}, {"w": 10}), ({"w": 3}, {"w": 30})]
    assert align([a], 7.5) == [(v,) for _, v in a]
    assert align([a, []], 7.5) == [] and align([], 7.5) == []


def test_idle_deployment_restarts_from_one_replica(stack_factory):
    s = stack_factory(models__max_replicas=2)
    s.start()
    lease, a, b = _two_replicas(s)
    for j in (a, b):                             # keep the pressure on: no scale-down before going idle
        s.job_ctl(j, metrics={"vllm:num_requests_waiting": 6})
    s.api("DELETE", f"/api/v1/leases/{lease['lease_id']}", s.tok["alice"])
    wait_for(lambda: not owned_running(s), what="replicas released")
    assert s.dep()["target_replicas"] == 1 and any(m == "target 2 -> 1 (idle)" for m in s.events())
    s.acquire()
    wait_for(lambda: s.pending_jobs(), what="queued")
    time.sleep(2)
    d = s.dep()
    assert d["target_replicas"] == 1 and len([x for x in d["slots"] if x["state"] == "racing"]) == 1


def test_budget_blocked_scale_up_does_not_hold_scale_down(stack_factory):
    s = stack_factory(models__max_replicas=3, scaling__max_submissions_per_day=4)
    s.start()
    lease, a, b = _two_replicas(s)               # 2 candidates per slot: the budget is used up
    for j in (a, b):
        s.job_ctl(j, metrics={"vllm:num_requests_waiting": 6})
    wait_for(lambda: s.dep()["target_replicas"] == 3, what="scale up")
    wait_for(lambda: any(m.startswith("new replica deferred: daily submission budget") for m in s.events()),
             what="scale-up deferred")
    assert s.pending_jobs() == [] and not [x for x in s.dep()["slots"] if x["state"] == "racing"]
    for j in (a, b):
        s.job_ctl(j)
    wait_for(lambda: s.dep()["target_replicas"] == 1, what="scale down despite the deferred replica")
    wait_for(lambda: len(owned_running(s)) == 1, what="one replica cancelled")


def test_scale_up_stuck_in_queue_is_cancelled(stack_factory):
    s = stack_factory(models__max_replicas=2, leases__acquire_timeout_s=5)
    s.start()
    lease, a = s.ready_one()
    s.job_ctl(a, metrics={"vllm:num_requests_waiting": 6})
    wait_for(lambda: len(s.pending_jobs()) == 2, what="second slot racing")
    s.job_ctl(a)
    wait_for(lambda: any("scale-up not ready within 5s" in m for m in s.events()), what="stuck scale-up released")
    wait_for(lambda: not s.pending_jobs(), what="queued jobs cancelled")
    assert s.dep()["target_replicas"] == 1 and s.dep()["ready_replicas"] == 1


def test_replacement_limit_counts_finished_handovers(stack_factory):
    s = stack_factory(scaling__replace_before_s=100, scaling__max_replacements_per_day=1)
    s.start()
    lease, a = s.ready_one()
    assert s.chat(lease["inference_key"]).status_code == 200     # replacements need recent traffic
    s.slurm("set", a, "time_left=50")
    b = wait_for(lambda: start_pending(s), what="replacement job started")
    wait_for(lambda: s.jobs()[a]["state"] == "CANCELLED", what="handover finished")
    s.slurm("set", b, "time_left=50")
    wait_for(lambda: any(m.startswith("replacement limit reached (1 in 24 h), next allowed at ") for m in s.events()),
             what="replacement limit")
    time.sleep(2)
    assert s.pending_jobs() == [] and sum("racing to replace" in m for m in s.events()) == 1


def test_walltime_replacement_drains_inflight_stream(stack_factory):
    s = stack_factory(scaling__replace_before_s=100, scaling__drain_deadline_s=60)
    s.start()
    lease, a = s.ready_one()
    s.slurm("set", a, "time_left=50")
    s.job_ctl(a, chunk_delay=1.0)                # ~8 s stream on the old replica
    out = {}
    t = threading.Thread(target=lambda: out.update(r=s.chat(lease["inference_key"], stream=True)))
    t.start()
    wait_for(lambda: any("racing to replace" in m for m in s.events()), what="replacement racing")
    b = wait_for(lambda: start_pending(s), what="replacement job started")
    wait_for(lambda: any(f"draining" in m for m in s.events()), what="old replica draining")
    assert s.jobs()[a]["state"] == "RUNNING"     # still serving the in-flight stream
    t.join(30)
    assert out["r"].status_code == 200 and out["r"].text.rstrip().endswith("data: [DONE]")
    wait_for(lambda: s.jobs()[a]["state"] == "CANCELLED", what="old replica cancelled after drain")
    assert s.jobs()[b]["state"] == "RUNNING"
    assert s.chat(lease["inference_key"]).status_code == 200


def test_no_replacement_without_recent_traffic(stack_factory):
    s = stack_factory(scaling__replace_before_s=100, scaling__replace_requires_activity_s=1)
    s.start()
    lease, a = s.ready_one()
    s.slurm("set", a, "time_left=50")
    time.sleep(4)
    assert not any("racing to replace" in m for m in s.events())
    assert s.pending_jobs() == []
