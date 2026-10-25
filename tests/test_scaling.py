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
