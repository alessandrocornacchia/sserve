# Scaling policy

**Goal:** improve GPU utilization by sharing model servers and matching their
number to real demand.

## Design principles

- **Share first:** another client does not automatically require another model server.
- **Scale on evidence:** add capacity for pressure; consolidate sustained low load.
- **Avoid oscillation:** observe demand over time and wait between adjustments.
- **Drain before stopping:** let active requests finish, with a deadline.
- **Respect cluster policy:** obey resource limits; never manufacture load to evade enforcement.

## Scaling decisions

`N` counts ready model servers; `target` is the desired count. These scaling
decisions exclude planned replacements. The pseudocode uses default settings.

Check approximately every **15 seconds**, but do not scale out while a model
server is starting, or scale at all within **10 minutes** of the previous
scaling change.

Each model server's samples carry their scrape time. A window point combines
the samples of all ready model servers taken within half a check interval of
each other. Points missing a model server are dropped, and fewer than two
complete points mean no metric decision.

### 1. Scale out: evidence of pressure

There are **two independent triggers**:

**A. Sustained pressure:** every sample in a complete **five-minute window** satisfies:

```python
total_waiting / N >= 2 or any(
    server.kv_usage >= 0.90 and server.waiting > 0
    for server in ready_servers
)
```

Cache occupancy alone is insufficient: the same model server must have waiting
requests. The qualifying condition may differ between samples.

**B. Repeated overload:** at least **five model-server HTTP 429/503 failures in
five minutes**, counted after routing retries. This does not require complete
metrics. Connection failures, invalid requests and the manager's own no-capacity
responses do not count.

Either trigger sets `target = N + 1` when below `max_replicas` (default: **2**).
Scale-out takes precedence over scale-in.

### 2. Scale in: evidence of spare capacity

Otherwise, with complete five-minute metrics and no overload failures:

```python
if N >= 2 and target > 1 and peak_total_running_and_waiting <= N - 1:
    target = N - 1
```

Counting running requests prevents an empty queue from being mistaken for idle
capacity. The configurable threshold is `(N - 1) * down_load_per_replica`, with
a default multiplier of **1**. A model server that is still starting does not
hold scale-in back: scaling in cancels it first.

## Lifecycle rules

| Condition | Action |
|---|---|
| Active reservations | Keep `target >= 1`; scaling to zero and waking on the next request is not implemented. |
| No active reservations | Cancel startup jobs; keep ready model servers for **10 minutes**, then drain and cancel their jobs and reset `target` to 1. This bypasses the scaling window and cooldown. |
| Demand returns after that | Start one model server. Scaling out again needs evidence from it; overload from before is ignored. |
| A budget blocks a submission | Do not take a slot: status events show the model server as `deferred` with the reason, and the next check tries again. Other scaling decisions continue. |
| An extra model server is not ready within `leases.acquire_timeout_s` | Cancel its jobs, set `target` to the ready count, and wait for the cooldown before scaling out again. |
| Stop a model server | Stop new requests, wait up to **15 minutes** for active requests, then cancel its job; unfinished responses may be interrupted. |
| Job has at most **30 minutes** left | If inference was active within the last **30 minutes**, start one replacement; drain the old model server only after the replacement is ready. |

The shutdown delay is `leases.linger_s`; zero disables it. Only one replacement
is pursued at a time, temporarily allowing one model server above the target.
`max_replacements_per_day` counts replacements started in the last 24 hours,
including finished and failed ones.
Only manager-owned jobs are cancelled—not benchmark programs or unrelated jobs.

## Constraints

- Model, GPU and submission limits still apply; queued jobs count toward GPU budgets.
- Missing request metrics block metric-based decisions, not the overload trigger;
  missing KV-cache data disables only the cache test.
- Target, cooldown and replacement starts persist across restarts; metrics and
  overload history do not.

> Request load is not GPU utilization: these rules do not prove compliance with
> cluster utilization requirements.

See [reservation expiry](manager.md#when-work-endsor-stops-making-requests),
[configuration](../examples/sserved.example.toml) and
[implementation](../sserved/manager.py) for operational details.
