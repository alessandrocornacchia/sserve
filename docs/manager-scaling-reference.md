# Scaling decision reference

Use [the policy explanation](manager-scaling.md) for the operator's view of when
capacity grows or shrinks. This reference records the current implementation's
exact conditions. Values below are package defaults unless stated otherwise;
the running configuration may override them.

## What is counted

A **replica** is one complete model server. Its GPU count and launch settings do
not change when the manager adds another replica. The **target** is the desired
replica count, which may differ from the number ready to serve.

Each replica position can have candidate jobs on several approved profiles.
These are alternatives for obtaining one replica: the first ready candidate
wins and its unused siblings are cancelled. They are not separate additions to
the target.

## Observation window and common gates

The metrics window is five minutes (`scaling.window_s = 300`). Checks normally
run every 15 seconds; scheduler and network delays can extend that interval.

Load-based scaling waits while an ordinary replica position is provisioning or
the current target has not been filled. After a target change, a ten-minute
cooldown (`scaling.cooldown_s = 600`) prevents another load-based change. A new
deployment has no earlier change to cool down from.

Active reservations keep the target at least one. Model and allocation limits
can prevent growth, and queues can delay a requested server becoming ready.

## Conditions for increasing the target

Either of the following triggers can increase the target by one, subject to the
common gates and the model's `max_replicas` limit.

### Sustained request pressure

Every observed sample in a complete window must satisfy at least one condition:

```text
total waiting requests / serving replicas >= 2

OR

one replica has BOTH:
    KV-cache usage >= 90%
    waiting requests > 0
```

The queue threshold is an average across replicas, not across time. The two
conditions may alternate between samples. A sample satisfying neither breaks
the sustained-pressure test. Cache occupancy and a waiting request must belong
to the same replica; high cache usage alone does not qualify.

The settings are `up_waiting_per_replica = 2.0` and `up_kv_usage = 0.90`.
KV cache holds model attention data; its occupancy is not GPU utilization.

### Repeated overload responses

Five backend-reported HTTP 429 or 503 failures within the last five minutes also
qualify (`up_capacity_errors = 5`). They can occur in a short burst and do not
require valid metrics or a full window of failures.

Counted events are failed client requests after routing retries, not individual
internal retry attempts. Connection errors, timeouts, invalid requests,
authentication errors and the manager's own no-capacity response do not count.

## Conditions for reducing the target

Reduction requires at least two serving replicas, a target above one, a complete
metrics window, no counted overload failures in that window, and satisfaction
of the common gates.

For `N` serving replicas, the test is:

```text
peak total (running + waiting requests) over the window
    <= (N - 1) * down_load_per_replica
```

`down_load_per_replica` defaults to one. Thus two serving replicas require a
combined peak of at most one request; three require at most two. These counts
measure concurrency, not the computational cost of requests.

The new target is based on `N - 1`, with a minimum of one while reservations
remain. It normally falls by one, but a mismatch between the stored target and
serving count can change the size of that adjustment. When choosing a position
to retire, provisioning positions take precedence; among serving positions,
the one with the least remaining walltime is preferred.

## Draining and release

A retiring server is removed from routing before its job is cancelled. Tracked
in-flight requests can finish until `drain_deadline_s = 900` elapses. Unused
non-winning candidates are cancelled without a serving drain.

When no active reservations remain, queued and provisioning positions are
retired immediately. Ready servers can remain for `leases.linger_s = 600`, with
no new scaling or walltime replacement. They then drain and are cancelled.
Setting `linger_s = 0` skips this reuse period.

Reservation expiry, preparation grace and inactivity are described in the
[client guide](manager.md#when-work-endsor-stops-making-requests). In-flight
inference protects against idle retirement; renewals and health probes are not
inference activity.

## Missing measurements and restarts

Metric-based decisions need running and waiting request counts from every
serving replica. A failed scrape or probe makes the window incomplete. Missing
KV-cache data disables that cache-pressure signal, but does not invalidate
otherwise valid request counts. The overload-response trigger is independent
of metrics availability.

Metrics samples and overload history are rebuilt after restart; the target and
last scaling time persist. A full metrics window normally has to accumulate
again after a restart or a new serving replica joins.

## Allocation limits and replacements

The following settings are under `[scaling]`:

| Setting | Package default | Shipped example |
|---|---:|---:|
| `max_total_gpus` | 16 | 8 |
| `max_startup_gpus` | 8 | 4 |
| `max_candidates_per_slot` | 3 | 3 |
| `max_submissions_per_day` | 12 | 12 |
| `max_replacements_per_day` | 8 | 8 |

The total GPU-request budget includes queued and running candidates. Per-profile
GPU limits also apply. The startup budget counts only duplicate candidates for
a position that are running but not ready, not all serving GPUs. Daily submission
and replacement counts are per deployment. **The daily replacement counter
currently undercounts promoted replacements; do not rely on that limit alone.**

The default model limit is two replicas; individual model configurations in the
example override it. See the [example configuration](../examples/sserved.example.toml)
for the complete settings rather than assuming its budgets match package defaults.

A walltime replacement maintains capacity rather than increasing the normal
target. The manager can request one in a job's last 30 minutes when real inference
was active within the preceding 30 minutes. One replacement is pursued at a
time. A ready replacement takes over while the old server drains, temporarily
allowing one replica above the target, subject to allocation budgets.

A policy or repeated-startup block prevents further submissions and replacements;
already ready replicas can continue serving. Resolve the cause before an
[operator reset](manager-operator.md#recovery-and-blocked-deployments).

## Implementation and open findings

The rules are implemented in [the manager](../sserved/manager.py), with defaults
in [configuration](../sserved/config.py). [Issue #9](https://github.com/alessandrocornacchia/sserve/issues/9)
tracks replacement accounting, stale targets after inactivity, budget-blocked
scaling and approximate alignment of metrics samples. These remain defects,
not changes to the policy described here.
