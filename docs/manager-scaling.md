# Scaling policy

The objective is to serve genuine inference demand with as few GPU allocations
as practical. Extra model servers are justified by sustained pressure; spare
servers should be released when demand can be consolidated. This prioritizes
useful resource utilization over minimizing latency at any cost.

## Design principles

1. **Share capacity before adding it.** Compatible clients share model servers.
   Another client reservation is not, by itself, a reason to allocate another
   server. Reservations establish a need for service; request pressure justifies
   additional capacity.
2. **Consolidate sparse demand.** Distributing intermittent requests across more
   servers can leave more GPUs underused. Count both running and waiting work
   when deciding whether fewer servers are sufficient.
3. **Avoid repeated allocation and cancellation.** Use sustained observations
   for metric-based decisions and a cooldown between adjustments. Treat repeated
   overload rejections as a separate pressure signal rather than waiting for
   complete metrics.
4. **Release resources without immediately interrupting work.** Stop sending
   requests to a retiring server, allow existing requests to finish, and bound
   that wait so retirement cannot hold an allocation indefinitely.
5. **Respect cluster authority and explicit budgets.** Use only approved compute
   profiles and bounded allocations. Do not generate artificial traffic or
   repeatedly resubmit underused work to evade utilization rules.

The current implementation uses request load as a proxy for useful capacity.
It does **not** measure GPU utilization or establish compliance with a cluster's
GPU-utilization policy.

## Mechanism

A **replica** is one complete model server. Scaling changes the number of
replicas, not the model's precision or GPUs per server. The **target** is the
desired replica count; scheduler queues and startup delays can keep the actual
serving count below it. Decisions apply to each deployment of a model with
compatible launch settings. All values below are defaults unless noted.

### 1. Scale out: evidence of pressure

There are **two independent triggers**. Either can justify one additional
replica, subject to the gates and limits in section 3.

#### Trigger A: sustained queue or cache pressure

Across a complete **five-minute window**, every observed sample must satisfy
at least one of these conditions:

```text
total waiting requests / number of serving replicas >= 2

OR

at least one replica has BOTH:
    KV-cache occupancy >= 90%
    waiting requests > 0
```

The first condition measures waiting requests per replica, not a time average.
The second requires cache pressure and waiting work on the same replica. KV
cache holds model attention data; high occupancy alone can reflect retained
data rather than unmet demand, so it does not trigger growth without waiting
requests.

The two conditions may alternate between samples. If a sample satisfies
neither, the sustained-pressure test fails. Checks normally run every 15
seconds, though scheduler or network operations can delay them.

#### Trigger B: repeated overload rejections

**Five backend-reported HTTP 429 or 503 failures within five minutes** also
qualify. These are failed client requests after routing retries, not individual
retry attempts. They can occur in a short burst; this trigger does not require
a full window of valid metrics.

Connection failures, timeouts, invalid requests, authentication errors and the
manager's own no-capacity response do not count. Adding GPUs does not repair
those failures. Backend overload rejections are different: they are direct
evidence that real work was not accepted.

With no ordinary replica still starting, a growth decision requests one more
replica, up to the model's configured limit. In steady state, this changes the
target from `N` serving replicas to `N + 1`.

### 2. Scale in: evidence of spare capacity

Scale-in requires at least two serving replicas, a target above one, a complete
five-minute metrics window, and **no counted overload failures** in that window.
It also obeys the common gates in section 3.

For `N` serving replicas, the default test is:

```text
peak total (running + waiting requests) over the window <= N - 1
```

More generally, the right-hand side is
`(N - 1) * scaling.down_load_per_replica`, whose multiplier defaults to one.
If the test passes, the target becomes `N - 1`, bounded by the model limit and
the floor of one while reservations remain.

For example, two replicas can become one if the combined load stays at most
one running or waiting request throughout the window. Three can become two
if the combined peak is at most two. These transitions still require the other
conditions above; sustained scale-out pressure takes precedence when growth is
allowed.

The rationale is to consolidate consistently sparse work, not react to an empty
queue at one instant. Counting running requests protects against mistaking a
server executing long requests for an idle server. The rule measures request
concurrency, not computational cost or a learned estimate of server capacity.

### 3. Stability and allocation limits

Two gates apply to both load-based growth and shrink decisions:

- **No unresolved ordinary replica startup.** Wait for capacity already being
  requested before making another load-based adjustment. Walltime replacements
  are handled separately in section 6.
- **Ten-minute cooldown after a scaling target change.** This separates
  adjustments so a traffic fluctuation does not repeatedly add and remove
  allocations. The cooldown persists across restarts; a new deployment has no
  earlier change to cool down from.

The model's `max_replicas` bounds the target and defaults to two. Submission
budgets separately bound the jobs requested to realize that target. A higher
target therefore does not guarantee another allocation.

| Allocation setting | Package default | Shipped example |
|---|---:|---:|
| Total GPU-request budget | 16 GPUs | 8 GPUs |
| Duplicate candidates' startup budget | 8 GPUs | 4 GPUs |
| Candidate profiles per replica position | 3 | 3 |
| Submissions per deployment per day | 12 | 12 |

A replica position may race candidate jobs on approved profiles. These jobs are
alternatives for obtaining one replica: the first ready candidate wins and its
unused siblings are cancelled. The total GPU budget counts queued and running
candidates; per-profile limits also apply. The duplicate-startup budget counts
additional candidates for the same position that are running but not yet ready,
not all serving GPUs. See the [configuration example](../examples/sserved.example.toml)
for the corresponding settings and per-model limits.

### 4. Scale to zero: end of client demand

While any client reservation remains valid, the target has a **minimum of one**.
This trades some idle capacity between requests for avoiding a new allocation
and model startup after every pause. It is a desired baseline, not a guarantee
that the cluster can supply a running server.

Once no active reservations remain:

1. Retire queued and provisioning candidates immediately.
2. Keep ready replicas for a **ten-minute reuse period**, without further
   scaling or walltime replacement.
3. Drain and cancel the remaining replicas.

The reuse period is `leases.linger_s`; setting it to zero skips the wait. This
shutdown path does not wait for a low-load metrics window or scaling cooldown.
Releasing one client's reservation does not end another client's access.

Reservations end through release, expiry or inference inactivity. Renewals and
health probes are not inference activity; a request still in flight protects
against idle retirement. The [client guide](manager.md#when-work-endsor-stops-making-requests)
defines the expiry and inactivity timers.

Load-based one-to-zero scaling while retaining valid reservations is **not
implemented**. It would require a separate wake-up policy; otherwise the same
reservation would immediately request capacity again.

### 5. Retirement: drain before cancellation

A retiring replica is removed from routing so it receives no new requests.
Tracked in-flight requests may finish for up to **15 minutes** before its job is
cancelled. Responses still active at that deadline can be interrupted. Unused
non-winning candidates are cancelled without this serving drain.

When choosing a position to retire, provisioning positions take precedence;
among serving positions, the one with the least remaining walltime is preferred.
Cancellation is limited to manager-owned serving jobs. It does not kill or
re-enqueue benchmark programs, or cancel unrelated cluster jobs.

### 6. Walltime replacement: maintain capacity, not grow it

A replacement addresses an approaching scheduler time limit, not increased
demand. The manager can request one during a serving job's **last 30 minutes**
only if real inference was active within the preceding 30 minutes. Without
recent activity, it does not chain allocations merely to keep the model loaded.

Only one replacement is pursued at a time. Once it is ready, the old replica
drains. The handover can temporarily put one replica above the ordinary target,
subject to allocation budgets. The configured daily replacement limit defaults
to eight, but its accounting is currently defective, as described below.

## Missing measurements and implementation limits

Metric-based decisions require running and waiting counts from every serving
replica over the complete window. Failed scrapes or invalid measurements block
those decisions; the overload-rejection trigger remains independent. Missing
KV-cache data disables that signal without invalidating otherwise valid request
counts. After restart, metrics and overload history are rebuilt, while the
target and cooldown persist. A full metrics window normally has to accumulate
again after a restart or a new replica joins.

[Issue #9](https://github.com/alessandrocornacchia/sserve/issues/9) tracks defects
that can make current behavior depart from the intended controls:

- Promoted replacements disappear from the daily counter, so the replacement
  limit is not reliably enforced. Other submission and GPU limits still apply.
- An old target can survive inactivity and restore extra replicas without fresh
  pressure evidence.
- Budget-blocked startup can stall scaling; a blocked model can also change its
  stored target without being able to obtain capacity.
- Metrics streams are combined by sample position rather than exact timestamps,
  making the aggregate approximate when sampling is irregular.

A policy or repeated-startup block prevents new submissions and replacements;
already ready replicas can continue serving. Resolve the cause before an
[operator reset](manager-operator.md#recovery-and-blocked-deployments), rather
than retrying to evade enforcement.

Live GPU inference and utilization-policy compliance have not yet been validated;
the [operator guide](manager-operator.md#live-validation-status-25-october-2026)
records the evidence. The implementation is in [the manager](../sserved/manager.py),
with default thresholds and timers in [configuration](../sserved/config.py).
