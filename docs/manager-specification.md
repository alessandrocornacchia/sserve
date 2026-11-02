# Opportunistic inference: prototype requirements

## Goal

A client names a model and receives a stable inference address. The service
finds suitable KAUST compute capacity, maintains it while useful demand exists,
and releases it when that demand ends. Clients may run on different machines;
benchmarks should need endpoint, credential and model configuration, not source
changes.

## Required workflow

1. An operator approves cluster profiles, model launch settings and budgets.
2. An authenticated client requests a model with optional permitted resource or
   cluster preferences. Defaults make a model identifier sufficient for a
   configured model.
3. The client receives its own renewable reservation and can inspect queue,
   startup, readiness and blocked states.
4. The benchmark uses one stable address while the service changes the compute
   behind it. Compatible clients may share capacity without sharing credentials.
5. The client releases its reservation on exit. Expiry and genuine inactivity
   reclaim forgotten work; probes and heartbeats do not manufacture demand.

## Service requirements

| Area | Requirement |
|---|---|
| Remote clients | Authenticated access from independent machines; no client needs cluster credentials. Separate clients cannot manage each other's reservations. |
| Inference | OpenAI chat and Anthropic messages compatibility, streaming, tool calls and model context metadata for existing benchmark adapters. |
| Scheduler | Short jobs, approximately three hours by default, with requeue eligibility and separate reconciliation of failures and expiry. |
| Capacity search | Race approved batch, preemptible and explicitly enabled PI profiles. Declare a winner only when usable; cancel only its surplus candidates. |
| Recovery | Distinguish failed connectivity from failed allocation. Follow requeues and changed nodes; never infer termination from scheduler uncertainty alone. |
| Ownership | Persist job identity and the full enabled profile set. Resolve ambiguous submission before retrying. Cancel only manager-owned work. |
| Availability | Route only to ready model servers; report temporary unavailability when none can serve. Keep the client address stable across model-server changes. |
| Scaling | Add model servers for sustained capacity pressure, not arbitrary request errors or low GPU utilization. Use windows, cooldowns and explicit bounds. |
| Resource release | Stop unused model servers and release unused reservations. Protect requests already in progress and bound draining. |
| Policy | Stop on utilization-policy cancellation or persistent misconfiguration. Do not create artificial traffic or repeatedly resubmit to bypass site rules. |
| Operations | Single manager, persistent state, supervised child processes, private credentials and safe network defaults. |

## Lessons carried forward

Past benchmark operation exposed several failures this prototype must avoid:

- A dropped SSH tunnel can leave a healthy serving job unreachable. Restore
  connectivity before requesting another allocation.
- An allocated GPU is not necessarily a ready model server. Loading, cache
  allocation, graph capture and tool-parser checks can still fail.
- Restarting a manager must not silently forget enabled compute options.
- Extra model servers can divide already intermittent demand and worsen utilization.
- Process existence is not endpoint health; exited child processes must be
  collected rather than mistaken for useful capacity.
- Cluster utilization rules are not vLLM queue or cache metrics. Measure and
  report those signals separately.
- A model card's memory claim is not a validated launch profile. Never silently
  quantize or change precision to make a request fit.

## Boundaries

`--requeue` is not a guarantee that every timeout or cancellation restarts a job.
A stable address is not a promise of uninterrupted service: spare capacity can
disappear, and an interrupted stream cannot be transparently continued.

The prototype manages serving capacity, not benchmark processes or result
directories. It does not resume experiments, alter their concurrency or change
their scoring. Those remain the benchmark runner's responsibility.

## Validation standard

Exercise concurrent clients, expiry and inactivity, partition races, scheduler
uncertainty, node changes, terminal failures, policy stops, tunnel recovery,
restart, streaming interruption, scaling and credential isolation. Use real
LiteLLM in integration tests and validate remote/container addressing.

Simulated infrastructure tests and live cluster tests must be reported
separately. Live validation requires a bounded allocation and cleanup limited to
the test's own resources. Neither a passing configuration check nor a healthy
host-only endpoint establishes that a real benchmark container can use a model.

See the [client guide](manager.md), [operator guide](manager-operator.md) and
[management overview](manager-overview.md). The
[current scaling policy](manager-scaling.md) documents the implemented decision
rules and their rationale.
