# How the manager adds and releases capacity

An extra model server helps only if there is enough work to use it. When
requests are intermittent, spreading them across more servers can leave more
GPUs idle. The manager therefore shares existing capacity, requests more when
work builds up, and consolidates servers when demand stays low.

The aim is useful GPU utilization, not simply more servers or lower latency.
This page explains the current policy with its default settings. The
[decision reference](manager-scaling-reference.md) gives the exact tests and
configuration details.

## When does the manager request another server?

The manager looks for evidence that the current servers cannot keep up. It can
request one more server when work stays backed up throughout a five-minute
observation period, or when model servers repeatedly reject requests because
they are overloaded. A brief queue on its own is not enough.

Growth is gradual. The manager waits for an existing startup attempt to resolve
before requesting another server, and normally waits ten minutes between
capacity changes. This avoids accumulating allocations in response to a short
burst. The configured model limit and resource budgets still apply; a decision
to grow does not guarantee that the cluster will provide capacity.

Connection failures and invalid requests do not justify adding GPUs. Those
problems need connection recovery or a client-side correction instead.

## When can it release an extra server?

The manager can consolidate servers after five minutes of consistently low
combined demand, provided there have been no overload rejections. It counts
requests already running as well as those waiting: an empty queue does not
mean a server is idle. The ten-minute wait between changes applies here too.

With the defaults, two servers can be reduced to one if their combined number
of running and waiting requests never exceeds one during that period. This
counts requests, not how much computation each request needs.

A retiring server stops receiving new requests. Existing requests have up to
15 minutes to finish before its job is cancelled. This is called **draining**;
it avoids cutting off work immediately, but responses still running at the
deadline can be interrupted.

## Why does the last server stay?

A client reservation tells the manager that the model is still needed. While
any valid reservation remains, the manager seeks at least one server, even
between requests. Cluster availability still determines whether one can
actually run. The current policy does not deliberately sleep at zero servers
while retaining reservations and wake on the next request.

Once the last reservation ends, queued startup work is cancelled. Ready servers
can remain for ten minutes so another client can reuse them; after that, they
drain and are released. Reservations can end through explicit release, expiry
or inactivity. Renewing a reservation without using inference does not keep it
alive indefinitely. See [when client work ends](manager.md#when-work-endsor-stops-making-requests).

## Example: two researchers share a quiet model

Suppose two researchers have reservations for the same model and two servers
are ready. For five minutes, at most one request is running, none are waiting,
and neither client sees an overload rejection. Once the wait between capacity
changes has elapsed, the manager can retire one server and direct new work to
the other.

If one researcher finishes, the manager keeps the server for the other. Once
both reservations end, the server stays for the short reuse period, then drains
and is released. Only manager-owned serving jobs are cancelled, not the
researchers' benchmark programs or unrelated cluster jobs.

## What should an operator check?

**Request load is not GPU utilization.** The policy uses request queues and
model-server measurements, not the GPU utilization used by cluster enforcement.
Check actual cluster telemetry before concluding that an allocation is useful
or meets the site's rules. Do not manufacture traffic or repeatedly restart
allocations to avoid those rules.

The prototype also has known defects: it can restore extra servers without
fresh evidence of demand, leave scaling stuck when resource limits prevent a
startup, and undercount daily replacements. Inspect owned jobs rather than
relying on the daily replacement limit alone.
[Issue #9](https://github.com/alessandrocornacchia/sserve/issues/9)
tracks these findings and the measurement limitations.

Live GPU inference and utilization-policy compliance have not yet been
validated. The [operator guide](manager-operator.md#live-validation-status-25-october-2026)
records what was tested. For tuning and precise behavior, use the
[decision reference](manager-scaling-reference.md) and
[example configuration](../examples/sserved.example.toml).
