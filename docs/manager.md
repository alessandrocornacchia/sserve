# Use a shared inference service

The optional **sserved** manager lets you request a model and run an existing
client against one stable inference address. It handles the SLURM jobs and SSH
connections behind that address. You do not need cluster access, and your
benchmark can run on a different machine from the manager.

This is a prototype on `feat/manager-leases`, not a replacement for the existing
`sserve up` workflow. Start with the [operator guide](manager-operator.md) if you
are setting up the service rather than using one that already exists.

## Before you start

Ask the operator for:

- The manager URL, reachable from the machine or container running your client.
- Your personal client token, delivered through a private channel.
- A model identifier configured on that manager. A Hugging Face identifier alone
  does not guarantee that a suitable launch profile exists.

You need Python 3.11 or newer and Git on your client machine. No vLLM, LiteLLM,
SLURM installation or cluster SSH key is needed there.

## 1. Get the client

Run this in a terminal on **the machine that will start your benchmark**:

```bash
git clone --branch feat/manager-leases https://github.com/alessandrocornacchia/sserve.git
cd sserve
python3 ./sserve lease --help
```

The help should list `run`, `acquire`, `renew`, `release`, `wait`, `status`, `env`
and `rotate-key`. The existing installer follows `main`; use this branch's
client while trying the prototype.

## 2. Configure access

In the same terminal, set the URL supplied by your operator. This example uses
an HTTPS address; replace it with your service's address:

```bash
export SSERVE_MANAGER_URL='https://inference.example.org'
export SSERVE_MANAGER_TOKEN_FILE="$HOME/.config/sserve/client-token"
```

Save your token without putting it in shell history. These commands use Bash:

```bash
umask 077
mkdir -p "$HOME/.config/sserve"
read -rsp 'Client token: ' token; printf '\n'
printf '%s\n' "$token" > "$SSERVE_MANAGER_TOKEN_FILE"
unset token
chmod 600 "$SSERVE_MANAGER_TOKEN_FILE"
python3 ./sserve lease status
```

A successful first check may print nothing: you do not have a reservation yet.
An authentication or connection error means access is not ready; resolve it
before requesting compute.

For persistent configuration, add a `[manager]` table to
`~/.config/sserve/config.toml` instead of exporting the two variables:

```toml
[manager]
url = "https://inference.example.org"
token_file = "~/.config/sserve/client-token"
```

## 3. Request a model and run a client

Set `MODEL` to the identifier your operator supplied. For example, **only if the
operator has configured it**:

```bash
export MODEL='Qwen/Qwen3-0.6B'
python3 ./sserve lease run "$MODEL" -- python3 examples/client.py --stream
```

The wrapper requests a reservation, reports queue and startup progress, and
waits until a model server is ready. It then runs the example client, which
prints the model list and a generated answer. The wrapper renews the reservation
while the command runs and releases it when the command exits.

Use the same pattern for your own command:

```bash
python3 ./sserve lease run "$MODEL" -- ./run-my-benchmark.sh
```

The child receives `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL`,
`ANTHROPIC_AUTH_TOKEN`, `SSERVE_MODEL` and `SSERVE_LEASE_ID`. Software that reads
these variables can use them directly. Otherwise, configure its usual provider
settings to use these values; the wrapper does not rewrite benchmark config
files or choose the benchmark's model on its behalf.

The OpenAI base URL ends in `/v1`. The Anthropic base URL does not: its SDK adds
`/v1/messages`. Both credentials are scoped to this reservation and model.

If a benchmark creates containers, pass the endpoint and credential into those
containers through its normal configuration. A working request from the host
does not prove a container can reach the same address. `localhost` inside a
container or on another machine is not the manager.

## During a run

From another terminal with the same access configuration:

```bash
python3 ./sserve lease status
python3 ./sserve lease status --json
```

Status shows your reservations, queue reasons and serving state. Other clients
may share compatible capacity, but their reservations and credentials remain
separate. Ending your reservation does not end theirs.

By default, the wrapper waits up to two hours for readiness. Change that client
deadline with `--wait-timeout SECONDS`. A deadline ends this client's wait and
releases its reservation; it is not permission to cancel another client's jobs.

`--cluster NAME`, `--gpus N` and `--set KEY=VALUE` are optional. They work only
when the operator has configured and allowed them. Do not assume the manager
will resize, quantize or change a model to make an arbitrary request fit.

## When work ends—or stops making requests

Reservations have two different limits:

- **Expiry:** a client must renew before its reservation expires. `lease run`
  does this automatically. A disconnected or crashed client eventually loses
  its reservation.
- **Inactivity:** a client that keeps renewing but sends no inference requests
  can still become idle. Preparation grace allows initial setup; a request still
  in progress is not idle. Health checks and renewals do not count as inference.

Preparation grace begins when capacity first becomes available, not while you
are waiting in the scheduler queue. The defaults allow 30 minutes of preparation
followed by a 30-minute inference-idle interval. A reservation that never gets
capacity has a separate six-hour acquisition deadline. The wrapper's default
two-hour wait is shorter; change it if a longer wait is appropriate.

Ask the operator about limits for benchmarks with long pauses. `--prep-grace`
and `--idle-timeout` request different limits in seconds, within the service's
configured bounds.

Use Ctrl-C to stop a wrapped command normally. After the last reservation ends,
ready capacity can remain briefly for reuse before it is drained and released.
Unused queued candidates should not remain waiting for work that no longer
exists.

## Manual reservations

Prefer `lease run` for a whole benchmark. For interactive use, you can save a
reservation and load its environment in your shell:

```bash
python3 ./sserve lease acquire "$MODEL" --save experiment --wait 7200
eval "$(python3 ./sserve lease env experiment)"
python3 examples/client.py
python3 ./sserve lease renew experiment
python3 ./sserve lease release experiment
```

**Manual reservations do not renew themselves after the acquire command exits.**
The default expiry is five minutes; renew within that interval while you are
using one. `lease wait` by itself does not renew a reservation either. Saved
credentials live in `~/.config/sserve/leases/` with private file permissions.

If you lose the saved key but still have an active reservation, rotate it:

```bash
python3 ./sserve lease rotate-key LEASE_ID --save experiment
eval "$(python3 ./sserve lease env experiment)"
```

Replace `LEASE_ID` with the identifier from `lease status`. Rotation immediately
invalidates the old inference key. It does not create a new reservation or
extend its expiry.

## What interruptions mean

| What you see | What to do |
|---|---|
| Queued, with a scheduler reason | Capacity is not ready. Wait or discuss another approved profile with the operator. |
| Starting | The allocation exists, but the model is not yet admitted for requests. |
| HTTP 503 while your reservation is valid | No usable capacity is currently available, or the deployment is blocked. Check status and use bounded client retries. |
| HTTP 401 with `lease_expired`, `lease_idle` or `lease_released` | Acquire a new reservation; renewing an ended one does not restore it. |
| An unfulfilled reservation | The model never became ready within the service's acquisition deadline. Check the cause before acquiring again. |
| Blocked deployment | Give the operator the status reason. Policy and persistent startup failures require investigation, not an endless retry loop. |
| A stream ends prematurely | Treat that request as failed. The service cannot continue a response lost to preemption or a broken connection. |

The address remains stable when jobs change, but availability is opportunistic.
Queue delays and interruptions can still exhaust a benchmark's own retry budget.
Keep its usual result persistence and resume mechanisms enabled.

See the [operator guide](manager-operator.md) for setup and recovery, and the
[management overview](manager-overview.md) for the workflow and division of
responsibilities.
