# Operate an sserved manager

Run one manager on a persistent machine that can reach your cluster login nodes.
Researchers connect to that manager over HTTP or HTTPS; they do not need your
cluster keys. Each compute job runs vLLM, not the manager or a benchmark.

For client setup, use [the client guide](manager.md). This page is for the person
responsible for cluster access, launch profiles and service operation.

> **Resource safety:** requesting a model can submit GPU jobs. Review every
> enabled profile and the allocation budgets before granting clients access.
> Stopping the manager does **not** cancel its jobs; it leaves them for recovery.
> Release reservations and verify job cleanup before shutting the service down
> for an extended period. Never use `scancel --me` as this service's cleanup.

## Prerequisites

On **the manager machine**:

- Python 3.11 or newer, Git, OpenSSH and [uv](https://docs.astral.sh/uv/).
- Non-interactive SSH access to the cluster login nodes, with verified host keys
  and credentials that remain usable during an unattended run.
- Network access to the clusters. For ROCS, arrange its VPN separately; the
  manager does not start VPNs or change SSH configuration.
- A persistent private state directory. Run only one controller per state
  directory and do not copy its state to a second active controller.

On **each configured compute cluster**:

- An approved account, partition and resource request.
- A working vLLM environment and a tested `serve.sh` recipe accessible from the
  allocated node. Weights and caches must be staged under site-approved storage
  and download procedures.
- The login node must be able to reach the allocated node's vLLM port for SSH
  forwarding. The manager does not require direct SSH login to compute nodes.

The manager does not install vLLM, download arbitrary models or infer a model's
GPU requirements for you. Validate the launch environment first. Weight size
alone does not account for KV cache, graph capture or the requested context.

## 1. Install the prototype

In a terminal on **the manager machine**:

```bash
git clone --branch feat/manager-leases https://github.com/alessandrocornacchia/sserve.git
cd sserve
uv sync
uv run sserved --help
```

The manager is an optional package. The existing manual `sserve up`, `down` and
other commands retain their separate workflow and state.

## 2. Configure clusters, profiles and models

Start from the supplied example, on **the manager machine**:

```bash
umask 077
mkdir -p ~/.config/sserve
cp examples/sserved.example.toml ~/.config/sserve/sserved.toml
```

Edit `~/.config/sserve/sserved.toml` before starting the service. The example is
written for Alessandro's ORIX environment: replace its SSH destination and
review the account, paths and approved resource limits. Remove models and
profiles you do not intend to make available. Keep the small Qwen model for an
initial test; do not offer the large Ornith profile until its allocation is
approved and its recipe has been validated.

The configuration has three related parts:

- **Cluster:** the SSH login destination and remote recipe/run directories.
- **Profile:** an approved scheduler request, including partition, QoS, account,
  GPU type and walltime.
- **Model:** the Hugging Face identifier clients request, its recipe, GPU count,
  context limit and the profiles it may use.

Different profiles can race to serve the same replica. Put all permitted race
candidates in the model's `profiles` list. The first candidate that becomes
healthy wins; only its unused siblings are cancelled. Extra replicas are a
separate decision based on sustained demand.

The following settings illustrate a conservative starting point:

```toml
[server]
listen = "127.0.0.1:8470"
advertised_url = "http://127.0.0.1:8470"
state_dir = "~/.local/state/sserved"

[scaling]
max_total_gpus = 4
max_startup_gpus = 1
max_candidates_per_slot = 2
max_submissions_per_day = 12
max_replacements_per_day = 4
```

These are only service and budget settings, not a complete cluster/model
configuration. `max_total_gpus` includes queued candidate requests, so it must
allow the race and replacement capacity you intend to use. A deployment can be
waiting on its own budget even when SLURM has free GPUs.

Cluster profile names, accounts and GPU types are site-specific. Do not copy PI
account names from another user's examples. Set a reserved profile's
`opt_in = true` to keep it disabled until you explicitly enable it.

### Recipe contract

The manager starts `<deploy_dir>/<recipe>/serve.sh` under a login shell. The
recipe must honor:

| Variable | Purpose |
|---|---|
| `VLLM_PORT` | The backend port chosen for this candidate. |
| `SSERVE_KEY_FILE` | Private file containing the backend API key; require this key when serving. |
| `SSERVE_MODEL`, `SSERVE_SERVED_MODEL` | Requested and externally served model identifiers. |
| `SSERVE_GPUS`, `TENSOR_PARALLEL_SIZE` | Allocated GPU count; use this only if the recipe supports that parallel configuration. |
| `MAX_MODEL_LEN` | Requested context limit, when configured. |

The manager supplies these variables; it does not rewrite hardcoded flags in an
existing recipe. In particular, a recipe with a fixed tensor-parallel size must
not advertise arbitrary GPU overrides. Clients cannot change model precision or
enable remote model code through the lease interface.

The supplied small-model recipe is
[`recipes/orix/qwen3-0.6b/serve.sh`](../recipes/orix/qwen3-0.6b/serve.sh).
Review its vLLM environment and CUDA compatibility paths before deploying it;
they are specific to the example ORIX environment. The recipe expects cached
weights and runs offline by default. Prepare the model through your site's
approved download procedure rather than allocating idle GPUs for staging.

Copy that recipe to the cluster's configured deployment directory. In this
example, run these commands on **the manager machine**, after replacing the SSH
destination with the one from your configuration:

```bash
export CLUSTER_HOST='your-kaust-user@your-cluster-login'
ssh "$CLUSTER_HOST" 'mkdir -p ~/.vllm/deploy/qwen3-0.6b'
rsync -av recipes/orix/qwen3-0.6b/ "$CLUSTER_HOST:.vllm/deploy/qwen3-0.6b/"
```

Use the matching directory if you changed `deploy_dir`. Deploy every recipe
listed in the model catalog before giving clients access. The manager starts
`serve.sh` directly; it generates its own scheduler script rather than running
the recipe directory's `default.sbatch`.

Keep `max_model_len` consistent with the running server. The manager exposes
model context information to clients, including benchmark adapters that need it.
For tool-using workloads, enable the profile's bounded readiness inference check
and use the correct tool parser. An HTTP health response alone does not establish
tool-use compatibility.

### Validate before serving

```bash
uv run sserved -c ~/.config/sserve/sserved.toml check
uv run sserved -c ~/.config/sserve/sserved.toml profile list
```

`check` should report `"ok": true`. It validates configuration and listener
exposure; it does **not** prove that cluster access, model fit or the recipe works.
Resolve warnings and verify enabled profiles before continuing.

## 3. Give clients access and start the manager

Create one token per client or independently managed benchmark runner. Keep the
token files private and transfer them through an approved private channel:

```bash
umask 077
uv run sserved -c ~/.config/sserve/sserved.toml token create researcher-a > researcher-a.token
uv run sserved -c ~/.config/sserve/sserved.toml token create operator --admin > operator.token
uv run sserved -c ~/.config/sserve/sserved.toml token list
uv run sserved -c ~/.config/sserve/sserved.toml serve
```

The last command stays in the foreground. Keep that terminal open for an initial
test. For unattended operation, use your normal service supervisor to run the
same command under the same user, configuration and persistent state directory.
Do not start multiple daemon workers against one state directory.

No model is requested merely by starting the service. Give a client the manager
URL, its token and a configured model identifier, then follow
[the client guide](manager.md). Check a real request from the actual benchmark
machine or container, not only from this host.

## Remote access

Loopback is the default. For clients on other machines, expose the service only
through a trusted private network or a TLS-protected endpoint. The client token
and inference key are bearer credentials: anyone who obtains one can use it.

### TLS at a reverse proxy

Keep the manager bound to `127.0.0.1:8470`. Terminate HTTPS at your organization's
reverse proxy and forward to that address. For example, with Caddy and a domain
you control:

```caddyfile
inference.example.org {
    reverse_proxy 127.0.0.1:8470
}
```

Configure the manager's `advertised_url` as `https://inference.example.org`.
The URL must be reachable from every client, and its TLS certificate must be
trusted there. Preserve streaming and allow long-running inference responses
through any proxy or load balancer in front of the manager.

If the service is exposed under a path prefix, set `root_path` to that prefix
and include it in `advertised_url`. For instance, use `root_path = "/inference"`
with `advertised_url = "https://example.org/inference"`. Do not append `/v1` to
the advertised URL; the client helper supplies the protocol-specific base path.

### Direct TLS or a private listener

The manager also accepts `tls_cert` and `tls_key` paths for direct TLS. Supply
both and restrict access to the private key.

A private-network listener can use a specific private IP with a matching
advertised URL. Plain HTTP still transmits credentials unencrypted; a private IP
does not make an untrusted shared network safe. Wildcard or public plaintext
binding is refused unless the operator explicitly declares a trusted network.
Do not use that override as a substitute for TLS on a public network.

## Routine operation

Run these commands in another terminal on **the manager machine**:

```bash
uv run sserved -c ~/.config/sserve/sserved.toml status
uv run sserved -c ~/.config/sserve/sserved.toml profile list
```

The local status command shows deployment identifiers, target replicas, owned
jobs, queue reasons and blocked conditions. Clients see their own reservations
through `sserve lease status`; an administrator token can inspect all clients.

Profile switches persist in state across restarts:

```bash
uv run sserved -c ~/.config/sserve/sserved.toml profile enable PROFILE_NAME
uv run sserved -c ~/.config/sserve/sserved.toml profile disable PROFILE_NAME
```

Replace `PROFILE_NAME` with a configured profile name. Disabling a profile
prevents new submissions through it; verify what is already allocated before
making assumptions about resource release. Editing its original `enabled`
default is not a replacement for the persisted operator switch.

### Scaling and resource efficiency

Read [how the manager adds and releases capacity](manager-scaling.md) for the
implemented scale-out and scale-in rules, defaults, warm period, draining,
allocation limits and known gaps. The objective is useful GPU occupancy, but the
current decisions use request-load proxies rather than measured GPU utilization.

### Recovery and blocked deployments

- **Lost tunnel:** the manager repairs connectivity to the existing allocation
  before deciding to replace the job. A dropped connection is not proof that
  vLLM or its SLURM allocation has failed.
- **Preemption or requeue:** the manager checks the job's new location and
  readmits it only after readiness checks. A requeued job can keep its ID while
  moving to another node.
- **Timeout:** `--requeue` makes a job eligible for scheduler requeue; it does
  not promise automatic restart after walltime expires. Replacement is a
  separate manager action, subject to demand and budgets.
- **Ambiguous submission:** the manager reconciles the outcome before retrying,
  rather than assuming no job exists and creating a duplicate.
- **Policy or repeated startup failure:** investigate the status reason and
  cluster logs. A blocked deployment is deliberate protection against wasteful
  resubmission, not an invitation to restart the daemon until it works.

Blocks are model-wide: a new reservation, cluster choice or launch override
cannot bypass one. After resolving the cause, an operator can reset the model:

```bash
uv run sserved -c ~/.config/sserve/sserved.toml reset MODEL
```

Replace `MODEL` with its configured identifier; a deployment identifier from
`status` is also accepted. The reset clears the model-wide block, not just that
one reservation. Do not reset a utilization-related block merely to run the same
underused work again. Scale decisions use vLLM demand signals; they do not
establish compliance with a cluster's GPU-utilization policy. Check the current
[ORIX](orix.md) or [ROCS](rocs.md) rules and approved project scope.

### Release resources before an extended shutdown

Stop accepting new work operationally, then have clients end their wrapped
commands or release their reservations. An operator can use an administrator
token with the normal client command to release a known reservation:

```bash
export SSERVE_MANAGER_URL='https://inference.example.org'
export SSERVE_MANAGER_TOKEN_FILE="$PWD/operator.token"
python3 ./sserve lease status
python3 ./sserve lease release LEASE_ID
uv run sserved -c ~/.config/sserve/sserved.toml status
```

Replace `LEASE_ID` with the reservation being retired. Releasing one does not
release other clients' reservations. Wait for the configured linger and bounded
drain, and verify that owned jobs have ended before stopping the daemon.

Stopping and restarting the daemon with the **same state directory** is a
recovery operation: it closes local connections and reconciles the jobs it
previously owned. It does not adopt manually launched `sserve` jobs or unrelated
benchmark finalizers. Do not delete state while owned jobs remain allocated.

### Change launch configuration safely

Retire affected reservations and jobs before removing a model, cluster or
profile, or changing an active model's launch settings. Startup refuses
incompatible configuration changes while the old work is still tracked. If that
happens, restore the previous configuration, start the manager and retire the
affected work normally; do not delete its state to silence the check.

Persisted profile enable/disable switches are separate from launch settings.
They are the normal way to control which existing profiles accept new work.

Do not replace a deployed `serve.sh` while queued or running jobs use it:
requeued jobs run that file again. Use versioned recipe directories for changes.
The startup configuration check does not detect files edited directly on the
cluster.

## Prototype limits

- The service cannot guarantee GPU queue times or uninterrupted inference.
  Already-started streams are not replayed after an interruption.
- The HTTP surface is deliberately narrower than a full provider proxy: the
  prototype focuses on chat completions, messages and model metadata. Check the
  implementation's supported routes before relying on embeddings, Responses,
  file APIs or token-count endpoints.
- Inference extensions are restricted to supported request parameters. Clients
  cannot supply a different backend URL, credentials, provider or routing policy.
- Model defaults are operator-managed. There is no automatic model sizing,
  quantization, arbitrary launch-code execution or promise that every Hugging
  Face model works.
- One controller and local persistent state are not a highly available service.
  Back up state securely and coordinate maintenance with clients.
- Short walltimes can cause gaps when replacement capacity starts too late.
  Autoscaling and replacement are bounded by the configured scheduler and GPU
  budgets, not guaranteed capacity reservations.

## Validate changes without allocating GPUs

On a development machine, from this checkout:

```bash
uv sync --extra test
uv run pytest -q
```

The test harness uses a simulated scheduler and vLLM endpoints while exercising
the real LiteLLM integration. Passing these tests is not evidence that a given
KAUST profile, runtime, model or network path has been validated on live GPUs.

On 25 October 2026, the final source passed **76 tests**. Coverage includes
independent clients, lease expiry and inactivity, partition races, uncertain
scheduler responses, requeue relocation, replacement, scaling, policy stops,
stream interruption, authentication, container addressing and process cleanup.
The scheduler and inference backends in that suite are simulated; LiteLLM is
real. The suite also covers the connection-path and shutdown problems found
during the live attempt below.

### Live validation status: 25 October 2026

A bounded ORIX attempt submitted one batch-H100 job requesting one GPU for at
most ten minutes. Job `341920` remained queued and was cancelled at the
fifteen-minute queue cutoff. No second job was submitted for this check.

This exercised real cluster access and the submission/cancellation path, but
**did not validate GPU inference**: the model never became available. Real vLLM
generation, metrics, tool calls, preemption and benchmark-container access still
need a live allocation. Do not mistake the simulated integration tests for that
evidence or treat this prototype as a production availability guarantee.

The client also reached the real manager from a separate container for the
reservation workflow; that is not a successful inference test. Cleanup confirmed
no live test job, manager, client or connection helper remained. The prepared
public model cache and task-owned recipe/log artifacts were retained for a
future authorized check, not as a running service.
