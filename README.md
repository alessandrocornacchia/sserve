# sserve

Serve a vetted vLLM recipe on a SLURM cluster and reach it from your machine with one command.

```bash
sserve up deepseek-v4-flash          # submit, wait for /health, tunnel to localhost
eval "$(sserve env deepseek-v4-flash)"   # OPENAI_BASE_URL, OPENAI_API_KEY, SSERVE_MODEL
```

```
this machine ──ssh──► login node ──sbatch──► compute node: vLLM :$VLLM_PORT
     ▲                    │
     └── ssh -L tunnel ───┘     http://127.0.0.1:8000/v1
```

It only needs SSH access to the login node. It never logs into a compute node,
and it leaves nothing running on the login node except sshd forwarding the
tunnel. The login node checks `/health` with `curl`.

## Install

On the machine that runs the agent, you need `python3` and `ssh`. Python
>= 3.11 is needed only for the config file; if you don't have it, run
`pixi global install python`.

```bash
ln -s "$PWD/sserve/sserve" ~/.local/bin/sserve
mkdir -p ~/.config/sserve && cp sserve/config.example.toml ~/.config/sserve/config.toml   # edit host
```

`host` is an SSH destination, so put the user, key and any `ProxyJump` in
`~/.ssh/config`. Test it with `ssh -o BatchMode=yes <host> true`.

## The catalogue

sserve only runs models that already work on a given cluster. Recipes are
specific to a cluster (GPU types, partitions, environments), so the repo keeps
one catalogue per cluster, named as in `config.toml`:

```
recipes/
  orix/                           # [clusters.orix]: H100/H200
    deepseek-v4-flash/
      default.sbatch              sserve up deepseek-v4-flash             batch-h200, shared pool
      h100.sbatch                 sserve up deepseek-v4-flash:h100        batch-h100, shared pool
      pi.sbatch                   sserve up deepseek-v4-flash:pi          pi-caninim reservation (or -r pi)
      freecycle.sbatch            sserve up deepseek-v4-flash:freecycle   preemptible, requeued
      serve.sh                    the vLLM command all four recipes share
  mcm/                            # [clusters.mcm]: A100
    qwen3.6-35b-a3b-fp8/
      default.sbatch              sserve -c mcm up qwen3.6-35b-a3b-fp8
```

`sserve push` uploads the selected cluster's catalogue to its `deploy_dir`
(`~/.vllm/deploy/<label>/` by default). `sserve push <label>` uploads a single
model; a directory path works too. `sserve ls` lists the recipes on the
cluster, using each recipe's `# sserve: <description>` line, and marks where
the cluster and the repo disagree:

```
deepseek-v4-flash         DeepSeek-V4-Flash, 1 node x 4 H200, ...
deepseek-v4-flash:8xh200  [remote only] hand-made on the cluster
qwen-next                 [not pushed] ...
llama-x                   [differs] ...         # same name, different content
```

Set `recipes_dir` in the config to keep a cluster's catalogue somewhere else.

### ORIX notes

On ORIX ([cluster policy](https://orix.kaust.edu.sa/cluster_policy/)), the
recipe variant picks the partition:

- **`default` (`batch-h200`, QoS `batch`).** Not preemptible, but the QoS caps
  the whole PI account at 8 GPU / 128 CPU / 1438 GB, so a group-mate's big job
  leaves yours pending with `MaxMemoryPerAccount`. `/opt/orix/utils/susage`
  shows what is free. Name `batch-h200` explicitly: the bare `batch`
  meta-partition also offers the job to `batch-h100`, which reports
  `BadConstraints` for an H200 request (harmless, but noisy). An explicit
  partition also needs an explicit `--qos=batch`.
- **`h100` (`batch-h100`, QoS `batch`).** The same model on 4 x H100: weights take
  ~45.5 GiB per GPU, which leaves ~19 GiB of KV cache (~750k tokens per DP rank,
  about 5.7 concurrent 128k-token requests). Same batch cap as `default`, shared
  across both GPU types. A single job never mixes H100 and H200.
- **`pi` (`pi-caninim`).** Reserved and never preempted; it evicts freecycle
  jobs. It is a 4-GPU half node, so this recipe takes the whole group's
  reservation.
- **`freecycle` (`freecycle-h200`).** Higher limits, but any other job can
  preempt it (SIGTERM, 60 s grace). The recipe passes `--requeue`, so vLLM comes
  back with the same job ID, port and key, probably on another node. Run
  `sserve up <label>:freecycle` again to re-attach the tunnel.
- **Idle endpoints get cancelled.** A GPU averaging below 75% utilisation (50%
  on `pi-*`) for two consecutive 1-hour windows gets the job cancelled, and
  repeatedly holding idle GPUs can get the account suspended. The recipes ask
  for 4 h; run `sserve down` as soon as you are done, and pass
  `-- --time=...` for longer runs that keep the GPUs busy.

### Recipe contract

A recipe is a normal sbatch script plus three rules:

```bash
#!/bin/bash --login
# sserve: one-line description shown by `sserve ls`
#SBATCH --gres=gpu:...  --time=...           # resources are the recipe's business

: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"

exec vllm serve <model> --host 0.0.0.0 --port "$VLLM_PORT" ...
```

1. Serve on `$VLLM_PORT`. sserve picks a random port from 20000 to 29999 for
   each run, so two jobs on the same node don't clash.
2. Export `VLLM_API_KEY` from `$SSERVE_KEY_FILE`. Each run gets a fresh key.
   Without one, anyone on the cluster network could use your endpoint.
3. Run the API server on the job's first node.

Recipes still work with a plain `sbatch` (port 8000, no key). sserve also
exports `SSERVE_RUN_DIR`, `SSERVE_LABEL`, and `SSERVE_RECIPE_DIR`; the last
one is the recipe's directory, for files kept next to the recipe such as chat
templates.

sserve sets `--job-name`, `--chdir`, `--output` and `--error` itself, which
override the recipe's values. Logs go to the run directory.

## Commands

| | |
|---|---|
| `sserve ls` | recipes on the cluster, with `[not pushed]` / `[differs]` / `[remote only]` |
| `sserve up <label>[:<recipe>]` | submit (or attach to a running job), wait, tunnel; **Ctrl-C cancels the job** |
| `sserve up <label> -d` | same, but return once ready and leave job + tunnel running |
| `sserve up <label> -- --time=4:00:00` | anything after `--` goes to `sbatch` |
| `sserve env <label>` | `export OPENAI_*` lines (also saved as a `.env` file) |
| `sserve status` | sserve jobs on the cluster, time left, local tunnels |
| `sserve logs <label> [-f]` | SLURM stdout/stderr of the run |
| `sserve down <label>` / `--all` | cancel the job, close the tunnel |
| `sserve push [<label>…]` | upload `recipes/<cluster>/` (or some labels) to the cluster |

Global `-c <cluster>` picks a cluster from the config, or takes a bare SSH host.

## Behaviour worth knowing

- **One deployment per model per cluster.** The job is named `sserve-<label>`.
  Running `up` again for a model that is already running attaches to it,
  including from another machine: sserve reads the port and key from the job's
  run directory. Asking for a different recipe of a running model is an error
  until you run `sserve down`.
- **Ctrl-C only cancels jobs this machine submitted.** If `up` attached to an
  existing job, Ctrl-C closes only the tunnel. Use `--keep-job` to never cancel.
- **The tunnel heals.** In foreground mode it is checked every 30 s and
  reopened on the same local port, for example after the laptop sleeps. After
  `-d`, running `up` again reopens it.
- **Run directories** (`~/.vllm/runs/<label>-<recipe>-<timestamp>/`) keep the
  recipe snapshot and SLURM logs. `down` deletes the key file but keeps the
  rest.
- **Local state** is kept in `~/.local/state/sserve/<cluster>/` (mode 600,
  because it holds the key). The SSH sockets are in `~/.cache/sserve/`. All
  remote commands share one multiplexed SSH connection, so polling doesn't open
  a new login every 15 s.
- `--bind 0.0.0.0` exposes the tunnel to containers on this machine. Anyone
  who can reach that port still needs the key.

## Not done yet

- Multi-node recipes (for example the 2 × 8 GPU DeepSeek head/worker setup in
  `../slurm-vllm/`) should work if the API server runs on the first node, but
  this hasn't been tried.
- No retry when the node's port is already taken. sserve detects
  `address already in use` in the log and tells you to run `up` again.
- No check that the recipe follows the contract before submitting. A recipe
  that ignores the key still runs, but `up` warns that the endpoint answers
  without one. A recipe that ignores the port times out on `/health`.
