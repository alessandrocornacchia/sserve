# How it works

```
this machine ──ssh──► login node ──sbatch──► compute node: vLLM :$VLLM_PORT
     ▲                    │
     └── ssh -L tunnel ───┘     http://127.0.0.1:8000/v1
```

sserve only needs SSH access to the login node. It never logs into a compute
node, and it leaves nothing running on the login node except sshd forwarding
the tunnel.

`sserve up` does this:

1. Copies the recipe into a fresh run directory on the cluster, writes a new API
   key there, and submits the job with `sbatch`. Each run gets a random port
   from 20000 to 29999, so two jobs on the same node don't clash.
2. Every 15 s, asks SLURM for the job state. Once the job is running, it asks
   the login node to `curl` vLLM's `/health` on the compute node.
3. When `/health` answers, it opens an `ssh -L` tunnel from a local port (the
   first free one from 8000) to the node, through the login node.
4. Calls `/v1/models` through the tunnel with the key. This checks both the path
   and the authentication, and finds the served model name.

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
- **One SSH login.** All remote commands share one multiplexed SSH connection,
  so polling doesn't open a new login every 15 s.
- **`--bind 0.0.0.0`** exposes the tunnel to containers on this machine. Anyone
  who can reach that port still needs the key.
- **Extra `sbatch` options.** Anything after `--` goes to `sbatch`, for example
  `sserve up <label> -- --time=8:00:00`.

## Where things are stored

On the cluster, each run has its own directory,
`~/.vllm/runs/<label>-<recipe>-<timestamp>/` (mode 700):

| File | Contents |
|---|---|
| `api_key` | the run's API key (mode 600); `down` deletes it |
| `sserve.env` | label, recipe and port, used to re-attach from another machine |
| `recipe.sbatch` | a snapshot of the recipe that was submitted |
| `slurm-<jobid>.out` / `.err` | the job's logs |

The compute node isn't stored: sserve asks SLURM for it while the job runs.

On your machine:

- `~/.local/state/sserve/<cluster>/<label>.json` and `.env` keep the
  deployment details, including the key (mode 600).
- `~/.cache/sserve/` holds the SSH control sockets.

## Not done yet

- **Multi-node recipes** should work if the API server runs on the job's first
  node, but this hasn't been tried.
- **No retry when the node's port is taken.** sserve detects
  `address already in use` in the log and tells you to run `up` again.
- **No contract check before submitting.** A recipe that ignores the key still
  runs, but `up` warns that the endpoint answers without one. A recipe that
  ignores the port times out on `/health`.
- **Requeued jobs aren't followed.** In foreground mode, `up` cancels a
  preemptible job that SLURM requeued. After a requeue onto another node,
  re-running `up` can keep the old tunnel. With `-d`, nothing tells you the job
  was preempted.
- **Only one model server per model.** You can't run two jobs of the same label at
  once, for example behind a load balancer.
