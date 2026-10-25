# Testing

`tests/slurm/` runs sserve end to end against a small SLURM cluster in Docker:
a login node you reach over SSH and one CPU compute node, `c1`, built from the
prebuilt images of [slurm-docker-cluster](https://github.com/giovtorres/slurm-docker-cluster).
There is no GPU, so the `mock-llm` recipes run a stand-in for vLLM
(`serve.py`) that serves `/health` and the OpenAI endpoints, checks the API
key, and echoes each prompt back.

```bash
eval "$(tests/slurm/cluster.sh up)"   # ~30 s once the images are pulled; sets SSERVE_CONFIG
tests/slurm/smoke.sh                  # push, ls, cat, pull, up, chat, status, logs, down, a failing job
./sserve -c test up mock-llm -d       # or try commands by hand
ssh sserve-slurm squeue               # the login node, as root
tests/slurm/cluster.sh down           # stop it and delete its volumes
```

It needs Docker with Compose v2, and port 3022 on 127.0.0.1
(`SSERVE_SLURM_SSH_PORT` picks another).

`cluster.sh up` makes an SSH key in `tests/slurm/.state/`, and adds a
`Host sserve-slurm` block for it to the top of `~/.ssh/config`, between marker
comments. `down` removes the block. Its sserve config, in `.state/config.toml`,
names the cluster `test` and keeps the catalogue and run directories on
`/data`, the only directory the login and compute nodes share.

| Recipe | What it tests |
|---|---|
| `mock-llm` | a recipe that follows the contract: port, key, server on the first node |
| `mock-llm:broken` | a job that exits before serving |

When you add a command or change behaviour the cluster can show, add a check to
`smoke.sh`, and a recipe here if it needs one.

## In CI

`.github/workflows/slurm-test.yml` runs the smoke test on every push to `main`
and every pull request. The Claude implementation workflow
([automation](automation.md)) starts the same cluster for the agent to try its
change on, then runs the smoke test and reports the result in the PR it opens.

What this doesn't cover: GPUs and real vLLM, `module load` preludes, partitions,
accounts and queue limits, preemption and requeue, and a login node that isn't
the SLURM controller. Check those on a real cluster.
