# ROCS guide

The [ROCS testbed](https://sands.kaust.edu.sa/internal/rocs-testbed/slurm-environment/)
is the SANDS group's SLURM cluster: A100 (80 GB and 40 GB), V100 and P100
GPUs. It is more relaxed than ORIX (smaller user base, 14-day jobs, a 15%
idle threshold), but it's only reachable through its VPN.

> These recipes were written from the ROCS documentation and haven't been run
> through sserve yet.

## Connecting

1. **VPN.** Connect the ROCS VPN (`rocs.ovpn`, shared on The ML Hub Slack). Off
   campus, connect the KAUST VPN first. `ping 172.18.0.10` should answer.
2. **SSH.** The head node is `mcmgt01`, and jobs can only be submitted from
   there. Following the ROCS docs, add `172.18.0.10 mcmgt01` to `/etc/hosts`
   and a `Host mcmgt01` entry with your KAUST username to `~/.ssh/config`.
   `ssh -o BatchMode=yes mcmgt01 true` should succeed.
3. **Cluster.** `config.example.toml` already has `[clusters.rocs]` with
   `host = "mcmgt01"`. Use it with `sserve -c rocs ...`, or set
   `default = "rocs"`.

sserve runs everything through `mcmgt01`, including the tunnel, which matches
ROCS's rule that only the head node accepts SSH.

## Which recipe to pick

| Recipe | GPUs | QoS | Can be interrupted |
|---|---|---|---|
| `default` | 1 × A100 80 GB (SXM, `mcnode22`) | `normal` | no |
| `spot` | 1 × A100 80 GB (SXM), including idle reserved ones | `spot` | yes; requeued automatically |
| `a100-40gb` | 2 × A100 40 GB (SXM), tensor parallel | `normal` | no |

```bash
sserve -c rocs up qwen3.6-35b-a3b-fp8              # default
sserve -c rocs up qwen3.6-35b-a3b-fp8:spot -d
sserve -c rocs up qwen3.6-35b-a3b-fp8:a100-40gb
```

`ginfo` on `mcmgt01` shows idle GPUs by type, and `ninfo` shows each node's GPUs
and features.

## Quotas and limits

- **`normal` QoS (the default).** At most 2 × A100, 8 × V100 and 8 × P100 per
  user at once. `a100-40gb` uses both A100s.
- **`spot` QoS.** No resource cap, but any `normal` job or reservation holder
  can preempt it. To also run on *reserved* idle GPUs, a job needs
  `--signal=R:...`, which the `spot` recipe sets.
- **Time.** Jobs can run up to 14 days, and interactive `srun` sessions up to
  4 hours. The docs announce that jobs longer than 3 days will have to be
  preemptible.
- **Reservations.** Time-based GPU reservations (e.g. an A100 80 GB every
  weekday 8am–5pm) can be requested through the ROCS GitHub issue tracker,
  with about a week's notice. Pass one with `-- --reservation=<name>`.

## Preemption

ROCS requeues preempted batch jobs by default: same job ID, port and key,
possibly another node. The `spot` recipe adds `--open-mode=append` so the logs
of every attempt are kept. Re-run `sserve up qwen3.6-35b-a3b-fp8:spot` to
re-attach the tunnel, and use `-d` (see
[limitations](how-it-works.md#not-done-yet)).

## Idle endpoints get cancelled

If any GPU of a job stays below **15%** utilisation for two consecutive 1-hour
windows, the job is cancelled (state `CANCELLED BY 64030`). You get an email
warning after the first hour. This is much more lenient than ORIX's 75%, but an
endpoint nobody is using still gets cancelled, so run `sserve down` when you're
done.

## GPU choice

- The ROCS guidelines ask for the weakest GPU that does the job. Qwen3.6-35B-A3B
  in FP8 needs about 35 GB for weights, so it needs an A100: 1 × 80 GB, or
  2 × 40 GB with tensor parallelism. V100 and P100 (16 GB) aren't an option.
- **The recipes avoid `acclnode06`.** Its A100 80 GB PCIe GPUs have shown odd
  behaviour, so every recipe asks for `gpu_sxm` as well as the GPU size. This
  leaves `mcnode22` (4 × A100 80 GB) for `default` and `spot`, and
  `mcnode23-25` for `a100-40gb`. SXM also matters for tensor parallelism:
  traffic between PCIe A100s is about 3× slower. If `mcnode22` is full, use
  `:a100-40gb`, or override the constraint with
  `-- --constraint=gpu_a100_80gb`, which allows `acclnode06` again.

## Environment

The recipes use the mamba environment `vllm-0.19-env`, activated by a login
shell. `[clusters.rocs]` runs `module load slurm` before every remote command;
drop that `prelude` if `sbatch` is already on your `PATH` on `mcmgt01`.
