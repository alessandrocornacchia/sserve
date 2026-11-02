# ORIX guide

[ORIX](https://orix.kaust.edu.sa/) is KAUST's 128-GPU cluster: 8 nodes of
8 × H100 80 GB and 8 nodes of 8 × H200 141 GB. Its scheduling rules affect
when a model starts and how long it stays up, so the
`deepseek-v4-flash` recipes come in one version per partition.

## Which recipe to pick

| Recipe | Partition / QoS | Starts | Can be interrupted |
|---|---|---|---|
| `default` | `batch-h200` / `batch` | when the shared pool and your group's quota allow | no |
| `h100` | `batch-h100` / `batch` | same as `default`, on H100 instead | no |
| `pi` | `pi-caninim` / `pi-caninim` | right away, unless a group-mate is using the reservation | no |
| `freecycle` | `freecycle-h200` / `freecycle` | usually soon, on any idle H200 | yes, at any time |

`/opt/orix/utils/susage` on the login node shows what is free in each
partition and how much of your group's quota is in use.

## Partitions and quotas

- **`batch` (shared pool).** Not preemptible. The QoS caps the whole PI
  account, across `batch-h100` and `batch-h200` combined, at 8 GPU / 128 CPU /
  1438 GB at once. If a group-mate is using that, your job stays pending with
  `MaxMemoryPerAccount` or `MaxGRESPerAccount`.
- **`pi-<pi username>` (group reservation).** Guaranteed capacity for selected
  PI groups, and it preempts freecycle jobs. `pi-caninim` is a half node
  (4 H200 on `orix-worker-h200-7`), so the `pi` recipe takes the whole
  group's reservation.
- **`freecycle` (preemptible).** Has access to every idle GPU, with higher
  limits, but any other job can preempt it.

A single job never mixes H100 and H200, which is why there are separate
`*-h100` and `*-h200` partitions.

The recipes name `batch-h200` instead of the `batch` meta-partition. `batch`
also offers the job to `batch-h100`, which makes SLURM briefly report
`BadConstraints` for an H200 request (harmless, but noisy). An explicit
partition also needs an explicit `--qos=batch`; without it, SLURM rejects the
job with `Invalid qos specification`.

## Preemption

A preempted job gets SIGTERM and 60 s of grace. The `freecycle` recipe passes
`--requeue`, so the job comes back with the same job ID, port and key, probably
on another node, and `--open-mode=append` keeps the logs of every attempt. Run
`sserve up deepseek-v4-flash:freecycle` again to re-attach the tunnel. Use `-d`
with this recipe: in foreground mode, sserve currently cancels a requeued job
(see [limitations](how-it-works.md#not-done-yet)).

## Idle endpoints get cancelled

ORIX cancels jobs whose GPUs sit idle. If a GPU averages below **75%**
utilisation (**50%** on `pi-*`) for two consecutive 1-hour windows, the job is
cancelled. Repeatedly holding idle GPUs can get the account suspended.

A chat endpoint is often idle, so:

- run `sserve down` as soon as you're done;
- the recipes ask for 4 hours; pass `-- --time=...` only for runs that keep the
  GPUs busy;
- before starting a second model server, make sure there is enough traffic to keep
  both busy.

## DeepSeek-V4-Flash on H100 and H200

The weights are about 156 GB of FP8 (146.6 GiB of routed experts, 8.8 GiB of
everything else). With 4 data-parallel ranks and expert parallelism, each GPU
holds about 45.5 GiB of weights.

| | KV cache per GPU | 128k-token requests at once, per rank |
|---|---|---|
| 4 × H100 (measured) | 19.3 GiB, ~750k tokens | ~5.7 |
| 4 × H200 | about 80 GiB (estimate) | several times more |

Both start in about 4 to 5 minutes once the job is running.

The model's thinking mode is off by default. Turn it on for tasks that need
careful lookup or reasoning:

```python
client.chat.completions.create(..., extra_body={"chat_template_kwargs": {"thinking": True}})
```

## ORIX-specific setup

The recipes rely on a few things in the user's environment:

- vLLM 0.24.0 in `~/venvs/vllm`
- the model in the Hugging Face cache under `$HF_HOME` (set by the login shell;
  the recipes run offline)
- NVIDIA's CUDA 12.9 forward-compatibility library, unpacked under
  `/mnt/data/u/cornaca/cuda-compat-12-9/`, because the nodes' driver only
  supports CUDA 12.8 and vLLM 0.24.0 ships CUDA 12.9 builds
