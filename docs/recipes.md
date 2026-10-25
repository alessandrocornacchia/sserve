# Recipes

sserve only runs models that already work on a given cluster. Recipes are
specific to a cluster (GPU types, partitions, environments), so the repository
keeps one catalogue per cluster, named as in `config.toml`:

```
recipes/
  orix/                           # [clusters.orix]: H100/H200
    deepseek-v4-flash/
      default.sbatch              sserve up deepseek-v4-flash             batch-h200, shared pool
      h100.sbatch                 sserve up deepseek-v4-flash:h100        batch-h100, shared pool
      pi.sbatch                   sserve up deepseek-v4-flash:pi          pi-caninim reservation (or -r pi)
      freecycle.sbatch            sserve up deepseek-v4-flash:freecycle   preemptible, requeued
      serve.sh                    the vLLM command all four recipes share
  rocs/                           # [clusters.rocs]: A100/V100/P100
    qwen3.6-35b-a3b-fp8/
      default.sbatch              sserve -c rocs up qwen3.6-35b-a3b-fp8             1 x A100 80GB
      spot.sbatch                 sserve -c rocs up qwen3.6-35b-a3b-fp8:spot        preemptible, requeued
      a100-40gb.sbatch            sserve -c rocs up qwen3.6-35b-a3b-fp8:a100-40gb   2 x A100 40GB, TP=2
      serve.sh                    the vLLM command the three recipes share
```

A model's directory is its **label**. Each `*.sbatch` file in it is a
**recipe**: `default` runs with `sserve up <label>`, and any other one with
`sserve up <label>:<recipe>` or `-r <recipe>`.

## Uploading and listing

`sserve push` uploads the selected cluster's catalogue to its `deploy_dir`
(`~/.vllm/deploy/<label>/` by default). `sserve push <label>` uploads a single
model; a directory path works too, with `--label` to rename it.

`sserve ls` lists the recipes of the selected cluster (`TARGET`), using each
recipe's `# sserve: <description>` line:

```
RECIPE                    TARGET  DESCRIPTION
deepseek-v4-flash         orix    DeepSeek-V4-Flash, 1 node x 4 H200, ...
deepseek-v4-flash:8xh200  orix    hand-made on the cluster
```

It lists both the recipes on the cluster and the local ones in
`recipes/<cluster>/`. If a local recipe isn't pushed yet, or differs from the
cluster's copy, `ls` ends with a reminder to run `sserve push` (and `sserve pull`
for recipes that exist only on the cluster, or differ). Only `*.sbatch`
files are compared, so run `sserve push` yourself after editing a helper such
as `serve.sh`.

`sserve ls --wide` (`-w`) adds a `PATH` column on the right: `host:/path/to/recipe.sbatch`
for the copy on the cluster (the one `sserve up` runs), or the local file for
a recipe that isn't pushed yet.

`sserve pull` goes the other way: it copies the cluster's catalogue (or the
labels you name) into `recipes/<cluster>/`, e.g. to keep a recipe that was
hand-made on the cluster. `sserve sync` does the same and also deletes local
files the cluster's copy lacks, so each label ends up identical to the
cluster's; labels that exist only locally are left alone. Both list the changes
(`+` new, `~` overwritten, `-` deleted) and ask before overwriting or deleting
anything (`-y` skips the question).

Before submitting, `sserve up` compares the label's local directory, helpers
included, with the cluster's. If a local file is new or different, it lists the
files and asks whether to push the local copy and deploy it (`d` shows a diff).
Answering no deploys the cluster's copy as it is. `--push` pushes without
asking, and `--no-check` skips the comparison. Without a terminal to ask on, it
deploys the cluster's copy. Attaching to a running job skips the check.

`sserve cat <label>[:<recipe>]` shows a recipe and its path. It reads the local
copy in `recipes/<cluster>/` when there is one, and otherwise the cluster's
(`--remote` always reads the cluster's). On a terminal, a local recipe opens in
a file manager when one is installed: `spf` on Linux opens the recipe's
directory, `yazi` on macOS opens on the file. `--plain`, or piping the output,
prints the file instead, with a `==> path <==` header on stderr.

## Writing a recipe

A recipe is a normal sbatch script plus three rules:

```bash
#!/bin/bash --login
# sserve: one-line description shown by `sserve ls`
#SBATCH --gres=gpu:...  --time=...           # resources are the recipe's business

: "${VLLM_PORT:=8000}"
[[ -n "${SSERVE_KEY_FILE:-}" ]] && export VLLM_API_KEY="$(<"$SSERVE_KEY_FILE")"

exec vllm serve <model> --host 0.0.0.0 --port "$VLLM_PORT" ...
```

1. **Serve on `$VLLM_PORT`.** sserve picks a random port from 20000 to 29999
   for each run, so two jobs on the same node don't clash.
2. **Export `VLLM_API_KEY` from `$SSERVE_KEY_FILE`.** Each run gets a fresh
   key. Without one, anyone on the cluster network could use your endpoint.
3. **Run the API server on the job's first node.**

Recipes still work with a plain `sbatch` (port 8000, no key). sserve also
exports `SSERVE_RUN_DIR`, `SSERVE_LABEL`, and `SSERVE_RECIPE_DIR`. The last
one is the recipe's directory, for files kept next to the recipe, such as chat
templates or a shared launch script.

sserve sets `--job-name`, `--chdir`, `--output` and `--error` itself, and these
override the recipe's values. Logs go to the run directory.

When several recipes of one model only differ in where they run, keep the
`vllm serve` command in a shared script and `exec` it from each recipe, as
`recipes/orix/deepseek-v4-flash/` does with `serve.sh`.
