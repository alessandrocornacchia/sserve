# sserve

**Your own LLM endpoint on a SLURM GPU cluster, one command away.**

sserve takes a model setup that's already known to work on your cluster,
submits it as a SLURM job, waits until vLLM is ready, and connects it to your
laptop. You get an OpenAI-compatible URL on `localhost` that works with any
client, agent or SDK.

```console
$ sserve up deepseek-v4-flash -d
[09:37:06] submitting deepseek-v4-flash:default on orix
[09:38:39] job 298010 RUNNING (on orix-worker-h200-6, loading the model)
[09:42:06] vLLM is healthy on orix-worker-h200-6:22646

deepseek-v4-flash:default is up on orix-worker-h200-6 (job 298010)
  base url : http://127.0.0.1:8000/v1
  model    : deepseek-ai/DeepSeek-V4-Flash-0731

$ eval "$(sserve env deepseek-v4-flash)"     # OPENAI_BASE_URL, OPENAI_API_KEY, SSERVE_MODEL
```

## What it does

- **One command from nothing to a working endpoint.** It submits the job, waits
  through the queue and model loading, opens the connection and checks it.
- **Models known to work.** Each cluster has a small catalogue of tested
  setups ("recipes"), so you don't have to work out vLLM flags, GPU counts and
  CUDA quirks from scratch.
- **Works with your existing tools.** Anything that speaks the OpenAI API
  (Python and JS SDKs, LiteLLM, coding agents, `curl`) just needs two
  environment variables.
- **Private by default.** Every run gets a fresh API key, so other cluster
  users can't use your GPUs through your endpoint.
- **Survives the real world.** If your laptop sleeps or the network drops,
  the connection reopens on the same port. Running `up` again from another
  machine attaches to the job that's already running instead of starting a
  second one.
- **Cleans up after itself.** Ctrl-C (or `sserve down`) cancels the job and
  frees the GPUs.
- **Several clusters, one tool.** You can switch clusters with `-c`. All sserve
  needs is SSH access to the login node.

## Quick start

```bash
# 1. Install
git clone https://github.com/alessandrocornacchia/sserve && sserve/install.sh

# 2. See what you can run
sserve ls

# 3. Start a model and use it
sserve up deepseek-v4-flash
```

The starter config expects an `orix` alias in `~/.ssh/config`. See
[configuration](docs/configuration.md) to add other clusters.

Then, from any OpenAI client:

```python
from openai import OpenAI
client = OpenAI()   # picks up OPENAI_BASE_URL and OPENAI_API_KEY
client.chat.completions.create(model="deepseek-ai/DeepSeek-V4-Flash-0731",
                               messages=[{"role": "user", "content": "Hello!"}])
```

## Commands

| | |
|---|---|
| `sserve ls` | list the models you can launch |
| `sserve up <model>[:<recipe>]` | launch a model (or attach to it if it's already running) and connect |
| `sserve up <model> -d` | the same, but return once ready and leave it running |
| `sserve env <model>` | print the environment variables for your client |
| `sserve status` | show what's running, where, and how much time it has left |
| `sserve logs <model> [-f]` | show the job's logs |
| `sserve down <model>` | stop the job and close the connection |
| `sserve push` | upload your local recipes to the cluster |

## Available models

| Cluster | Model | Recipes |
|---|---|---|
| ORIX (H100/H200) | `deepseek-v4-flash` | `default` (H200), `h100`, `pi` (group reservation), `freecycle` (preemptible) |
| ROCS (A100) | `qwen3.6-35b-a3b-fp8` | `default` |

## Learn more

- [How it works](docs/how-it-works.md): architecture, where state and keys live, behaviour details, limitations
- [Configuration](docs/configuration.md): installing, `config.toml`, multiple clusters
- [Recipes](docs/recipes.md): the catalogue, `push`/`ls`, and how to write a recipe
- [ORIX guide](docs/orix.md): partitions, quotas, preemption and idle-job rules on ORIX
