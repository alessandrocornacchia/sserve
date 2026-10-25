# sserve

**Run a vLLM server as a SLURM job and use it from your laptop as an
OpenAI-compatible endpoint.**

sserve is a small command-line utility. Given an SSH-reachable SLURM login node
and a recipe (an `sbatch` script that starts vLLM), it submits the job, waits
until the server is healthy, and tunnels it to `localhost` with a per-run API
key. Any OpenAI client can then use it.

sserve does not write recipes for you. It runs the ones you give it. You can
add your own (see [recipes](docs/recipes.md)), and you should check that a
recipe works on your cluster with a plain `sbatch` before you use it through
sserve. The repo ships a few [example recipes](#examples-for-sands-lab).

```console
$ sserve up my-model -d
[09:37:06] submitting my-model:default on login1
[09:38:39] job 298010 RUNNING (on gpu-node-6, loading the model)
[09:42:06] vLLM is healthy on gpu-node-6:22646

my-model:default is up on gpu-node-6 (job 298010)
  base url : http://127.0.0.1:8000/v1
  model    : org/My-Model
  env      : eval "$(sserve env my-model)"

$ eval "$(sserve env my-model)"     # OPENAI_BASE_URL, OPENAI_API_KEY, SSERVE_MODEL
```

## What it does

- Submits the recipe, waits for `/health`, opens an `ssh -L` tunnel through the
  login node, and checks it with the key.
- Gives every run a fresh API key, so other cluster users can't reach your server.
- Reopens the tunnel after a laptop sleep. Running `up` from another
  machine attaches to the running job.
- Ctrl-C or `sserve down` cancels the job.

It only needs `python3` and `ssh` locally, and SSH access to the login node.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/alessandrocornacchia/sserve/main/install.sh | sh
```

This fetches the repo into `~/.local/share/sserve` (re-run it to update), links
`sserve` into `~/.local/bin`, and creates `~/.config/sserve/config.toml` if
you don't have one. From a clone, run `./install.sh` instead.

Edit the config to point at your cluster's login node (an alias from
`~/.ssh/config`). See [configuration](docs/configuration.md). Then:

```bash
sserve config show     # where sserve is installed and what it read from the config
sserve ls              # the recipes on the cluster
sserve up <model>      # start one
```

## Test the endpoint

After `eval "$(sserve env <model>)"` (add `-c <cluster>` if it isn't your default),
`OPENAI_BASE_URL`, `OPENAI_API_KEY` and `SSERVE_MODEL` are set.

With the bundled client (standard library only):

```bash
python3 examples/client.py                          # list models, send one prompt
python3 examples/client.py "Write a haiku" --stream
```

With `curl`:

```bash
curl -s "$OPENAI_BASE_URL/models" -H "Authorization: Bearer $OPENAI_API_KEY"

curl -s "$OPENAI_BASE_URL/chat/completions" \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model": "'"$SSERVE_MODEL"'", "messages": [{"role": "user", "content": "Hello!"}]}'
```

With the OpenAI SDK (`pip install openai`):

```python
import os
from openai import OpenAI
client = OpenAI()   # reads OPENAI_BASE_URL and OPENAI_API_KEY
r = client.chat.completions.create(model=os.environ["SSERVE_MODEL"],
                                   messages=[{"role": "user", "content": "Hello!"}])
print(r.choices[0].message.content)
```

## Commands

| | |
|---|---|
| `sserve ls [-w]` | list the recipes on the cluster (`-w`: with their paths) |
| `sserve cat <model>[:<recipe>]` | show a recipe and its path (opens in `spf`/`yazi` if installed; `--plain` to print) |
| `sserve up <model>[:<recipe>]` | launch a model (or attach to it if it's already running) and connect |
| `sserve up <model> -d` | the same, but return once ready and leave it running |
| `sserve env <model>` | print the environment variables for your client |
| `sserve status` | show what's running, where, and how much time it has left |
| `sserve logs <model> [-f]` | show the job's logs |
| `sserve down <model>` | stop the job and close the connection |
| `sserve push` | upload your local recipes to the cluster |
| `sserve pull [<model>]` | download the cluster's recipes into your local copy |
| `sserve sync [<model>]` | like `pull`, and delete local files the cluster doesn't have |
| `sserve config show` | show the install location, config file and resolved clusters |

`-c <cluster>` picks a cluster from the config (or any SSH host). Without it,
sserve uses `$SSERVE_CLUSTER`, then `default` in the config.

## Examples for SANDS Lab

Example recipes for the two clusters the SANDS Lab uses. Each guide covers
access, quotas and which recipe to pick.

| Cluster | Model | Recipes | Guide |
|---|---|---|---|
| ORIX | `deepseek-v4-flash` | `default`, `h100`, `pi`, `freecycle` | [docs/orix.md](docs/orix.md) |
| ROCS | `qwen3.6-35b-a3b-fp8` | `default`, `spot`, `a100-40gb` | [docs/rocs.md](docs/rocs.md) |

## Learn more

- [How it works](docs/how-it-works.md): architecture, where state and keys live, limitations
- [Configuration](docs/configuration.md): `config.toml`, multiple clusters, environment variables
- [Recipes](docs/recipes.md): the recipe contract, `push`/`ls`, and how to write one
- [Testing](docs/testing.md): the Docker SLURM cluster and end-to-end test
- [Issue automation](docs/automation.md): how Claude plans and implements issues in this repo
