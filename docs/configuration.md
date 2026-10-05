# Configuration

## Install

```bash
git clone https://github.com/alessandrocornacchia/sserve && sserve/install.sh
```

`install.sh` links `sserve` into `~/.local/bin` (or `$SSERVE_BIN`) and copies
`config.example.toml` to `~/.config/sserve/config.toml` (or `$SSERVE_CONFIG`)
if you don't have a config yet. It's safe to run again: an existing config is
never overwritten. It checks for `python3` and `ssh`, and tells you if
`~/.local/bin` isn't on your `PATH`. The config file needs Python 3.11 or newer;
if you don't have it, run `pixi global install python`.

## The config file

`~/.config/sserve/config.toml` (or wherever `$SSERVE_CONFIG` points) lists your
clusters. Each cluster's name is also the name of its recipe folder:
`[clusters.orix]` uses `recipes/orix/`.

```toml
default = "orix"

[clusters.orix]
host = "orix"                        # ssh destination
deploy_dir = "~/.vllm/deploy"        # recipe catalogue on the cluster (default)
runs_dir = "~/.vllm/runs"            # one directory per run (default)
# sbatch_args = ["--account=myproject"]   # added to every submission
# recipes_dir = "~/my-recipes/orix"       # local catalogue (default: recipes/<cluster name>)

[clusters.rocs]
host = "mcmgt01"
prelude = "module load slurm"        # run before every remote command, if sbatch isn't on PATH
```

`host` is an SSH destination, so put the user, key and any `ProxyJump` in
`~/.ssh/config`:

```
Host orix
  User <username>
  Hostname orix-login.kaust.edu.sa
  IdentityFile ~/.ssh/<key>
```

Test it with `ssh -o BatchMode=yes orix true`. sserve uses batch mode, so it
can't answer password or passphrase prompts. Use a key loaded in your agent.

## Picking a cluster

- `-c <name>` picks a cluster from the config.
- `-c <ssh host>` works without any config, using the defaults above.
- Otherwise sserve uses `$SSERVE_CLUSTER`, then `default` from the config.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SSERVE_CONFIG` | `~/.config/sserve/config.toml` | config file |
| `SSERVE_STATE` | `~/.local/state/sserve` | local deployment state |
| `SSERVE_CLUSTER` | none | cluster to use when `-c` isn't given |
| `NO_COLOR` | unset | turn off coloured logs |
