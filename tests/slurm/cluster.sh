#!/usr/bin/env bash
# Start or stop the Docker SLURM cluster that sserve's tests run against.
#
#   tests/slurm/cluster.sh up      start it and print the SSERVE_CONFIG to use
#   tests/slurm/cluster.sh down    stop it and delete its volumes
#
# `up` also adds a `Host sserve-slurm` block (root@127.0.0.1, its own key) to the
# top of ~/.ssh/config, between marker comments; `down` removes it. The sserve
# cluster is called `test`:
#
#   export SSERVE_CONFIG=tests/slurm/.state/config.toml
#   ./sserve -c test push && ./sserve -c test up mock-llm -d
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
state="$here/.state"
host=sserve-slurm
begin="# >>> $host (sserve tests/slurm/cluster.sh) >>>"
end="# <<< $host <<<"
export COMPOSE_PROJECT_NAME=sserve-slurm
export SSERVE_SLURM_SSH_PORT=${SSERVE_SLURM_SSH_PORT:-3022}
export SSERVE_SLURM_KEYS="$state/authorized_keys"
compose=(docker compose -f "$here/compose.yml")

say() { printf '[cluster] %s\n' "$*" >&2; }

drop_ssh_block() {
    [ -f ~/.ssh/config ] || return 0
    awk -v b="$begin" -v e="$end" '$0==b{skip=1} skip==2{skip=0; if ($0=="") next} !skip{print} $0==e{skip=2}' ~/.ssh/config > ~/.ssh/config.sserve-tmp
    mv ~/.ssh/config.sserve-tmp ~/.ssh/config
}

wait_for() {  # wait_for <seconds> <what> <command...>
    local t=$1 what=$2; shift 2
    for _ in $(seq "$t"); do "$@" >/dev/null 2>&1 && return 0; sleep 1; done
    say "timed out after ${t}s waiting for $what"
    return 1
}

up() {
    mkdir -p "$state" ~/.ssh && chmod 700 ~/.ssh
    [ -f "$state/id_ed25519" ] || ssh-keygen -q -t ed25519 -N '' -C "$host" -f "$state/id_ed25519"
    cp "$state/id_ed25519.pub" "$SSERVE_SLURM_KEYS"

    say "starting the cluster (ssh on 127.0.0.1:$SSERVE_SLURM_SSH_PORT)"
    "${compose[@]}" up -d --quiet-pull

    # First match wins in ssh_config, so the block goes on top, before any `Host *`.
    drop_ssh_block
    { printf '%s\n' "$begin" \
        "Host $host" \
        "  HostName 127.0.0.1" \
        "  Port $SSERVE_SLURM_SSH_PORT" \
        "  User root" \
        "  IdentityFile $state/id_ed25519" \
        "  IdentitiesOnly yes" \
        "  StrictHostKeyChecking no" \
        "  UserKnownHostsFile /dev/null" \
        "  LogLevel ERROR" \
        "$end" ""
      cat ~/.ssh/config 2>/dev/null || true; } > ~/.ssh/config.sserve-tmp
    mv ~/.ssh/config.sserve-tmp ~/.ssh/config && chmod 600 ~/.ssh/config

    wait_for 180 "ssh to the login node" ssh -o BatchMode=yes "$host" true
    wait_for 180 "compute node c1 to be idle" \
        ssh -o BatchMode=yes "$host" 'sinfo -h -N -o "%N %T" | grep -q "^c1 idle"'

    cat > "$state/config.toml" <<EOF
# Written by tests/slurm/cluster.sh. The run and recipe dirs live on /data,
# which the login and compute nodes share (their home directories are not shared).
default = "test"

[clusters.test]
host = "$host"
deploy_dir = "/data/sserve/deploy"
runs_dir = "/data/sserve/runs"
recipes_dir = "$here/recipes"
EOF
    say "ready: sinfo on the login node says"
    ssh -o BatchMode=yes "$host" sinfo >&2
    echo "export SSERVE_CONFIG=$state/config.toml"
}

down() {
    say "stopping the cluster and deleting its volumes"
    "${compose[@]}" down -v --remove-orphans
    drop_ssh_block
    [ -s ~/.ssh/config ] || rm -f ~/.ssh/config  # we created it
    rm -f "$state/config.toml"
}

case "${1:-}" in
    up) up ;;
    down) down ;;
    *) echo "usage: $0 up|down" >&2; exit 2 ;;
esac
