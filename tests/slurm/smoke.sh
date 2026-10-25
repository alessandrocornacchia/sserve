#!/usr/bin/env bash
# End-to-end test of sserve against the Docker SLURM cluster (cluster.sh up),
# with the mock-llm recipes standing in for vLLM. Run from anywhere:
#
#   eval "$(tests/slurm/cluster.sh up)" && tests/slurm/smoke.sh
#
# Each check prints a line; the first failure stops the script.
set -euo pipefail

repo=$(cd "$(dirname "$0")/../.." && pwd)
: "${SSERVE_CONFIG:=$repo/tests/slurm/.state/config.toml}"
export SSERVE_CONFIG NO_COLOR=1
export SSERVE_STATE=$(mktemp -d)      # don't touch the real deployments' state
out=$(mktemp)
trap 'rm -rf "$SSERVE_STATE" "$out"' EXIT
[ -f "$SSERVE_CONFIG" ] || { echo "no $SSERVE_CONFIG: run tests/slurm/cluster.sh up first" >&2; exit 2; }

sserve() { "$repo/sserve" -c test "$@"; }
step() { printf '\n=== %s\n' "$*"; }
ok() { printf -- '--- ok: %s\n' "$*"; }
fail() { printf -- '--- FAIL: %s\n' "$*" >&2; cat "$out" >&2; exit 1; }
# expect <pattern> <description>: the output of the last `run` must match
expect() { grep -Eq -- "$1" "$out" || fail "$2 (no /$1/ in the output above)"; ok "$2"; }
run() { "$@" 2>&1 | tee "$out"; return "${PIPESTATUS[0]}"; }

sserve down --all >/dev/null 2>&1 || true

step "config show"
run sserve config show || fail "config show"
expect 'sserve-slurm' "config resolves the test cluster"

step "push and ls"
run sserve push || fail "push"
run sserve ls -w || fail "ls"
expect '^mock-llm +test' "ls lists mock-llm"
expect 'mock-llm:broken' "ls lists mock-llm:broken"
expect 'sserve-slurm:/data/sserve/deploy/mock-llm/default.sbatch' "ls -w shows the cluster path"

step "cat"
run sserve cat mock-llm --plain --remote || fail "cat"
expect 'fake vLLM' "cat prints the cluster's recipe"

step "pull is a no-op right after push"
run sserve pull -y || fail "pull"
if grep -Eq '^ *[~+-] ' "$out"; then fail "pull changed local files"; fi
ok "pull found nothing to change"

step "up -d"
run sserve up mock-llm -d --ready-timeout 300 || fail "up"
expect 'is healthy on c1' "the job became healthy on c1"
expect 'mock-llm:default is up' "up reports the deployment"

step "env and the endpoint"
eval "$(sserve env mock-llm)"
[ -n "${OPENAI_BASE_URL:-}" ] && [ -n "${OPENAI_API_KEY:-}" ] || fail "env did not export the variables"
ok "env exports OPENAI_BASE_URL and OPENAI_API_KEY"
run python3 "$repo/examples/client.py" "hello cluster" || fail "client.py"
expect 'models: sserve/mock-llm' "client lists the model through the tunnel"
expect 'echo: hello cluster' "client gets a chat answer"
run python3 "$repo/examples/client.py" "streamed words" --stream || fail "client.py --stream"
expect 'echo: streamed words' "client streams an answer"
code=$(curl -s -o /dev/null -w '%{http_code}' "$OPENAI_BASE_URL/models")
[ "$code" = 401 ] || { echo "HTTP $code" > "$out"; fail "the endpoint answers without the key"; }
ok "the endpoint rejects requests without the key"

step "status and logs"
run sserve status || fail "status"
expect 'mock-llm' "status shows the job"
run sserve logs mock-llm -n 20 || fail "logs"
expect 'mock vLLM serving' "logs show the server's output"

step "up again attaches"
run sserve up mock-llm -d --ready-timeout 120 || fail "second up"
expect 'attach|already running|is up' "a second up attaches to the running job"
[ "$(ssh -o BatchMode=yes sserve-slurm 'squeue -h -n sserve-mock-llm | wc -l')" = 1 ] \
    || fail "a second up submitted another job"
ok "still one job"

step "down"
run sserve down mock-llm || fail "down"
sleep 2
[ "$(ssh -o BatchMode=yes sserve-slurm 'squeue -h -n sserve-mock-llm | wc -l')" = 0 ] || fail "down left the job queued"
ok "down cancelled the job"

step "a failing recipe"
if run sserve up mock-llm:broken -d --ready-timeout 120; then fail "up of a failing recipe succeeded"; fi
expect 'ended in state' "up reports the failed job"
expect 'failing on purpose' "up shows the job's log"

printf '\nall smoke tests passed\n'
