"""SLURM access through the cluster's login node.

Every call is a bash script sent over ssh stdin (secrets never reach argv or a
process list), sharing one multiplexed connection per cluster. Calls are
bounded; a timeout tears the mux down, because a wedged control socket
otherwise makes every later call hang.

Ownership: each job carries `--comment=sserved:<instance>:<candidate>` and a
`sservem-` job name; the run directory (mode 700) holds a nonce, the job id and
the backend key. Cancellation re-checks the comment on the login node, so the
manager can only ever cancel its own jobs.
"""
from __future__ import annotations

import asyncio
import base64
import os
import shlex
import signal
from dataclasses import dataclass
from pathlib import Path

from .config import Cluster

JOB_PREFIX = "sservem-"
SSH_BASE = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3"]

_EXPAND = 'x() { case "$1" in "~"|"~/"*) printf %s "$HOME${1#\\~}";; *) printf %s "$1";; esac; }\n'

SUBMIT_SH = _EXPAND + r'''
set -uo pipefail
run="$(x "$1")/$2"; name=$3; comment=$4
mkdir -p "$run" && chmod 700 "$run" || { echo "SBATCH_ERR cannot create run dir"; exit 4; }
(umask 077; printf '%s\n' "$SSERVE_KEY" > "$run/api_key"; printf '%s\n' "$SSERVE_NONCE" > "$run/nonce")
printf %s "$SSERVE_SBATCH_B64" | base64 -d > "$run/job.sbatch"
cd "$run"
out=$(env -u SSERVE_KEY -u SSERVE_NONCE -u SSERVE_SBATCH_B64 \
      sbatch --parsable --export=ALL --job-name="$name" --comment="$comment" --chdir="$run" \
      --output="$run/slurm-%j.out" --error="$run/slurm-%j.err" "$run/job.sbatch" 2>"$run/sbatch.stderr")
rc=$?
id=$(printf '%s\n' "$out" | sed -n 's/^\([0-9][0-9]*\)\(;.*\)\{0,1\}$/\1/p' | head -1)
if [ $rc -eq 0 ] && [ -n "$id" ]; then
    printf '%s\n' "$id" > "$run/jobid"; echo "JOBID $id"
elif [ $rc -eq 0 ]; then
    echo "SBATCH_ERR sbatch succeeded without a numeric job id (outcome unknown; resolve by name)"; exit 3
else
    cat "$run/sbatch.stderr" > "$run/sbatch.err"; echo "SBATCH_ERR $(head -c 2000 "$run/sbatch.stderr")"; exit 4
fi
'''

# Job ids come from stdout only (stderr may carry warnings) and must be numeric; anything else is UNKNOWN.
RESOLVE_SH = _EXPAND + r"""
run="$(x "$1")/$2"; name=$3
num() { case "$1" in ''|*[!0-9]*) return 1;; esac; }
if [ -s "$run/jobid" ]; then
    id=$(head -1 "$run/jobid")
    num "$id" && { echo "JOBID $id"; exit 0; }
    echo "UNKNOWN unexpected content in jobid file"; exit 0
fi
out=$(squeue -h -u "$USER" -n "$name" -o '%i' 2>/dev/null); rc=$?
[ $rc -eq 0 ] || { echo "UNKNOWN squeue failed (rc=$rc)"; exit 0; }
id=$(printf '%s\n' "$out" | sed '/^[[:space:]]*$/d' | head -1)
if [ -n "$id" ]; then num "$id" && { echo "JOBID $id"; exit 0; }; echo "UNKNOWN unexpected squeue output"; exit 0; fi
out=$(sacct -n -X -P --name="$name" -S now-2days -o JobID 2>/dev/null); rc=$?
[ $rc -eq 0 ] || { echo "UNKNOWN sacct failed (rc=$rc)"; exit 0; }
id=$(printf '%s\n' "$out" | sed '/^[[:space:]]*$/d' | head -1)
if [ -n "$id" ]; then num "$id" && { echo "JOBID $id"; exit 0; }; echo "UNKNOWN unexpected sacct output"; exit 0; fi
if [ -d "$run" ]; then
    [ -s "$run/sbatch.err" ] && { echo "REJECTED $(head -c 500 "$run/sbatch.err")"; exit 0; }
    echo "NOJOB"
else
    echo "NORUN"
fi
"""

# Q|id|state|node|reason|comment|timeleft|restarts|workdir   (restarts "?" when scontrol failed)
# A|id|rc|State|ExitCode                                      (rc != 0: sacct failed, nothing known)
QUERY_SH = r"""
echo "ME $(id -u)"
out=$(squeue -h -u "$USER" -o '%i|%j|%T|%N|%r|%k|%L|%Z' 2>/dev/null) || { echo "SQUEUE_FAIL rc=$?"; exit 6; }
printf '%s\n' "$out" | while IFS='|' read -r id name st nodes reason comment left wd; do
    [ -n "$id" ] || continue
    case "$name" in SSERVE_PREFIX*) ;; *) continue;; esac
    case "$comment" in "sserved:$1:"*) ;; *) continue;; esac
    node=""
    if [ -n "$nodes" ] && [ "$nodes" != "(null)" ]; then node=$(scontrol show hostnames "$nodes" 2>/dev/null | head -1); fi
    if js=$(scontrol show job -o "$id" 2>/dev/null) && r=$(printf '%s' "$js" | grep -o 'Restarts=[0-9]*'); then
        restarts=${r#Restarts=}
    else
        restarts="?"
    fi
    printf 'Q|%s|%s|%s|%s|%s|%s|%s|%s\n' "$id" "$st" "$node" "$reason" "$comment" "$left" "$restarts" "$wd"
done
echo "SQUEUE_OK"
shift
for id in "$@"; do
    s=$(sacct -n -X -P -j "$id" -o State,ExitCode 2>/dev/null); rc=$?
    printf 'A|%s|%s|%s\n' "$id" "$rc" "$(printf '%s' "$s" | head -1)"
done
""".replace("SSERVE_PREFIX", JOB_PREFIX)

# CANCELLED only when scancel itself succeeded; UNKNOWN/FAILED leave the outcome open.
CANCEL_SH = _EXPAND + r"""
c=$(squeue -h -j "$1" -o '%k' 2>/dev/null); rc=$?
if [ $rc -ne 0 ]; then
    # squeue -j fails for jobs it no longer knows; tell that apart from a broken squeue
    squeue -h -u "$USER" -o '%i' >/dev/null 2>&1 && { echo GONE; exit 0; }
    echo "UNKNOWN squeue failed"; exit 6
fi
if [ -z "$c" ]; then echo GONE; exit 0; fi
[ "$c" = "$2" ] || { echo NOT_OURS; exit 5; }
if err=$(scancel "$1" 2>&1); then
    echo CANCELLED
    rm -f "$(x "$3")/api_key"
else
    echo "FAILED $err" | head -c 300; exit 7
fi
"""


class RemoteError(Exception):
    pass


class Unreachable(RemoteError):
    """ssh/transport failure or timeout: says nothing about the job."""


@dataclass
class JobRow:
    job_id: str
    state: str
    node: str
    reason: str
    comment: str
    time_left: float | None
    restarts: int | None          # None: scontrol could not tell (never read as a change)
    workdir: str = ""


def parse_time_left(s: str) -> float | None:
    s = s.strip()
    if not s or s in ("UNLIMITED", "INVALID", "NOT_SET"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = [int(p) for p in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    h, m, sec = parts[-3:]
    return days * 86400 + h * 3600 + m * 60 + sec


class Runner:
    def __init__(self, cluster: Cluster, sock_dir: Path):
        self.c = cluster
        self.sock_dir = sock_dir

    def _mux(self) -> list[str]:
        self.sock_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        return ["-o", "ControlMaster=auto", "-o", f"ControlPath={self.sock_dir}/m-%C", "-o", "ControlPersist=10m"]

    def argv(self, args: tuple[str, ...]) -> list[str]:
        if self.c.runner == "local":
            return ["bash", "-s", "--", *args]
        return ["ssh", *SSH_BASE, *self.c.ssh_opts, *self._mux(), self.c.host,
                "bash -s -- " + " ".join(shlex.quote(a) for a in args)]

    async def run(self, script: str, *args: str, secrets_env: dict[str, str] | None = None,
                  timeout: float = 90) -> tuple[int, str, str]:
        prefix = "".join(f"export {k}={shlex.quote(v)}\n" for k, v in (secrets_env or {}).items())
        body = (self.c.prelude + "\n" if self.c.prelude else "") + prefix + script
        env = None
        if self.c.runner == "local":
            env = {**os.environ, **self.c.env}
        p = await asyncio.create_subprocess_exec(*self.argv(args), stdin=asyncio.subprocess.PIPE,
                                                 stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                                                 env=env, start_new_session=True)
        try:
            out, err = await asyncio.wait_for(p.communicate(body.encode()), timeout)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await p.wait()
            await self.reset_mux()
            raise Unreachable(f"{self.c.name}: remote call timed out after {timeout:.0f}s")
        o, e = out.decode(errors="replace"), err.decode(errors="replace").strip()
        if self.c.runner != "local" and p.returncode == 255:
            raise Unreachable(f"{self.c.name}: ssh failed: {e[-300:]}")
        return p.returncode, o, e

    async def reset_mux(self) -> None:
        if self.c.runner == "local":
            return
        p = await asyncio.create_subprocess_exec("ssh", *self._mux(), "-O", "exit", self.c.host,
                                                 stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                                                 start_new_session=True)
        try:
            await asyncio.wait_for(p.wait(), 10)
        except asyncio.TimeoutError:
            os.killpg(p.pid, signal.SIGKILL)
            await p.wait()
            for s in self.sock_dir.glob("m-*"):
                s.unlink(missing_ok=True)


class Slurm:
    """The four operations the reconciler needs, with tri-state results."""

    def __init__(self, cluster: Cluster, sock_dir: Path):
        self.cluster = cluster
        self.r = Runner(cluster, sock_dir)

    async def submit(self, run_id: str, name: str, comment: str, sbatch_text: str, key: str, nonce: str
                     ) -> tuple[str, str]:
        """-> ("submitted", job_id) | ("rejected", why) | ("unknown", why). Never raises for transport."""
        try:
            rc, out, err = await self.r.run(SUBMIT_SH, self.cluster.runs_dir, run_id, name, comment, timeout=120,
                                            secrets_env={"SSERVE_KEY": key, "SSERVE_NONCE": nonce,
                                                         "SSERVE_SBATCH_B64": base64.b64encode(sbatch_text.encode()).decode()})
        except Unreachable as e:
            return "unknown", str(e)
        for line in out.splitlines():
            if line.startswith("JOBID ") and line.split()[1:2] and line.split()[1].isdigit():
                return "submitted", line.split()[1]
        why = next((l[11:] for l in out.splitlines() if l.startswith("SBATCH_ERR ")), err or out)[:500]
        if rc == 4 and not any(s in why.lower() for s in ("timed out", "socket", "unable to contact", "connection")):
            return "rejected", why
        return "unknown", why

    async def resolve(self, run_id: str, name: str) -> tuple[str, str]:
        """For a submission whose outcome is unknown: -> ("submitted", id) | ("not_submitted", why) |
        ("rejected", why) | ("unknown", why)."""
        try:
            rc, out, err = await self.r.run(RESOLVE_SH, self.cluster.runs_dir, run_id, name, timeout=60)
        except Unreachable as e:
            return "unknown", str(e)
        line = out.strip().splitlines()[-1] if out.strip() else ""
        if rc != 0 or line.startswith("UNKNOWN"):
            return "unknown", (line or err or out)[:300]
        if line.startswith("JOBID "):
            jid = line.split()[1] if len(line.split()) > 1 else ""
            return ("submitted", jid) if jid.isdigit() else ("unknown", f"non-numeric job id {jid[:40]!r}")
        if line.startswith("REJECTED"):
            return "rejected", line[9:]
        if line in ("NOJOB", "NORUN"):
            return "not_submitted", line
        return "unknown", (err or out)[:300]

    async def query(self, instance: str, ids: list[str]
                    ) -> tuple[str, dict[str, JobRow], dict[str, tuple[bool, str]]]:
        """-> (my uid, live jobs tagged by this instance keyed by id, {id: (sacct ok, State)}).
        Raises Unreachable when squeue could not be listed: then nothing may be inferred."""
        rc, out, err = await self.r.run(QUERY_SH, instance, *ids, timeout=90)
        if rc != 0 or "SQUEUE_OK" not in out:
            raise Unreachable(f"{self.cluster.name}: squeue failed: {(err or out)[-300:]}")
        uid, live, acct = "", {}, {}
        for line in out.splitlines():
            if line.startswith("ME "):
                uid = line[3:].strip()
            elif line.startswith("Q|"):
                _, jid, st, node, reason, comment, left, restarts, wd = (line.split("|") + [""] * 9)[:9]
                if not jid.isdigit():
                    continue
                live[jid] = JobRow(jid, st, node, reason, comment, parse_time_left(left),
                                   int(restarts) if restarts.isdigit() else None, wd)
            elif line.startswith("A|"):
                _, jid, arc, s = (line.split("|", 3) + ["", ""])[:4]
                acct[jid] = (arc.strip() == "0", s.strip())
        if not uid:
            raise Unreachable(f"{self.cluster.name}: could not determine the scheduler uid")
        return uid, live, acct

    async def cancel(self, job_id: str, comment: str, run_dir: str) -> str:
        """-> CANCELLED (scancel succeeded) | GONE (not in the queue) | NOT_OURS | FAILED ... | UNKNOWN ...
        Never raises: transport failures are UNKNOWN."""
        try:
            rc, out, err = await self.r.run(CANCEL_SH, job_id, comment, run_dir, timeout=60)
        except Unreachable as e:
            return f"UNKNOWN {e}"
        res = (out.strip().splitlines() or [f"UNKNOWN rc={rc} {err[-200:]}"])[0]
        if res == "CANCELLED" and rc != 0:
            return f"UNKNOWN rc={rc}"
        return res


def sbatch_script(*, profile, model, gpus: int, recipe_dir: str, port: int, max_model_len: int,
                  served: str) -> str:
    """The job script: SBATCH resources from the operator profile, then vLLM via the recipe's serve.sh.
    The job runs vLLM only."""
    lines = ["#!/bin/bash --login",
             f"#SBATCH --time={profile.time}", "#SBATCH --nodes=1", "#SBATCH --ntasks=1",
             f"#SBATCH --cpus-per-task={model.cpus_per_task}",
             f"#SBATCH --gres={profile.gres.format(gpus=gpus)}"]
    if model.mem:
        lines.append(f"#SBATCH --mem={model.mem}")
    if profile.requeue:
        lines += ["#SBATCH --requeue", "#SBATCH --open-mode=append"]
    lines += [f"#SBATCH {a}" for a in profile.sbatch_args]
    env = {"VLLM_PORT": str(port), "SSERVE_MODEL": model.name, "SSERVE_SERVED_MODEL": served,
           "SSERVE_GPUS": str(gpus), "TENSOR_PARALLEL_SIZE": str(gpus), "SSERVE_RECIPE_DIR": recipe_dir,
           **({"MAX_MODEL_LEN": str(max_model_len)} if max_model_len else {}), **model.env}
    lines += ['export SSERVE_RUN_DIR="$PWD" SSERVE_KEY_FILE="$PWD/api_key"']
    lines += [f"export {k}={_sh_path(v) if k == 'SSERVE_RECIPE_DIR' else shlex.quote(v)}" for k, v in env.items()]
    lines += ['echo "sserved job $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname)"',
              'exec bash "$SSERVE_RECIPE_DIR/serve.sh"']
    return "\n".join(lines) + "\n"


def _sh_path(p: str) -> str:
    """Quote a path for bash, keeping a leading ~/ expandable."""
    if p == "~" or p.startswith("~/"):
        return '"$HOME"' + (shlex.quote(p[1:]) if len(p) > 1 else "")
    return shlex.quote(p)
