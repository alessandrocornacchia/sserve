#!/usr/bin/env python3
"""A deterministic SLURM stand-in. Installed as sbatch/squeue/sacct/scancel/scontrol
(symlinks to this file); state lives in $FAKESLURM_DB (JSON, flock-protected).

Jobs never start by themselves: a test drives them with
    fakeslurm.py ctl start JOB [--node N]      PENDING -> RUNNING, runs the job script
    fakeslurm.py ctl requeue JOB [--node N]    kill, back to PENDING (restarts+1)
    fakeslurm.py ctl end JOB STATE [--by UID]  kill, terminal (CANCELLED gets "by UID")
    fakeslurm.py ctl set JOB key=value ...     reason / time_left (seconds) / hide (absent from squeue)
    fakeslurm.py ctl reject PATTERN MESSAGE    sbatch fails when the script contains PATTERN
    fakeslurm.py ctl sbatch-timeout on|off     sbatch submits but reports a socket timeout
    fakeslurm.py ctl fail CMD on|off           CMD (squeue/sacct/scancel/scontrol) exits 1 with an error
    fakeslurm.py ctl warn CMD on|off           CMD prints a warning on stderr but succeeds
    fakeslurm.py ctl list | killall
A RUNNING job whose script process died is reported FAILED.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

DB = Path(os.environ.get("FAKESLURM_DB", "/tmp/fakeslurm.json"))


@contextmanager
def state(write=True):
    DB.parent.mkdir(parents=True, exist_ok=True)
    with open(str(DB) + ".lock", "a+") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        s = json.loads(DB.read_text()) if DB.exists() and DB.stat().st_size else {"next": 1000, "jobs": {}, "ctl": {}}
        yield s
        if write:
            tmp = DB.with_suffix(".tmp")
            tmp.write_text(json.dumps(s, indent=1))
            os.replace(tmp, DB)


def alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return st not in ("Z", "X")
    except (FileNotFoundError, IndexError):
        return False


def refresh(j: dict) -> None:
    if j["state"] == "RUNNING" and not alive(j.get("pid")):
        j.update(state="FAILED", exit="1:0", pid=None)


def kill(j: dict) -> None:
    if alive(j.get("pid")):
        try:
            os.killpg(j["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(50):
            if not alive(j["pid"]):
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(j["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
    j["pid"] = None


def opt(args: list[str], name: str, default=None):
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            return args[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return default


def fmt_left(j: dict) -> str:
    left = j.get("time_left", 3 * 3600)
    if j["state"] == "RUNNING" and j.get("started"):
        left = max(0, left - (time.time() - j["started"]))
    left = int(left)
    return f"{left // 3600}:{left % 3600 // 60:02d}:{left % 60:02d}"


def field(j: dict, f: str) -> str:
    return {"i": j["id"], "j": j["name"], "T": j["state"], "N": j.get("node") or "",
            "r": j.get("reason") or "None", "k": j.get("comment") or "", "L": fmt_left(j),
            "Z": j.get("chdir", ""), "M": "0:00"}.get(f, "")


def squeue(args):
    fmt = opt(args, "-o", "%i %j %T")
    jid, name = opt(args, "-j"), opt(args, "-n")
    with state() as s:
        for j in s["jobs"].values():
            refresh(j)
            if j["state"] not in ("PENDING", "RUNNING", "CONFIGURING") or j.get("hide"):
                continue
            if jid and j["id"] != jid or name and j["name"] != name:
                continue
            print(re.sub(r"%(\w)", lambda m: field(j, m.group(1)), fmt))


def sacct(args):
    jid, name = opt(args, "-j"), opt(args, "--name")
    fields = (opt(args, "-o") or "State").split(",")
    with state() as s:
        for j in s["jobs"].values():
            refresh(j)
            if j.get("hide_acct"):
                continue
            if jid and j["id"] != jid or name and j["name"] != name:
                continue
            vals = {"JobID": j["id"], "State": j.get("acct_state") or j["state"], "ExitCode": j.get("exit", "0:0")}
            print("|".join(vals.get(f, "") for f in fields))


def sbatch(args):
    script = args[-1]
    text = Path(script).read_text()
    with state() as s:
        for pat, msg in s["ctl"].get("reject", {}).items():
            if pat in text:
                print(f"sbatch: error: {msg}", file=sys.stderr)
                sys.exit(1)
        jid = str(s["next"])
        s["next"] += 1
        chdir = opt(args, "--chdir", os.getcwd())
        s["jobs"][jid] = {
            "id": jid, "name": opt(args, "--job-name", "job"), "comment": opt(args, "--comment", ""),
            "state": "PENDING", "reason": "Priority", "script": str(Path(chdir, script) if not os.path.isabs(script) else script),
            "chdir": chdir, "out": os.path.join(chdir, opt(args, "--output", "slurm-%j.out").replace("%j", jid)),
            "err": os.path.join(chdir, opt(args, "--error", "slurm-%j.err").replace("%j", jid)), "restarts": 0,
            "env": {k: v for k, v in os.environ.items()}, "submitted": time.time(),
            "requeue": "#SBATCH --requeue" in text, "text": text}
        timeout = s["ctl"].get("sbatch_timeout")
    if timeout:
        print("sbatch: error: Batch job submission failed: Socket timed out on send/recv operation", file=sys.stderr)
        sys.exit(1)
    print(jid)


def scancel(args):
    jid = args[-1]
    with state() as s:
        j = s["jobs"].get(jid)
        if j and j["state"] in ("PENDING", "RUNNING"):
            kill(j)
            j.update(state="CANCELLED", acct_state=f"CANCELLED by {os.getuid()}", ended=time.time())


def scontrol(args):
    if args[:2] == ["show", "hostnames"]:
        print(args[2])
        return
    if args[:2] == ["show", "job"]:
        jid = args[-1]
        with state(write=False) as s:
            j = s["jobs"].get(jid)
            if j:
                print(f"JobId={jid} JobName={j['name']} JobState={j['state']} Restarts={j['restarts']} "
                      f"NodeList={j.get('node') or '(null)'}")


def start(s, j, node):
    env = {**j["env"], "SLURM_JOB_ID": j["id"], "SLURM_RESTART_COUNT": str(j["restarts"])}
    out, err = open(j["out"], "a"), open(j["err"], "a")
    p = subprocess.Popen(["bash", j["script"]], cwd=j["chdir"], env=env, stdout=out, stderr=err,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    j.update(state="RUNNING", node=node, pid=p.pid, reason="None", started=time.time())


def ctl(args):
    cmd = args[0]
    with state() as s:
        if cmd == "start":
            j = s["jobs"][args[1]]
            start(s, j, opt(args, "--node", "node-a"))
        elif cmd == "requeue":
            j = s["jobs"][args[1]]
            kill(j)
            j.update(state="PENDING", reason="BeginTime", node=None, restarts=j["restarts"] + 1)
        elif cmd == "end":
            j = s["jobs"][args[1]]
            kill(j)
            st = args[2]
            by = opt(args, "--by")
            j.update(state=st, acct_state=f"{st} by {by}" if by else st, ended=time.time())
        elif cmd == "set":
            j = s["jobs"][args[1]]
            for kv in args[2:]:
                k, v = kv.split("=", 1)
                j[k] = float(v) if k == "time_left" else (v == "1") if k in ("hide", "hide_acct") else v
        elif cmd == "reject":
            s["ctl"].setdefault("reject", {})[args[1]] = args[2]
        elif cmd == "unreject":
            s["ctl"].get("reject", {}).pop(args[1], None)
        elif cmd == "fail":
            s["ctl"].setdefault("fail", {})[args[1]] = args[2] == "on"
        elif cmd == "warn":
            s["ctl"].setdefault("warn", {})[args[1]] = args[2] == "on"
        elif cmd == "sbatch-timeout":
            s["ctl"]["sbatch_timeout"] = args[1] == "on"
        elif cmd == "list":
            for j in s["jobs"].values():
                refresh(j)
            print(json.dumps({k: {f: v for f, v in j.items() if f != "env"} for k, j in s["jobs"].items()}))
        elif cmd == "killall":
            for j in s["jobs"].values():
                kill(j)


def main():
    me = Path(sys.argv[0]).name
    args = sys.argv[1:]
    if me == "fakeslurm.py":
        me, args = args[0], args[1:]
    if me != "ctl":
        with state(write=False) as s:
            if s["ctl"].get("fail", {}).get(me):
                print(f"{me}: error: Unable to contact slurm controller (connect failure)", file=sys.stderr)
                sys.exit(1)
            if s["ctl"].get("warn", {}).get(me):
                print(f"{me}: warning: 12345 slurm.conf deprecated option (not a job id)", file=sys.stderr)
    {"sbatch": sbatch, "squeue": squeue, "sacct": sacct, "scancel": scancel, "scontrol": scontrol, "ctl": ctl}[me](args)


if __name__ == "__main__":
    main()
