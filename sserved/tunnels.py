"""SSH tunnels to vLLM on compute nodes, supervised as direct children.

One `ssh -N -L 127.0.0.1:<local>:<node>:<port> <login>` per replica, in the
foreground (no -f), so the manager owns, watches and reaps it. Children get
PR_SET_PDEATHSIG, so they die with the manager, and carry SSERVED_TUNNEL=<instance>
in their environment so a restarted manager can find and stop leftovers whose
parent died before the signal could be delivered.
"""
from __future__ import annotations

import asyncio
import ctypes
import os
import signal
import socket
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import Cluster
from .slurm import SSH_BASE

_libc = ctypes.CDLL(None, use_errno=True)
PR_SET_PDEATHSIG = 1


def _die_with_parent() -> None:
    _libc.prctl(PR_SET_PDEATHSIG, signal.SIGTERM)
    os.setsid()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@dataclass
class Tunnel:
    key: str                     # candidate id + generation
    node: str
    port: int
    lport: int
    proc: asyncio.subprocess.Process | None = None
    started: float = 0.0
    restarts: int = 0
    last_exit: str = ""
    history: list[float] = field(default_factory=list)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None


class Tunnels:
    def __init__(self, instance: str, log_dir: Path):
        self.instance = instance
        self.log_dir = log_dir
        self.t: dict[str, Tunnel] = {}

    def argv(self, c: Cluster, node: str, port: int, lport: int) -> list[str]:
        if c.tunnel_cmd:
            return [a.format(lport=lport, node=node, port=port, host=c.host) for a in c.tunnel_cmd]
        return ["ssh", *SSH_BASE, *c.ssh_opts, "-o", "ExitOnForwardFailure=yes", "-o", "ControlMaster=no",
                "-N", "-L", f"127.0.0.1:{lport}:{node}:{port}", c.host]

    async def ensure(self, key: str, c: Cluster, node: str, port: int) -> Tunnel:
        """Start (or restart) the tunnel for `key`; a changed node/port gets a new tunnel."""
        t = self.t.get(key)
        if t and (t.node != node or t.port != port):
            await self.close(key)
            t = None
        if t and t.alive:
            return t
        if t is None:
            t = Tunnel(key, node, port, free_port())
            self.t[key] = t
        elif t.proc is not None:
            t.last_exit = f"exit {t.proc.returncode}"
            t.restarts += 1
        # rate-limit restarts: at most 6 per 5 minutes per tunnel
        now = time.time()
        t.history = [h for h in t.history if now - h < 300]
        if len(t.history) >= 6:
            return t
        t.history.append(now)
        self.log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        log = open(self.log_dir / f"tunnel-{key}.log", "ab")
        try:
            t.proc = await asyncio.create_subprocess_exec(
                *self.argv(c, node, port, t.lport), stdin=asyncio.subprocess.DEVNULL, stdout=log, stderr=log,
                env={**os.environ, **c.env, "SSERVED_TUNNEL": self.instance}, preexec_fn=_die_with_parent)
        finally:
            log.close()
        t.started = now
        asyncio.get_running_loop().create_task(self._reap(t.proc))
        return t

    async def _reap(self, proc: asyncio.subprocess.Process) -> None:
        await proc.wait()      # always awaited: no zombies

    async def close(self, key: str) -> None:
        t = self.t.pop(key, None)
        if not t or not t.proc:
            return
        if t.proc.returncode is None:
            try:
                os.killpg(t.proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(t.proc.wait(), 5)
            except asyncio.TimeoutError:
                os.killpg(t.proc.pid, signal.SIGKILL)
                await t.proc.wait()

    async def close_all(self) -> None:
        for k in list(self.t):
            await self.close(k)

    def kill_leftovers(self) -> list[int]:
        """Stop tunnels a previous incarnation of this instance left behind (same uid, our marker,
        not one of our current children)."""
        mine = {t.proc.pid for t in self.t.values() if t.proc}
        killed = []
        marker = f"SSERVED_TUNNEL={self.instance}".encode()
        for d in Path("/proc").iterdir():
            if not d.name.isdigit() or int(d.name) in mine or int(d.name) == os.getpid():
                continue
            try:
                if d.stat().st_uid != os.getuid():
                    continue
                if marker not in (d / "environ").read_bytes().split(b"\0"):
                    continue
                os.kill(int(d.name), signal.SIGTERM)
                killed.append(int(d.name))
            except (OSError, ValueError):
                continue
        return killed
