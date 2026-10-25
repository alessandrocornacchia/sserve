"""Durable state: one SQLite file (WAL, synchronous=FULL), mode 600, and a flock
that keeps a second controller from running against the same state.

Every write is a transaction. Secrets in here: token and lease-key *hashes*,
and backend (vLLM) keys, which the manager must present to its own jobs.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS tokens (
    name TEXT PRIMARY KEY, hash TEXT NOT NULL UNIQUE, admin INTEGER NOT NULL DEFAULT 0,
    created REAL NOT NULL, revoked REAL);
CREATE TABLE IF NOT EXISTS profile_state (
    name TEXT PRIMARY KEY, enabled INTEGER NOT NULL, changed REAL NOT NULL, note TEXT);
CREATE TABLE IF NOT EXISTS deployments (
    id TEXT PRIMARY KEY, spec TEXT NOT NULL, model TEXT NOT NULL, state TEXT NOT NULL,
    target INTEGER NOT NULL DEFAULT 0, created REAL NOT NULL, last_demand REAL,
    last_scale REAL, startup_failures INTEGER NOT NULL DEFAULT 0, max_model_len INTEGER,
    fingerprint TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS breakers (
    model TEXT PRIMARY KEY, reason TEXT NOT NULL, at REAL NOT NULL, deployment TEXT);
CREATE TABLE IF NOT EXISTS leases (
    id TEXT PRIMARY KEY, client TEXT NOT NULL, deployment TEXT NOT NULL, key_hash TEXT NOT NULL UNIQUE,
    label TEXT, created REAL NOT NULL, expires REAL NOT NULL, prep_grace REAL NOT NULL,
    idle_timeout REAL NOT NULL, acquire_deadline REAL NOT NULL, available_since REAL,
    last_activity REAL, state TEXT NOT NULL, ended REAL, end_reason TEXT);
CREATE TABLE IF NOT EXISTS slots (
    id TEXT PRIMARY KEY, deployment TEXT NOT NULL, state TEXT NOT NULL, created REAL NOT NULL,
    winner TEXT, reason TEXT, drain_deadline REAL, replaces TEXT);
CREATE TABLE IF NOT EXISTS candidates (
    id TEXT PRIMARY KEY, slot TEXT NOT NULL, deployment TEXT NOT NULL, profile TEXT NOT NULL,
    cluster TEXT NOT NULL, gpus INTEGER NOT NULL, run_dir TEXT NOT NULL, port INTEGER NOT NULL,
    backend_key TEXT NOT NULL, nonce TEXT NOT NULL, job_id TEXT, state TEXT NOT NULL,
    slurm_state TEXT, reason TEXT, node TEXT, restarts INTEGER NOT NULL DEFAULT 0,
    time_left REAL, submitted REAL NOT NULL, running_since REAL, ready_since REAL, ended REAL,
    end_state TEXT, cancel_requested REAL, cancel_acked REAL, cancel_attempt REAL,
    generation INTEGER NOT NULL DEFAULT 0, missing_since REAL, unhealthy_since REAL);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, deployment TEXT, kind TEXT NOT NULL, msg TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS leases_dep ON leases(deployment, state);
CREATE INDEX IF NOT EXISTS cand_dep ON candidates(deployment, state);
"""

LIVE_CANDIDATE = ("submitting", "unknown", "queued", "starting", "ready", "draining", "cancelling")


def hash_secret(s: str) -> str:
    # Secrets are 256-bit random values, so a fast hash is enough: there is nothing to brute-force.
    return hashlib.sha256(s.encode()).hexdigest()


def new_secret(prefix: str) -> str:
    return f"{prefix}{secrets.token_urlsafe(32)}"


class AlreadyRunning(Exception):
    pass


class Store:
    def __init__(self, state_dir: Path, lock: bool = True):
        self.dir = state_dir
        self.dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.dir, 0o700)
        self._lock_fd = None
        if lock:
            self._lock_fd = os.open(self.dir / "controller.lock", os.O_RDWR | os.O_CREAT, 0o600)
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(self._lock_fd)
                raise AlreadyRunning(f"another sserved holds {self.dir / 'controller.lock'}")
            os.ftruncate(self._lock_fd, 0)
            os.write(self._lock_fd, str(os.getpid()).encode())
        path = self.dir / "state.db"
        old = os.umask(0o077)
        try:
            self.db = sqlite3.connect(path, isolation_level=None, check_same_thread=False, timeout=30)
        finally:
            os.umask(old)
        os.chmod(path, 0o600)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript(SCHEMA)
        if not self.meta("instance"):
            self.set_meta("instance", secrets.token_hex(6))

    def close(self) -> None:
        self.db.close()
        if self._lock_fd is not None:
            os.close(self._lock_fd)

    # -- generic helpers ------------------------------------------------------
    def tx(self):
        return _Tx(self.db)

    def q(self, sql: str, *args) -> list[dict]:
        return [dict(r) for r in self.db.execute(sql, args).fetchall()]

    def one(self, sql: str, *args) -> dict | None:
        r = self.db.execute(sql, args).fetchone()
        return dict(r) if r else None

    def x(self, sql: str, *args) -> int:
        return self.db.execute(sql, args).rowcount

    def insert(self, table: str, row: dict) -> None:
        cols = ",".join(row)
        self.db.execute(f"INSERT INTO {table} ({cols}) VALUES ({','.join('?' * len(row))})", tuple(row.values()))

    def update(self, table: str, id_: str, **kv) -> None:
        sets = ",".join(f"{k}=?" for k in kv)
        self.db.execute(f"UPDATE {table} SET {sets} WHERE id=?", (*kv.values(), id_))

    def meta(self, k: str) -> str | None:
        r = self.one("SELECT v FROM meta WHERE k=?", k)
        return r["v"] if r else None

    def set_meta(self, k: str, v: str) -> None:
        self.x("INSERT INTO meta(k,v) VALUES(?,?) ON CONFLICT(k) DO UPDATE SET v=excluded.v", k, v)

    @property
    def instance(self) -> str:
        return self.meta("instance") or ""

    def event(self, deployment: str | None, kind: str, msg: str) -> None:
        self.x("INSERT INTO events(ts,deployment,kind,msg) VALUES(?,?,?,?)", time.time(), deployment, kind, msg)

    # -- tokens -------------------------------------------------------------------
    def create_token(self, name: str, admin: bool = False) -> str:
        tok = new_secret("sst_")
        with self.tx():
            if self.one("SELECT 1 FROM tokens WHERE name=? AND revoked IS NULL", name):
                raise ValueError(f"token {name!r} exists; revoke it first")
            self.x("DELETE FROM tokens WHERE name=?", name)
            self.insert("tokens", {"name": name, "hash": hash_secret(tok), "admin": int(admin), "created": time.time()})
        return tok

    def revoke_token(self, name: str) -> bool:
        return self.x("UPDATE tokens SET revoked=? WHERE name=? AND revoked IS NULL", time.time(), name) > 0

    def principal(self, token: str) -> dict | None:
        return self.one("SELECT name, admin FROM tokens WHERE hash=? AND revoked IS NULL", hash_secret(token))

    # -- operator profile switches (persisted; survive restarts) ---------------------
    def profile_enabled(self, name: str, default: bool) -> bool:
        r = self.one("SELECT enabled FROM profile_state WHERE name=?", name)
        return bool(r["enabled"]) if r else default

    def set_profile(self, name: str, enabled: bool, note: str = "") -> None:
        self.x("INSERT INTO profile_state(name,enabled,changed,note) VALUES(?,?,?,?) "
               "ON CONFLICT(name) DO UPDATE SET enabled=excluded.enabled, changed=excluded.changed, note=excluded.note",
               name, int(enabled), time.time(), note)


class _Tx:
    def __init__(self, db):
        self.db = db

    def __enter__(self):
        self.db.execute("BEGIN IMMEDIATE")
        return self

    def __exit__(self, et, ev, tb):
        self.db.execute("ROLLBACK" if et else "COMMIT")
        return False


def dumps(o) -> str:
    return json.dumps(o, sort_keys=True, separators=(",", ":"))
