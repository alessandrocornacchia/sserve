"""Leases, deployments and the reconciler.

Vocabulary
- lease: one client's TTL reservation of a model deployment (acquire/renew/release)
- deployment: the shared serving group for one (model, cluster, gpus, overrides) spec;
  compatible leases share it, incompatible ones never do
- slot: one replica position of a deployment; it races candidate jobs on approved
  profiles and is served by the first candidate that becomes healthy
- candidate: one SLURM job (owned, tagged) that may become a slot's replica

The reconciler is level-triggered: every pass reads leases, SLURM and endpoint
health, then moves each object one step. Scheduler uncertainty is never read
as "ended", and nothing is submitted while an earlier submission is unresolved.
"""
from __future__ import annotations

import asyncio
import collections
import hashlib
import json
import logging
import re
import secrets
import time
from dataclasses import dataclass, field

import httpx

from . import slurm as S
from .config import OVERRIDES, Config, Model
from .routing import Hub
from .store import LIVE_CANDIDATE, Store, dumps, hash_secret, new_secret
from .tunnels import Tunnels

log = logging.getLogger("sserved")

TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "PREEMPTED", "OUT_OF_MEMORY",
            "BOOT_FAIL", "DEADLINE", "SPECIAL_EXIT", "REVOKED"}
RUNNING = {"RUNNING", "CONFIGURING", "COMPLETING", "STAGE_OUT"}
PENDING = {"PENDING", "REQUEUED", "REQUEUE_HOLD", "REQUEUE_FED", "RESV_DEL_HOLD", "SUSPENDED", "STOPPED", "RESIZING"}


class ApiError(Exception):
    def __init__(self, status: int, code: str, msg: str, retry_after: int | None = None):
        super().__init__(msg)
        self.status, self.code, self.msg, self.retry_after = status, code, msg, retry_after


@dataclass
class ReplicaRuntime:
    """In-memory per-candidate observations (not durable; rebuilt after restart)."""
    fails: int = 0
    routed: bool = False
    lport: int = 0
    samples: collections.deque = field(default_factory=lambda: collections.deque(maxlen=2000))
    last_probe: str = ""


def now() -> float:
    return time.time()


class Manager:
    def __init__(self, cfg: Config, store: Store, hub: Hub | None = None):
        self.cfg = cfg
        self.db = store
        self.hub = hub or Hub()
        self.instance = store.instance
        self.tunnels = Tunnels(self.instance, cfg.state_dir / "logs")
        self.slurm = {n: S.Slurm(c, mux_dir(cfg)) for n, c in cfg.clusters.items()}
        self.rt: dict[str, ReplicaRuntime] = collections.defaultdict(ReplicaRuntime)
        self.http = httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0))
        self.cluster_ok: dict[str, str] = {}      # cluster -> last query error ("" when fine)
        self.my_uid: dict[str, str] = {}
        self._swept: dict[str, float] = {}
        self._lock = asyncio.Lock()
        for p in cfg.profiles.values():           # persist the operator's profile set on first sight
            if self.db.one("SELECT 1 FROM profile_state WHERE name=?", p.name) is None:
                self.db.set_profile(p.name, p.enabled, "config default")

    # ======================================================================= leases
    def _spec(self, req: dict) -> tuple[Model, dict]:
        name = req.get("model")
        m = self.cfg.models.get(name or "")
        if m is None:
            raise ApiError(404, "model_not_in_catalog",
                           f"model {name!r} has no launch profile on this server; ask the operator to add one")
        ov = dict(req.get("overrides") or {})
        if req.get("gpus") is not None:
            ov["gpus"] = req["gpus"]
        for k, v in ov.items():
            if k not in m.allowed_overrides:
                raise ApiError(400, "override_not_allowed", f"override {k!r} is not allowed for {m.name}")
            try:
                ov[k] = OVERRIDES[k](v)
            except (TypeError, ValueError):
                raise ApiError(400, "bad_override", f"override {k!r} must be {OVERRIDES[k].__name__}")
        cluster = req.get("cluster") or None
        clusters = {self.cfg.profiles[p].cluster for p in m.profiles}
        if cluster and cluster not in clusters:
            raise ApiError(400, "cluster_not_allowed", f"{m.name} is not configured on cluster {cluster!r}")
        spec = {"model": m.name, "cluster": cluster, "gpus": ov.get("gpus", m.gpus),
                "max_model_len": ov.get("max_model_len", m.max_model_len)}
        return m, spec

    def eligible_profiles(self, m: Model, spec: dict) -> list[str]:
        out = []
        for pn in m.profiles:
            p = self.cfg.profiles[pn]
            if spec["cluster"] and p.cluster != spec["cluster"]:
                continue
            if not self.db.profile_enabled(pn, p.enabled):
                continue
            out.append(pn)
        return out

    def acquire(self, principal: dict, req: dict) -> dict:
        m, spec = self._spec(req)
        if not self.eligible_profiles(m, spec):
            raise ApiError(409, "no_enabled_profile",
                           f"no enabled compute profile for {m.name}" + (f" on {spec['cluster']}" if spec["cluster"] else ""))
        L = self.cfg.leases
        ttl = min(_num(req, "ttl_s", L.default_ttl_s), L.max_ttl_s)
        idle = min(_num(req, "idle_timeout_s", L.idle_timeout_s), L.max_idle_timeout_s)
        prep = min(_num(req, "prep_grace_s", L.prep_grace_s), idle * 4)
        acq = min(_num(req, "acquire_timeout_s", L.acquire_timeout_s), L.max_acquire_timeout_s)
        if ttl < L.min_ttl_s:
            raise ApiError(400, "ttl_too_short", f"ttl_s must be >= {L.min_ttl_s:.0f}")
        dep_id = "d-" + hashlib.sha256(dumps(spec).encode()).hexdigest()[:10]
        lid, key, t = "l-" + secrets.token_hex(8), new_secret("ssk_"), now()
        with self.db.tx():
            n = self.db.one("SELECT COUNT(*) n FROM leases WHERE client=? AND state='active'", principal["name"])["n"]
            if n >= L.max_leases_per_client:
                raise ApiError(429, "too_many_leases", f"client has {n} active leases (max {L.max_leases_per_client})")
            d = self.db.one("SELECT * FROM deployments WHERE id=?", dep_id)
            fp = launch_fingerprint(self.cfg, m.name, spec)
            if d is not None and d["fingerprint"] != fp:
                busy = self.db.one(f"SELECT 1 FROM candidates WHERE deployment=? AND state IN "
                                   f"({','.join('?' * len(LIVE_CANDIDATE))})", dep_id, *LIVE_CANDIDATE)
                if busy:
                    raise ApiError(409, "launch_settings_changed", "this deployment still runs jobs with the previous "
                                   "launch settings; retry once they have ended")
                self.db.update("deployments", dep_id, fingerprint=fp)
            if d is None:
                self.db.insert("deployments", {"id": dep_id, "spec": dumps(spec), "model": m.name, "state": "active",
                                               "target": 1, "created": t, "last_demand": t, "fingerprint": fp})
                self.db.event(dep_id, "deployment", f"created for {m.name} gpus={spec['gpus']}"
                                                    f"{' cluster=' + spec['cluster'] if spec['cluster'] else ''}")
            else:
                self.db.update("deployments", dep_id, state="active", target=max(1, d["target"]), last_demand=t)
            self.db.insert("leases", {"id": lid, "client": principal["name"], "deployment": dep_id,
                                      "key_hash": hash_secret(key), "label": str(req.get("label") or "")[:80],
                                      "created": t, "expires": t + ttl, "prep_grace": prep,
                                      "idle_timeout": idle, "acquire_deadline": t + acq,
                                      "available_since": t if self.hub.ready_since.get(dep_id) else None,
                                      "last_activity": None, "state": "active"})
            self.db.event(dep_id, "lease", f"{lid} acquired by {principal['name']}")
        return {**self.lease_view(self._lease(lid)), "inference_key": key}

    def _lease(self, lid: str) -> dict:
        r = self.db.one("SELECT * FROM leases WHERE id=?", lid)
        if r is None:
            raise ApiError(404, "lease_not_found", "no such lease")
        return r

    def _own(self, principal: dict, lid: str) -> dict:
        r = self.db.one("SELECT * FROM leases WHERE id=?", lid)
        if r is None or (r["client"] != principal["name"] and not principal.get("admin")):
            raise ApiError(404, "lease_not_found", "no such lease")    # do not reveal other clients' leases
        return r

    def lease_view(self, r: dict) -> dict:
        d = self.db.one("SELECT * FROM deployments WHERE id=?", r["deployment"])
        base = self.cfg.server.base_url
        return {"lease_id": r["id"], "client": r["client"], "label": r["label"], "model": d["model"],
                "deployment": r["deployment"], "state": r["state"], "end_reason": r["end_reason"],
                "created": r["created"], "expires": r["expires"], "prep_grace_s": r["prep_grace"],
                "available_since": r["available_since"], "acquire_deadline": r["acquire_deadline"],
                "idle_timeout_s": r["idle_timeout"], "last_activity": r["last_activity"],
                "inflight": self.hub.inflight_lease(r["id"]),
                "base_url": base, "openai_base_url": base + "/v1", "anthropic_base_url": base,
                "service": self.deployment_status(d)}

    def renew(self, principal: dict, lid: str, ttl: float | None) -> dict:
        r = self._own(principal, lid)
        self._maybe_end(r)
        r = self._lease(lid)
        if r["state"] != "active":
            raise ApiError(410, f"lease_{r['state']}", f"lease ended ({r['end_reason']}); acquire a new one")
        ttl = min(_num({"ttl_s": ttl}, "ttl_s", self.cfg.leases.default_ttl_s), self.cfg.leases.max_ttl_s)
        if ttl < self.cfg.leases.min_ttl_s:
            raise ApiError(400, "ttl_too_short", f"ttl_s must be >= {self.cfg.leases.min_ttl_s:.0f}")
        self.db.update("leases", lid, expires=now() + ttl)      # renewal is not activity
        return self.lease_view(self._lease(lid))

    def release(self, principal: dict, lid: str) -> dict:
        r = self._own(principal, lid)
        if r["state"] == "active":
            self._end(r, "released", "released by client")
        return self.lease_view(self._lease(lid))

    def rotate_key(self, principal: dict, lid: str) -> dict:
        r = self._own(principal, lid)
        if r["state"] != "active":
            raise ApiError(410, f"lease_{r['state']}", f"lease ended ({r['end_reason']})")
        key = new_secret("ssk_")
        self.db.update("leases", lid, key_hash=hash_secret(key))
        self.db.event(r["deployment"], "lease", f"{lid} key rotated")
        return {**self.lease_view(self._lease(lid)), "inference_key": key}

    def list_leases(self, principal: dict) -> list[dict]:
        if principal.get("admin"):
            rows = self.db.q("SELECT * FROM leases WHERE state='active' ORDER BY created")
        else:
            rows = self.db.q("SELECT * FROM leases WHERE client=? AND (state='active' OR ended>?) ORDER BY created",
                             principal["name"], now() - 3600)
        return [self.lease_view(r) for r in rows]

    def _end(self, r: dict, state: str, why: str) -> None:
        self.db.x("UPDATE leases SET state=?, ended=?, end_reason=? WHERE id=? AND state='active'",
                  state, now(), why, r["id"])
        self.db.event(r["deployment"], "lease", f"{r['id']} {state}: {why}")

    def _idle(self, r: dict, t: float) -> bool:
        """Idle time accrues only while the model is servable (a client waiting in the queue cannot
        send traffic), and only after the preparation grace that starts at first availability.
        In-flight requests are activity until they end; renewals and health probes are not."""
        if self.hub.inflight_lease(r["id"]) > 0 or r["available_since"] is None:
            return False
        ready = self.hub.ready_since.get(r["deployment"])
        if ready is None:
            return False
        return t > max(r["available_since"] + r["prep_grace"], ready, r["last_activity"] or 0) + r["idle_timeout"]

    def _maybe_end(self, r: dict) -> None:
        if r["state"] != "active":
            return
        t = now()
        if r["available_since"] is None and self.hub.ready_since.get(r["deployment"]):
            self.db.update("leases", r["id"], available_since=t)     # durable: grace starts once, here
            r["available_since"] = t
        if r["expires"] < t and self.hub.inflight_lease(r["id"]) == 0:
            self._end(r, "expired", "not renewed before its TTL")
        elif r["available_since"] is None and t > r["acquire_deadline"]:
            self._end(r, "unfulfilled", f"no ready replica within {r['acquire_deadline'] - r['created']:.0f}s "
                                        "of acquiring (acquire timeout)")
        elif self._idle(r, t):
            self._end(r, "idle", f"no inference for {r['idle_timeout']:.0f}s")

    # ------------------------------------------------------------- inference auth
    def authorize(self, key: str, model: str | None) -> tuple[dict, dict]:
        """Lease key -> (lease, deployment), or ApiError with an explicit code:
        401 invalid_api_key / lease_expired / lease_idle / lease_released, 404 model_not_found,
        503 deployment_blocked / no_ready_replica."""
        r = self.db.one("SELECT * FROM leases WHERE key_hash=?", hash_secret(key or ""))
        if r is None:
            raise ApiError(401, "invalid_api_key", "invalid API key")
        self._maybe_end(r)
        r = self._lease(r["id"])
        if r["state"] != "active":
            raise ApiError(401, f"lease_{r['state']}", f"lease {r['id']} ended ({r['end_reason']}); acquire a new lease")
        d = self.db.one("SELECT * FROM deployments WHERE id=?", r["deployment"])
        m = self.cfg.models[d["model"]]
        if model is not None and model not in (m.name, m.served):
            raise ApiError(404, "model_not_found", f"this key serves {m.name!r} only")
        return r, d

    def blocked(self, model: str) -> dict | None:
        return self.db.one("SELECT reason, at, deployment FROM breakers WHERE model=?", model)

    def admission(self, d: dict) -> None:
        if not self.hub.admitted(d["id"]):
            b = self.blocked(d["model"])
            if b:
                raise ApiError(503, "deployment_blocked", f"model blocked: {b['reason']}; operator reset needed",
                               retry_after=600)
            raise ApiError(503, "no_ready_replica", "no ready replica for this model yet (queued or starting); retry later",
                           retry_after=30)

    def touch(self, lid: str) -> None:
        self.db.update("leases", lid, last_activity=now())

    # ================================================================== status
    def deployment_status(self, d: dict) -> dict:
        slots = self.db.q("SELECT * FROM slots WHERE deployment=? AND state!='retired' ORDER BY created", d["id"])
        cands = self.db.q(f"SELECT * FROM candidates WHERE deployment=? AND state IN ({','.join('?' * len(LIVE_CANDIDATE))}) "
                          "ORDER BY submitted", d["id"], *LIVE_CANDIDATE)
        ready = sorted(self.hub.admitted(d["id"]))
        leases = self.db.one("SELECT COUNT(*) n FROM leases WHERE deployment=? AND state='active'", d["id"])["n"]
        blocked = self.blocked(d["model"])
        if blocked and not ready:
            state = "blocked"
        elif ready:
            state = "ready"
        elif any(c["state"] in ("starting", "ready") for c in cands):
            state = "starting"
        elif any(c["state"] in ("queued", "submitting", "unknown") for c in cands):
            state = "queued"
        elif leases:
            state = "waiting"
        else:
            state = "idle"
        recent = self.db.q("SELECT ts, kind, msg FROM events WHERE deployment=? ORDER BY id DESC LIMIT 8", d["id"])
        return {
            "deployment": d["id"], "model": d["model"], "spec": json.loads(d["spec"]), "state": state,
            "blocked": blocked,
            "active_leases": leases, "target_replicas": d["target"], "ready_replicas": len(ready),
            "max_model_len": d["max_model_len"],
            "slots": [{"slot": s["id"], "state": s["state"], "winner": s["winner"], "replaces": s["replaces"]} for s in slots],
            "jobs": [{"candidate": c["id"], "slot": c["slot"], "profile": c["profile"], "cluster": c["cluster"],
                      "job_id": c["job_id"], "state": c["state"], "slurm_state": c["slurm_state"],
                      "reason": c["reason"], "node": c["node"], "restarts": c["restarts"],
                      "time_left_s": c["time_left"], "routed": self.rt[c["id"]].routed if c["id"] in self.rt else False,
                      "metrics": self._last_sample(c["id"])} for c in cands],
            "recent_events": [{"ts": e["ts"], "kind": e["kind"], "msg": e["msg"]} for e in recent],
            "scheduler_reachable": {k: (v == "") for k, v in self.cluster_ok.items()},
        }

    def _last_sample(self, cid: str) -> dict | None:
        rt = self.rt.get(cid)
        if not rt or not rt.samples:
            return None
        ts, m = rt.samples[-1]
        return None if m is None else {"at": ts, **m}

    def status(self, principal: dict) -> dict:
        if principal.get("admin"):
            deps = self.db.q("SELECT * FROM deployments ORDER BY created")
        else:
            deps = self.db.q("SELECT d.* FROM deployments d WHERE d.id IN "
                             "(SELECT deployment FROM leases WHERE client=? AND state='active')", principal["name"])
        out = {"deployments": [self.deployment_status(d) for d in deps]}
        if principal.get("admin"):
            out["profiles"] = {n: {"enabled": self.db.profile_enabled(n, p.enabled), "opt_in": p.opt_in,
                                   "cluster": p.cluster, "policy": p.policy}
                               for n, p in self.cfg.profiles.items()}
        return out

    # =============================================================== reconciler
    async def run_forever(self, stop: asyncio.Event) -> None:
        killed = self.tunnels.kill_leftovers()
        if killed:
            self.db.event(None, "restart", f"stopped {len(killed)} leftover tunnel(s) from a previous run")
        while not stop.is_set():
            try:
                await self.reconcile()
            except Exception:  # noqa: BLE001 - one bad pass must not stop the controller
                log.exception("reconcile pass failed")
            try:
                await asyncio.wait_for(stop.wait(), self.cfg.server.reconcile_interval_s)
            except asyncio.TimeoutError:
                pass

    async def shutdown(self) -> None:
        await self.tunnels.close_all()
        await self.http.aclose()
        for sl in self.slurm.values():          # our own ControlPersist master would outlive us otherwise
            await sl.r.reset_mux()

    async def reconcile(self) -> None:
        async with self._lock:
            t = now()
            for r in self.db.q("SELECT * FROM leases WHERE state='active'"):
                self._maybe_end(r)
            await self._observe_jobs()
            await self._retry_cancels()
            for d in self.db.q("SELECT * FROM deployments WHERE state!='retired'"):
                await self._reconcile_deployment(d, t)

    # ---------------------------------------------------------------- observe
    async def _observe_jobs(self) -> None:
        live = self.db.q(f"SELECT * FROM candidates WHERE state IN ({','.join('?' * len(LIVE_CANDIDATE))})",
                         *LIVE_CANDIDATE)
        by_cluster: dict[str, list[dict]] = collections.defaultdict(list)
        for c in live:
            by_cluster[c["cluster"]].append(c)
        for cl in self.cfg.clusters:      # orphan sweep for clusters with nothing live
            if cl not in by_cluster and now() - self._swept.get(cl, 0) > self.cfg.scaling.orphan_sweep_s:
                by_cluster[cl] = []
        for cl, cands in by_cluster.items():
            self._swept[cl] = now()
            sl = self.slurm[cl]
            for c in [c for c in cands if c["state"] in ("submitting", "unknown")]:
                await self._resolve(sl, c)
            ids = [c["job_id"] for c in self.db.q(
                "SELECT job_id FROM candidates WHERE cluster=? AND job_id IS NOT NULL AND state IN "
                f"({','.join('?' * len(LIVE_CANDIDATE))})", cl, *LIVE_CANDIDATE)]
            try:
                uid, rows, acct = await sl.query(self.instance, ids)
            except S.Unreachable as e:
                if self.cluster_ok.get(cl) != str(e):
                    self.db.event(None, "scheduler", f"{cl} unreachable: {e} (job states unchanged)")
                self.cluster_ok[cl] = str(e)
                continue
            self.cluster_ok[cl], self.my_uid[cl] = "", uid
            known = set()
            for c in self.db.q("SELECT * FROM candidates WHERE cluster=? AND job_id IS NOT NULL AND state IN "
                               f"({','.join('?' * len(LIVE_CANDIDATE))})", cl, *LIVE_CANDIDATE):
                known.add(c["job_id"])
                self._apply(c, rows.get(c["job_id"]), acct.get(c["job_id"]), uid)
            for jid, row in rows.items():
                if jid not in known:
                    await self._unexpected_job(sl, cl, jid, row)

    async def _unexpected_job(self, sl: S.Slurm, cl: str, jid: str, row: S.JobRow) -> None:
        """A live job tagged by this instance that no live candidate tracks under that id."""
        cid = row.comment.split(":", 2)[2] if row.comment.count(":") >= 2 else ""
        c = self.db.one("SELECT * FROM candidates WHERE id=?", cid)
        if c is not None:
            run_id = c["run_dir"].rstrip("/").rsplit("/", 1)[-1]
            if c["cluster"] != cl or (row.workdir and not row.workdir.rstrip("/").endswith(run_id)) or \
                    (c["job_id"] and c["job_id"] != jid):
                self.db.event(c["deployment"], "orphan", f"job {jid} claims candidate {cid} but does not match it; left alone")
                return
            if c["state"] in ("submitting", "unknown"):
                self.db.update("candidates", cid, job_id=jid, state="queued")      # resolution by tag
                self.db.event(c["deployment"], "submit", f"{cid}: submission found as job {jid} (tag)")
                return
            if c["state"] == "ended":
                # we concluded it ended (e.g. LOST/NOT_SUBMITTED) but it is alive: track it again and let the
                # slot logic keep or cancel it; never leave it running untracked
                slot = self.db.one("SELECT state FROM slots WHERE id=?", c["slot"])
                revive = "queued" if row.state in PENDING else "starting"
                self.db.update("candidates", cid, job_id=jid, state=revive, end_state=None, ended=None,
                               missing_since=None, running_since=now() if revive == "starting" else None,
                               node=row.node or None)
                self.db.event(c["deployment"], "orphan", f"{cid} job {jid} was considered ended but is {row.state}; tracking again")
                if not slot or slot["state"] in ("retiring", "retired"):
                    c = self.db.one("SELECT * FROM candidates WHERE id=?", cid)
                    await self._cancel(c, "revived job belongs to a retired slot")
                return
            return
        self.db.event(None, "orphan", f"job {jid} on {cl} carries this manager's tag but no known candidate; cancelling")
        res = await sl.cancel(jid, row.comment, "/nonexistent")
        if res != "CANCELLED":
            self.db.event(None, "orphan", f"cancel of orphan job {jid}: {res}; will retry")

    async def _resolve(self, sl: S.Slurm, c: dict) -> None:
        res, info = await sl.resolve(c["run_dir"].rsplit("/", 1)[-1], self._job_name(c))
        if res == "submitted":
            # a pending cancellation intent applies as soon as the job is identified
            self.db.update("candidates", c["id"], job_id=info, state="cancelling" if c["cancel_requested"] else "queued")
            self.db.event(c["deployment"], "submit", f"{c['id']}: unresolved submission found as job {info}")
        elif res == "not_submitted" and now() - c["submitted"] > self.cfg.scaling.submit_resolve_age_s:
            self.db.update("candidates", c["id"], state="ended", end_state="NOT_SUBMITTED", ended=now())
            self.db.event(c["deployment"], "submit", f"{c['id']}: confirmed not submitted ({info})")
        elif res == "rejected":
            self._rejected(c, info)

    def _job_name(self, c: dict) -> str:
        return f"{S.JOB_PREFIX}{c['deployment']}-{c['id'][-6:]}"

    def _comment(self, c: dict) -> str:
        return f"sserved:{self.instance}:{c['id']}"

    def _apply(self, c: dict, row: S.JobRow | None, acct: tuple[bool, str] | None, uid: str) -> None:
        """Move a candidate one step from what SLURM reported."""
        cid, t = c["id"], now()
        if row is not None:
            st = row.state
            upd = {"slurm_state": st, "reason": row.reason or None, "time_left": row.time_left, "missing_since": None}
            if st in PENDING:
                if c["state"] in ("starting", "ready", "draining"):
                    self._lost_endpoint(c, f"job {c['job_id']} back in queue ({st}, {row.reason}): preempted/requeued")
                    upd.update(generation=c["generation"] + 1, running_since=None, ready_since=None, node=None)
                if c["state"] in ("unknown", "submitting", "starting", "ready"):
                    upd["state"] = "queued"
            elif st in RUNNING:
                if c["state"] in ("ready", "starting", "draining") and (
                        (row.restarts is not None and row.restarts != c["restarts"]) or
                                                                        (c["node"] and row.node and row.node != c["node"])):
                    self._lost_endpoint(c, f"job {c['job_id']} restarted (restarts {c['restarts']}->{row.restarts}, "
                                           f"node {c['node']}->{row.node})")
                    upd.update(generation=c["generation"] + 1, ready_since=None, running_since=t, state="starting"
                               if c["state"] != "draining" else "draining")
                elif c["state"] in ("queued", "unknown", "submitting"):
                    upd.update(state="starting", running_since=t)
                    self.db.event(c["deployment"], "job", f"{cid} job {c['job_id']} running on {row.node} ({c['profile']})")
                upd.update(node=row.node or c["node"], restarts=c["restarts"] if row.restarts is None else row.restarts)
            self.db.update("candidates", cid, **upd)
            return
        # not in a successful squeue listing: sacct decides. A failed sacct says nothing; an empty
        # one only starts a grace period.
        if acct is None or not acct[0]:
            return
        state = acct[1].split("|")[0].strip()
        base = state.split()[0] if state else ""
        if base in TERMINAL:
            self._ended(c, state, uid)
        elif base in PENDING or base in RUNNING:
            pass                                    # squeue/sacct race; look again next pass
        else:
            if c["missing_since"] is None:
                self.db.update("candidates", cid, missing_since=t)
            elif t - c["missing_since"] > self.cfg.scaling.unknown_terminal_grace_s:
                self._ended(c, "LOST (absent from squeue and sacct)", uid)

    def _lost_endpoint(self, c: dict, why: str) -> None:
        self._unroute(c)
        asyncio.get_running_loop().create_task(self.tunnels.close(c["id"]))
        self.db.event(c["deployment"], "replica", f"{c['id']}: {why}")

    def _unroute(self, c: dict) -> None:
        self.hub.evict(c["deployment"], c["id"])
        if c["id"] in self.rt:
            self.rt[c["id"]].routed = False

    def _ended(self, c: dict, state: str, uid: str) -> None:
        cid, t = c["id"], now()
        self._lost_endpoint(c, f"job {c['job_id']} ended: {state}")
        self.db.update("candidates", cid, state="ended", end_state=state[:200], ended=t)
        d = self.db.one("SELECT * FROM deployments WHERE id=?", c["deployment"])
        base = state.split()[0]
        m = re.search(r"by (\d+)", state)
        who = m.group(1) if m else None
        if base == "CANCELLED" and c["cancel_requested"] and (who == uid or (who is None and c["cancel_acked"])):
            return                                  # ours: sacct attributes it to us, or our scancel was acknowledged
        if c["cancel_requested"] and base != "CANCELLED":
            return                                  # ended some other way while we were cancelling it
        if base == "CANCELLED":
            who = who or "unknown"
            if who != uid:
                self._breaker(d, f"job {c['job_id']} ({c['profile']}) cancelled by uid {who}, not by this manager: "
                                 "treating as a site-policy cancellation")
            else:
                self._breaker(d, f"job {c['job_id']} cancelled by this account outside the manager")
            return
        if base in ("TIMEOUT", "PREEMPTED", "DEADLINE"):
            return                                  # walltime / preemption: replaceable within budgets
        if c["ready_since"] is None:                # never became healthy: a startup failure
            n = d["startup_failures"] + 1
            self.db.update("deployments", d["id"], startup_failures=n)
            if n >= self.cfg.scaling.max_startup_failures:
                self._breaker(d, f"{n} consecutive startup failures (last: job {c['job_id']} {state}; "
                                 f"see {c['run_dir']}/slurm-{c['job_id']}.err)")

    def _rejected(self, c: dict, why: str) -> None:
        self.db.update("candidates", c["id"], state="ended", end_state="REJECTED", ended=now(), reason=why[:300])
        self.db.event(c["deployment"], "submit", f"{c['id']} sbatch rejected on {c['profile']}: {why[:300]}")
        d = self.db.one("SELECT * FROM deployments WHERE id=?", c["deployment"])
        n = d["startup_failures"] + 1
        self.db.update("deployments", d["id"], startup_failures=n)
        if n >= self.cfg.scaling.max_startup_failures:
            self._breaker(d, f"sbatch rejected {n} times in a row (last on {c['profile']}: {why[:200]})")

    def _breaker(self, d: dict, reason: str) -> None:
        """Model-wide and persistent: a new lease, another cluster or an override cannot reopen it;
        only `sserved reset MODEL` does. Not-ready candidates are cancelled by the reconciler."""
        if self.blocked(d["model"]):
            return
        self.db.x("INSERT OR IGNORE INTO breakers(model, reason, at, deployment) VALUES(?,?,?,?)",
                  d["model"], reason, now(), d["id"])
        self.db.event(d["id"], "blocked", f"{reason} -- {d['model']} blocked: no further submissions until "
                                          f"`sserved reset {d['model']}`")

    # --------------------------------------------------------- per deployment
    async def _reconcile_deployment(self, d: dict, t: float) -> None:
        sc, m = self.cfg.scaling, self.cfg.models.get(d["model"])
        if m is None:
            return
        spec = json.loads(d["spec"])
        blocked = self.blocked(d["model"])
        demand = self.db.one("SELECT COUNT(*) n FROM leases WHERE deployment=? AND state='active'", d["id"])["n"]
        if demand:
            self.db.update("deployments", d["id"], last_demand=t)
            d["last_demand"] = t
        lingering = not demand and d["last_demand"] and t - d["last_demand"] < self.cfg.leases.linger_s
        cands = self.db.q("SELECT * FROM candidates WHERE deployment=? AND state IN "
                          f"({','.join('?' * len(LIVE_CANDIDATE))})", d["id"], *LIVE_CANDIDATE)

        # 1. endpoints: tunnels, probes, admission, startup deadlines
        for c in cands:
            if c["state"] in ("starting", "ready", "draining") and c["node"]:
                await self._endpoint(d, m, c, spec)

        slots = {s["id"]: s for s in self.db.q("SELECT * FROM slots WHERE deployment=? AND state!='retired'", d["id"])}
        cands = self.db.q("SELECT * FROM candidates WHERE deployment=? AND state IN "
                          f"({','.join('?' * len(LIVE_CANDIDATE))})", d["id"], *LIVE_CANDIDATE)
        by_slot = collections.defaultdict(list)
        for c in cands:
            by_slot[c["slot"]].append(c)

        # 1a. a blocked model keeps serving what is ready but holds nothing else
        if blocked:
            for c in cands:
                if c["state"] in ("submitting", "unknown", "queued", "starting"):
                    await self._cancel(c, "model blocked: " + blocked["reason"][:120])

        # 1b. bound duplicate start-up capacity: a slot's 2nd+ RUNNING-not-ready candidates
        dup = [c for c in sorted((c for c in cands if c["state"] == "starting"), key=lambda c: c["running_since"] or 0)
               if any(o["slot"] == c["slot"] and o["id"] != c["id"] and o["state"] in ("starting", "ready") and
                      (o["running_since"] or 0) <= (c["running_since"] or 0) for o in cands)]
        budget = sc.max_startup_gpus
        for c in dup:
            if c["gpus"] <= budget:
                budget -= c["gpus"]
            else:
                await self._cancel(c, f"duplicate start-up capacity cap ({sc.max_startup_gpus} GPUs) reached")

        # 2. slot bookkeeping: winners, sibling cancellation, lost winners.
        # A replacement whose original is gone (retiring/retired) becomes a primary slot.
        for s in slots.values():
            if s["replaces"] and (s["replaces"] not in slots or slots[s["replaces"]]["state"] == "retiring"):
                self.db.update("slots", s["id"], replaces=None)
                s["replaces"] = None
        for s in slots.values():
            mine = by_slot.get(s["id"], [])
            ready = [c for c in mine if c["state"] == "ready"]
            if s["state"] == "retiring":
                continue
            if s["winner"] and not any(c["id"] == s["winner"] and c["state"] == "ready" for c in mine):
                repl = self.db.one("SELECT id FROM slots WHERE replaces=? AND state!='retired'", s["id"])
                if s["replaces"] is None and repl:
                    with self.db.tx():
                        self.db.update("slots", repl["id"], replaces=None)       # promoted now
                        self._retire_slot(s, "winner lost; its replacement slot takes over")
                    continue
                self.db.update("slots", s["id"], winner=None, state="racing")
                s.update(winner=None, state="racing")
                self.db.event(d["id"], "slot", f"{s['id']} lost its replica; racing again")
            if not s["winner"] and ready:
                w = min(ready, key=lambda c: c["ready_since"])
                self.db.update("slots", s["id"], winner=w["id"], state="serving")
                s.update(winner=w["id"], state="serving")
                self.db.event(d["id"], "slot", f"{s['id']} served by {w['id']} (job {w['job_id']}, {w['profile']})")
                if s["replaces"]:          # hand over atomically: the replacement becomes the primary
                    with self.db.tx():
                        old = s["replaces"]
                        self.db.update("slots", s["id"], replaces=None)
                        if old in slots:
                            self._retire_slot(slots[old], f"replaced by {s['id']}")
                    s["replaces"] = None
            if s["winner"]:
                for c in mine:
                    if c["id"] != s["winner"] and c["state"] in ("submitting", "unknown", "queued", "starting", "ready"):
                        await self._cancel(c, f"sibling of {s['winner']} which won slot {s['id']}")

        # 3. desired replicas
        slots = {s["id"]: s for s in self.db.q("SELECT * FROM slots WHERE deployment=? AND state!='retired'", d["id"])}
        active_slots = [s for s in slots.values() if s["state"] in ("racing", "serving")]
        primary = [s for s in active_slots if not s["replaces"]]
        if demand or lingering:
            target = max(1, d["target"])
            if not demand:               # lingering: keep serving replicas, never hold queued jobs
                for s in active_slots:
                    if s["state"] == "racing":
                        self._retire_slot(s, "no demand (lingering)")
            else:
                target = await self._autoscale(d, m, primary, t)
                if len(primary) < target and not blocked:
                    sid = "s-" + secrets.token_hex(5)
                    self.db.insert("slots", {"id": sid, "deployment": d["id"], "state": "racing", "created": t})
                    self.db.event(d["id"], "slot", f"{sid} created (target {target})")
                elif len(primary) > target:
                    victim = sorted(primary, key=lambda s: (s["state"] == "serving", self._time_left(s)))[0]
                    self._retire_slot(victim, f"scale down to {target}")
                self._replacements(d, m, primary, by_slot, t)
        else:
            for s in active_slots:
                self._retire_slot(s, "no active leases" + (" (linger elapsed)" if d["last_demand"] else ""))
            if not cands:
                self.db.update("deployments", d["id"], state="idle")

        # 4. submissions for racing slots
        slots = {s["id"]: s for s in self.db.q("SELECT * FROM slots WHERE deployment=? AND state!='retired'", d["id"])}
        if demand and not blocked:
            for s in slots.values():
                if s["state"] == "racing":
                    await self._race(d, m, spec, s, by_slot.get(s["id"], []), t)

        # 5. retiring slots: drain, then cancel; close when nothing is live
        for s in slots.values():
            if s["state"] != "retiring":
                continue
            mine = self.db.q("SELECT * FROM candidates WHERE slot=? AND state IN "
                             f"({','.join('?' * len(LIVE_CANDIDATE))})", s["id"], *LIVE_CANDIDATE)
            for c in mine:
                if c["state"] in ("ready", "draining") and c["id"] == s["winner"]:
                    if c["state"] == "ready":
                        self._unroute(c)
                        self.db.update("candidates", c["id"], state="draining")
                        self.db.event(d["id"], "drain", f"{c['id']} draining (deadline {sc.drain_deadline_s:.0f}s)")
                    if self.hub.drained(d["id"], c["id"]) or t > (s["drain_deadline"] or t):
                        await self._cancel(c, "drained" if self.hub.drained(d["id"], c["id"]) else "drain deadline")
                elif c["state"] != "cancelling":
                    await self._cancel(c, s["reason"] or "slot retired")
            if not mine:
                self.db.update("slots", s["id"], state="retired")

    def _time_left(self, s: dict) -> float:
        if not s["winner"]:
            return 0
        c = self.db.one("SELECT time_left FROM candidates WHERE id=?", s["winner"])
        return (c or {}).get("time_left") or 0

    def _retire_slot(self, s: dict, why: str) -> None:
        if s["state"] in ("retiring", "retired"):
            return
        self.db.update("slots", s["id"], state="retiring", reason=why, drain_deadline=now() + self.cfg.scaling.drain_deadline_s)
        s["state"] = "retiring"
        self.db.event(s["deployment"], "slot", f"{s['id']} retiring: {why}")

    async def _cancel(self, c: dict, why: str, force: bool = False) -> None:
        """Durable intent first (crash-safe), then scancel; only a successful scancel is acknowledged.
        Unacknowledged cancellations are retried by _retry_cancels until SLURM shows the job ended."""
        if c["state"] == "cancelling" and c["cancel_acked"] and not force:
            return
        self._unroute(c)
        t = now()
        if not c["job_id"]:
            if c["state"] in ("submitting", "unknown"):
                if not c["cancel_requested"]:
                    self.db.update("candidates", c["id"], cancel_requested=t)   # applied once resolved
                    self.db.event(c["deployment"], "cancel", f"{c['id']}: {why} (after its submission is resolved)")
                return
            self.db.update("candidates", c["id"], state="ended", end_state="NOT_SUBMITTED", ended=t)
            return
        if c["state"] != "cancelling" or not c["cancel_requested"]:
            self.db.update("candidates", c["id"], state="cancelling", cancel_requested=c["cancel_requested"] or t)
            self.db.event(c["deployment"], "cancel", f"{c['id']} job {c['job_id']}: {why}")
        self.db.update("candidates", c["id"], cancel_attempt=t)
        res = await self.slurm[c["cluster"]].cancel(c["job_id"], self._comment(c), c["run_dir"])
        if res == "CANCELLED":
            self.db.update("candidates", c["id"], cancel_acked=now())
            await self.tunnels.close(c["id"])
        elif res == "GONE":
            await self.tunnels.close(c["id"])      # outcome (and who ended it) comes from sacct
        elif res == "NOT_OURS":
            self.db.update("candidates", c["id"], state="ended", end_state="NOT_OURS (tag mismatch)", ended=now())
            self.db.event(c["deployment"], "cancel", f"{c['id']} job {c['job_id']} tag mismatch: NOT cancelled, untracked")
        else:
            self.db.event(c["deployment"], "cancel", f"{c['id']} job {c['job_id']} cancel not confirmed ({res[:120]}); will retry")

    async def _retry_cancels(self) -> None:
        for c in self.db.q("SELECT * FROM candidates WHERE state='cancelling' AND cancel_acked IS NULL AND job_id IS NOT NULL"):
            if now() - (c["cancel_attempt"] or 0) >= 10:
                await self._cancel(c, "retry")

    # --------------------------------------------------------------- endpoint
    async def _endpoint(self, d: dict, m: Model, c: dict, spec: dict) -> None:
        sc, rt, t = self.cfg.scaling, self.rt[c["id"]], now()
        cl = self.cfg.clusters[c["cluster"]]
        tun = await self.tunnels.ensure(c["id"], cl, c["node"], c["port"])
        rt.lport = tun.lport
        ok, why, info = await self._probe(tun.lport, c, m)
        rt.last_probe = why
        if c["state"] == "starting":
            if ok and m.readiness_inference:
                ok, why = await self._readiness_inference(tun.lport, c, m)
            if ok:
                self.db.update("candidates", c["id"], state="ready", ready_since=t, unhealthy_since=None)
                self.db.update("deployments", d["id"], startup_failures=0,
                               max_model_len=info.get("max_model_len") or d["max_model_len"])
                self.db.event(d["id"], "ready", f"{c['id']} healthy on {c['node']} (job {c['job_id']}, {c['profile']})")
                c = self.db.one("SELECT * FROM candidates WHERE id=?", c["id"])
                slot = self.db.one("SELECT * FROM slots WHERE id=?", c["slot"])
                if slot and (slot["winner"] in (None, c["id"])) and slot["state"] in ("racing", "serving"):
                    self._route(d, m, c, rt)
            elif c["running_since"] and t - c["running_since"] > m.startup_timeout_s:
                n = d["startup_failures"] + 1
                self.db.update("deployments", d["id"], startup_failures=n)
                await self._cancel(c, f"not healthy {m.startup_timeout_s:.0f}s after starting ({why})")
                if n >= sc.max_startup_failures:
                    self._breaker(d, f"{n} consecutive startup failures (last: {why})")
            return
        if c["state"] == "draining":
            return
        # ready
        sample = await self._metrics(tun.lport, c) if ok else None
        rt.samples.append((t, sample))
        if ok:
            rt.fails = 0
            if c["unhealthy_since"]:
                self.db.update("candidates", c["id"], unhealthy_since=None)
            slot = self.db.one("SELECT * FROM slots WHERE id=?", c["slot"])
            if slot and slot["winner"] == c["id"] and slot["state"] == "serving" and not rt.routed:
                self._route(d, m, c, rt)
            return
        rt.fails += 1
        if rt.fails >= sc.health_fail_unroute and rt.routed:
            self._unroute(c)
            self.db.event(d["id"], "replica", f"{c['id']} unrouted after {rt.fails} failed probes ({why})")
        if not c["unhealthy_since"]:
            self.db.update("candidates", c["id"], unhealthy_since=t)
        elif t - c["unhealthy_since"] > sc.health_fail_replace_s:
            await self._cancel(c, f"unhealthy for {t - c['unhealthy_since']:.0f}s while RUNNING ({why})")

    def _route(self, d: dict, m: Model, c: dict, rt: ReplicaRuntime) -> None:
        self.hub.admit(d["id"], c["id"], m.served, rt.lport, c["backend_key"])
        rt.routed = True
        self.db.event(d["id"], "route", f"{c['id']} admitted to routing")

    async def _probe(self, lport: int, c: dict, m: Model) -> tuple[bool, str, dict]:
        base = f"http://127.0.0.1:{lport}"
        try:
            h = await self.http.get(base + "/health", timeout=5)
            if h.status_code != 200:
                return False, f"/health {h.status_code}", {}
            r = await self.http.get(base + "/v1/models", headers={"Authorization": f"Bearer {c['backend_key']}"}, timeout=10)
            if r.status_code != 200:
                return False, f"/v1/models {r.status_code}", {}
            data = r.json().get("data", [])
            ids = {x.get("id") for x in data}
            if m.served not in ids:
                return False, f"serves {sorted(ids)} not {m.served}", {}
            card = next(x for x in data if x.get("id") == m.served)
            return True, "ok", {"max_model_len": card.get("max_model_len")}
        except (httpx.HTTPError, ValueError) as e:
            return False, f"{type(e).__name__}", {}

    async def _readiness_inference(self, lport: int, c: dict, m: Model) -> tuple[bool, str]:
        """One tiny tool-call request (excluded from demand accounting) to verify the profile's
        tool parser before admission."""
        body = {"model": m.served, "max_tokens": 64, "messages": [{"role": "user", "content": "Call the ping tool."}],
                "tools": [{"type": "function", "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}}}],
                "tool_choice": "required"}
        try:
            r = await self.http.post(f"http://127.0.0.1:{lport}/v1/chat/completions", json=body, timeout=120,
                                     headers={"Authorization": f"Bearer {c['backend_key']}"})
            msg = r.json()["choices"][0]["message"]
            return (bool(msg.get("tool_calls")), "ok" if msg.get("tool_calls") else "no tool call in readiness check")
        except Exception as e:  # noqa: BLE001
            return False, f"readiness inference failed: {type(e).__name__}"

    async def _metrics(self, lport: int, c: dict) -> dict | None:
        try:
            r = await self.http.get(f"http://127.0.0.1:{lport}/metrics", timeout=5,
                                    headers={"Authorization": f"Bearer {c['backend_key']}"})
            if r.status_code != 200:
                return None
            v = parse_prom(r.text)
            if "vllm:num_requests_running" not in v or "vllm:num_requests_waiting" not in v:
                return None
            return {"running": v["vllm:num_requests_running"], "waiting": v["vllm:num_requests_waiting"],
                    "kv": v.get("vllm:kv_cache_usage_perc")}
        except (httpx.HTTPError, ValueError):
            return None

    # ------------------------------------------------------------- racing
    async def _race(self, d: dict, m: Model, spec: dict, s: dict, mine: list[dict], t: float) -> None:
        sc = self.cfg.scaling
        if any(c["state"] in ("submitting", "unknown") for c in mine):
            return                                      # resolve before submitting anything else
        used = {c["profile"] for c in mine}
        last_fail = self.db.one("SELECT MAX(ended) t FROM candidates WHERE deployment=? AND state='ended' AND "
                                "end_state NOT IN ('NOT_SUBMITTED') AND cancel_requested IS NULL", d["id"])["t"] or 0
        if t - last_fail < sc.submit_backoff_s * (1 + 0.25 * secrets.randbelow(100) / 100):
            return
        day = self.db.one("SELECT COUNT(*) n FROM candidates WHERE deployment=? AND submitted>?", d["id"], t - 86400)["n"]
        held = self.db.one(f"SELECT COALESCE(SUM(gpus),0) g FROM candidates WHERE state IN "
                           f"({','.join('?' * len(LIVE_CANDIDATE))})", *LIVE_CANDIDATE)["g"]
        for pn in self.eligible_profiles(m, spec):
            if len(mine) >= sc.max_candidates_per_slot:
                break
            if pn in used:
                continue
            if day >= sc.max_submissions_per_day:
                self._note_once(d, "budget", f"daily submission budget ({sc.max_submissions_per_day}) used up")
                return
            if held + spec["gpus"] > sc.max_total_gpus:
                self._note_once(d, "budget", f"GPU budget: {held} held + {spec['gpus']} > {sc.max_total_gpus}")
                return
            p = self.cfg.profiles[pn]
            if p.max_gpus:
                ph = self.db.one(f"SELECT COALESCE(SUM(gpus),0) g FROM candidates WHERE profile=? AND state IN "
                                 f"({','.join('?' * len(LIVE_CANDIDATE))})", pn, *LIVE_CANDIDATE)["g"]
                if ph + spec["gpus"] > p.max_gpus:
                    continue
            recent_rej = self.db.one("SELECT 1 FROM candidates WHERE deployment=? AND profile=? AND end_state='REJECTED' "
                                     "AND ended>?", d["id"], pn, t - 3600)
            if recent_rej:
                continue
            c = await self._submit(d, m, spec, s, pn)
            mine.append(c)
            used.add(pn)
            day += 1
            held += spec["gpus"]

    def _note_once(self, d: dict, kind: str, msg: str) -> None:
        last = self.db.one("SELECT msg FROM events WHERE deployment=? AND kind=? ORDER BY id DESC LIMIT 1", d["id"], kind)
        if not last or last["msg"] != msg:
            self.db.event(d["id"], kind, msg)

    async def _submit(self, d: dict, m: Model, spec: dict, s: dict, pn: str) -> dict:
        p = self.cfg.profiles[pn]
        cl = self.cfg.clusters[p.cluster]
        cid = "c-" + secrets.token_hex(6)
        run_id = f"sserved-{d['id']}-{cid}"
        port = 20000 + secrets.randbelow(10000)
        key, nonce = secrets.token_urlsafe(32), secrets.token_hex(16)
        row = {"id": cid, "slot": s["id"], "deployment": d["id"], "profile": pn, "cluster": p.cluster,
               "gpus": spec["gpus"], "run_dir": f"{cl.runs_dir.rstrip('/')}/{run_id}", "port": port,
               "backend_key": key, "nonce": nonce, "state": "submitting", "submitted": now()}
        self.db.insert("candidates", row)              # durable *before* sbatch: a crash leaves "submitting"
        text = S.sbatch_script(profile=p, model=m, gpus=spec["gpus"], recipe_dir=f"{cl.deploy_dir.rstrip('/')}/{m.recipe}",
                               port=port, max_model_len=spec["max_model_len"], served=m.served)
        res, info = await self.slurm[p.cluster].submit(run_id, self._job_name(row), self._comment(row), text, key, nonce)
        if res == "submitted":
            self.db.update("candidates", cid, job_id=info, state="queued")
            self.db.event(d["id"], "submit", f"{cid} job {info} on {pn} for slot {s['id']}")
        elif res == "rejected":
            self._rejected(row, info)
        else:
            self.db.update("candidates", cid, state="unknown", reason=info[:300])
            self.db.event(d["id"], "submit", f"{cid} submission outcome unknown ({info[:200]}); resolving before any retry")
        return self.db.one("SELECT * FROM candidates WHERE id=?", cid)

    # ------------------------------------------------------- scaling, replacement
    async def _autoscale(self, d: dict, m: Model, primary: list[dict], t: float) -> int:
        sc = self.cfg.scaling
        target = max(1, d["target"])
        if d["last_scale"] and t - d["last_scale"] < sc.cooldown_s:
            return min(target, m.max_replicas)
        serving = [s for s in primary if s["state"] == "serving"]
        racing = [s for s in primary if s["state"] == "racing"]
        win = t - sc.window_s
        series = []
        complete = bool(serving)
        for s in serving:
            rt = self.rt.get(s["winner"])
            pts = [(ts, v) for ts, v in (rt.samples if rt else []) if ts >= win]
            if not rt or not rt.samples or rt.samples[0][0] > win or not pts or any(v is None for _, v in pts):
                complete = False
            series.append(pts)
        cap_err = self.hub.capacity_errors_since(d["id"], win)
        n_pts = min((len(p) for p in series), default=0)
        # per sample index i: (sum waiting, sum running+waiting, any kv-hot replica with waiting) across replicas
        agg = [(sum(p[i][1]["waiting"] for p in series),
                sum(p[i][1]["running"] + p[i][1]["waiting"] for p in series),
                any((p[i][1].get("kv") or 0) >= sc.up_kv_usage and p[i][1]["waiting"] > 0 for p in series))
               for i in range(n_pts)] if complete else []
        pressure = cap_err >= sc.up_capacity_errors
        if agg:   # sustained: every sample in the window over the threshold
            pressure = pressure or all(w / len(series) >= sc.up_waiting_per_replica or hot for w, _, hot in agg)
        new = target
        if pressure and not racing and len(primary) < m.max_replicas:
            new = len(primary) + 1
        elif agg and len(serving) >= 2 and target > 1 and cap_err == 0 and not racing:
            n = len(serving)
            if max(load for _, load, _ in agg) <= (n - 1) * sc.down_load_per_replica:
                new = n - 1
        new = min(max(1, new), m.max_replicas)
        if new != target:
            self.db.update("deployments", d["id"], target=new, last_scale=t)
            self.db.event(d["id"], "scale", f"target {target} -> {new} "
                                            f"({'pressure' if new > target else 'sustained low load'}; "
                                            f"capacity errors in window: {cap_err})")
        return new

    def _replacements(self, d: dict, m: Model, primary: list[dict], by_slot: dict, t: float) -> None:
        sc = self.cfg.scaling
        if self.blocked(d["model"]):
            return
        recent = self.db.one("SELECT MAX(last_activity) a FROM leases WHERE deployment=?", d["id"])["a"] or 0
        if t - recent > sc.replace_requires_activity_s:
            return                                   # no real traffic: let it end, do not chain allocations
        done = self.db.one("SELECT COUNT(*) n FROM slots WHERE deployment=? AND replaces IS NOT NULL AND created>?",
                           d["id"], t - 86400)["n"]
        if self.db.one("SELECT 1 FROM slots WHERE deployment=? AND replaces IS NOT NULL AND state IN ('racing','serving')",
                       d["id"]):
            return                                   # one renewal at a time: bounded +1 surge
        for s in sorted(primary, key=self._time_left):
            if s["state"] != "serving":
                continue
            left = self._time_left(s)
            if not left or left > sc.replace_before_s:
                continue
            if done >= sc.max_replacements_per_day:
                self._note_once(d, "budget", f"replacement budget ({sc.max_replacements_per_day}/day) used up")
                return
            sid = "s-" + secrets.token_hex(5)
            self.db.insert("slots", {"id": sid, "deployment": d["id"], "state": "racing", "created": t, "replaces": s["id"]})
            self.db.event(d["id"], "replace", f"{sid} racing to replace {s['id']} ({left:.0f}s walltime left)")
            return

    # ------------------------------------------------------------------ admin
    def reset_breaker(self, model: str) -> bool:
        return reset_breaker(self.db, model)


def parse_prom(text: str) -> dict[str, float]:
    """Sum Prometheus samples per metric name (labels dropped)."""
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        try:
            name_labels, val = line.rsplit(" ", 1) if "}" not in line else (line[:line.rindex("}") + 1], line[line.rindex("}") + 1:])
            name = name_labels.split("{", 1)[0].strip()
            out[name] = out.get(name, 0.0) + float(val.strip().split()[0])
        except (ValueError, IndexError):
            continue
    return out


def _num(req: dict, key: str, default: float) -> float:
    v = req.get(key)
    if v is None:
        return float(default)
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ApiError(400, "bad_number", f"{key} must be a number")
    try:
        f = float(v)
    except ValueError:
        raise ApiError(400, "bad_number", f"{key} must be a number")
    if not (f >= 0 and f < 1e9):
        raise ApiError(400, "bad_number", f"{key} out of range")
    return f


def launch_fingerprint(cfg: Config, model: str, spec: dict) -> str:
    """Everything that decides what a job of this deployment runs and where. Deployments keep the
    fingerprint they were created with; a config change must not silently alter live deployments."""
    m = cfg.models[model]
    prof = {pn: {k: v for k, v in vars(cfg.profiles[pn]).items() if k not in ("enabled", "policy", "max_gpus")}
            for pn in m.profiles}
    mv = {k: v for k, v in vars(m).items() if k not in ("max_replicas", "startup_timeout_s", "allowed_overrides")}
    cl = {n: {"deploy_dir": cfg.clusters[n].deploy_dir, "runs_dir": cfg.clusters[n].runs_dir}
          for n in {p["cluster"] for p in prof.values()}}
    return hashlib.sha256(dumps({"spec": spec, "model": mv, "profiles": prof, "clusters": cl}).encode()).hexdigest()[:16]


def check_config_against_state(cfg: Config, db: Store) -> list[str]:
    """Problems that make it unsafe to start with this config: live jobs or leases that reference
    clusters/profiles/models the config no longer has, or deployments whose launch settings changed."""
    problems = []
    live = db.q(f"SELECT * FROM candidates WHERE state IN ({','.join('?' * len(LIVE_CANDIDATE))})", *LIVE_CANDIDATE)
    for c in live:
        if c["cluster"] not in cfg.clusters:
            problems.append(f"candidate {c['id']} (job {c['job_id']}) is on cluster {c['cluster']!r}, which the config no longer has")
        if c["profile"] not in cfg.profiles:
            problems.append(f"candidate {c['id']} (job {c['job_id']}) uses profile {c['profile']!r}, which the config no longer has")
    busy = {c["deployment"] for c in live}
    busy |= {r["deployment"] for r in db.q("SELECT deployment FROM leases WHERE state='active'")}
    for d in db.q("SELECT * FROM deployments"):
        if d["id"] not in busy:
            continue
        if d["model"] not in cfg.models:
            problems.append(f"deployment {d['id']} ({d['model']}) has live jobs or leases but the model is not configured")
            continue
        try:
            fp = launch_fingerprint(cfg, d["model"], json.loads(d["spec"]))
        except KeyError as e:
            problems.append(f"deployment {d['id']} ({d['model']}): profile/cluster {e} is no longer configured")
            continue
        if fp != d["fingerprint"]:
            problems.append(f"deployment {d['id']} ({d['model']}) has live jobs or leases and its launch settings "
                            "changed (recipe, env, GPUs, profiles or clusters)")
    return problems


def reset_breaker(db: Store, target: str) -> bool:
    """Operator reset, by model name or by one of its deployment ids."""
    d = db.one("SELECT model FROM deployments WHERE id=?", target)
    model = d["model"] if d else target
    n = db.x("DELETE FROM breakers WHERE model=?", model)
    if n:
        db.x("UPDATE deployments SET startup_failures=0 WHERE model=?", model)
        db.event(None, "blocked", f"{model}: breaker reset by operator")
    return n > 0


def mux_dir(cfg: Config):
    """Directory for this manager's ssh ControlMaster sockets. Unix socket paths are limited to ~104-108
    bytes, so it is short and derived from the state dir (unique per manager, never another tool's mux)."""
    from pathlib import Path
    h = hashlib.sha256(str(cfg.state_dir.resolve()).encode()).hexdigest()[:10]
    return Path("~/.cache/sserved").expanduser() / h
