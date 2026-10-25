"""sserved: the manager daemon and its operator commands.

    sserved -c CONFIG serve                     run the controller + HTTP API (one per state dir)
    sserved -c CONFIG token create NAME [--admin]   print a new client token (shown once)
    sserved -c CONFIG token revoke NAME | token list
    sserved -c CONFIG profile list | enable NAME | disable NAME
    sserved -c CONFIG reset MODEL|DEPLOYMENT    clear a blocked model (model-wide; operator only)
    sserved -c CONFIG status                    deployments, slots and jobs from the state file
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time

from . import config as C
from .store import AlreadyRunning, Store


def _store(cfg: C.Config) -> Store:
    return Store(cfg.state_dir, lock=False)


async def _serve(cfg: C.Config) -> int:
    import uvicorn

    from .api import build_app
    from .manager import Manager

    try:
        store = Store(cfg.state_dir, lock=True)
    except AlreadyRunning as e:
        print(f"sserved: {e}", file=sys.stderr)
        return 3
    from .manager import check_config_against_state
    problems = check_config_against_state(cfg, store)
    if problems:
        for p in problems:
            print(f"sserved: refusing to start: {p}", file=sys.stderr)
        print("sserved: restore the previous settings until those jobs/leases end (see `sserved status`)",
              file=sys.stderr)
        store.close()
        return 4
    mgr = Manager(cfg, store)
    app = build_app(mgr)
    s = cfg.server
    ucfg = uvicorn.Config(app, host=s.host, port=s.port, root_path=s.root_path, log_level="info",
                          ssl_certfile=s.tls_cert or None, ssl_keyfile=s.tls_key or None,
                          timeout_graceful_shutdown=30, proxy_headers=False)
    server = uvicorn.Server(ucfg)
    # uvicorn restores the previous handlers after a graceful stop and re-raises the signal; with the
    # default SIGTERM action that kills us before the shutdown below (tunnels, ssh mux, state) runs.
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: None)
    stop = asyncio.Event()
    loop_task = asyncio.create_task(mgr.run_forever(stop))
    store.event(None, "start", f"sserved pid {os.getpid()} listening on {s.listen}, advertised {s.base_url}")
    try:
        await server.serve()
    finally:
        stop.set()
        await loop_task
        await mgr.shutdown()
        store.event(None, "stop", "sserved stopped (jobs left running; adopted on next start)")
        store.close()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sserved", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("-c", "--config", default=os.environ.get("SSERVED_CONFIG", "~/.config/sserve/sserved.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("serve")
    tk = sub.add_parser("token")
    tk.add_argument("action", choices=["create", "revoke", "list"])
    tk.add_argument("name", nargs="?")
    tk.add_argument("--admin", action="store_true")
    pr = sub.add_parser("profile")
    pr.add_argument("action", choices=["list", "enable", "disable"])
    pr.add_argument("name", nargs="?")
    rs = sub.add_parser("reset", help="clear a blocked model (by model name or deployment id)")
    rs.add_argument("target", metavar="MODEL|DEPLOYMENT")
    sub.add_parser("status")
    sub.add_parser("check", help="validate the config and the listener exposure")
    a = ap.parse_args(argv)

    try:
        cfg = C.load(a.config)
        warnings = C.exposure_check(cfg.server)
    except (C.ConfigError, FileNotFoundError) as e:
        print(f"sserved: config: {e}", file=sys.stderr)
        return 2
    for w in warnings:
        print(f"sserved: WARNING: {w}", file=sys.stderr)

    if a.cmd == "serve":
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
        logging.getLogger("LiteLLM").setLevel(logging.WARNING)
        logging.getLogger("LiteLLM Router").setLevel(logging.WARNING)
        return asyncio.run(_serve(cfg))
    if a.cmd == "check":
        print(json.dumps({"ok": True, "advertised": cfg.server.base_url, "warnings": warnings}))
        return 0

    st = _store(cfg)
    if a.cmd == "token":
        if a.action == "list":
            for r in st.q("SELECT name, admin, created, revoked FROM tokens ORDER BY created"):
                print(f"{r['name']}\t{'admin' if r['admin'] else 'client'}\t"
                      f"{'revoked' if r['revoked'] else 'active'}\t{time.strftime('%F %T', time.localtime(r['created']))}")
            return 0
        if not a.name:
            ap.error("token create/revoke needs NAME")
        if a.action == "create":
            try:
                print(st.create_token(a.name, a.admin))
            except ValueError as e:
                print(f"sserved: {e}", file=sys.stderr)
                return 1
            return 0
        return 0 if st.revoke_token(a.name) else 1
    if a.cmd == "profile":
        if a.action == "list":
            for n, p in cfg.profiles.items():
                print(f"{n}\t{'enabled' if st.profile_enabled(n, p.enabled) else 'disabled'}\t"
                      f"cluster={p.cluster}{' opt-in' if p.opt_in else ''}\t{p.policy}")
            return 0
        if a.name not in cfg.profiles:
            print(f"sserved: unknown profile {a.name!r}", file=sys.stderr)
            return 1
        st.set_profile(a.name, a.action == "enable", "operator")
        st.event(None, "profile", f"{a.name} {a.action}d by operator")
        return 0
    if a.cmd == "reset":
        from .manager import reset_breaker
        n = reset_breaker(st, a.target)
        print("reset" if n else "not blocked")
        return 0 if n else 1
    if a.cmd == "status":
        for b in st.q("SELECT * FROM breakers"):
            print(f"BLOCKED {b['model']}: {b['reason']}  (reset: sserved reset {json.dumps(b['model'])})")
        for d in st.q("SELECT * FROM deployments ORDER BY created"):
            leases = st.one("SELECT COUNT(*) n FROM leases WHERE deployment=? AND state='active'", d["id"])["n"]
            print(f"{d['id']}  {d['model']}  state={d['state']} leases={leases} target={d['target']}")
            for c in st.q("SELECT * FROM candidates WHERE deployment=? AND state!='ended' ORDER BY submitted", d["id"]):
                print(f"    {c['id']} slot={c['slot']} {c['profile']} job={c['job_id']} {c['state']} "
                      f"slurm={c['slurm_state']} reason={c['reason']} node={c['node']}")
        return 0
    return 1


if __name__ == "__main__":
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    sys.exit(main())
