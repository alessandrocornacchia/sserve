"""sserved configuration: one TOML file, validated at startup.

Operator decisions live here: clusters, approved SLURM profiles (PI profiles are
opt-in), the model catalogue with launch settings, lease and scaling limits, and
where the server listens / how clients reach it. Clients only pick a model and
optional overrides that the catalogue explicitly allows.
"""
from __future__ import annotations

import ipaddress
import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse


class ConfigError(Exception):
    pass


@dataclass
class Server:
    listen: str = "127.0.0.1:8470"
    advertised_url: str = ""            # base clients use, without /v1; default http://<listen>
    root_path: str = ""                 # path prefix added by a reverse proxy (e.g. /sserve)
    trusted_network: bool = False       # allow plaintext on a non-loopback address
    tls_cert: str = ""
    tls_key: str = ""
    state_dir: str = "~/.local/state/sserved"
    reconcile_interval_s: float = 15.0

    @property
    def host(self) -> str:
        return self.listen.rsplit(":", 1)[0].strip("[]")

    @property
    def port(self) -> int:
        return int(self.listen.rsplit(":", 1)[1])

    @property
    def base_url(self) -> str:
        if self.advertised_url:
            return self.advertised_url.rstrip("/")
        scheme = "https" if self.tls_cert else "http"
        return f"{scheme}://{self.listen}{self.root_path.rstrip('/')}"


@dataclass
class Leases:
    default_ttl_s: float = 300
    min_ttl_s: float = 30
    max_ttl_s: float = 3600
    prep_grace_s: float = 1800          # from first availability: no traffic needed yet
    acquire_timeout_s: float = 6 * 3600  # a lease whose model never became ready ends ("unfulfilled")
    max_acquire_timeout_s: float = 24 * 3600
    idle_timeout_s: float = 1800        # no inference activity (renewals don't count)
    max_idle_timeout_s: float = 6 * 3600
    linger_s: float = 600               # keep ready replicas after the last lease ends
    max_leases_per_client: int = 16


@dataclass
class Cluster:
    name: str
    host: str = ""                      # ssh destination of the login node
    runs_dir: str = "~/.vllm/runs"
    deploy_dir: str = "~/.vllm/deploy"
    prelude: str = ""
    runner: str = "ssh"                 # "ssh", or "local" (tests: run scripts on this machine)
    tunnel_cmd: list[str] = field(default_factory=list)   # override; placeholders {lport} {node} {port} {host}
    ssh_opts: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)      # extra env for "local" runner (tests)


@dataclass
class Profile:
    name: str
    cluster: str
    sbatch_args: list[str] = field(default_factory=list)   # partition/qos/account, operator-approved
    gres: str = "gpu:{gpus}"
    time: str = "3:00:00"
    requeue: bool = True
    enabled: bool = True                # default when first seen; operator changes persist in state
    opt_in: bool = False                # PI profiles: never enabled unless the operator enables them
    max_gpus: int = 0                   # manager-side cap on GPUs held via this profile (0 = none)
    policy: str = ""                    # site utilisation policy, informational (e.g. "75% 1h avg, 2 h")


@dataclass
class Model:
    name: str                           # HF id clients ask for
    recipe: str                         # <deploy_dir>/<recipe>/serve.sh on the cluster
    profiles: list[str]
    gpus: int = 1
    served_model_name: str = ""
    max_model_len: int = 0
    cpus_per_task: int = 8
    mem: str = ""
    env: dict[str, str] = field(default_factory=dict)
    allowed_overrides: list[str] = field(default_factory=list)   # subset of OVERRIDES
    startup_timeout_s: float = 2400     # RUNNING -> healthy: staging, load, KV cache, graph capture
    readiness_inference: bool = False   # one tiny tool-call request before admission (not demand)
    max_replicas: int = 2
    clusters: list[str] = field(default_factory=list)

    @property
    def served(self) -> str:
        return self.served_model_name or self.name


OVERRIDES = {"gpus": int, "max_model_len": int}


@dataclass
class Scaling:
    window_s: float = 300
    up_waiting_per_replica: float = 2.0     # mean vllm:num_requests_waiting per replica, sustained
    up_kv_usage: float = 0.90               # with waiting > 0
    up_capacity_errors: int = 5             # 429/503/connection errors in the window
    down_load_per_replica: float = 1.0      # max(running+waiting) must fit in n-1 replicas
    cooldown_s: float = 600
    max_startup_gpus: int = 8               # GPUs of *duplicate* candidates (2nd+ of a slot) RUNNING, not yet healthy
    submit_resolve_age_s: float = 600       # a crash-interrupted submission is "not submitted" only after this
    max_total_gpus: int = 16                # GPUs held (queued+running), all deployments
    max_candidates_per_slot: int = 3
    max_submissions_per_day: int = 12       # per deployment
    submit_backoff_s: float = 120
    max_startup_failures: int = 3           # consecutive, per deployment, before the breaker opens
    replace_before_s: float = 1800          # start a replacement when walltime left drops below
    replace_requires_activity_s: float = 1800
    max_replacements_per_day: int = 8
    drain_deadline_s: float = 900
    health_fail_unroute: int = 2            # consecutive failed probes before unrouting a replica
    health_fail_replace_s: float = 600      # a RUNNING job unhealthy this long is replaced
    unknown_terminal_grace_s: float = 300
    orphan_sweep_s: float = 600             # also query clusters without live candidates this often   # job missing from squeue and sacct: wait before "lost"


@dataclass
class Config:
    server: Server
    leases: Leases
    scaling: Scaling
    clusters: dict[str, Cluster]
    profiles: dict[str, Profile]
    models: dict[str, Model]
    path: Path | None = None

    @property
    def state_dir(self) -> Path:
        return Path(os.path.expanduser(self.server.state_dir))


def _take(cls, d: dict, **extra):
    known = set(cls.__dataclass_fields__)
    bad = set(d) - known
    if bad:
        raise ConfigError(f"{cls.__name__}: unknown keys {sorted(bad)}")
    return cls(**{**d, **extra})


def parse(raw: dict, path: Path | None = None) -> Config:
    bad = set(raw) - {"server", "leases", "scaling", "clusters", "profiles", "models"}
    if bad:
        raise ConfigError(f"unknown sections {sorted(bad)}")
    clusters = {n: _take(Cluster, d, name=n) for n, d in raw.get("clusters", {}).items()}
    profiles = {n: _take(Profile, d, name=n) for n, d in raw.get("profiles", {}).items()}
    models = {n: _take(Model, d, name=n) for n, d in raw.get("models", {}).items()}
    cfg = Config(_take(Server, raw.get("server", {})), _take(Leases, raw.get("leases", {})),
                 _take(Scaling, raw.get("scaling", {})), clusters, profiles, models, path)
    for p in profiles.values():
        if p.cluster not in clusters:
            raise ConfigError(f"profile {p.name}: unknown cluster {p.cluster}")
        if p.opt_in and p.enabled:
            p.enabled = False           # opt-in profiles start disabled; only the operator enables them
    for m in models.values():
        for pn in m.profiles:
            if pn not in profiles:
                raise ConfigError(f"model {m.name}: unknown profile {pn}")
        for o in m.allowed_overrides:
            if o not in OVERRIDES:
                raise ConfigError(f"model {m.name}: override {o} not supported (allowed: {sorted(OVERRIDES)})")
        if m.gpus < 1:
            raise ConfigError(f"model {m.name}: gpus must be >= 1")
    if cfg.server.root_path and not cfg.server.root_path.startswith("/"):
        raise ConfigError("server.root_path must start with /")
    if bool(cfg.server.tls_cert) != bool(cfg.server.tls_key):
        raise ConfigError("server.tls_cert and server.tls_key go together")
    return cfg


def load(path: str | os.PathLike) -> Config:
    p = Path(path).expanduser()
    with p.open("rb") as f:
        return parse(tomllib.load(f), p)


def exposure_check(s: Server) -> list[str]:
    """Refuse a plaintext listener reachable from untrusted networks; return warnings.

    Loopback is always fine. With TLS any address is allowed. Plaintext on a private
    address warns; plaintext on a public or wildcard address is refused unless the
    operator declares the network trusted (e.g. a private overlay or a TLS proxy
    in front on the same host)."""
    warnings: list[str] = []
    host = s.host
    try:
        ip = ipaddress.ip_address(host) if host not in ("", "localhost") else ipaddress.ip_address("127.0.0.1")
    except ValueError:
        raise ConfigError(f"server.listen must be an IP address, not {host!r}")
    wildcard = ip.is_unspecified
    if not s.tls_cert and not ip.is_loopback:
        if (wildcard or not ip.is_private) and not s.trusted_network:
            raise ConfigError(f"refusing plaintext listener on {'wildcard' if wildcard else 'public'} address {host}: "
                              "set tls_cert/tls_key, bind a private or loopback address, "
                              "or set trusted_network = true if this network is private")
        warnings.append(f"plaintext listener on non-loopback {host}: inference and lease keys travel unencrypted")
    if not ip.is_loopback:
        adv = urlparse(s.base_url).hostname or ""
        if adv in ("127.0.0.1", "localhost", "::1") or adv.startswith("127."):
            warnings.append("advertised_url points at localhost: remote clients and containers cannot use it")
    return warnings
