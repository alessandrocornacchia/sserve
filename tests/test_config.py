"""Config validation and listener exposure rules (no processes)."""
from __future__ import annotations

from pathlib import Path

import pytest

from sserved import config as C
from sserved.manager import parse_prom
from sserved.slurm import parse_time_left, sbatch_script

ROOT = Path(__file__).resolve().parent.parent


def test_example_config_parses_and_pi_is_opt_in():
    cfg = C.load(ROOT / "examples" / "sserved.example.toml")
    assert cfg.profiles["orix-pi-caninim"].opt_in and not cfg.profiles["orix-pi-caninim"].enabled
    assert cfg.models["Qwen/Qwen3-0.6B"].recipe == "qwen3-0.6b"
    assert (ROOT / "recipes" / "orix" / "qwen3-0.6b" / "serve.sh").exists()
    assert C.exposure_check(cfg.server) == []


def S(**kw):
    return C.Server(**kw)


@pytest.mark.parametrize("listen", ["0.0.0.0:8470", "[::]:8470", "8.8.8.8:8470"])
def test_plaintext_public_or_wildcard_refused(listen):
    with pytest.raises(C.ConfigError):
        C.exposure_check(S(listen=listen))


def test_trusted_network_and_tls_allowed():
    w = C.exposure_check(S(listen="0.0.0.0:8470", trusted_network=True, advertised_url="http://10.0.0.5:8470"))
    assert any("plaintext" in x for x in w)
    assert not [x for x in C.exposure_check(S(listen="0.0.0.0:8470", tls_cert="c", tls_key="k",
                                                advertised_url="https://h.example/sserve")) if "plaintext" in x]


def test_private_plaintext_warns_and_localhost_advertised_warns():
    w = C.exposure_check(S(listen="172.30.0.1:8470"))
    assert any("plaintext" in x for x in w) and any("advertised_url" in x for x in w) is False
    w = C.exposure_check(S(listen="10.1.2.3:8470", advertised_url="http://127.0.0.1:8470"))
    assert any("localhost" in x for x in w)


def test_base_url_root_path():
    assert S(listen="127.0.0.1:1", root_path="/sserve").base_url == "http://127.0.0.1:1/sserve"
    assert S(advertised_url="https://x.org/sserve/").base_url == "https://x.org/sserve"


def test_unknown_keys_and_bad_refs_rejected():
    with pytest.raises(C.ConfigError):
        C.parse({"server": {"lisen": "x"}})
    with pytest.raises(C.ConfigError):
        C.parse({"clusters": {"a": {}}, "profiles": {"p": {"cluster": "b"}}})
    with pytest.raises(C.ConfigError):
        C.parse({"clusters": {"a": {}}, "profiles": {"p": {"cluster": "a"}},
                 "models": {"m": {"recipe": "r", "profiles": ["p"], "allowed_overrides": ["api_base"]}}})


def test_parsers():
    assert parse_time_left("1-02:03:04") == 93784 and parse_time_left("5:00") == 300
    assert parse_time_left("UNLIMITED") is None
    v = parse_prom('# HELP x\nvllm:num_requests_waiting{model_name="a"} 2\nvllm:num_requests_waiting{model_name="b"} 3\n'
                   'vllm:kv_cache_usage_perc{m="a"} 0.5\n')
    assert v["vllm:num_requests_waiting"] == 5 and v["vllm:kv_cache_usage_perc"] == 0.5


def test_sbatch_script_contract():
    cfg = C.load(ROOT / "examples" / "sserved.example.toml")
    t = sbatch_script(profile=cfg.profiles["orix-freecycle-h200"], model=cfg.models["ornith-ai/Ornith-1.5-122B"],
                      gpus=2, recipe_dir="~/.vllm/deploy/ornith-1.5-122b", port=23456, max_model_len=262144,
                      served="ornith-ai/Ornith-1.5-122B")
    for line in ("#SBATCH --gres=gpu:nvidia_h200:2", "#SBATCH --requeue", "#SBATCH --time=3:00:00",
                 "#SBATCH --partition=freecycle-h200", "export TENSOR_PARALLEL_SIZE=2", "export VLLM_PORT=23456",
                 'export SSERVE_RECIPE_DIR="$HOME"/.vllm/deploy/ornith-1.5-122b'):
        assert line in t, line
    assert "trust-remote-code" not in t and "quant" not in t


def test_mux_socket_path_fits_unix_limit(tmp_path):
    from sserved.manager import mux_dir
    from sserved.slurm import Runner
    deep = tmp_path / ("x" * 60) / "very" / "deep" / "state"
    cfg = C.parse({"server": {"state_dir": str(deep)}, "clusters": {"c": {"host": "user@login.example.org"}}})
    d = mux_dir(cfg)
    sock = d / ("m-" + "f" * 40)                       # ssh %C expands to 40 hex chars
    assert len(str(sock).encode()) < 104
    assert str(d).startswith(str(Path("~/.cache/sserved").expanduser()))
    other = C.parse({"server": {"state_dir": str(tmp_path / "other")}, "clusters": {"c": {"host": "h"}}})
    assert mux_dir(other) != d                         # unique per manager state dir
    opts = Runner(cfg.clusters["c"], d)._mux()
    assert f"ControlPath={d}/m-%C" in opts
