"""Tests for the E10-V100 / E12 runner code paths (CPU only): the PBS idle parser in b_e10.py and the E12 settings
and seed scheme in b_e12.py."""
import importlib.util
import os

import pytest

pytest.importorskip("numpy")
HERE = os.path.dirname(__file__)
SCRIPTS = os.path.abspath(os.path.join(HERE, "..", "bench", "scripts"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def b10():
    return _load("b_e10")


@pytest.fixture(scope="module")
def b12():
    return _load("b_e12")


def test_parse_gpu_apps(b10):
    gpus = "0, GPU-aaa\n1, GPU-bbb\n"
    apps = "111, GPU-aaa, 300\n222, GPU-bbb, 5000\n999, GPU-aaa, 10\n"
    out = b10.parse_gpu_apps(apps, gpus, {"222": "alice", "111": "cyyou68"}, "999")
    assert out == [("0", "111", "cyyou68", "300"), ("1", "222", "alice", "5000")]
    assert b10.parse_gpu_apps("No running processes found\n", gpus, {}, "1") == []


def test_is_idle_pbs_only_blocks_on_own_gpu(b10, monkeypatch):
    monkeypatch.setattr(b10, "IDLE_MODE", "pbs")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert b10.is_idle(dict(gpu_other="gpu1:222/alice/5000", other_py="none", runner=0))
    assert not b10.is_idle(dict(gpu_other="gpu1:222/alice/5000;gpu0:5/bob/9", other_py="none", runner=0))
    assert not b10.is_idle(dict(gpu_other="none", other_py="123", runner=0))
    monkeypatch.setattr(b10, "IDLE_MODE", "u94")                  # default: anything on any GPU blocks
    assert not b10.is_idle(dict(gpu_other="gpu1:222/alice/5000", other_py="none", runner=0))
    assert b10.is_idle(dict(gpu_other="none", other_py="none", runner=0))


def test_e12_settings_and_seeds(b12):
    assert b12.SETTINGS == ((37, 100), (74, 100), (296, 100), (74, 10), (74, 1000))
    g = b12.by_M(b12.SETTINGS)
    assert list(g) == [10, 100, 1000] and g[100] == [37, 74, 296] and g[10] == [74] and g[1000] == [74]
    assert sorted((p, m) for m, ps in g.items() for p in ps) == sorted(b12.SETTINGS)
    k = ("ad", "brickwork", "s7", 20, 8, 0.01)
    # pilot seed shared by all settings (nested pilots); production seeds distinct per setting and from E10's
    assert b12.pilot_seed("e12", k, "zz") == b12.E.seed_of("e12-pilot", *k, "zz")
    prods = {b12.prod_seed("e12", p, m, k, "zz") for p, m in b12.SETTINGS}
    assert len(prods) == len(b12.SETTINGS)
    assert b12.E.seed_of("e10-static", "pilot", *k, "zz") not in prods
    assert b12.label(296, 100) == "p296_M100"
