from pathlib import Path

import torch

import torch_type_nn

ROOT = Path(__file__).resolve().parents[1]


def test_version():
    assert isinstance(torch_type_nn.__version__, str)


def test_torch_available():
    assert torch.ones(2).sum().item() == 2.0


def test_wheel_config_does_not_force_include_native_c():
    """force-include of csrc → native/c plus package-data native/c packs ORIGIN twice."""
    import tomllib
    cfg = tomllib.loads((ROOT / "pyproject.toml").read_text())
    hatch = cfg.get("tool", {}).get("hatch", {}).get("build", {})
    wheel = hatch.get("targets", {}).get("wheel", {})
    assert "force-include" not in wheel
    assert (ROOT / "src" / "torch_type_nn" / "native" / "c" / "tnn_py.c").is_file()
    assert (ROOT / "src" / "torch_type_nn" / "native" / "c" / "ORIGIN").is_file()
    assert not (ROOT / "csrc").exists()
