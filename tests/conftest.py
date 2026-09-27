"""Device selection for the test-suite.

Every test that takes the ``device`` fixture runs on the CPU and, when a CUDA
device is visible, again on ``cuda``. ``TNN_REQUIRE_CUDA=1`` (set by the GPU
check of ``flake.nix``) turns a missing CUDA device into an error instead of a
skip, so a GPU run can never pass by silently testing only the CPU.
"""

import os

import pytest
import torch

REQUIRE_CUDA = os.environ.get("TNN_REQUIRE_CUDA") == "1"


def _device_banner() -> str:
    built = torch.version.cuda or "cpu-only"
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cap = torch.cuda.get_device_capability(0)
        return (f"torch-type-nn device: CUDA {name} "
                f"cap {cap[0]}.{cap[1]} (torch {torch.__version__}, built {built})")
    return (f"torch-type-nn device: CPU "
            f"(torch {torch.__version__}, built {built}; no CUDA device)")


def pytest_report_header(config):
    return [_device_banner()]


def pytest_sessionstart(session):
    # -q hides the header; print once so `nix flake check -L` always shows it
    print(_device_banner(), flush=True)


def pytest_configure(config):
    if REQUIRE_CUDA and not torch.cuda.is_available():
        raise pytest.UsageError(
            "TNN_REQUIRE_CUDA=1 but torch.cuda.is_available() is False "
            f"(torch {torch.__version__}, built with CUDA {torch.version.cuda})")


def cuda_or_skip():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        cuda_or_skip()
    return torch.device(request.param)
