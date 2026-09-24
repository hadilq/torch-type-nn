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
