"""The pinned datasets: manifest, loaders, and (when present) the files.

The files are not in git. ``nix flake check`` points ``TORCH_TYPE_NN_DATA`` at
the pinned store copies and sets ``TNN_REQUIRE_DATA=1``, which turns a missing
file into a failure; elsewhere those tests skip until ``nix develop`` or
``python benchmarks/fetch_data.py`` has fetched the data.
"""

import gzip
import math
import os
import sys
from pathlib import Path

import pytest
import torch

BENCH = Path(__file__).parents[1] / "benchmarks"
sys.path.insert(0, str(BENCH))

import bench  # noqa: E402
import fetch_data  # noqa: E402
import scale  # noqa: E402

REQUIRE = os.environ.get("TNN_REQUIRE_DATA") == "1"
SHAPES = {"iris": (150, 4, 3), "wine": (178, 13, 3), "wdbc": (569, 30, 1),
          "diabetes": (442, 10, 1), "ionosphere": (351, 34, 1)}


def need(name):
    path = bench.DATA / name
    if not path.exists():
        if REQUIRE:
            pytest.fail(f"TNN_REQUIRE_DATA=1 but {path} is missing")
        pytest.skip(f"{name} not fetched (nix develop, or benchmarks/fetch_data.py)")
    return path


def test_manifest_covers_every_task():
    entries = fetch_data.manifest()
    files = {e["file"] for e in entries}
    assert {spec[0] for t, spec in bench.TASKS.items() if spec[0]} <= files
    assert set(scale.FILES.values()) <= files
    for e in entries:
        assert e["url"].startswith("https://") and e["hash"].startswith("sha256-")
        assert len(e["hash"]) == len("sha256-") + 44


def test_data_files_are_ignored_by_git():
    ignore = (Path(__file__).parents[1] / ".gitignore").read_text().split()
    assert "benchmarks/data/*" in ignore and "!benchmarks/data/README.md" in ignore


def test_sri_matches_nix():
    assert fetch_data.sri(b"") == "sha256-47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU="


@pytest.mark.parametrize("header", [True, False])
def test_friedman_loader(tmp_path, header):
    g = torch.Generator().manual_seed(0)
    X = torch.rand(20, 10, dtype=torch.float64, generator=g)
    y = torch.randn(20, dtype=torch.float64, generator=g)
    lines = (["\t".join([f"x{i}" for i in range(1, 11)] + ["target"])] if header else [])
    lines += ["\t".join(f"{v:.17g}" for v in [*X[r].tolist(), y[r].item()]) for r in range(20)]
    path = tmp_path / "f.tsv.gz"
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    Xl, yl = scale.load_friedman(path)
    assert torch.equal(Xl, X) and torch.equal(yl[:, 0], y)


@pytest.mark.parametrize("task", list(SHAPES))
def test_board_datasets_load(task):
    need(bench.TASKS[task][0])
    X, Y, _ = bench.load(task)
    n, n_in, n_out = SHAPES[task]
    assert (len(X), len(X[0]), len(Y[0])) == (n, n_in, n_out)


def test_pinned_files_have_their_hashes():
    for e in fetch_data.manifest():
        path = need(e["file"])
        assert fetch_data.sri(path.read_bytes()) == e["hash"], e["file"]


def test_digits_file():
    need("digits.csv.gz")
    X, Y, classify = scale.load("digits")
    assert classify and X.shape == (1797, 64) and Y.shape == (1797, 10)
    assert float(X.min()) == 0.0 and float(X.max()) == 16.0
    assert torch.equal(Y.sum(1), torch.ones(1797, dtype=Y.dtype))


def test_friedman_file_is_friedman_one():
    """The file must be Friedman #1: inputs uniform on [0, 1], and
    y - f(x) a unit-variance zero-mean noise, f the Friedman #1 function."""
    need("564_fried.tsv.gz")
    X, Y, classify = scale.load("friedman")
    assert not classify and X.shape == (40768, 10) and Y.shape == (40768, 1)
    assert float(X.min()) >= 0.0 and float(X.max()) <= 1.0
    f = (10 * torch.sin(math.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4])
    e = Y[:, 0] - f
    assert abs(float(e.mean())) < 0.05 and abs(float(e.var()) - 1.0) < 0.05
    *_, floor = scale.split("friedman")
    assert 0 < floor < 0.01
