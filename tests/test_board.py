"""board.py writes every table of BOARD.md; hold it to its arithmetic and markers."""

import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import board  # noqa: E402


def row(impl, task, m, sd, seeds=5, params=100.0, **kw):
    return {"impl": impl, "task": task, "hold_mse": m, "hold_mse_sd": sd, "seeds": seeds,
            "params": params, "params_sd": 0.0, "hold_acc": None, "mse": m, "acc": None,
            "init_layers": 2, "layers": 2.0, "train_s": 1.0, "us_per_infer": 1.0, **kw}


def test_verdict_is_two_standard_errors_of_the_difference():
    a, b = row("x", "t", 0.10, 0.02), row("y", "t", 0.13, 0.01)
    d, se2, clear = board.verdict(a, b)
    assert d == pytest.approx(-0.03)
    assert se2 == pytest.approx(2 * math.sqrt(0.02 ** 2 / 5 + 0.01 ** 2 / 5))
    assert clear is (abs(d) > se2)
    assert board.verdict(row("x", "t", None, 0), b) is None


def test_write_fills_markers_is_idempotent_and_says_when_data_is_missing(tmp_path):
    ref = tmp_path / "ref.jsonl"
    res = tmp_path / "res.jsonl"
    ref.write_text(json.dumps(row("type-nn", "iris", 0.02, 0.005)) + "\n"
                   + json.dumps(row("c-mlp", "iris", 0.03, 0.005, params=67.0)) + "\n")
    res.write_text(json.dumps(row("torch-ref-type-nn", "iris", 0.021, 0.004)) + "\n")
    md = tmp_path / "B.md"
    md.write_text("intro\n" + "".join(f"<!-- board:{n} -->\nOLD\n<!-- /board:{n} -->\nprose\n"
                                      for n in ("per-sample", "port", "like-for-like",
                                                "c-board", "scale")))
    secs = board.sections([str(res)], [str(tmp_path / "none.jsonl")], str(ref))
    board.write(md, secs)
    text = md.read_text()
    assert "OLD" not in text and text.count("prose") == 5
    assert "torch ref-type-nn vs C type-nn" in text          # port pair rendered
    assert "C type-nn vs C c-mlp" in text                    # C board pair rendered
    assert text.count(board.MISSING) == 2                    # like-for-like and scale
    board.write(md, secs)
    assert md.read_text() == text


def test_write_refuses_a_board_without_the_markers(tmp_path):
    md = tmp_path / "B.md"
    md.write_text("no markers\n")
    with pytest.raises(SystemExit):
        board.write(md, {"port": ["x"]})


def test_a_missing_c_reference_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="C reference"):
        board.sections([], [], str(tmp_path / "missing.jsonl"))
