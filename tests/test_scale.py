"""scale.py accepts C / CUDA models on the large-data tasks."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "benchmarks"))
import bench  # noqa: E402
import scale  # noqa: E402


def test_scale_families_and_default_models():
    assert scale._family("c-type-nn") == "c"
    assert scale._family("c-type-nn-overfit") == "c"
    assert scale._family("cuda-type-nn") == "cuda"
    assert scale._family("type-nn") == "type-nn"
    assert scale._family("mlp-16") == "fixed"
    text = Path(scale.__file__).read_text()
    assert "c-type-nn,c-type-nn-overfit" in text
    assert bench.MODELS["c-type-nn"][0] == "c"


def test_board_cuda_markers_and_write(tmp_path):
    import board
    ref = tmp_path / "ref.jsonl"
    res = tmp_path / "res.jsonl"
    ref.write_text(json.dumps({
        "impl": "type-nn", "task": "iris", "hold_mse": 0.02, "hold_mse_sd": 0.005,
        "seeds": 5, "params": 100.0, "params_sd": 0.0, "hold_acc": None,
        "mse": 0.02, "acc": None, "init_layers": 3, "layers": 3.0,
        "train_s": 1.0, "us_per_infer": 1.0,
    }) + "\n")
    res.write_text(json.dumps({
        "impl": "cuda-type-nn", "task": "iris", "hold_mse": 0.03,
        "hold_mse_sd": 0.01, "seeds": 5, "params": 80.0, "params_sd": 0.0,
        "hold_acc": None, "mse": 0.03, "acc": None, "init_layers": 3,
        "layers": 3.0, "train_s": 1.0, "us_per_infer": 1.0,
        "backend": "cudatypenn", "device": "cuda",
    }) + "\n")
    md = tmp_path / "BOARD.CUDA.md"
    md.write_text("".join(f"<!-- board:{n} -->\nOLD\n<!-- /board:{n} -->\n"
                          for n in ("per-sample", "pairs", "scale")))
    secs = board.sections_cuda([str(res)], [str(tmp_path / "none.jsonl")], str(ref))
    board.write(md, secs)
    text = md.read_text()
    assert "OLD" not in text
    assert "cuda type-nn" in text
    assert board.MISSING in text
    root = Path(__file__).resolve().parents[1] / "BOARD.CUDA.md"
    body = root.read_text()
    for name in ("per-sample", "pairs", "scale"):
        assert f"<!-- board:{name} -->" in body
        assert f"<!-- /board:{name} -->" in body


def test_cuda_board_does_not_merge_30seed_reference(tmp_path):
    """C cells on BOARD.CUDA.md must not come from the 30-seed binary file."""
    import board
    ref = tmp_path / "ref.jsonl"
    res = tmp_path / "res.jsonl"
    ref.write_text(json.dumps({
        "impl": "type-nn", "task": "iris", "hold_mse": 0.999, "hold_mse_sd": 0.0,
        "seeds": 30, "params": 1.0, "params_sd": 0.0, "hold_acc": None,
        "mse": 0.999, "acc": None, "init_layers": 3, "layers": 3.0,
        "train_s": 1.0, "us_per_infer": 1.0,
    }) + "\n")
    res.write_text(json.dumps({
        "impl": "cuda-type-nn", "task": "iris", "hold_mse": 0.03,
        "hold_mse_sd": 0.01, "seeds": 5, "params": 80.0, "params_sd": 0.0,
        "hold_acc": None, "mse": 0.03, "acc": None, "init_layers": 3,
        "layers": 3.0, "train_s": 1.0, "us_per_infer": 1.0,
        "backend": "cudatypenn", "device": "cuda",
    }) + "\n")
    secs = board.sections_cuda([str(res)], [], str(ref))
    text = "\n".join(secs["per-sample"])
    assert "0.999" not in text
    assert "cuda type-nn" in text
