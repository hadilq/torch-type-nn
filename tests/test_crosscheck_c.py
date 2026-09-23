"""Number-for-number comparison with the reference C implementation.

Runs when ``TYPE_NN_SRC`` points at a checkout of
https://github.com/hadilq/type-nn and a C compiler is available
(``nix flake check`` provides both).
"""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import torch
from helpers import D

from torch_type_nn import AndOr, TypeAdam, mse_loss

SRC = os.environ.get("TYPE_NN_SRC")
CC = shutil.which(os.environ.get("CC", "cc")) or shutil.which("gcc")
pytestmark = pytest.mark.skipif(
    not (SRC and Path(SRC, "type_nn.c").exists() and CC),
    reason="set TYPE_NN_SRC to a type-nn checkout (and have a C compiler)")


@pytest.fixture(scope="module")
def harness(tmp_path_factory):
    exe = tmp_path_factory.mktemp("c") / "crosscheck"
    here = Path(__file__).parent / "c" / "crosscheck.c"
    subprocess.run([CC, "-O2", "-std=c11", f"-I{SRC}", str(here), f"{SRC}/type_nn.c",
                    f"{SRC}/type_nn_scale.c", "-lm", "-o", str(exe)], check=True)
    return exe


def run(harness, seed):
    out = subprocess.run([str(harness), str(seed)], check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def build(spec):
    layers = []
    for L in spec:
        R = max(len(u) for u in L["units"])
        l = AndOr(L["n_in"], L["n_out"], R, dtype=D)
        with torch.no_grad():
            l.mask.fill_(False)
            l.project_()
            for k, unit in enumerate(L["units"]):
                for r, o in enumerate(unit):
                    l.mask[k, r] = True
                    l.weight[k, r] = torch.tensor(o["w"], dtype=D)
                    l.bias[k, r] = o["b"]
                    l.assembly[k, r] = o["a"]
        layers.append(l)
    return torch.nn.Sequential(*layers)


def close(a, b, tol=1e-12):
    a, b = torch.as_tensor(a, dtype=D), torch.as_tensor(b, dtype=D)
    return float(((a - b).abs() / b.abs().clamp(min=1.0)).max()) < tol


@pytest.mark.parametrize("seed", [12345, 7, 2024])
def test_forward_backward_and_adam_match_c(harness, seed):
    ref = run(harness, seed)
    net = build(ref["before"])
    x = torch.tensor([ref["x"]], dtype=D, requires_grad=True)
    t = torch.tensor([ref["t"]], dtype=D)

    inputs = []
    hooks = [l.register_forward_pre_hook(lambda m, a: inputs.append(a[0].detach()))
             for l in net]
    y = net(x)
    for h in hooks:
        h.remove()
    assert close(y[0].detach(), ref["y"]), "forward"
    opt = TypeAdam(net, lr=0.0)
    opt.zero_grad()
    x.grad = None
    mse_loss(y, t).backward()
    assert close(x.grad[0], ref["dx"]), "dL/dx through the stack"
    for l, L, xin in zip(net, ref["grads"], inputs, strict=True):
        for k, unit in enumerate(L["units"]):
            for r, o in enumerate(unit):
                assert close(l.bias.grad[k, r], o["gb"]), "dL/db"
                assert close(l.assembly.grad[k, r], o["ga"]), "dL/da"
                assert close(l.weight.grad[k, r], o["gb"] * xin[0]), "dL/dw"
    opt.step()                         # lr 0: moments and step counts only

    opt.param_groups[0]["lr"] = 0.01
    opt.zero_grad()
    mse_loss(net(x.detach()), t).backward()
    opt.step()
    for l, L in zip(net, ref["after"], strict=True):
        for k, unit in enumerate(L["units"]):
            for r, o in enumerate(unit):
                assert close(l.weight[k, r].detach(), o["w"]), "w after two Adam steps"
                assert close(l.bias[k, r].detach(), o["b"]), "b after two Adam steps"
                assert close(l.assembly[k, r].detach(), o["a"]), "a after two Adam steps"
