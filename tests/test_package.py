import torch

import torch_type_nn


def test_version():
    assert isinstance(torch_type_nn.__version__, str)


def test_torch_available():
    assert torch.ones(2).sum().item() == 2.0
