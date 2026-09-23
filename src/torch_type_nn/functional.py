"""Functional core of type-nn.

For a batch x of shape (B, n) and a layer with m units of up to R Ors each:

    O[b,k,r] = w[k,r] . x[b] + b[k,r]                 Or, a sum type
    A[b,k]   = prod_r sign(O) |O| ^ a[k,r]             And, a product type
    z[b,k]   = sign(A) ln(1 + |A|)                     partition function

The forward runs in log space so a deep product never overflows:

    l = sum_r a ln|O|,  s = prod_r sign(O),  z = s ln(1 + e^l)

and the backward is written by hand (``AndOrFunction``) so it can take the
exact one-sided rule at a factor that is exactly zero, which autograd
through ``log|O|`` would turn into NaN:

    dz/dO_r = s sigma(l) a_r / O_r             (O_r != 0)
    dz/da_r = s sigma(l) ln|O_r|               (O_r != 0, else 0)
    at O_r = 0: dz/dO_r = cofactor if a_r = 1 and it is the only zero, else 0

Slots with ``mask == False`` are padding. They hold the identity Or
(w = 0, b = 1, a = 1), whose value is exactly 1, so they do not change A;
their parameter gradients are zeroed so no optimizer can move them.
"""

from __future__ import annotations

import torch
from torch.autograd.function import once_differentiable

__all__ = ["readout", "readout_grad", "and_or", "AndOrFunction"]


def readout(u: torch.Tensor) -> torch.Tensor:
    """F(u) = sign(u) ln(1 + |u|): odd, monotone, F(0) = 0."""
    return torch.sign(u) * torch.log1p(u.abs())


def readout_grad(u: torch.Tensor) -> torch.Tensor:
    """F'(u) = 1 / (1 + |u|)."""
    return 1.0 / (1.0 + u.abs())


def _ors(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    m, R, n = weight.shape
    return (x @ weight.reshape(m * R, n).T).reshape(x.shape[0], m, R) + bias


class AndOrFunction(torch.autograd.Function):
    """z = sign(A) ln(1 + |A|), A = prod_r (w_r . x + b_r)^{a_r}. See module docs."""

    @staticmethod
    def forward(ctx, x, weight, bias, assembly, mask):
        O = _ors(x, weight, bias)                          # (B, m, R)
        logabs = torch.log(O.abs())                        # -inf at an exact zero
        ell = (assembly * logabs).sum(-1)                  # (B, m); -inf if a factor is 0
        sgn = torch.sign(O).prod(-1)                       # 0 if a factor is 0
        z = sgn * torch.logaddexp(ell, torch.zeros_like(ell))
        ctx.save_for_backward(x, weight, assembly, mask, O, logabs, ell, sgn)
        return z

    @staticmethod
    @once_differentiable
    def backward(ctx, g):
        x, weight, assembly, mask, O, logabs, ell, sgn = ctx.saved_tensors
        m, R, n = weight.shape
        B = x.shape[0]
        a = assembly.unsqueeze(0)                          # (1, m, R)
        ss = (sgn * torch.sigmoid(ell)).unsqueeze(-1)      # s sigma(l), (B, m, 1)
        nz = O != 0
        safe_O = torch.where(nz, O, torch.ones_like(O))
        dz_dO = torch.where(nz, ss * a / safe_O, torch.zeros_like(O))
        dz_da = torch.where(nz, ss * torch.where(nz, logabs, torch.zeros_like(O)),
                            torch.zeros_like(O))
        zero = ~nz
        if bool(zero.any()):
            # The only zero factor with a = 1 has the cofactor as its slope
            # (dz/dA = 1 at A = 0); every other case at a zero is 0.
            n_zero = zero.sum(-1, keepdim=True)            # (B, m, 1)
            ell_nz = (a * torch.where(nz, logabs, torch.zeros_like(O))).sum(-1, keepdim=True)
            sgn_nz = torch.where(nz, torch.sign(O), torch.ones_like(O)).prod(-1, keepdim=True)
            cof = sgn_nz * torch.exp(ell_nz)
            use = zero & (n_zero == 1) & (a <= 1)
            dz_dO = torch.where(use, cof.expand_as(O), dz_dO)
        gk = g.unsqueeze(-1)                               # (B, m, 1)
        dO = gk * dz_dO                                    # dL/dO, (B, m, R)
        grad_x = grad_w = grad_b = grad_a = None
        if ctx.needs_input_grad[0]:
            grad_x = dO.reshape(B, m * R) @ weight.reshape(m * R, n)
        maskf = mask.to(O.dtype)
        dOp = dO * maskf
        if ctx.needs_input_grad[1]:
            grad_w = (dOp.reshape(B, m * R).T @ x).reshape(m, R, n)
        if ctx.needs_input_grad[2]:
            grad_b = dOp.sum(0)
        if ctx.needs_input_grad[3]:
            grad_a = (gk * dz_da).sum(0) * maskf
        return grad_x, grad_w, grad_b, grad_a, None


def and_or(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    assembly: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply one type-nn layer.

    Args:
        x: input of shape ``(*, n)``.
        weight: ``(m, R, n)``, the dense weights of every Or.
        bias: ``(m, R)``.
        assembly: ``(m, R)``, assembly indices, expected ``>= 1``.
        mask: optional ``(m, R)`` bool; ``False`` marks a padding slot, which
            must hold the identity Or (w = 0, b = 1).

    Returns:
        ``(*, m)``.
    """
    m, R, n = weight.shape
    if mask is None:
        mask = torch.ones(m, R, dtype=torch.bool, device=weight.device)
    lead = x.shape[:-1]
    z = AndOrFunction.apply(x.reshape(-1, n), weight, bias, assembly, mask)
    return z.reshape(*lead, m)
