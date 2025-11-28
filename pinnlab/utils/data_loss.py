from __future__ import annotations
from typing import Optional
import torch

def data_loss_mse(residual: torch.Tensor) -> torch.Tensor:
    """
    Standard L2 data loss, but returned per-point (no reduction):
        ℓ_i = (r_i)^2
    """
    return residual ** 2

def data_loss_l1(residual: torch.Tensor) -> torch.Tensor:
    """
    LAD-PINN (L1) data loss, per-point:
        ℓ_i = |r_i|
    """
    return residual.abs()

def data_loss_q_gaussian(
    residual: torch.Tensor,
    q: float = 1.2,
    beta: Optional[float] = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    r"""
    OrPINN-style Tsallis q-Gaussian negative log-likelihood, per-point.

    For a q-Gaussian PDF of the form

        p_q(r) ∝ [ 1 + (q-1) * β * r^2 ]^{-1/(q-1)}

    the negative log-likelihood (up to an additive constant) is

        ℓ_q(r) = 1/(q-1) * log( 1 + (q-1) * β * r^2 )

    OrPINN chooses β = 1 / [ 2 (3 - q) ] so that variance is ~1
    (you can override via the `beta` argument).

    When q → 1 this converges to a quadratic loss.
    """
    if q == 1.0:
        # In the q→1 limit, this becomes proportional to r^2
        return 0.5 * residual ** 2

    if beta is None:
        # Default consistent with OrPINN: q in (1, 3)
        beta = 0.5 / (3.0 - q)

    z = 1.0 + (q - 1.0) * beta * (residual ** 2)
    z = torch.clamp(z, min=eps)
    return torch.log(z) / (q - 1.0)


def aggregate_data_loss(point_losses: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    """
    Helper to aggregate per-point losses into a scalar.
    """
    if reduction == "mean":
        return point_losses.mean()
    elif reduction == "sum":
        return point_losses.sum()
    else:
        raise ValueError(f"Unknown reduction: {reduction}")