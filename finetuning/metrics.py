"""
Implementing usefull metrics and loss 
"""

import torch
from torch import Tensor
from torch.nn import PairwiseDistance, Module
from torch.nn.functional import relu, mse_loss

def reduce_loss(
    loss: Tensor,
    reduction: str,
) -> Tensor:
    """
    Apply a "mean" or "sum" reduction to a vector of losses.

    Raises
    ------
    ValueError
        If `reduction` is not "mean" or "sum".
    """
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(
            f"Invalid reduction type: {reduction}. Must be 'mean' or 'sum'."
        )

class EuclideanSimilarity(torch.nn.Module):
    """
    Compute similarity between two embedding tensors based on their Euclidean distance.
    Smaller distance = higher similarity.
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.distance = PairwiseDistance(p=2, eps=eps)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Compute L2 distance
        dist = self.distance(x1, x2)
        # Convert distance to similarity
        sim = 1 / (1 + dist)
        return sim


def euclidean_similarity(y_true, y_pred):
    distance = PairwiseDistance()
    # Compute L2 distance
    dist = distance(y_true, y_pred)
    # Convert distance to similarity
    sim = 1 / (1 + dist)
    return sim


def pearson(x, y, dim=0, eps=1e-8):
    x = x - x.mean(dim=dim, keepdim=True)
    y = y - y.mean(dim=dim, keepdim=True)
    cov = (x * y).mean(dim=dim)
    # Biased std, consistent with the biased covariance above.
    std_x = x.std(dim=dim, unbiased=False)
    std_y = y.std(dim=dim, unbiased=False)

    return cov / (std_x * std_y + eps)


def coefficient_of_determination(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int = 0,
) -> torch.Tensor:
    """
    Coefficient of determination (R2) along `dim`.

    Parameters
    ----------
    y_true, y_pred : torch.Tensor
        Targets and predictions with the same shape, e.g. (N, D).
    dim : int, default=0
        Sample dimension.

    Returns
    -------
    torch.Tensor
        R2 per remaining index, e.g. (D,). NaN where `y_true` is constant.
    """
    residual_sum = ((y_true - y_pred) ** 2).sum(dim=dim)
    total_sum = ((y_true - y_true.mean(dim=dim, keepdim=True)) ** 2).sum(dim=dim)

    return torch.where(
        total_sum > 0,
        1.0 - residual_sum / total_sum.clamp_min(torch.finfo(total_sum.dtype).tiny),
        torch.full_like(total_sum, float("nan")),
    )


def concordance_correlation(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    dim: int = 0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Lin's concordance correlation coefficient (CCC) along `dim`.

    CCC = 2 cov(x, y) / (var(x) + var(y) + (mean(x) - mean(y))^2)

    Parameters
    ----------
    y_true, y_pred : torch.Tensor
        Targets and predictions with the same shape, e.g. (N, D).
    dim : int, default=0
        Sample dimension.
    eps : float, default=1e-8
        Numerical stability term in the denominator.

    Returns
    -------
    torch.Tensor
        CCC per remaining index, e.g. (D,).
    """
    mean_true = y_true.mean(dim=dim, keepdim=True)
    mean_pred = y_pred.mean(dim=dim, keepdim=True)
    covariance = ((y_true - mean_true) * (y_pred - mean_pred)).mean(dim=dim)
    variance_true = y_true.var(dim=dim, unbiased=False)
    variance_pred = y_pred.var(dim=dim, unbiased=False)
    mean_gap = (mean_true - mean_pred).squeeze(dim) ** 2

    return 2.0 * covariance / (variance_true + variance_pred + mean_gap + eps)


class RMSELoss(Module):
    """
    Root mean squared error: sqrt(MSE + eps).

    Parameters
    ----------
    eps : float, default=1e-8
        Keeps the gradient finite when the MSE is 0.
    """

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(mse_loss(y_pred, y_true) + self.eps)


INEQUALITY_CONSTRAINTS = [
    # total cells
    ("CD45+", "cells"),
    ("CD68+", "cells"),
    ("CD163+", "cells"),
    ("CD3e+", "cells"),
    ("CD8a+", "cells"),
    ("CD45R0+", "cells"),
    ("CD20+", "cells"),
    ("CD4+", "cells"),
    ("FOXP3+", "cells"),
    ("FOXP3+", "cells"),
    ("PDL1+", "cells"),
    ("Ki67+", "cells"),
    ("PanCK+", "cells"),
    ("Ecadherin+", "cells"),
    ("CD31+", "cells"),
    ("SMA+", "cells"),
    # --- Immune backbone ---
    ("CD3e+", "CD45+"),
    ("CD20+", "CD45+"),
    ("CD68+", "CD45+"),
    # --- Myeloid ---
    ("CD163+", "CD68+"),
    ("CD163+", "CD45+"),
    # --- T cell hierarchy ---
    ("CD8a+", "CD3e+"),
    ("CD8a+", "CD45+"),
    ("CD4+", "CD3e+"),
    ("CD4+", "CD45+"),
    ("FOXP3+", "CD4+"),
    ("FOXP3+", "CD3e+"),
    ("FOXP3+", "CD45+"),
    # --- Memory / activation ---
    ("CD45R0+", "CD3e+"),
    ("CD45R0+", "CD45+"),
    ("PD1+", "CD3e+"),
    ("PD1+", "CD45+"),
    # --- Epithelial ---
    ("Ecadherin+", "PanCK+"),
]

marker_to_idx = {
    "cells": 0,
    "CD45+": 1,
    "CD68+": 2,
    "CD163+": 3,
    "CD3e+": 4,
    "CD8a+": 5,
    "CD45R0+": 6,
    "CD20+": 7,
    "CD4+": 8,
    "FOXP3+": 9,
    "PDL1+": 10,
    "PD1+": 11,
    "Ki67+": 12,
    "PanCK+": 13,
    "Ecadherin+": 14,
    "CD31+": 15,
    "SMA+": 16,
    "ALL-": 17,
    "BIO-": 18,
}


def hierarchy_penalty(
    y_pred,
    marker_to_idx=marker_to_idx,
    constraints=INEQUALITY_CONSTRAINTS,
    alpha=1.0,
    power=2,
):
    """
    Enforces hierarchical constraints A <= B in linear space.

    Args:
        y_pred: Tensor (B, D) — cell-count or log-space predictions
        marker_to_idx: dict {marker_name: index}
        constraints: list of tuples (A, B) meaning A <= B
        alpha: scaling factor
        eps: numerical stability for exp

    Returns:
        scalar loss
    """
    loss = 0.0

    for A, B in constraints:
        idx_A = marker_to_idx[A]
        idx_B = marker_to_idx[B]

        z_A = y_pred[:, idx_A]
        z_B = y_pred[:, idx_B]

        # violation: A > B
        violation = relu(z_A - z_B)

        loss += (violation**power).mean()

    return alpha * loss


class HierarchyLoss(Module):
    def __init__(
        self,
        marker_to_idx=marker_to_idx,
        constraints=INEQUALITY_CONSTRAINTS,
        mse_weight=1.0,
        hierarchy_weight=1.0,
        power=2,
    ):
        """
        Combines MSE loss with hierarchy constraints.

        Args:
            marker_to_idx: dict {marker_name: index}
            constraints: list of (A, B) meaning A <= B
            mse_weight: weight for MSE loss
            hierarchy_weight: weight for hierarchy penalty
            power: exponent for violation penalty
            eps: numerical stability
        """

        super().__init__()

        self.marker_to_idx = marker_to_idx
        self.constraints = constraints

        self.mse_weight = mse_weight
        self.hierarchy_weight = hierarchy_weight

        self.power = power

        # Precompute index pairs for speed
        self.A_idx = torch.tensor([marker_to_idx[a] for a, _ in constraints])
        self.B_idx = torch.tensor([marker_to_idx[b] for _, b in constraints])

    def hierarchy_penalty(self, y_pred):
        """
        Vectorized hierarchy penalty: A <= B
        """

        # No need to convert to linear space for inequality check
        z_A = y_pred[:, self.A_idx]
        z_B = y_pred[:, self.B_idx]

        # violation: A > B
        violation = relu(z_A - z_B)

        loss = (violation**self.power).mean()

        return loss

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, D)
            y_true: (B, D)
        """

        mse = mse_loss(y_pred, y_true)

        hierarchy = self.hierarchy_penalty(y_pred)

        loss = self.mse_weight * mse + self.hierarchy_weight * hierarchy

        return loss
