from __future__ import annotations

from typing import Iterable, Tuple

import torch

from .data import COEFFICIENTS, MAX_L


Coefficient = Tuple[float, int, int, int]


def _ensure_points(points: torch.Tensor) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError("points must be a torch.Tensor")
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points


def _mul_power(term: torch.Tensor, base: torch.Tensor, exponent: int) -> torch.Tensor:
    if exponent == 0:
        return term
    if exponent == 1:
        return term * base
    return term * torch.pow(base, exponent)


def _evaluate_terms(
    coeffs: Iterable[Coefficient], x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    values = torch.zeros_like(x)
    for coeff, i, j, k in coeffs:
        term = torch.full_like(x, coeff)
        term = _mul_power(term, x, i)
        term = _mul_power(term, y, j)
        term = _mul_power(term, z, k)
        values = values + term
    return values


def _evaluate_grad_terms(
    coeffs: Iterable[Coefficient], x: torch.Tensor, y: torch.Tensor, z: torch.Tensor
) -> torch.Tensor:
    grad_x = torch.zeros_like(x)
    grad_y = torch.zeros_like(y)
    grad_z = torch.zeros_like(z)

    for coeff, i, j, k in coeffs:
        if i:
            term_x = torch.full_like(x, coeff * i)
            term_x = _mul_power(term_x, x, i - 1)
            term_x = _mul_power(term_x, y, j)
            term_x = _mul_power(term_x, z, k)
            grad_x = grad_x + term_x
        if j:
            term_y = torch.full_like(y, coeff * j)
            term_y = _mul_power(term_y, x, i)
            term_y = _mul_power(term_y, y, j - 1)
            term_y = _mul_power(term_y, z, k)
            grad_y = grad_y + term_y
        if k:
            term_z = torch.full_like(z, coeff * k)
            term_z = _mul_power(term_z, x, i)
            term_z = _mul_power(term_z, y, j)
            term_z = _mul_power(term_z, z, k - 1)
            grad_z = grad_z + term_z

    return torch.stack((grad_x, grad_y, grad_z), dim=-1)


def _coefficients(l: int, m: int) -> Iterable[Coefficient]:
    if l < 0 or l > MAX_L:
        raise ValueError(f"Degree l must satisfy 0 <= l <= {MAX_L}, got {l}")
    if abs(m) > l:
        raise ValueError(f"Order m must satisfy |m| <= l, got (l={l}, m={m})")
    try:
        return COEFFICIENTS[(l, m)]
    except KeyError as exc:
        raise KeyError(f"No coefficients found for (l={l}, m={m})") from exc


def evaluate(l: int, m: int, points: torch.Tensor) -> torch.Tensor:
    """Evaluate the solid harmonic R_l^m at the provided points."""

    pts = _ensure_points(points)
    coeffs = _coefficients(l, m)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return _evaluate_terms(coeffs, x, y, z)


def evaluate_gradient(l: int, m: int, points: torch.Tensor) -> torch.Tensor:
    """Evaluate the gradient of R_l^m with respect to x, y, z."""

    pts = _ensure_points(points)
    coeffs = _coefficients(l, m)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    return _evaluate_grad_terms(coeffs, x, y, z)
