from __future__ import annotations

from abc import ABC, abstractmethod

import torch

from . import mathematics as maths
from . import solid_harmonics as sh


class TestFunction(ABC):
    """Base class for boundary test functions u(y)."""

    def __init__(self, name: str, epsilon: float) -> None:
        self.name = name
        self.epsilon = float(epsilon)

    def eval_func(self, points: torch.Tensor) -> torch.Tensor:
        return self._compute_func(points)

    def eval_grad(self, points: torch.Tensor) -> torch.Tensor:
        return self._compute_grad(points)

    def eval_smoothed(self, points: torch.Tensor) -> torch.Tensor:
        """The function convolved with a Gaussian of width epsilon."""
        return self._compute_smoothed(points)

    @abstractmethod
    def _compute_func(self, points: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _compute_grad(self, points: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def _compute_smoothed(self, points: torch.Tensor) -> torch.Tensor: ...


class FundamentalSolution(TestFunction):
    def __init__(self, source: torch.Tensor, epsilon: float) -> None:
        source = source.reshape(1, 3)
        super().__init__(
            name=f"fundamental_source_{source.cpu().numpy().tolist()}",
            epsilon=epsilon,
        )
        self._source = source.detach().clone()

    @property
    def source(self) -> torch.Tensor:
        return self._source

    def _compute_func(self, points: torch.Tensor) -> torch.Tensor:
        source = self._source.to(device=points.device, dtype=points.dtype)
        values = maths.fund_sol(source, points).squeeze(0).unsqueeze(-1)
        return values

    def _compute_grad(self, points: torch.Tensor) -> torch.Tensor:
        source = self._source.to(device=points.device, dtype=points.dtype)
        gradients = maths.fund_sol_grad_y(source, points).squeeze(0)
        return gradients

    def _compute_smoothed(self, points: torch.Tensor) -> torch.Tensor:
        source = self._source.to(device=points.device, dtype=points.dtype)
        values = maths.fund_sol_eps(source, points, self.epsilon).squeeze(0)
        return values.unsqueeze(-1)


class RegularSolidHarmonic(TestFunction):
    def __init__(self, l: int, m: int, epsilon: float) -> None:
        name = f"solid_harmonic_l{l}_m{m}"
        super().__init__(name=name, epsilon=epsilon)
        self.l = int(l)
        self.m = int(m)

    def _compute_func(self, points: torch.Tensor) -> torch.Tensor:
        values = sh.evaluate(self.l, self.m, points)
        return values.reshape(-1, 1)

    def _compute_grad(self, points: torch.Tensor) -> torch.Tensor:
        return sh.evaluate_gradient(self.l, self.m, points)

    def _compute_smoothed(self, points: torch.Tensor) -> torch.Tensor:
        # Solid harmonics are harmonic everywhere; convolution with the Gaussian kernel
        # leaves the function unchanged, so reuse eval_func.
        return self._compute_func(points)
