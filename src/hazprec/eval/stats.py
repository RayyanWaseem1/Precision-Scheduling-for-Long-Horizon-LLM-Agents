"""Small, dependency-free statistics helpers."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def bootstrap(stat: Callable[[np.ndarray], float], n_units: int, n_boot: int = 2000, seed: int = 0,
              alpha: float = 0.05) -> tuple[float, float, np.ndarray]:
    """Percentile bootstrap over resampled unit indices. ``stat`` receives an index array."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat(rng.integers(0, n_units, n_units)) for _ in range(n_boot)], float)
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi), vals


def weighted_linfit(x: Sequence[float], y: Sequence[float], w: Sequence[float] | None = None) -> tuple[float, float]:
    """Weighted least squares y = a + b x; returns (a, b)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    w = np.ones_like(x) if w is None else np.asarray(w, float)
    xm, ym = np.average(x, weights=w), np.average(y, weights=w)
    b = float(np.sum(w * (x - xm) * (y - ym)) / np.sum(w * (x - xm) ** 2))
    return float(ym - b * xm), b


def fit_logistic(X: np.ndarray, y: np.ndarray, ridge: float = 1e-6, iters: int = 100,
                 tol: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by iteratively reweighted least squares.
    Returns (coefficients, standard errors)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        p = 1 / (1 + np.exp(-eta))
        W = p * (1 - p)
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1])
        g = X.T @ (y - p) - ridge * beta
        step = np.linalg.solve(H, g)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    eta = np.clip(X @ beta, -30, 30)
    p = 1 / (1 + np.exp(-eta))
    H = X.T @ (X * (p * (1 - p))[:, None]) + ridge * np.eye(X.shape[1])
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    return beta, se
