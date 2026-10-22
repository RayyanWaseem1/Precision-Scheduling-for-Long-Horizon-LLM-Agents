"""
Hazard-optimal bit allocation (Proposal Section 5.2)

Problem: minimize sum_i Delta_i(b_i) subject to sum_i c_i * b_i <= B, b_i in levels

* ``waterfill``             continous solution b_i = 1/2 log2(s_i / c_i) + beta, clipped
* ``allocate_greedy``       discrete, upgrades by best marginal hazard reduction per bit-cost
* ``allocate_exact``        discrete brute force (small problems, tests, phase tables)
* ``phase_table``           per-phase bits for a phase static policy at average budget b-bar
"""

from __future__ import annotations 

import itertools
from typing import Callable, Mapping, Sequence 

import numpy as np 
from numpy.typing import ArrayLike

Curve = Callable[[float, float], float]

def excess_exp2(s: float, b: float) -> float:
    """ Excess hazard model Delta(b) = s * 2^(-2b)"""
    return float(s) * 2.0 ** (-2.0 * float(b))

def delta_table(sens: ArrayLike, levels: Sequence[int], curve: Curve = excess_exp2) -> np.ndarray:
    sens = np.asarray(sens, float)
    return np.array([[curve(s, b) for b in levels] for s in sens], dtype = float)

def _check(delta: np.ndarray, cost: np.ndarray, levels: Sequence[int], budget: float) -> None:
    if delta.shape != (len(cost), len(levels)):
        raise ValueError("delta must have shape (n_items, n_levels)")
    if list(levels) != sorted(levels):
        raise ValueError("levels must be ascending")
    if float(np.dot(cost, np.full(len(cost), levels[0]))) > budget + 1e-9:
        raise ValueError("budget is below the cost of the lowest precision everywhere")

def allocate_greedy(delta: np.ndarray, cost: ArrayLike, levels: Sequence[int], budget: float) -> np.ndarray:
    """ Greedy marginal allocation. Optimal for convex curves with equal costs; close to 
    optimal otherwise (compare with ``allocate_exact`` on small instances)"""
    delta = np.asarray(delta, float)
    cost = np.asarray(cost, float)
    _check(delta, cost, levels, budget)
    idx = np.zeros(len(cost), dtype = int)
    spent = float(np.dot(cost, np.full(len(cost), levels[0])))
    while True:
        best, best_gain  = -1, 0.0
        for i in range(len(cost)):
            j = idx[i]
            if j + 1 >= len(levels):
                continue 
            extra = cost[i] * (levels[j + 1] - levels[j])
            if spent + extra > budget + 1e-9:
                continue 
            gain = (delta[i, j] - delta[i, j + 1]) / max(extra, 1e-12)
            if gain > best_gain:
                best, best_gain = i, gain 
        if best < 0:
            break 
        j = idx[best]
        spent += cost[best] * (levels[j + 1] - levels[j])
        idx[best] += 1 
    return np.array([levels[j] for j in idx])


def allocate_exact(delta: np.ndarray, cost: ArrayLike, levels: Sequence[int], budget: float,
                   max_combos: int = 2_000_000) -> np.ndarray:
    delta = np.asarray(delta, float)
    cost = np.asarray(cost, float)
    _check(delta, cost, levels, budget)
    n, L = delta.shape
    if L**n > max_combos:
        raise ValueError("problem too large for exact search; use allocate_greedy")
    lv = np.asarray(levels, float)
    best, best_val = None, np.inf
    for combo in itertools.product(range(L), repeat = n):
        c = float(np.dot(cost, lv[list(combo)]))
        if c > budget + 1e-9:
            continue 
        v = float(sum(delta[i, j] for i, j in enumerate(combo)))
        if v < best_val - 1e-15 or (abs(v - best_val) <= 1e-15 and best is not None and c < best[1]):
            best, best_val = (combo, c), v
    assert best is not None 
    return np.array([levels[j] for j in best[0]])

def allocate_dp(delta: np.ndarray, cost: ArrayLike, levels: Sequence[int], budget: float,
                max_states: int = 50_000_000) -> np.ndarray:
    """ Exact multiple-choice knapsack by dynamic programming over the integer budget.
    
    Costs are rounded to integers (token counts already are). Each item chooses one
    level; the state is bit-tokens spent. Complexity O(n * L * B)"""
    delta = np.asarray(delta, float)
    cost = np.rint(np.asarray(cost, float)).astype(np.int64)
    _check(delta, cost.astype(float), levels, budget)
    lv = np.asarray(levels, np.int64)
    B = int(np.floor(budget + 1e-9))
    n, L = delta.shape
    if (B + 1) * n > max_states:
        raise ValueError("budget too large for DP; use allocate_greedy")
    INF = np.inf 
    dp = np.full(B + 1, INF)
    dp[0] = 0.0
    choice = np.zeros((n, B + 1), dtype = np.int16)
    for i in range(n):
        new = np.full(B + 1, INF)
        best = np.zeros(B + 1, dtype = np.int16)
        for j in range(L):
            w = int(cost[i] * lv[j])
            if w > B:
                continue 
            cand = np.full(B + 1, INF)
            cand[w:] = dp[: B + 1 - w] + delta[i, j]
            better = cand < new 
            new[better] = cand[better]
            best[better] = j 
        dp, choice[i] = new, best 
    x = int(np.argmin(dp))
    if not np.isfinite(dp[x]):
        raise ValueError("no feasible allocation")
    out = np.zeros(n, dtype = int)
    for i in range(n -1, -1, -1):
        j = int(choice[i, x])
        out[i] = levels[j]
        x -= int(cost[i] * lv[j])
    return out 

def allocate_bits(sens: ArrayLike, cost: ArrayLike, levels: Sequence[int], budget: float,
                  curve: Curve = excess_exp2) -> np.ndarray:
    """ Exact when feasible (brute force for tiny problems, DP for integer costs),
    greedy otherwise"""
    delta = delta_table(sens, levels, curve)
    if len(levels) ** len(delta) <= 200_000:
        return allocate_exact(delta, cost, levels, budget)
    try:
        return allocate_dp(delta, cost, levels, budget)
    except ValueError:
        return allocate_greedy(delta, cost, levels, budget)

def waterfill(sens: ArrayLike, cost: ArrayLike, budget: float, 
              bmin: float = 0.0, bmax: float = 16.0, iters: int = 200) -> np.ndarray:
    """ Continuous optimum of sum s_i 2^(-2 b_i) s.t. sum c_i b_i = budget, bmin<=b<=bmax"""
    s = np.asarray(sens, float)
    c = np.asarray(cost, float)
    if np.any(s <= 0) or np.any(c <= 0):
        raise ValueError("sensitivities and costs must be positive")
    base = 0.5 * np.log2(s / c)

    def spend(beta: float) -> float:
        return float(np.dot(c, np.clip(base + beta, bmin, bmax)))

    lo, hi = -64.0, 64.0
    if spend(lo) > budget or spend(hi) < budget:
        raise ValueError("budget outside the feasible range [sum c * bmin, sum c * bmax]")
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if spend(mid) < budget:
            lo = mid 
        else:
            hi = mid 
    return np.clip(base + 0.5 * (lo + hi), bmin, bmax)


def phase_table(
        sens: Mapping[str, float],
        tokens: Mapping[str, float],
        freq: Mapping[str, float],
        levels: Sequence[int],
        bbar: float,
        curve: Curve = excess_exp2,
        curves: Mapping[str, Callable[[int], float]] | None = None,
) -> dict[str, int]:
    """
    Bits per phase for a phase-static policy.

    ``freq[p]``     expected number of segments of phase p per episode (or per hop)
    ``tokens[p]``   expected tokens per segment of phase p (the cost unit)
    The budget is a token-weighted average of ``bbar`` bits. ``curves`` (fitted 
    per-phase Delta(b) functions) override the parametric ``sens``/``curve`` model.
    """
    phases = [p for p in freq if freq[p] > 0]
    cost = np.array([freq[p] * tokens[p] for p in phases], float)
    if curves is not None:
        delta = np.array([[freq[p] * curves[p](b) for b in levels] for p in phases], float)
    else:
        delta = np.array([[freq[p] * curve(sens[p], b) for b in levels] for p in phases], float)
    budget = bbar * float(cost.sum())
    if len(levels) ** len(phases) <= 200_000:
        bits = allocate_exact(delta, cost, levels, budget)
    else: 
        bits = allocate_greedy(delta, cost, levels, budget)
    return {p: int(b) for p, b in zip(phases, bits)}

def average_bits(table: Mapping[str, int], tokens: Mapping[str, float], freq: Mapping[str, float]) -> float:
    num = sum(freq[p] * tokens[p] * table[p] for p in table if freq.get(p, 0) > 0)
    den = sum(freq[p] * tokens[p] for p in table if freq.get(p, 0) > 0)
    return num / den 
