"""
Closed-form predictions (proposal Section 5.3) and curve-based prediction for HorizonGym episodes.

Propositions:
    1. Delta_adapt / Delta_unif = GM(s) / AM(s)                 (equal costs)
    2. log(S_adapt / S_unif) = kappa * H, kappa = 2^(-2b) (AM - GM)
    3. H* = ln((h0 + A) / (h0 + G)) / (A - G)                   (peak absolute gain)
"""

from __future__ import annotations 

import math 
from dataclasses import dataclass 
from typing import Callable, Mapping

import numpy as np 
from numpy.typing import ArrayLike

from hazprec.phases import Phase 

def am_gm(s: ArrayLike, weights: ArrayLike | None = None) -> tuple[float, float]:
    x = np.asarray(s, float)
    if np.any(x <= 0):
        raise ValueError("sensitivities must be positive")
    w = np.ones_like(x) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    return float(np.dot(w, x)), float(np.exp(np.dot(w, np.log(x))))

def amgm_ratio(s: ArrayLike, weights: ArrayLike | None = None) -> float:
    am, gm = am_gm(s, weights)
    return gm / am 

def kappa(s: ArrayLike, bbar: float, weights: ArrayLike | None = None) -> float:
    am, gm = am_gm(s, weights)
    return 2.0 ** (-2.0 * bbar) * (am - gm)

def h_star_from_hazards(h0: float, A: float, G: float) -> float:
    """ Horizon (in steps) where S_adapt - S_unif peaks; A, G = per-step excess hazards"""
    if A <= G:
        raise ValueError("uniform excess hazard A must exceed adaptive G")
    return math.log((h0 + A) / (h0 + G)) / (A - G)

def h_star_from_lines(a_adapt: float, m_adapt: float, a_unif: float, m_unif: float) -> float:
    """ Peak of exp(a_a + m_a H) - exp(a_u + m_u H) for log-linear survival lines
    (m < 0 are per-step log-survival slopes)."""
    if not (m_unif < m_adapt < 0):
        raise ValueError("need m_unif < m_adapt < 0")
    return (math.log(m_unif / m_adapt) + a_unif - a_adapt) / (m_adapt - m_unif)


### HorizonGym - specific prediction from fitted per-phase curves ###

DeltaFn = Callable[[Phase, int], float]

@dataclass 
class SurvivalLine:
    intercept: float # log S at zero hops (first-turn and submit corrections)
    slope: float # log-survival per hop (negative)

    def log_s(self, hops: float) -> float:
        return self.intercept + self.slope * hops 

    def to_dict(self) -> dict:
        return {"intercept": self.intercept, "slope_per_hop": self.slope}

def _log_keep(delta: DeltaFn, base: float, read_phase: Phase, act_phase: Phase, 
              bits: Mapping[Phase, int], think: bool) -> float:
    keep = math.log1p(-base)
    keep += math.log1p(-min(delta(read_phase, bits[read_phase]), 0.999999))
    if think:
        keep += math.log1p(-min(delta(Phase.THINK, bits[Phase.THINK]), 0.999999))
    keep += math.log1p(-min(delta(act_phase, bits[act_phase]), 0.999999))
    return keep 

def horizongym_line(delta: DeltaFn, base_hazard: float, bits: Mapping[Phase, int], think: bool = True) -> SurvivalLine:
    """
    Predicted log-survival line for HorizonGym under a phase-static schedule.

    Turn structure for n hops: one (SYSTEM, ACT) turn, (2n - 1) (READ, ACT) turns, and 
    one (READ, STOP) turn. ``bits`` maps each phase to its precision; for a uniform 
    schedule every phase maps to the same value
    """
    l_sys = _log_keep(delta, base_hazard, Phase.SYSTEM, Phase.ACT, bits, think)
    l_act = _log_keep(delta, base_hazard, Phase.READ, Phase.ACT, bits, think)
    l_stop = _log_keep(delta, base_hazard, Phase.READ, Phase.STOP, bits, think)
    return SurvivalLine(intercept = l_sys - l_act + l_stop, slope = 2.0 * l_act)

def predict_horizon_gain(delta: DeltaFn, base_hazard: float, uniform_bits: int, 
                         adaptive_bits: Mapping[Phase, int], think: bool = True) -> dict:
    uni = {p: uniform_bits for p in Phase}
    lu = horizongym_line(delta, base_hazard, uni, think)
    la = horizongym_line(delta, base_hazard, adaptive_bits, think)
    out = {
        "uniform": lu.to_dict(),
        "adaptive": la.to_dict(),
        "log_ratio_intercept": la.intercept - lu.intercept,
        "log_ratio_slope_per_hop": la.slope - lu.slope,
    }
    try:
        out["h_star_hops"] = h_star_from_lines(la.intercept, la.slope, lu.intercept, lu.slope)
    except ValueError:
        out["h_star_hops"] = None 
    return out 
