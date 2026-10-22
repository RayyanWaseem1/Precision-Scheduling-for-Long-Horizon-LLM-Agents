"""Glue: from a fitted sensitivity atlas to an adaptive schedule and pre-registered
predictions for the horizon sweep."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from hazprec.eval.stats import weighted_linfit
from hazprec.phases import Phase
from hazprec.sensitivity.fit import AtlasFit
from hazprec.theory.allocation import average_bits, phase_table, waterfill
from hazprec.theory.predictions import horizongym_line, predict_horizon_gain

SCHED_PHASES = ("system", "read", "think", "act", "stop")


def episode_frequencies(n_hops: int, think: bool = True) -> dict[str, float]:
    """Segments of each phase in one HorizonGym episode (successful path)."""
    n_turns = 2 * n_hops + 1
    freq = {"system": 1.0, "read": float(n_turns - 1), "act": float(2 * n_hops), "stop": 1.0}
    if think:
        freq["think"] = float(n_turns)
    return freq


def _tokens(fit: AtlasFit) -> dict[str, float]:
    t = dict(fit.tokens)
    t.setdefault("stop", t.get("act", 24.0))
    t.setdefault("system", t.get("read", 100.0))
    return t


def adaptive_schedule(fit: AtlasFit, bbar: float, levels: Sequence[int], alloc_hops: int, think: bool = True,
                      mode: str = "parametric") -> dict[str, Any]:
    delta = fit.delta_fn(mode)
    freq = episode_frequencies(alloc_hops, think)
    tokens = _tokens(fit)
    missing = set(freq) - (set(fit.tokens) & set(fit.curves))
    if missing:
        raise ValueError(f"atlas lacks measured phase coverage: {sorted(missing)}")
    curves = {p: (lambda b, p=p: delta(p, b)) for p in freq}
    table = phase_table({}, tokens, freq, levels, bbar, curves=curves)
    return {
        "table": table,
        "avg_bits_at_alloc_hops": average_bits(table, tokens, freq),
        "tokens": tokens,
        "freq": freq,
    }


def continuous_diagnostics(fit: AtlasFit, bbar: float, alloc_hops: int, think: bool = True) -> dict[str, float]:
    """Continuous water-filling vs uniform excess hazard per episode (Proposition 1's
    quantity with unequal costs). Uses fitted parametric sensitivities."""
    freq = episode_frequencies(alloc_hops, think)
    tokens = _tokens(fit)
    phases = [p for p in freq if fit.delta_fn()(p, 0) > 0]
    if len(phases) < 2:
        return {}
    s = np.array([freq[p] * fit.delta_fn()(p, 0) for p in phases])  # delta at b=0 equals s
    c = np.array([freq[p] * tokens[p] for p in phases])
    b = waterfill(s, c, bbar * c.sum(), bmin=0.0, bmax=16.0)
    adapt = float(np.sum(s * 2.0 ** (-2 * b)))
    unif = float(np.sum(s) * 2.0 ** (-2 * bbar))
    return {"continuous_excess_ratio": adapt / unif, "continuous_bits": dict(zip(phases, map(float, b)))}


def build_predictions(fit: AtlasFit, bbar: float, levels: Sequence[int], alloc_hops: int, think: bool = True,
                      mode: str = "parametric", sweep_hops: Sequence[int] = ()) -> dict[str, Any]:
    """Adaptive schedules and predictions.

    * ``adaptive_table`` is allocated for ``alloc_hops`` (the long-horizon schedule) and
      defines the asymptotic slope and H*.
    * For each horizon in ``sweep_hops`` a table is allocated for *that* horizon, so the
      adaptive policy never spends more than the uniform b-bar at any horizon. The
      pre-registered slope is the regression of the predicted log ratios on hops, which is
      exactly the statistic the analysis computes on the measured ratios.
    """
    if int(bbar) != bbar or int(bbar) not in levels:
        raise ValueError(f"uniform baseline needs b-bar to be an available level; got {bbar} with {levels}")
    delta = fit.delta_fn(mode)
    # Remove the excess hazard already present at ref bits from the measured base rate.
    ref = fit.ref_bits
    at_ref = delta(Phase.READ, ref) + delta(Phase.ACT, ref) + (delta(Phase.THINK, ref) if think else 0.0)
    base = max(fit.base_hazard - at_ref, 1e-6)

    sched = adaptive_schedule(fit, bbar, levels, alloc_hops, think, mode)
    table = {Phase.parse(k): v for k, v in sched["table"].items()}
    asym = predict_horizon_gain(delta, base, int(bbar), table, think)
    uniform = {p: int(bbar) for p in Phase}

    per_hops = []
    for h in sweep_hops:
        s_h = adaptive_schedule(fit, bbar, levels, int(h), think, mode)
        t_h = {Phase.parse(k): v for k, v in s_h["table"].items()}
        la = horizongym_line(delta, base, t_h, think).log_s(h)
        lu = horizongym_line(delta, base, uniform, think).log_s(h)
        per_hops.append({"hops": int(h), "table": s_h["table"], "avg_bits": s_h["avg_bits_at_alloc_hops"],
                         "log_S_adaptive": la, "log_S_uniform": lu, "log_ratio": la - lu})
    pred = dict(asym)
    pred["asymptotic_log_ratio_slope_per_hop"] = asym["log_ratio_slope_per_hop"]
    if len(per_hops) >= 2:
        _, slope = weighted_linfit([r["hops"] for r in per_hops], [r["log_ratio"] for r in per_hops])
        pred["log_ratio_slope_per_hop"] = slope
    return {
        "bbar": bbar,
        "uniform_bits": int(bbar),
        "adaptive_table": sched["table"],
        "alloc_hops": alloc_hops,
        "avg_bits_at_alloc_hops": sched["avg_bits_at_alloc_hops"],
        "per_hops": per_hops,
        "base_hazard_per_turn": base,
        "curve_mode": mode,
        "prediction": pred,
        "diagnostics": continuous_diagnostics(fit, bbar, alloc_hops, think),
    }


def adaptive_policy_for(prereg: dict):
    """A function hops -> PhaseTablePolicy using the per-horizon tables when present."""
    from hazprec.scheduler.policies import PhaseTablePolicy

    tables = {r["hops"]: r["table"] for r in prereg.get("per_hops", [])}

    def make(hops: int) -> PhaseTablePolicy:
        return PhaseTablePolicy(tables.get(int(hops), prereg["adaptive_table"]))

    return make
