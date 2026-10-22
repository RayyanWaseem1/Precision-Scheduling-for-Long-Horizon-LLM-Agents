"""
Discrete-time survival analysis of a trajectories (proposal Section 7.7)

Every turn on episode survives to is a row "at risk"; the outcome is whether the turn 
failed. A logistic hazard model on turn index, phase and segment precisions estimates
how each factor moves the per-step hazard
"""

from __future__ import annotations 

from collections import defaultdict
from typing import Iterable, Sequence

import numpy as np 

from .stats import fit_logistic, wilson_interval

def at_risk_rows(trajectories: Iterable[dict]) -> list[dict]:
    rows = []
    for tr in trajectories:
        for t in tr["turns"]:
            segs = {s["phase"]: s for s in t["segments"]}
            read = segs.get("read") or segs.get("system")
            act = segs.get("act") or segs.get("stop")
            rows.append({
                "seed": tr["seed"],
                "turn": t["turn"],
                "step_index": t["step_index"],
                "phase": t["expected_phase"],
                "read_bits": read["bits"] if read else None,
                "think_bits": segs["think"]["bits"] if "think" in segs else None,
                "act_bits": act["bits"] if act else None,
                "failed": int(not t["correct"]),
                "policy": tr.get("policy", {}).get("name"),
            })
    return rows 

def hazard_by_turn(trajectories: Iterable[dict]) -> list[dict]:
    """ Empirical per-turn hazard: failures / episodes at risk, with Wilson intervals"""
    at = defaultdict(lambda: [0, 0])
    for r in at_risk_rows(trajectories):
        at[r["turn"]][0] += r["failed"]
        at[r["turn"]][1] += 1
    out = []
    for turn in sorted(at):
        f, n = at[turn] 
        lo, hi = wilson_interval(f, n) 
        out.append({"turn": turn, "at_risk": n, "failures": f, "hazard": f / n, "lo": lo, "hi": hi})
    return out 

def fit_hazard_model(trajectories: Iterable[dict], features: Sequence[str] = ("turn", "is_stop", "act_bits",
                                                                              "read_bits", "think_bits")) -> dict:
    """ Logistic discrete-time hazard model. Bits enter as 2^(-2b) so coefficients
    are on the scale the theory predicts (excess hazard proportional to 2^(-2b))"""
    rows = at_risk_rows(trajectories)
    if not rows:
        raise ValueError("no turns to fit")
    cols = ["intercept"]
    X = [np.ones(len(rows))]
    for f in features:
        if f == "is_stop":
            X.append(np.array([1.0 if r["phase"] == "stop" else 0.0 for r in rows]))
        elif f.endswith("_bits"):
            X.append(np.array([0.0 if r[f] is None else 2.0 ** (-2 * r[f]) for r in rows]))
            f = f.replace("_bits", "_2^-2b")
        else:
            X.append(np.array([float(r[f]) for r in rows]))
        cols.append(f)
    Xm = np.column_stack(X)
    keep = [i for i in range(Xm.shape[1]) if i == 0 or np.std(Xm[:, i]) > 0]
    Xm = Xm[:, keep]
    cols = [cols[i] for i in keep]
    y = np.array([r["failed"] for r in rows], float)
    beta, se = fit_logistic(Xm, y)
    return {"n_rows": len(rows), "events": int(y.sum()),
            "coef": {c: float(b) for c, b in zip(cols, beta)}, "se": {c: float(s) for c, s in zip(cols, se)}}
