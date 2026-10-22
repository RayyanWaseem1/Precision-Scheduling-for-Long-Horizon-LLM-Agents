"""
Horizon sweep and the test of Propositions 2 and 3 (proposal Sections 7.1, 8).

Uniform and adaptive policies run on the *same task seeds* at each horizon, so the
comparison is paired. The log success ration is regressed on the number of hops; its
slope is compared with the pre-registered prediction using a paired bootstrap over seeds
"""

from __future__ import annotations 

import math 
from collections import defaultdict
from dataclasses import replace 
from typing import Callable, Mapping, Sequence 

import numpy as np 

from hazprec.agent.harness import AgentHarness
from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
from hazprec.quant.backends.base import PrecisionBackend 
from hazprec.scheduler.policies import PrecisionPolicy 
from hazprec.utils.io import JsonlWriter, completed_keys
from hazprec.utils.experiment import ensure_experiment 

from .stats import bootstrap, weighted_linfit 

KEY_FIELDS = ("hops", "policy_label", "seed")

def run_horizon_sweep(
    backend: PrecisionBackend,
    env_cfg: HorizonGymConfig,
    hops_list: Sequence[int],
    policies: Mapping[str, PrecisionPolicy | Callable[[int], PrecisionPolicy]],
    n_episodes: int, 
    out_path: str, 
    seed0: int = 100_000,
    think: bool = True,
    harness_kwargs: dict | None = None,
    experiment: dict | None = None,
    prereg_sha: str | None = None,
    progress: bool = True, 
) -> None:
    if experiment is None:
        raise ValueError("resumable runs require experiment=identity(resolved_config)")
    ensure_experiment(out_path, experiment, "horizon", prereg_sha)
    done = completed_keys(out_path, KEY_FIELDS)
    with JsonlWriter(out_path) as w:
        for hops in hops_list:
            env = HorizonGymEnv(replace(env_cfg, n_hops = hops))
            for label, pol in policies.items():
                if not isinstance(pol, PrecisionPolicy):
                    pol = pol(hops) # per-horizon policy factory
                h = AgentHarness(backend, pol, think = think, **(harness_kwargs or {}))
                for i in range(n_episodes):
                    seed = seed0 + i
                    if (hops, label, seed) in done:
                        continue 
                    tr = h.run_episode(env, seed)
                    w.write({
                        "hops": hops, "policy_label": label, "seed": seed, "success": tr.success,
                        "avg_bits": tr.avg_bits, "tokens": tr.tokens, "n_turns": len(tr.turns),
                        "first_failure_step": tr.first_failure_step,
                    })
                if progress:
                    print(f"[horizon] hops = {hops} policy = {label} done", flush = True)

def _smoothed_log_rate(successes: np.ndarray) -> float:
    # Add-half smoothing keeps log() finite when a cell has zero successes
    return math.log((successes.sum() + 0.5) / (len(successes) + 1.0))

def analyze_horizon(records: Sequence[dict], adaptive: str = "adaptive", uniform: str = "uniform",
                    prediction: dict | None = None, per_hops: Sequence[dict] | None = None,
                    n_boot: int = 2000, seed: int = 0) -> dict:
    seen = set()
    by = defaultdict(dict)
    bits = defaultdict(list)
    for r in records:
        key = (r["hops"], r["policy_label"], r["seed"])
        if key in seen:
            raise ValueError(f"duplicate sweep record: {key}")
        seen.add(key)
        by[(r["hops"], r["policy_label"])][r["seed"]] = int(r["success"])
        bits[(r["hops"], r["policy_label"])].append(r["avg_bits"])
    hops_list = sorted({h for h, _ in by})
    paired = {}
    for h in hops_list:
        a, u = by.get((h, adaptive), {}), by.get((h, uniform), {})
        if set(a) != set(u):
            raise ValueError(f"unpaired policy seeds at horizon {h}; finish the sweep")
        seeds = sorted(a)
        if seeds:
            paired[h] = (np.array([a[s] for s in seeds]), np.array([u[s] for s in seeds]))
    if len(paired) < 2:
        raise ValueError("need at least two horizons with paired adaptive/uniform runs")

    predicted = {int(r["hops"]): r["log_ratio"] for r in (per_hops or [])}
    rows = []
    for h, (sa, su) in paired.items():
        rows.append({
            "pred_log_ratio": predicted.get(int(h)),
            "hops": h, "n": len(sa), "S_adaptive": float(sa.mean()), "S_uniform": float(su.mean()),
            "log_ratio": _smoothed_log_rate(sa) - _smoothed_log_rate(su),
            "abs_gain": float(sa.mean() - su.mean()),
            "avg_bits_adaptive": float(np.mean(bits[(h, adaptive)])),
            "avg_bits_uniform": float(np.mean(bits[(h, uniform)])),

        })
    hs = np.array([r["hops"] for r in rows], float)
    lr = np.array([r["log_ratio"] for r in rows])
    intercept, slope = weighted_linfit(hs, lr, [r["n"] for r in rows])

    # A completed sweep uses the same seed IDs at every horizon. Resample seed
    # clusters jointly across horizons and policies; never truncate larger groups.
    seed_sets = [set(by[(h, adaptive)]) for h in paired]
    if any(s != seed_sets[0] for s in seed_sets[1:]):
        raise ValueError("horizons must have identical paired seed sets; finish the sweep")
    n_units = len(seed_sets[0])
    if n_units < 2 or n_boot < 2:
        raise ValueError("need at least two paired seeds and bootstrap samples")

    def stat(idx: np.ndarray) -> float:
        ys = [_smoothed_log_rate(paired[h][0][idx]) - _smoothed_log_rate(paired[h][1][idx])
              for h in sorted(paired)]
        return weighted_linfit(hs, ys, [r["n"] for r in rows])[1]

    lo, hi, _ = bootstrap(stat, n_units, n_boot, seed)
    peak = max(rows, key = lambda r: r["abs_gain"])
    out = {
        "rows": rows,
        "fit": {"intercept": intercept, "slope_per_hop": slope, "slope_ci95": [lo, hi]},
        "empirical_peak_gain_hops": peak["hops"],
    }
    if prediction is not None:
        pred = prediction.get("log_ratio_slope_per_hop")
        out["prediction"] = {
            "slope_per_hop": pred,
            "h_star_hops": prediction.get("h_star_hops"),
            "slope_inside_ci": bool(pred is not None and lo <= pred <= hi),
        }
    return out 

def format_horizon(an: dict) -> str:
    lines = [f"{'hops':>5}{'n':>6}{'S_adapt':>10}{'S_unif':>10}{'log_ratio':>11}{'predicted':>11}{'gain':>8}"
             f"{'bits a/u':>13}"]
    for r in an["rows"]:
        pred = "" if r.get("pred_log_ratio") is None else f"{r['pred_log_ratio']:.3f}"
        lines.append(
            f"{r['hops']:>5}{r['n']:>6}{r['S_adaptive']:>10.3f}{r['S_uniform']:>10.3f}{r['log_ratio']:>11.3f}"
            f"{pred:>11}{r['abs_gain']:>8.3f}{r['avg_bits_adaptive']:>7.2f}/{r['avg_bits_uniform']:.2f}"
        )
    f = an["fit"]
    lines.append(f"slope of log ratio per hop: {f['slope_per_hop']:.4f} 95% CI [{f['slope_ci95'][0]:.4f}, "
                 f"{f['slope_ci95'][1]:.4f}]")
    lines.append(f"largest absolute gain at hops = {an['empirical_peak_gain_hops']}")
    if "prediction" in an:
        p = an["prediction"]
        verdict = "INSIDE" if p["slope_inside_ci"] else "OUTSIDE"
        hstar = "n/a" if p["h_star_hops"] is None else f"{p['h_star_hops']:.1f}"
        lines.append(f"pre-registered slope: {p['slope_per_hop']:.4f} -> {verdict} the CI; "
                     f" predicted peak-gain horizon H* = {hstar} hops")
    return "\n".join(lines) 