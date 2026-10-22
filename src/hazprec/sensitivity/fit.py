"""
Fit per-phase excess-hazard curves from atlas records (proposal Section 6.3).

Phase keys: the READ probe on turn 0 is the ``system`` prefill; the ACT probe is
``act`` or ``stop`` depending on which oracle action the turn was checked against.
"""

from __future__ import annotations

import math 
from collections import defaultdict
from dataclasses import dataclass, field 
from typing import Iterable
from statistics import NormalDist 

import numpy as np 

from hazprec.eval.stats import wilson_interval
from hazprec.phases import Phase 

def phase_key(rec: dict) -> str:
    probe = rec["probe"]
    if probe == Phase.READ.value:
        return Phase.SYSTEM.value if rec["turn"] == 0 else Phase.READ.value 
    if probe == Phase.ACT.value:
        return rec["expected_phase"]
    return probe 

@dataclass
class PhaseCurve:
    phase: str 
    s: float #fitted sensitivity in Delta(b) = s * 2^(-2b)
    log2_slope: float | None #fitted d log2(Delta) / d b; the model predicts -2
    table: dict[int, dict] = field(default_factory = dict) # bits -> {rate, delta, lo, hi, n, failures}

    def delta(self, bits: int, mode: str = "parametric") -> float:
        if bits >= 16:
            return 0.0
        if mode == "empirical" and bits in self.table:
            return max(self.table[bits]["delta"], 0.0)
        return self.s * 2.0 ** (-2 * bits)

    def to_dict(self) -> dict:
        return {"phase": self.phase, "s": self.s, "log2_slope": self.log2_slope,
                "table": {str(k): v for k, v in self.table.items()}}

@dataclass 
class AtlasFit:
    base_hazard: float #per turn failure rate with every segment at ref bits 
    base_n: int 
    ref_bits: int 
    curves: dict[str, PhaseCurve]
    tokens: dict[str, float] # mean tokens per segment by phase (reference runs)
    freq_per_hop: dict[str, float] #segments of each phase per hop (reference runs)
    n_tasks: int 

    def delta_fn(self, mode: str = "parametric"):
        def fn(phase, bits: int) -> float:
            key = Phase.parse(phase).value
            c = self.curves.get(key)
            if c is None and key == Phase.SYSTEM.value:
                c = self.curves.get(Phase.READ.value)
            if c is None and key == Phase.STOP.value:
                c = self.curves.get(Phase.ACT.value)
            return 0.0 if c is None else c.delta(bits, mode)
        return fn 

    def to_dict(self) -> dict:
        return {
            "base_hazard": self.base_hazard, "base_n": self.base_n, "ref_bits": self.ref_bits,
            "curves": {k: v.to_dict() for k, v in self.curves.items()},
            "tokens": self.tokens, "freq_per_hop": self.freq_per_hop, "n_tasks": self.n_tasks,
        }

def _fit_s(bits: list[int], deltas: list[float], ns: list[int]) -> float:
    x = np.array([2.0 ** (-2 * b) for b in bits])
    y = np.array(deltas)
    w = np.array(ns, float)
    den = float(np.sum(w * x * x))
    return max(float(np.sum(w * x * y)) / den, 0.0) if den > 0 else 0.0

def _fit_log2_slope(bits: list[int], deltas: list[float], ns: list[int]) -> float | None:
    pts = [(b, d, n) for b, d, n in zip(bits, deltas, ns) if d > 0]
    if len(pts) < 2:
        return None
    b = np.array([p[0] for p in pts], float)
    y = np.log2([p[1] for p in pts])
    w = np.array([p[2] for p in pts], float)
    bm = np.average(b, weights = w)
    ym = np.average(y, weights = w)
    den = float(np.sum(w * (b - bm) ** 2))
    return float(np.sum(w * (b - bm) * (y - ym)) / den) if den > 0 else None 

def fit_atlas(records: Iterable[dict]) -> AtlasFit:
    records = list(records)
    probes = [r for r in records if r.get("type") == "probe"]
    refs = [r for r in records if r.get("type") == "reference"]
    if not probes:
        raise ValueError("no probe records in atlas")
    ref_bits = int(probes[0]["ref_bits"])
    references = {}
    seen = set()
    for r in probes:
        if r.get("task_seed") is None or r.get("turn") is None:
            raise ValueError("atlas records require task_seed and turn for matched estimation")
        if int(r["ref_bits"]) != ref_bits:
            raise ValueError("atlas mixes reference precisions")
        if type(r["n"]) is not int or r["n"] <= 0 or type(r["failures"]) is not int or not 0 <= r["failures"] <= r["n"]:
            raise ValueError("invalid probe sample/failure counts")
        key = (r["task_seed"], r["turn"], r["probe"], r["bits"])
        if key in seen:
            raise ValueError(f"duplicate atlas record: {key}")
        seen.add(key)
        if r["probe"] == "ref":
            if r["bits"] != ref_bits:
                raise ValueError("reference record has wrong precision")
            references[(r["task_seed"], r["turn"])] = r
    if not references:
        raise ValueError("atlas has no reference probes")
    base_n = sum(r["n"] for r in references.values())
    base_rate = sum(r["failures"] for r in references.values()) / base_n
    pooled = defaultdict(lambda: defaultdict(list))
    for r in probes:
        if r["probe"] == "ref":
            continue
        ref = references.get((r["task_seed"], r["turn"]))
        if ref is None or ref["expected_phase"] != r["expected_phase"]:
            raise ValueError("probe has no matching task/turn/phase reference")
        pooled[phase_key(r)][int(r["bits"])].append((r, ref))

    curves = {}
    for key, by_bits in pooled.items():
        table, bits_l, deltas, ns = {}, [], [], []
        for b, pairs in sorted(by_bits.items()):
            n = sum(r["n"] for r, _ in pairs)
            f = sum(r["failures"] for r, _ in pairs)
            # Match each reference rate to exactly the contexts/weights of this probe.
            matched_base = sum(r["n"] * ref["failures"] / ref["n"] for r, ref in pairs) / n
            delta = f / n - matched_base
            # Simultaneous Wilson intervals for both rates in every stratum, then
            # project their weighted differences. Conservative, conditional on the
            # observed task contexts; this includes reference sampling uncertainty.
            z = NormalDist().inv_cdf(1 - 0.05 / (4 * len(pairs)))
            lo = hi = 0.0
            for r, ref in pairs:
                pl, ph = wilson_interval(r["failures"], r["n"], z)
                rl, rh = wilson_interval(ref["failures"], ref["n"], z)
                weight = r["n"] / n
                lo += weight * (pl - rh)
                hi += weight * (ph - rl)
            table[b] = {"rate": f / n, "reference_rate": matched_base, "delta": delta,
                        "lo": lo, "hi": hi, "n": n, "failures": f,
                        "ci_method": "matched-stratum Bonferroni-Wilson (conditional contexts)"}
            bits_l.append(b)
            deltas.append(delta)  # retain negative estimates; constrain fitted s only
            ns.append(n)
        curves[key] = PhaseCurve(key, _fit_s(bits_l, deltas, ns), _fit_log2_slope(bits_l, deltas, ns), table)

    # token costs and phase frequencies from the reference trajectories
    tok_sum: dict[str, float] = defaultdict(float)
    tok_cnt: dict[str, int] = defaultdict(int)
    hops_total = 0
    for r in refs:
        hops_total += int(r.get("n_hops", 1))
        for t in r["turns"]:
            for s in t["segments"]:
                p = s["phase"]
                tok_sum[p] += s["tokens"]
                tok_cnt[p] += 1
    tokens = {p: tok_sum[p] / tok_cnt[p] for p in tok_cnt}
    freq = {p: tok_cnt[p] / max(hops_total, 1) for p in tok_cnt}
    return AtlasFit(base_rate, base_n, ref_bits, curves, tokens, freq, len(refs))

def format_fit(fit: AtlasFit) -> str:
    lines = [f"Reference ({fit.ref_bits}-bits) per-turn failure rate: {fit.base_hazard:.4f} (n={fit.base_n}),"
             f" tasks={fit.n_tasks}"]
    lines.append(f"{'phase':<8}{'s':>10}{'log2 slope':>12} excess hazard by bits [95% CI]")
    for key in sorted(fit.curves):
        c = fit.curves[key]
        slope = "n/a" if c.log2_slope is None else f"{c.log2_slope:2f}"
        cells = " ".join(
            f"{b}b:{v['delta']:+.4f}[{v['lo']:+.3f},{v['hi']:+.3f}]" for b, v in sorted(c.table.items())
        )
        lines.append(f"{key:<8}{c.s:>10.3f}{slope:>12} {cells}")
    lines.append("tokens/segment: " + ", ".join(f"{k} = {v:.1f}" for k, v in sorted(fit.tokens.items())))
    lines.append("segments/hop: " + ", ".join(f"{k} = {v:.2f}" for k, v in sorted(fit.freq_per_hop.items())))
    return "\n".join(lines)

def isfine(x: float | None) -> bool:
    return x is not None and math.isfinite(x)