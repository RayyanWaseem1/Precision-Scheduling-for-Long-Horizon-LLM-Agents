from __future__ import annotations

from typing import Any

from hazprec.phases import Phase

from .policies import OraclePhasePolicy, PacedPolicy, PhaseTablePolicy, PrecisionPolicy, UniformPolicy


def make_policy(cfg: dict[str, Any]) -> PrecisionPolicy:
    kind = cfg.get("kind", "uniform")
    if kind == "uniform":
        return UniformPolicy(int(cfg["bits"]))
    if kind == "phase_table":
        return PhaseTablePolicy(cfg["table"], cfg.get("default"))
    if kind == "oracle_phase":
        return OraclePhasePolicy(cfg["table"], cfg.get("default"))
    if kind == "paced":
        return PacedPolicy(cfg["levels"], float(cfg["bbar"]), cfg["sensitivity"], eta=float(cfg.get("eta", 0.5)),
                           ref_ratio=float(cfg.get("ref_ratio", 1.0)))
    raise ValueError(f"unknown policy kind: {kind}")


__all__ = ["Phase", "PrecisionPolicy", "UniformPolicy", "PhaseTablePolicy", "OraclePhasePolicy", "PacedPolicy",
           "make_policy"]
