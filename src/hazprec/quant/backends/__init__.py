"""Backends and a config-driven factory. Heavy dependencies are imported lazily."""

from __future__ import annotations

from typing import Any

from .base import GenResult, PrecisionBackend, TokenBackend
from .simulated import HazardModel, SimulatedBackend


def make_backend(cfg: dict[str, Any], seed: int = 0) -> PrecisionBackend:
    kind = cfg.get("kind", "sim")
    levels = tuple(cfg.get("levels", (3, 4, 5, 6, 8)))
    ref_bits = int(cfg.get("ref_bits", max(levels)))
    if kind == "sim":
        sim = cfg.get("sim", {})
        hazard = HazardModel(
            sensitivity=dict(sim.get("sensitivity", HazardModel().sensitivity)),
            base_hazard=float(sim.get("base_hazard", 0.002)),
        )
        return SimulatedBackend(
            hazard, levels, ref_bits, int(sim.get("think_tokens", 40)), int(sim.get("act_tokens", 24))
        )
    if kind == "hf":
        from .hf_fakequant import HFFakeQuantBackend

        hf = cfg.get("hf", {})
        return HFFakeQuantBackend(
            hf["model"],
            levels=levels,
            ref_bits=ref_bits,
            group_size=int(hf.get("group_size", 128)),
            device=hf.get("device"),
            dtype=hf.get("dtype", "bfloat16"),
            temperature=float(hf.get("temperature", 0.7)),
            chat_template_kwargs=hf.get("chat_template_kwargs"),
            tool_role=hf.get("tool_role", "tool"),
            seed=seed,
        )
    if kind == "mlx":
        from .mlx_backend import MLXBackend

        mc = cfg.get("mlx", {})
        return MLXBackend(
            mc["model"],
            levels=levels,
            ref_bits=ref_bits,
            group_size=int(mc.get("group_size", 64)),
            cache_dir=mc.get("cache_dir", "~/.cache/hazprec/mlx"),
            temperature=float(mc.get("temperature", 0.7)),
            chat_template_kwargs=mc.get("chat_template_kwargs"),
            tool_role=mc.get("tool_role", "tool"),
            seed=seed,
        )
    raise ValueError(f"unknown backend kind: {kind}")


__all__ = ["GenResult", "PrecisionBackend", "TokenBackend", "HazardModel", "SimulatedBackend", "make_backend"]
