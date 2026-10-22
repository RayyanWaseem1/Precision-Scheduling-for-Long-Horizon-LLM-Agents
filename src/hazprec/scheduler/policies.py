"""
Precision policies (proposal Section 6.4)

P0 UniformPolicy            same bits for every segment
P1 PhaseTablePolicy         bits per phase, typically from ``theory.phase_table``
P2 PacedPolicy              per-segment water-filling with an online budget price 
P3 OraclePhasePolicy        P1, but told the true phase of the upcoming action
"""

from __future__ import annotations 

import math 
from abc import ABC, abstractmethod
from typing import Any, Callable, Mapping, Sequence 

from hazprec.phases import Phase, SegmentContext, SegmentRecord 

class PrecisionPolicy(ABC):
    name = "policy"

    def begin_episode(self) -> None:
        pass 

    @abstractmethod
    def bits_for(self, ctx: SegmentContext) -> int: ...

    def observe(self, ctx: SegmentContext, record: SegmentRecord) -> None:
        pass 

    def describe(self) -> dict[str, Any]:
        return {"name": self.name}

class UniformPolicy(PrecisionPolicy):
    name = "uniform"

    def __init__(self, bits: int):
        self.bits = int(bits)

    def bits_for(self, ctx: SegmentContext) -> int:
        return self.bits 

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "bits": self.bits}

def _table_lookup(table: Mapping[Phase, int], phase: Phase, default: int) -> int:
    if phase in table:
        return table[phase]
    if phase is Phase.SYSTEM and Phase.READ in table:
        return table[Phase.READ]
    if phase is Phase.STOP and Phase.ACT in table:
        return table[Phase.ACT]
    return default 

class PhaseTablePolicy(PrecisionPolicy):
    """ Bits by phase. The tool-call segment uses STOP bits when the harness's hinter
    predicts the all will end the episode, ACT bits otherwise"""

    name = "phase_table"

    def __init__(self, table: Mapping[str | Phase, int], default: int | None = None):
        self.table = {Phase.parse(k): int(v) for k, v in table.items()}
        self.default = int(default if default is not None else max(self.table.values()))

    def _phase_for(self, ctx: SegmentContext) -> Phase:
        if ctx.phase in (Phase.ACT, Phase.STOP) and ctx.hint_phase is not None:
            return ctx.hint_phase 
        return ctx.phase 

    def bits_for(self, ctx: SegmentContext) -> int:
        return _table_lookup(self.table, self._phase_for(ctx), self.default)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "table": {p.value: b for p, b in self.table.items()}}


class OraclePhasePolicy(PhaseTablePolicy):
    """ Upper-bound variant: uses the true phase of the upcoming action, which the 
    harness passes as ``ctx.extra['oracle_phase']`` when an oracle is available"""

    name = "oracle_phase"

    def _phase_for(self, ctx: SegmentContext) -> Phase:
        if ctx.phase in (Phase.ACT, Phase.STOP) and "oracle_phase" in ctx.extra:
            return ctx.extra["oracle_phase"]
        return super()._phase_for(ctx)

class PacedPolicy(PrecisionPolicy):
    """ Online water-filling: b = round(1/2 log2(s_hat / c_hat) + beta), clipped to the 
    available levels, where beta is nudged after every segment so the token-weighted 
    average precision tracks ``target_bbar``.
    
    ``sensitivity`` maps a SegmentContext to an estimated sensitivity: a per-phase prior
    by default, or a learned probe on ``ctx.features``.
    """

    name = "paced"

    def __init__(
        self,
        levels: Sequence[int],
        target_bbar: float, 
        sensitivity: Callable[[SegmentContext], float] | Mapping[str | Phase, float],
        beta0: float | None = None,
        eta: float = 0.5,
        ref_ratio: float = 1.0,
    ):
        # ``ref_ratio`` is a typical s/c value; centering on it makes beta start near b-bar
        self.ref_ratio = float(ref_ratio)
        self.levels = tuple(sorted(levels))
        self.target = float(target_bbar)
        if callable(sensitivity):
            self.sens_fn = sensitivity 
        else:
            prior = {Phase.parse(k): float(v) for k, v in sensitivity.items()}
            self.sens_fn = lambda ctx: prior.get(
                ctx.hint_phase if ctx.phase in (Phase.ACT, Phase.STOP) and ctx.hint_phase else ctx.phase,
                prior.get(Phase.READ, 1.0),
            )
        self.beta0 = beta0 if beta0 is not None else target_bbar 
        self.eta = eta 
        self.begin_episode()

    def begin_episode(self) -> None:
        self.beta = self.beta0
        self.bit_tokens = 0.0
        self.tokens = 0.0

    def _snap(self, b: float) -> int:
        lo, hi = self.levels[0], self.levels[-1]
        b = min(max(b, lo), hi)
        return min(self.levels, key = lambda lv: (abs(lv - b), lv))

    def bits_for(self, ctx: SegmentContext) -> int:
        s = max(self.sens_fn(ctx), 1e-12)
        c = max(ctx.est_tokens, 1)
        return self._snap(0.5 * math.log2(s / c / self.ref_ratio) + self.beta)

    def observe(self, ctx: SegmentContext, record: SegmentRecord) -> None:
        self.bit_tokens += record.bits * record.tokens 
        self.tokens += record.tokens
        avg = self.bit_tokens / max(self.tokens, 1.0)
        self.beta += self.eta * (self.target - avg)

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "target_bbar": self.target, "levels": list(self.levels), "eta": self.eta}
