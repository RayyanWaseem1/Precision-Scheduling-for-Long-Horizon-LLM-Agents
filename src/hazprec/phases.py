"""Phase taxonomy shared by the environment, harness, scheduler and analysis.

A *turn* is one model response plus the observation that follows it. Each turn is
split into segments, and every segment runs at a single weight precision:

    READ   prefill of new context (the previous tool observation, or the task prompt)
    THINK  free-form reasoning decoded before the tool call
    ACT    the tool call itself (name + arguments)
    STOP   a tool call that ends the episode (``submit``); scheduled like ACT but
           measured separately because its failure mode differs

SYSTEM is the very first prefill (system + user prompt).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Phase(str, Enum):
    SYSTEM = "system"
    READ = "read"
    THINK = "think"
    ACT = "act"
    STOP = "stop"

    @classmethod
    def parse(cls, value: "str | Phase") -> "Phase":
        return value if isinstance(value, Phase) else cls(str(value).lower())


DECODE_PHASES = (Phase.THINK, Phase.ACT, Phase.STOP)
PREFILL_PHASES = (Phase.SYSTEM, Phase.READ)


@dataclass(frozen=True)
class SegmentContext:
    """Everything a precision policy may look at before choosing bits for a segment."""

    phase: Phase
    turn_index: int
    est_tokens: int
    # Phase predicted for the upcoming tool call (ACT or STOP) by a deployable hinter.
    hint_phase: Phase | None = None
    # Optional features for learned policies (e.g. last hidden state as a numpy array).
    features: Any = None
    extra: dict = field(default_factory=dict)


@dataclass
class SegmentRecord:
    """What actually happened in one segment (logged into trajectories)."""

    phase: Phase
    bits: int
    tokens: int
    text: str = ""
    boundary_tokens: int = 0

    def to_dict(self) -> dict:
        return {"phase": self.phase.value, "bits": self.bits, "tokens": self.tokens, "text": self.text, "boundary_tokens": self.boundary_tokens}
