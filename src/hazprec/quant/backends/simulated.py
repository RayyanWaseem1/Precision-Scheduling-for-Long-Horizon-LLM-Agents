"""
An oracle agent with a known, controllable hazard model.

It answers every step with the oracle action, except that with probability

    q = 1 - (1 - h0) * prod_phase (1 - s_phase * 2^(-2 * bits_phase))

it corrupts the action the way a compressed model plausibly would (reads the decoy 
toke, gets a count off by one, follows the wrong link, ...). Because the true 
sensitivieis are known, it lets us check that the measurement pipeline recovers them 
and that the theory's predictions hold before spending any GPU time.
"""

from __future__ import annotations 

import json 
import random
from dataclasses import dataclass, field 
from typing import Any, Sequence 

from hazprec.agent.toolcall import CLOSE_TAG, OPEN_TAG, ToolCall
from hazprec.phases import Phase 

from .base import GenResult, PrecisionBackend

@dataclass
class HazardModel:
    # Excess hazard at b bits = sensitivity * 2^(-2b). Keys are phase names
    sensitivity: dict[str, float] = field(
        default_factory=lambda: {"system": 0.3, "read": 0.3, "think": 0.5, "act": 2.0, "stop": 6.0}
    )
    base_hazard: float = 0.002 

    def excess(self, phase: Phase | str, bits: int) -> float:
        key = Phase.parse(phase).value 
        s = self.sensitivity.get(key, 0.0)
        return min(1.0, s * 2.0 ** (-2 * bits)) if bits < 16 else 0.0

    def turn_failure_prob(self, read_bits: int, think_bits: int | None, act_bits: int, act_phase: Phase,
                          read_phase: Phase = Phase.READ) -> float:
        keep = 1.0 - self.base_hazard 
        keep *= 1.0 - self.excess(read_phase, read_bits)
        if think_bits is not None:
            keep *= 1.0 - self.excess(Phase.THINK, think_bits)
        keep *= 1.0 - self.excess(act_phase, act_bits)
        return 1.0 - keep

def corrupt(call: ToolCall, rng: random.Random, env: Any) -> ToolCall:
    """ A plausible wronga ction for ``call``."""
    args = dict(call.arguments)
    if call.name == "read":
        visited = [r.record_id for r in env.task.chain[: max(1, env.state.step // 2)]]
        wrong = rng.choice(visited + [f"r_{rng.randint(1000, 9999)}"])
        if wrong == args["record_id"]:
            wrong = f"r_{rng.randint(1000, 9999)}"
        args["record_id"] = wrong 
        return ToolCall("read", args)
    if call.name == "update":
        keys = [k for k in ("ack", "seq", "src") if k in args]
        k = rng.choice(keys)
        if k == "ack":
            rec = env.task.records[args["record_id"]]
            args["ack"] = rec.old_token 
        elif k == "seq":
            args["seq"] = int(args["seq"]) + rng.choice([-1,1])
        else:
            args["src"] = "START" if args["src"] != "START" else env.task.chain[-1].record_id
        return ToolCall("update", args)
    if call.name == "submit":
        args["count"] = int(args["count"]) + rng.choice([-1,1])
        return ToolCall("submit", args)
    return call 

class SimulatedBackend(PrecisionBackend):
    def __init__(
        self,
        hazard: HazardModel | None = None, 
        levels: Sequence[int] = (3, 4, 5, 6, 8),
        ref_bits: int = 8, 
        think_tokens: int = 40,
        act_tokens: int = 24,
        chars_per_token: float = 4.0,
    ):
        self.hazard = hazard or HazardModel()
        self.levels = tuple(levels)
        self.ref_bits = ref_bits 
        self.think_tokens = think_tokens
        self.act_tokens = act_tokens 
        self.chars_per_token = chars_per_token
        self.env: Any = None 
        self.rng = random.Random(0)
        self._seen = 0
        self._read_bits = ref_bits 
        self._read_phase = Phase.SYSTEM
        self._think_bits: int | None = None 

    def bind_env(self, env: Any) -> None:
        self.env = env 

    def reset(self, seed: int | None = None) -> None:
        self._seen = 0
        self._read_bits = self.ref_bits 
        self._read_phase = Phase.SYSTEM
        self._think_bits = None 
        if seed is not None:
            self.rng = random.Random(seed * 7919 + 17)

    def truncate_to(self, messages: list[dict]) -> None:
        self._seen = min(self._seen, len(messages))

    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)

    def ingest(self, messages: list[dict], bits: int) -> int:
        self.check_bits(bits)
        new = messages[self._seen:]
        self._read_phase = Phase.SYSTEM if len(messages) <= 2 else Phase.READ
        self._seen = len(messages)
        self._read_bits = bits 
        self._think_bits = None 
        chars = sum(len(m.get("content", "")) for m in new)
        return max(1, int(chars / self.chars_per_token))

    def generate(self, bits: int, max_tokens: int, stop: Sequence[str] = (), prefix: str = "") -> GenResult:
        self.check_bits(bits)
        if self.env is None:
            raise RuntimeError("SimulatedBackend needs bind_env(env) before generating")
        if OPEN_TAG in stop: # thinking segment 
            self._think_bits = bits 
            n = min(self.think_tokens, max_tokens)
            return GenResult("Checking the record and choosing the next call.", n, OPEN_TAG, "", False)
        # tool-call segment 
        expected: ToolCall = self.env.expected 
        q = self.hazard.turn_failure_prob(
            self._read_bits, self._think_bits, bits, self.env.expected_phase, self._read_phase
        )
        call = corrupt(expected, self.rng, self.env) if self.rng.random() < q else expected 
        text = json.dumps({"name": call.name, "arguments": call.arguments})
        hit = CLOSE_TAG if CLOSE_TAG in stop else None 
        return GenResult(text, min(self.act_tokens, max_tokens), hit, "", False)

    # The simulated sate is tiny, so snapshots are cheap copies 
    def snapshot(self) -> Any:
        return (self._seen, self._read_bits, self._read_phase, self._think_bits)

    def restore(self, snap: Any) -> None:
        self._seen, self._read_bits, self._read_phase, self._think_bits = snap