"""
Procedural task generation for HorizonGym (deterministic given a seed)
"""

from __future__ import annotations

import random 
from dataclasses import dataclass 

from hazprec.agent.toolcall import ToolCall 
from hazprec.phases import Phase 

from .config import HorizonGymConfig

# Unambiguous characters only (no 0/O, 1/I/L) so a "misread" is a real error, not OCR noise 
_TOKEN_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"
_FILLER_WORDS = (
    "amber delta river copper signal harbor lattice meadow quartz ridge summit timber "
    "vector willow beacon cinder falcon glacier harvest island juniper kernel lantern "
    "marble nectar orbit pillar quiver raven saddle thistle umber valley wander yonder"
).split()
FIELD_ORDER = ("ack", "seq", "src")

@dataclass 
class Record:
    record_id: str 
    token: str 
    old_token: str 
    next_id: str # "END" for the last record 
    status: str 
    notes: list[str]

    def render(self, decoy_field: bool) -> str:
        lines = [f"RECORD {self.record_id}", f"status: {self.status}", f"token: {self.token}"]
        if decoy_field:
            lines.append(f"old_token: {self.old_token}")
        lines.append(f"next: {self.next_id}")
        lines += [f"note: {n}" for n in self.notes]
        return "\n".join(lines)

@dataclass 
class Task:
    seed: int 
    config: HorizonGymConfig
    start_id: str 
    chain: list[Record]
    oracle: list[ToolCall]
    oracle_phases: list[Phase] 

    @property
    def records(self) -> dict[str, Record]:
        return {r.record_id: r for r in self.chain}


def _token(rng: random.Random, n: int) -> str:
    return "".join(rng.choice(_TOKEN_ALPHABET) for _ in range(n))

def generate_task(config: HorizonGymConfig, seed: int) -> Task:
    rng = random.Random(seed)
    ids: set[str] = set() 
    chain: list[Record] = []
    while len(ids) < config.n_hops:
        ids.add(f"r_{rng.randint(1000, 9999)}")
    order = sorted(ids)
    rng.shuffle(order)
    for i, rid in enumerate(order):
        tok = _token(rng, config.token_len)
        old = _token(rng, config.token_len)
        while old == tok:
            old = _token(rng, config.token_len)
        notes = [" ".join(rng.choice(_FILLER_WORDS) for _ in range(8)) for _ in range(config.filler_lines)]
        nxt = order[i + 1] if i + 1 < len(order) else "END"
        chain.append(Record(rid, tok, old, nxt, rng.choice(["open", "pending", "active"]), notes))

    fields = FIELD_ORDER[: config.n_fields]
    oracle: list[ToolCall] = []
    phases: list[Phase] = []
    prev = "START"
    for i, rec in enumerate(chain):
        oracle.append(ToolCall("read", {"record_id": rec.record_id}))
        phases.append(Phase.ACT)
        args: dict = {"record_id": rec.record_id}
        values = {"ack": rec.token, "seq": i + 1, "src": prev}
        for f in fields:
            args[f] = values[f]
        oracle.append(ToolCall("update", args))
        phases.append(Phase.ACT)
        prev = rec.record_id 
    oracle.append(ToolCall("submit", {"count": len(chain)}))
    phases.append(Phase.STOP)
    return Task(seed, config, chain[0].record_id, chain, oracle, phases)