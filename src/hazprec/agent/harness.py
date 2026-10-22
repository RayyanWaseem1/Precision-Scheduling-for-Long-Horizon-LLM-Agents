"""
Agent loop with per-segment precision.

Each turn: prefill the new context (READ, or SYSTEM on the first turn), decode
reasoning up to the opening tool-call tag (THINK), then decode the tool call (ACT, or
STOP when the hinter predicts a terminating call). Every segment asks the policy for 
its bits, and every segment is logged with its bits and token count.

Each turn also records how its decode segments ended: "tag" (the expected tag was
emitted), "eos" (the model ended its turn), "cap" (the token budget ran out), or, for
ACT, "skipped" (no tool call was decoded because the model ended its turn in THINK).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping 

from hazprec.envs.horizongym import HorizonGymEnv
from hazprec.envs.horizongym.hint import HorizonGymStopHinter
from hazprec.phases import Phase, SegmentContext, SegmentRecord 
from hazprec.quant.backends.base import GenResult, PrecisionBackend
from hazprec.scheduler.policies import PrecisionPolicy 

from .toolcall import CLOSE_TAG, OPEN_TAG, ToolCall, parse_tool_call 


def end_reason(g: GenResult, tag: str) -> str:
    """How a decode segment ended: its expected tag, end of turn, or the token cap."""
    if g.stop_hit == tag:
        return "tag"
    return "eos" if g.eos else "cap"


@dataclass 
class TurnRecord:
    turn: int 
    segments: list[SegmentRecord]
    call: ToolCall | None 
    correct: bool 
    reason: str 
    step_index: int 
    expected_phase: Phase 
    sched_phase: Phase 
    done: bool 
    think_end: str | None = None  # None when THINK is disabled
    act_end: str | None = None

    def to_dict(self, with_text: bool = False) -> dict:
        return {
            "turn": self.turn, 
            "segments": [
                {k: v for k, v in s.to_dict().items() if with_text or k != "text"} for s in self.segments
            ],
            "call": self.call.to_dict() if self.call else None,
            "correct": self.correct,
            "reason": self.reason, 
            "step_index": self.step_index,
            "expected_phase": self.expected_phase.value,
            "sched_phase": self.sched_phase.value,
            "done": self.done,
            "think_end": self.think_end,
            "act_end": self.act_end,
        }


@dataclass 
class Trajectory:
    seed: int 
    success: bool 
    turns: list[TurnRecord]
    policy: dict 
    env_config: dict 
    meta: dict = field(default_factory = dict)

    @property 
    def tokens(self) -> int:
        return sum(s.tokens for t in self.turns for s in t.segments)

    @property
    def bit_tokens(self) -> float:
        return float(sum(s.bits * s.tokens for t in self.turns for s in t.segments))

    @property 
    def avg_bits(self) -> float:
        return self.bit_tokens / max(self.tokens, 1)

    @property 
    def first_failure_step(self) -> int | None:
        for t in self.turns:
            if not t.correct:
                return t.step_index
        return None 

    def to_dict(self, with_text: bool = False) -> dict:
        return {
            "seed": self.seed,
            "success": self.success,
            "n_turns": len(self.turns),
            "tokens": self.tokens,
            "avg_bits": self.avg_bits,
            "first_failure_step": self.first_failure_step,
            "policy": self.policy,
            "env": self.env_config,
            "meta": self.meta,
            "turns": [t.to_dict(with_text) for t in self.turns],
        }

class AgentHarness:
    def __init__(
        self,
        backend: PrecisionBackend,
        policy: PrecisionPolicy,
        think: bool = True, 
        max_think_tokens: int = 160, 
        max_act_tokens: int = 96,
        hinter_factory: Callable[[], Any] | None = HorizonGymStopHinter,
        pass_oracle_phase: bool = False, 
        chars_per_token: float = 4.0,
    ):
        self.backend = backend 
        self.policy = policy 
        self.think = think 
        self.max_think_tokens = max_think_tokens
        self.max_act_tokens = max_act_tokens
        self.hinter_factory = hinter_factory 
        self.pass_oracle_phase = pass_oracle_phase
        self.chars_per_token = chars_per_token
        self.hinter: Any = None 

    ####

    def _bits(self, ctx: SegmentContext, override: Mapping[Phase, int] | None) -> int:
        if override:
            key = Phase.READ if ctx.phase in (Phase.SYSTEM, Phase.READ) else (
                Phase.ACT if ctx.phase in (Phase.ACT, Phase.STOP) else ctx.phase)
            if key in override:
                return int(override[key])
        return self.policy.bits_for(ctx)

    def _record(self, ctx: SegmentContext, seg: SegmentRecord, override) -> SegmentRecord:
        if not override:
            self.policy.observe(ctx, seg)
        return seg 

    def run_turn(
        self,
        env: HorizonGymEnv,
        messages: list[dict],
        turn: int,
        bits_override: Mapping[Phase, int] | None = None,
    ) -> TurnRecord:
        segments: list[SegmentRecord] = []
        prefill_phase = Phase.SYSTEM if turn == 0 else Phase.READ
        est = int(len(messages[-1].get("content", "")) / self.chars_per_token) + 1

        # READ / SYSTEM: prefill the new context 
        ctx = SegmentContext(prefill_phase, turn, est)
        b = self._bits(ctx, bits_override)
        n = self.backend.ingest(messages, b) 
        segments.append(self._record(ctx, SegmentRecord(prefill_phase, b, n), bits_override))

        hint = self.hinter.hint() if self.hinter is not None else None 
        extra = {"oracle_phase": env.expected_phase} if self.pass_oracle_phase else {}

        # THINK: free-form reasoning up to the tool-call tag 
        think_text, overflow, need_open = "", "", True 
        think_end: str | None = None
        if self.think:
            ctx = SegmentContext(Phase.THINK, turn, self.max_think_tokens // 2, hint, extra = extra)
            b = self._bits(ctx, bits_override)
            g = self.backend.generate(b, self.max_think_tokens, stop = (OPEN_TAG, ))
            think_text, overflow = g.text, g.overflow
            need_open = g.stop_hit != OPEN_TAG
            think_end = end_reason(g, OPEN_TAG)
            segments.append(self._record(ctx, SegmentRecord(Phase.THINK, b, g.n_tokens, g.text, g.boundary_tokens), bits_override))

        # ACT / STOP: the tool call 
        sched_phase = hint if hint is not None else Phase.ACT
        ctx = SegmentContext(sched_phase, turn, self.max_act_tokens // 3, hint, extra = extra)
        b = self._bits(ctx, bits_override)
        if think_end == "eos":
            # The model ended its turn without opening a tool call. Forcing the tag after
            # its end-of-turn token would write an off-template continuation into the KV
            # history of every later turn, so the turn has no tool call (a format failure).
            act_text, n_act, act_boundary, act_end = "", 0, 0, "skipped"
        elif overflow and CLOSE_TAG in overflow:
            act_text, n_act = overflow.split(CLOSE_TAG, 1)[0], 0
            act_boundary, act_end = 0, "tag"
        else:
            g = self.backend.generate(b, self.max_act_tokens, stop = (CLOSE_TAG,), prefix = OPEN_TAG if need_open else "")
            act_text, n_act = overflow + g.text, g.n_tokens
            act_boundary, act_end = g.boundary_tokens, end_reason(g, CLOSE_TAG)
        segments.append(self._record(ctx, SegmentRecord(sched_phase, b, n_act, act_text, act_boundary), bits_override))

        if act_end == "skipped":
            call, content = None, think_text.strip()
        else:
            call = parse_tool_call(OPEN_TAG + act_text + CLOSE_TAG)
            content = (think_text.strip() + "\n" if think_text.strip() else "") + OPEN_TAG + act_text.strip() + CLOSE_TAG
        messages.append({"role": "assistant", "content": content})

        result = env.step(call)
        messages.append({"role": "tool", "content": result.observation})
        if self.hinter is not None:
            self.hinter.update(call, result.observation, result.correct)
        return TurnRecord(
            turn, segments, call, result.correct, result.reason, result.step_index,
            result.expected_phase, sched_phase, result.done, think_end, act_end,
        )

    def run_episode(self, env: HorizonGymEnv, seed: int, max_turns: int | None = None,
                    meta: dict | None = None) -> Trajectory:
        env.reset(seed)
        self.backend.reset(seed)
        self.backend.bind_env(env)
        self.policy.begin_episode()
        self.hinter = self.hinter_factory() if self.hinter_factory else None 
        messages = env.messages()
        cfg = env.config 
        limit = max_turns or cfg.horizon * (1 + (cfg.max_retries if cfg.error_mode == "recoverable" else 0)) + 2
        turns: list[TurnRecord] = []
        while not env.state.done and len(turns) < limit:
            turns.append(self.run_turn(env, messages, len(turns)))
        return Trajectory(seed, env.state.success, turns, self.policy.describe(), cfg.to_dict(), dict(meta or {}))