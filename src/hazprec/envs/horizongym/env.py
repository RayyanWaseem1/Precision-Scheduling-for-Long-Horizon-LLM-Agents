"""
HorizonGym: a stateful tool world where every step is checkable against an oracle
"""

from __future__ import annotations 

from dataclasses import dataclass, field 
from typing import Any 

from hazprec.agent.toolcall import ToolCall 
from hazprec.phases import Phase 

from .config import HorizonGymConfig
from .generator import FIELD_ORDER, Task, generate_task 

_FIELD_DOCS = {
    "ack": "ack (str) = the record's `token` value (NOT `old_token`)",
    "seq": "seq (int) = the record's position in the chain, starting at 1",
    "src": "src (str) = the id of the record you read just before this one, or START for the first",
}

def system_prompt(config: HorizonGymConfig) -> str:
    fields = FIELD_ORDER[: config.n_fields]
    update_sig = ", ".join(["record_id: str"] + [f"{f}: {'int' if f == 'seq' else 'str'}" for f in fields])
    return (
        "You are an operations agent working in a records system. Call exactly one tool per turn. \n"
        "Tools: \n"
        "- read(record_id: str): returns the record's fields. \n"
        f"- update({update_sig}): writes an acknowledgement to a record. \n"
        "- submit(count: int): finishes the task. \n"
        "Respond with brief reasoning, then exactly one tool call in this format: \n"
        '<tool_call>{"name": "<tool>", "arguments": {...}}</tool_call>'
    )

def user_prompt(task: Task) -> str:
    cfg = task.config 
    fields = FIELD_ORDER[: cfg.n_fields]
    field_lines = "\n".join(f" - {_FIELD_DOCS[f]}" for f in fields)
    return (
        f"Start at record {task.start_id}. Follow the chain of `next` links until you reach a record "
        "whose `next` is END. \n"
        "For every record, first read it, then call update on that same record with: \n"
        f"{field_lines}\n"
        "After updating the record whose `next` is END, call submit with count = the number of "
        "records you visited."
    )

def _norm(value: Any) -> Any:
    if isinstance(value, bool):
        return value 
    if isinstance(value, (int, float)):
        return int(value) if float(value).is_integer() else value 
    if isinstance(value, str):
        s = value.strip()
        try: 
            return int(s)
        except ValueError:
            return s
    return value 

def check_action(call: ToolCall | None, expected: ToolCall) -> tuple[bool, str]:
    """ Return (correct, reason). Extra arguments are ignored; missing or wrong ones fail"""
    if call is None:
        return False, "could not parse a tool call"
    if call.name != expected.name:
        return False, f"expected tool '{expected.name}', got '{call.name}'"
    for key, want in expected.arguments.items():
        if key not in call.arguments:
            return False, f"missing argument '{key}'"
        if _norm(call.arguments[key]) != _norm(want):
            return False, f"wrong value for '{key}'"
    return True, "ok"


@dataclass 
class StepResult:
    correct: bool 
    reason: str 
    observation: str 
    done: bool 
    success: bool 
    step_index: int #index into the oracle plan that this action was checked against
    expected: ToolCall 
    expected_phase: Phase 
    retries_used: int = 0 

@dataclass 
class EnvState:
    step: int = 0 
    retries: int = 0 
    done: bool = False 
    success: bool = False 
    errors: int = 0 
    log: list[dict] = field(default_factory = list) 

class HorizonGymEnv:
    def __init__(self, config: HorizonGymConfig | None = None):
        self.config = config or HorizonGymConfig()
        self.task: Task | None = None 
        self.state = EnvState()

    # lifecycle #
    def reset(self, seed: int) -> Task:
        self.task = generate_task(self.config, seed)
        self.state = EnvState()
        return self.task 

    def messages(self) -> list[dict]:
        assert self.task is not None, "call reset() first"
        return [
            {"role": "system", "content": system_prompt(self.config)},
            {"role": "user", "content": user_prompt(self.task)},
        ]

    # oracle access (for analysis, simulated agents, and replay) #
    @property 
    def expected(self) -> ToolCall:
        assert self.task is not None
        return self.task.oracle[self.state.step]

    @property 
    def expected_phase(self) -> Phase:
        assert self.task is not None 
        return self.task.oracle_phases[self.state.step]

    def get_state(self) -> EnvState:
        s = self.state 
        return EnvState(s.step, s.retries, s.done, s.success, s.errors, list(s.log))

    def set_state(self, state: EnvState) -> None:
        self.state = EnvState(state.step, state.retries, state.done, state.success, state.errors, list(state.log))

    # transition #
    def step(self, call: ToolCall | None) -> StepResult:
        assert self.task is not None, "call reset() first"
        if self.state.done:
            raise RuntimeError("episode is over")
        idx = self.state.step 
        expected = self.task.oracle[idx]
        exp_phase = self.task.oracle_phases[idx]
        ok, reason = check_action(call, expected)
        self.state.log.append({"step": idx, "correct": ok, "reason": reason})

        if ok:
            obs = self._execute(expected)
            retries = self.state.retries 
            self.state.step += 1
            self.state.retries = 0 
            if expected.name == "submit":
                self.state.done = True 
                self.state.success = True 
            return StepResult(True, reason, obs, self.state.done, self.state.success, idx, expected, exp_phase, retries)

        self.state.errors += 1
        if self.config.error_mode == "fatal":
            self.state.done = True 
            return StepResult(False, reason, f"ERROR: {reason}. Episode failed." , True, False, idx, expected, exp_phase)

        self.state.retries += 1
        if self.state.retries > self.config.max_retries:
            self.state.done = True 
            return StepResult(
                False, reason, f"ERROR: {reason}. Too many failed attempts; episode failed.",
                True, False, idx, expected, exp_phase, self.state.retries,
            )
        return StepResult(
            False, reason, f"ERROR: {reason}. The action was rejected; try again.",
            False, False, idx, expected, exp_phase, self.state.retries,
        )

    def _execute(self, call: ToolCall) -> str:
        assert self.task is not None 
        if call.name == "read":
            rec = self.task.records[call.arguments["record_id"]]
            return rec.render(self.config.decoy_field)
        if call.name == "update":
            return f"OK: record {call.arguments['record_id']} updated."
        return "OK: task submitted."
