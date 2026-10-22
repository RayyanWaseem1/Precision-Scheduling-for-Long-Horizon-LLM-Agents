"""Sensitivity measurement by replay (proposal Section 6.2).

For each task: run a reference episode at ``ref_bits``. For every turn on its correct
prefix, restore the conversation, environment, hinter and backend cache to just before
that turn (before its observation is prefilled), then regenerate the turn
``n_samples`` times with one segment (READ, THINK or ACT) at a probe precision and the
others at ``ref_bits``. The fraction of regenerated
turns the checker rejects is that turn's failure rate at that setting; subtracting
the reference rate gives the excess hazard.

Records are appended to JSONL as they are produced; a rerun skips completed keys.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator, Sequence

from hazprec.agent.harness import AgentHarness
from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
from hazprec.phases import Phase
from hazprec.quant.backends.base import PrecisionBackend
from hazprec.scheduler.policies import UniformPolicy
from hazprec.utils.io import JsonlWriter, completed_keys, read_jsonl
from hazprec.utils.experiment import ensure_experiment

PROBES = (Phase.READ, Phase.THINK, Phase.ACT)
KEY_FIELDS = ("type", "task_seed", "turn", "probe", "bits")


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256(repr(parts).encode()).hexdigest()[:8], 16)


@dataclass
class _Checkpoint:
    messages: list[dict]
    env_state: Any
    hinter: Any
    backend: Any  # backend snapshot taken before the turn's observation was prefilled


def _reference(harness: AgentHarness, env: HorizonGymEnv, seed: int) -> tuple[Any, list[_Checkpoint]]:
    """Run the reference episode, keeping a checkpoint before every turn."""
    env.reset(seed)
    harness.backend.reset(seed)
    harness.backend.bind_env(env)
    harness.policy.begin_episode()
    harness.hinter = harness.hinter_factory() if harness.hinter_factory else None
    messages = env.messages()
    cps: list[_Checkpoint] = []
    turns = []
    cfg = env.config
    limit = cfg.horizon * (1 + (cfg.max_retries if cfg.error_mode == "recoverable" else 0)) + 2
    while not env.state.done and len(turns) < limit:
        cps.append(_Checkpoint(copy.deepcopy(messages), env.get_state(), copy.deepcopy(harness.hinter),
                               harness.backend.snapshot()))
        turns.append(harness.run_turn(env, messages, len(turns)))
    return turns, cps


def measure_task(
    backend: PrecisionBackend,
    env_cfg: HorizonGymConfig,
    seed: int,
    levels: Sequence[int],
    ref_bits: int,
    n_samples: int,
    probes: Sequence[Phase] = PROBES,
    think: bool = True,
    max_turns: int | None = None,
    done_keys: set | None = None,
    harness_kwargs: dict | None = None,
) -> Iterator[dict]:
    if n_samples <= 0 or (max_turns is not None and max_turns <= 0):
        raise ValueError("sample counts and max_turns must be positive")
    if ref_bits not in levels or any(b not in backend.levels for b in levels):
        raise ValueError("atlas precisions must be available and include the reference")
    probes = tuple(p for p in probes if think or p != Phase.THINK)
    env = HorizonGymEnv(env_cfg)
    harness = AgentHarness(backend, UniformPolicy(ref_bits), think=think, **(harness_kwargs or {}))
    turns, cps = _reference(harness, env, seed)
    done_keys = done_keys or set()

    yield {
        "type": "reference",
        "task_seed": seed,
        "turn": None,
        "probe": None,
        "bits": ref_bits,
        "success": env.state.success,
        "n_hops": env_cfg.n_hops,
        "turns": [t.to_dict(with_text=True) for t in turns],
        "prefix_sha256": [hashlib.sha256(json.dumps(cp.backend[:2], default=str).encode()).hexdigest()
                          for cp in cps],
    }

    # Only turns on the correct prefix are replayable (a later turn's context assumes it).
    replayable = []
    for t in turns:
        replayable.append(t)
        if not t.correct:
            break
    if max_turns is not None:
        replayable = replayable[:max_turns]

    settings = [("ref", ref_bits)] + [(p.value, b) for p in probes for b in levels if b != ref_bits]
    for t, cp in zip(replayable, cps):
        for probe, bits in settings:
            key = ("probe", seed, t.turn, probe, bits)
            if key in done_keys:
                continue
            override = {Phase.READ: ref_bits, Phase.THINK: ref_bits, Phase.ACT: ref_bits}
            if probe != "ref":
                override[Phase.parse(probe)] = bits
            failures = 0
            for k in range(n_samples):
                env.set_state(cp.env_state)
                msgs = copy.deepcopy(cp.messages)
                harness.hinter = copy.deepcopy(cp.hinter)
                backend.restore(cp.backend)
                backend.reseed(_seed(seed, t.turn, probe, bits, k))
                rec = harness.run_turn(env, msgs, t.turn, bits_override=override)
                failures += int(not rec.correct)
            yield {
                "type": "probe",
                "task_seed": seed,
                "turn": t.turn,
                "step_index": t.step_index,
                "expected_phase": t.expected_phase.value,
                "probe": probe,
                "bits": bits,
                "ref_bits": ref_bits,
                "n": n_samples,
                "failures": failures,
            }


def run_atlas(
    backend: PrecisionBackend,
    env_cfg: HorizonGymConfig,
    seeds: Sequence[int],
    levels: Sequence[int],
    ref_bits: int,
    n_samples: int,
    out_path: str,
    probes: Sequence[Phase] = PROBES,
    think: bool = True,
    max_turns: int | None = None,
    harness_kwargs: dict | None = None,
    experiment: dict | None = None,
    prereg_sha: str | None = None,
    progress: bool = True,
) -> int:
    if experiment is None:
        raise ValueError("resumable runs require experiment=identity(resolved_config)")
    ensure_experiment(out_path, experiment, "atlas", prereg_sha)
    done = completed_keys(out_path, KEY_FIELDS)
    done_tasks = {r["task_seed"]: r for r in read_jsonl(out_path) if r.get("type") == "reference"}
    n_written = 0
    with JsonlWriter(out_path) as w:
        for i, seed in enumerate(seeds):
            for rec in measure_task(backend, env_cfg, seed, levels, ref_bits, n_samples, probes, think,
                                    max_turns, done, harness_kwargs):
                if rec["type"] == "reference" and seed in done_tasks:
                    if rec != done_tasks[seed]:
                        raise ValueError("reference trajectory changed on resume; use a new experiment")
                    continue
                w.write(rec)
                n_written += 1
            if progress:
                print(f"[atlas] task {i + 1}/{len(seeds)} (seed {seed}) done", flush=True)
    return n_written


def validate_atlas_records(records: Sequence[dict], cfg: dict) -> None:
    """Require the complete registered set of reference and probe records."""
    at = cfg['atlas']
    seeds = set(range(at['seed0'], at['seed0'] + at['n_tasks']))
    refs = [r for r in records if r.get('type') == 'reference']
    if len(refs) != len(seeds) or {r['task_seed'] for r in refs} != seeds:
        raise ValueError('incomplete or duplicate atlas reference tasks')
    probes = [p for p in PROBES if cfg['agent']['think'] or p != Phase.THINK]
    settings = [('ref',cfg['backend']['ref_bits'])] + [
        (p.value,b) for p in probes for b in cfg['backend']['levels'] if b != cfg['backend']['ref_bits']]
    expected = set()
    for ref in refs:
        if ref['n_hops'] != cfg['env']['n_hops'] or ref['bits'] != cfg['backend']['ref_bits']:
            raise ValueError('atlas reference configuration mismatch')
        turns = []
        for t in ref['turns']:
            turns.append(t)
            if not t['correct']: break
        if at['max_turns'] is not None: turns = turns[:at['max_turns']]
        if not turns: raise ValueError('atlas reference contains no turns')
        for t in turns:
            for probe,bits in settings:
                expected.add((ref['task_seed'],t['turn'],probe,bits))
    actual_rows = [r for r in records if r.get('type') == 'probe']
    actual = {(r['task_seed'],r['turn'],r['probe'],r['bits']) for r in actual_rows}
    if actual != expected or len(actual_rows) != len(expected) or len(records) != len(refs)+len(actual_rows):
        raise ValueError('incomplete, duplicate, or unexpected atlas probe records')
    if any(r['n'] != at['n_samples'] for r in actual_rows):
        raise ValueError('atlas sample counts differ from configuration')
