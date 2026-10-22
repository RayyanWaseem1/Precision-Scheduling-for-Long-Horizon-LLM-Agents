"""Phase 1 gate: is this model + config ready for the Phase 2 sensitivity atlas?

Runs uniform-precision episodes at each requested bit-width and horizon on task seeds
disjoint from the atlas and the horizon sweep, then checks six things:

  1. format           the model emits parseable tool calls at the reference precision
  2. reliable         reference-precision success meets the bar at every tested horizon
  3. signal           the lowest bit-width has a measurably higher per-turn failure rate
  4. operating_point  uniform precision at theory.bbar is predicted to keep enough
                      survival at every sweep horizon for the horizon test to measure
  5. cache            history is append-only: no assistant text is re-prefilled
  6. cost             what the atlas in the config will take on this machine

By default the tested horizons are the shortest sweep horizon and the atlas horizon
(env.n_hops), and the tested precisions are the reference, theory.bbar and the lowest
level. Each failing episode records what the model wrote, what the checker expected,
and how its THINK and ACT segments ended (tag, eos, cap or skipped).

Usage:
  python scripts/phase1_gate.py -c configs/mac_pilot.yaml --quick
  python scripts/phase1_gate.py -c configs/mac_pilot.yaml --out runs/gate_mac_qwen3_1p7b.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

from hazprec.agent.harness import AgentHarness
from hazprec.config import env_config, harness_kwargs, load_config
from hazprec.envs.horizongym import HorizonGymEnv
from hazprec.eval.stats import wilson_interval
from hazprec.quant.backends import make_backend
from hazprec.quant.backends.base import TokenBackend
from hazprec.scheduler import UniformPolicy
from hazprec.sensitivity.replay import measure_task
from hazprec.utils.experiment import identity


def failure_kind(reason: str) -> str:
    if reason.startswith("could not parse"):
        return "unparseable"
    if reason.startswith("expected tool"):
        return "wrong_tool"
    if reason.startswith("missing argument"):
        return "missing:" + reason.split("'")[1]
    if reason.startswith("wrong value"):
        return "wrong:" + reason.split("'")[1]
    return reason


def failure_detail(env: HorizonGymEnv, turn) -> dict:
    """What the model wrote on the failing turn, next to what the checker expected."""
    assert env.task is not None, "call reset() before collecting failure details"
    text = {s.phase.value: s.text for s in turn.segments if s.text}
    return {
        "turn": turn.turn,
        "step": turn.step_index,
        "reason": turn.reason,
        "expected": env.task.oracle[turn.step_index].to_json(),
        "think_end": turn.think_end,
        "act_end": turn.act_end,
        "think": text.get("think", "")[:600],
        "act": (text.get("act") or text.get("stop") or "")[:300],
    }


def run_episode(harness: AgentHarness, env: HorizonGymEnv, seed: int) -> dict:
    """``AgentHarness.run_episode``, unrolled so we can see each observation and time it."""
    be = harness.backend
    env.reset(seed)
    be.reset(seed)
    be.bind_env(env)
    harness.policy.begin_episode()
    harness.hinter = harness.hinter_factory() if harness.hinter_factory else None
    stats0 = dict(getattr(be, "stats", {}))
    msgs = env.messages()
    cfg = env.config
    limit = cfg.horizon * (1 + (cfg.max_retries if cfg.error_mode == "recoverable" else 0)) + 2
    turns, read_overhead = [], []
    t0 = time.perf_counter()
    while not env.state.done and len(turns) < limit:
        obs = msgs[-1]["content"] if len(turns) > 0 else None
        rec = harness.run_turn(env, msgs, len(turns))
        if obs is not None and isinstance(be, TokenBackend):
            read_overhead.append(rec.segments[0].tokens - len(be._encode(obs)))
        turns.append(rec)
    secs = time.perf_counter() - t0
    stats1 = dict(getattr(be, "stats", {}))
    prefill = sum(s.tokens for t in turns for s in t.segments if s.phase.value in ("system", "read"))
    decode = sum(s.tokens for t in turns for s in t.segments if s.phase.value in ("think", "act", "stop"))
    fail = next((t for t in turns if not t.correct), None)
    expected_bits = Counter()
    for turn in turns:
        for segment in turn.segments:
            expected_bits[segment.bits] += segment.tokens - segment.boundary_tokens
    provenance = (Counter(be.cached_bits) == expected_bits) if isinstance(be, TokenBackend) else None
    think_tokens = sum(s.tokens - s.boundary_tokens for t in turns for s in t.segments if s.phase.value == "think")
    return {
        "provenance_ok": provenance,
        "seed": seed,
        "success": env.state.success,
        "turns": len(turns),
        "correct_turns": sum(t.correct for t in turns),
        "failure": None if fail is None else failure_kind(fail.reason),
        "failure_phase": None if fail is None else fail.expected_phase.value,
        "parse_failures": sum(t.call is None for t in turns),
        "prefill_tokens": prefill,
        "decode_tokens": decode,
        "seconds": secs,
        "full_renders": stats1.get("full", 0) - stats0.get("full", 0),
        "read_overhead": read_overhead,
        "example_call": next((t.call.to_json() for t in turns if t.call is not None), None),
        "think_tokens": think_tokens,
        "think_end": dict(Counter(t.think_end for t in turns if t.think_end)),
        "act_end": dict(Counter(t.act_end for t in turns if t.act_end)),
        "failure_detail": None if fail is None else failure_detail(env, fail),
    }


def summarize(eps: list[dict]) -> dict:
    n = len(eps)
    if not n:
        raise ValueError("cannot summarize empty episode results")
    succ = sum(e["success"] for e in eps)
    turns = sum(e["turns"] for e in eps)
    fails = sum(e["turns"] - e["correct_turns"] for e in eps)
    lo, hi = wilson_interval(succ, n)
    hlo, hhi = wilson_interval(fails, turns)
    secs = sum(e["seconds"] for e in eps)
    return {
        "episodes": n,
        "success": succ / n,
        "success_ci95": [lo, hi],
        "turns": turns,
        "turn_failures": fails,
        "turn_hazard": fails / max(turns, 1),
        "turn_hazard_ci95": [hlo, hhi],
        "parse_rate": 1 - sum(e["parse_failures"] for e in eps) / max(turns, 1),
        "failure_kinds": dict(Counter(e["failure"] for e in eps if e["failure"])),
        "sec_per_turn": secs / max(turns, 1),
        "decode_tok_per_sec": sum(e["decode_tokens"] for e in eps) / max(secs, 1e-9),
        "tokens_per_turn": (sum(e["prefill_tokens"] + e["decode_tokens"] for e in eps)) / max(turns, 1),
        "think_tokens_per_turn": sum(e["think_tokens"] for e in eps) / max(turns, 1),
        "think_end": dict(sum((Counter(e["think_end"]) for e in eps), Counter())),
        "act_end": dict(sum((Counter(e["act_end"]) for e in eps), Counter())),
    }


def sweep_survival(turn_hazard: float, hops) -> dict:
    """Predicted success at each sweep horizon (2H+1 actions) if the per-turn hazard
    stayed constant. A screening estimate for the operating point, not a prediction."""
    return {int(h): (1 - turn_hazard) ** (2 * int(h) + 1) for h in hops}


def print_failures(eps: list[dict], k: int = 2) -> None:
    for e in [e for e in eps if e["failure_detail"]][:k]:
        d = e["failure_detail"]
        print(f"    seed {e['seed']} turn {d['turn']}: {d['reason']}; expected {d['expected']}\n"
              f"      think[{d['think_end']}]: {d['think'][:240]!r}\n      act[{d['act_end']}]: {d['act'][:160]!r}",
              flush=True)


def measure_atlas_cost(be, cfg, seed0, n_tasks=1, n_samples=2):
    """Time actual snapshot restoration and mixed-precision replay on held-out tasks.

    Scale to full reference-path coverage so an early failure cannot make the
    full atlas appear cheap. This is a calibration estimate, not a runtime guarantee.
    """
    elapsed, reference_turns, probe_samples, replay_turns = 0.0, 0, 0, 0
    for i in range(n_tasks):
        start = time.perf_counter()
        rows = list(measure_task(be, env_config(cfg), seed0+i, cfg['backend']['levels'],
                                cfg['backend']['ref_bits'], n_samples,
                                think=cfg['agent']['think'], max_turns=cfg['atlas']['max_turns'],
                                harness_kwargs=harness_kwargs(cfg)))
        elapsed += time.perf_counter() - start
        reference_turns += len(rows[0]['turns'])
        probes = [r for r in rows if r['type']=='probe']
        probe_samples += sum(r['n'] for r in probes)
        replay_turns += len({r['turn'] for r in probes})
    if not probe_samples or not reference_turns or not replay_turns:
        raise ValueError('cost calibration produced no replay measurements')
    ec = env_config(cfg)
    full_turns = ec.horizon * (1 + (ec.max_retries if ec.error_mode=='recoverable' else 0)) + 2
    if ec.error_mode == 'fatal':
        full_turns = ec.horizon
    planned_replay = min(full_turns, cfg['atlas']['max_turns'] or full_turns)
    coverage_scale = max(full_turns / (reference_turns / n_tasks),
                         planned_replay / (replay_turns / n_tasks), 1.0)
    sample_scale = max(cfg['atlas']['n_samples'] / n_samples, 1.0)
    return {'measured_seconds': elapsed, 'measured_probe_samples': probe_samples,
            'calibration_tasks': n_tasks, 'calibration_samples': n_samples,
            'coverage_scale': coverage_scale,
            'hours_total': elapsed / n_tasks * cfg['atlas']['n_tasks'] * coverage_scale * sample_scale / 3600,
            'method': 'timed replay; conservatively scaled to full reference-path coverage'}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-c','--config',required=True)
    ap.add_argument('--set',dest='overrides',action='append',default=[])
    ap.add_argument('--bits',type=int,nargs='+')
    ap.add_argument('--hops',type=int,nargs='+',help='default: the shortest sweep horizon and the atlas horizon')
    ap.add_argument('--gate-hops',type=int,help='horizon of the signal and operating-point checks; default: env.n_hops')
    ap.add_argument('--n',type=int,default=30)
    ap.add_argument('--quick',action='store_true')
    ap.add_argument('--min-ref-success',type=float,default=.8)
    ap.add_argument('--min-parse-rate',type=float,default=.98)
    ap.add_argument('--min-sweep-survival',type=float,default=.1)
    ap.add_argument('--max-read-overhead',type=int,default=48)
    ap.add_argument('--atlas-budget-hours',type=float,default=24)
    ap.add_argument('--seed0',type=int,default=200000)
    ap.add_argument('--cost-tasks',type=int,default=1)
    ap.add_argument('--cost-samples',type=int,default=2)
    ap.add_argument('--out')
    a=ap.parse_args(argv)
    cfg=load_config(a.config,a.overrides)
    # Defaults follow the experiment: horizon-specific failures (Qwen3-1.7B failed 30/30
    # 1-hop episodes at 8 bits) must be caught at the horizons the atlas and sweep use.
    atlas_hops=int(cfg['env']['n_hops']); sweep_hops=sorted(int(h) for h in cfg['horizon']['hops'])
    if a.quick:
        a.n,a.hops,a.gate_hops=10,[atlas_hops],atlas_hops
    if a.gate_hops is None: a.gate_hops=atlas_hops
    if a.hops is None: a.hops=sorted({sweep_hops[0],a.gate_hops})
    # load_config guarantees theory.bbar is an available level (the uniform baseline).
    levels=cfg['backend']['levels']; ref=cfg['backend']['ref_bits']; low=min(levels); bbar=int(cfg['theory']['bbar'])
    tested=a.bits if a.bits is not None else sorted({ref,bbar,low},reverse=True)
    if (ref not in tested or low not in tested or bbar not in tested or low==ref or not set(tested)<=set(levels)
            or len(set(tested))!=len(tested)):
        ap.error('--bits must include the reference, theory.bbar and a distinct lowest configured level, without duplicates')
    if min(a.n,a.gate_hops,a.cost_tasks,a.cost_samples,*a.hops)<1 or a.seed0<0 or len(set(a.hops))!=len(a.hops):
        ap.error('counts and horizons must be positive; horizons unique; seed0 nonnegative')
    if a.gate_hops not in a.hops or a.gate_hops>9000 or max(a.hops)>9000:
        ap.error('--hops must include --gate-hops and fit the record ID space')
    if (not (0<a.min_ref_success<=1 and 0<a.min_parse_rate<=1 and 0<a.min_sweep_survival<1
             and 0<a.atlas_budget_hours<float('inf')) or a.max_read_overhead<0):
        ap.error('invalid gate thresholds')
    # Check actual ranges, including calibration tasks, against both experiment splits.
    gate_seeds=set(range(a.seed0,a.seed0+a.n))
    cost_seed=a.seed0+a.n
    used=gate_seeds | set(range(cost_seed,cost_seed+a.cost_tasks))
    for section,count in [('atlas','n_tasks'),('horizon','n_episodes')]:
        start=cfg[section]['seed0']
        if any(start<=s<start+cfg[section][count] for s in used):
            ap.error('gate/calibration seeds overlap the atlas or evaluation split')
    be=make_backend(cfg['backend'],cfg['seed'])
    print(f'[gate] backend={cfg["backend"]["kind"]}; checking every precision',flush=True)
    smoke={}
    for b in levels:
        h=AgentHarness(be,UniformPolicy(b),think=cfg['agent']['think'],**harness_kwargs(cfg))
        smoke[b]=run_episode(h,HorizonGymEnv(env_config(cfg,n_hops=1)),a.seed0)
    results={}
    for hops in a.hops:
        for b in tested:
            h=AgentHarness(be,UniformPolicy(b),think=cfg['agent']['think'],**harness_kwargs(cfg))
            env=HorizonGymEnv(env_config(cfg,n_hops=hops))
            eps=[run_episode(h,env,a.seed0+i) for i in range(a.n)]
            results[b,hops]=eps
            print(f'[gate] {b} bits, {hops} hops: {summarize(eps)}',flush=True)
            print_failures(eps)
    ref_eps=[e for (b,_),eps in results.items() if b==ref for e in eps]
    rs=summarize(ref_eps)
    # Compare hazard at the same gate horizon, rather than pooling different horizons.
    high=summarize(results[ref,a.gate_hops]); low_s=summarize(results[low,a.gate_hops])
    ref_success={h:summarize(results[ref,h])['success'] for h in a.hops}
    # The horizon test needs the uniform arm off the floor: zero-success cells make the
    # smoothed log ratio, and so the measured slope, an artifact of the smoothing constant.
    base=summarize(results[bbar,a.gate_hops]); survival=sweep_survival(base['turn_hazard'],sweep_hops)
    hz_lo,hz_hi=base['turn_hazard_ci95']; worst,best=sweep_survival(hz_hi,sweep_hops),sweep_survival(hz_lo,sweep_hops)
    all_eps=[e for eps in results.values() for e in eps]+list(smoke.values())
    overhead=[o for e in all_eps for o in e['read_overhead']]
    cache_ok=(bool(overhead) and all(e['provenance_ok'] is True and e['full_renders']==1 for e in all_eps)
              and all(0<=o<=a.max_read_overhead for o in overhead))
    print('[gate] timing actual atlas replay',flush=True)
    cost=measure_atlas_cost(be,cfg,cost_seed,a.cost_tasks,a.cost_samples)
    checks=[
        {'name':'format','pass':rs['parse_rate']>=a.min_parse_rate,'value':rs['parse_rate']},
        {'name':'reliable','pass':min(ref_success.values())>=a.min_ref_success,'value':ref_success},
        {'name':'signal','pass':low_s['turn_hazard_ci95'][0]>high['turn_hazard_ci95'][1],
         'value':{'low_ci':low_s['turn_hazard_ci95'],'reference_ci':high['turn_hazard_ci95']}},
        {'name':'operating_point','pass':min(survival.values())>=a.min_sweep_survival,
         'value':{'bbar':bbar,'gate_hops':a.gate_hops,'turn_hazard':base['turn_hazard'],'predicted_success':survival,
                  'predicted_success_ci95':{h:[worst[h],best[h]] for h in sweep_hops}}},
        {'name':'cache','pass':cache_ok,'value':{'observations_checked':len(overhead)}},
        {'name':'cost','pass':cost['hours_total']<=a.atlas_budget_hours,'value':cost},
    ]
    ready=all(c['pass'] for c in checks) and cfg['backend']['kind']!='sim' and not a.quick
    report={'ready':ready,'mode':'diagnostic' if a.quick or cfg['backend']['kind']=='sim' else 'full',
            'config':cfg,'experiment':identity(cfg),'arguments':vars(a),'checks':checks,
            'smoke':smoke,'table':{f'{b}b_{h}h':summarize(e) for (b,h),e in results.items()},
            'episodes':{f'{b}b_{h}h':e for (b,h),e in results.items()},'atlas_cost':cost}
    for c in checks: print(f'[{"PASS" if c["pass"] else "FAIL"}] {c["name"]}: {c["value"]}')
    print(f'Phase 2 ready: {ready}')
    if a.out:
        Path(a.out).parent.mkdir(parents=True,exist_ok=True)
        Path(a.out).write_text(json.dumps(report,indent=2,allow_nan=False))
    return 0 if ready else 1


if __name__=='__main__':
    sys.exit(main())
