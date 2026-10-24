"""Which low-precision segments break the final submit? Whole episodes, held-out seeds.

The atlas lowers one segment of one turn and scores only that turn. Uniform 4-bit
Qwen3-1.7B nevertheless fails every STOP step at 3 hops, while 4-bit STOP probes on an
8-bit history never fail. The damage must therefore come from precision carried in
the history (KV computed earlier), which the atlas does not measure. This check runs
whole episodes with chosen bits per phase to locate it.

It uses the gate's held-out seeds and writes only to runs/interaction-<time>/, so it
does not touch the pre-registered sweep. It lives outside src/ and scripts/, so it does
not change the experiment identity.

Usage (repo root):
  .venv/bin/python interaction_check.py runs/pilot_q17_nothink_b6/config.yaml \
      --prereg prereg/pilot_q17_nothink_b6.json

Extra cells, with or without the default ones (bits are system,read,act,stop):
  .venv/bin/python interaction_check.py runs/pilot_q17_nothink_b6/config.yaml --no-defaults \
      --cell "instructions at 6:3:6,8,8,8" --cell "instructions at 4, 6 hops:6:4,8,8,8"

Sampling is seeded per episode, so cells from separate runs on the same seeds pair up.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from hazprec.agent.harness import AgentHarness
from hazprec.config import env_config, harness_kwargs, load_config
from hazprec.envs.horizongym import HorizonGymEnv
from hazprec.eval.stats import wilson_interval
from hazprec.quant.backends import make_backend
from hazprec.scheduler.policies import PhaseTablePolicy
from hazprec.utils.experiment import identity

# name, hops, bits for system, read, act, stop
CELLS = [
    ("reference", 3, 8, 8, 8, 8),
    ("uniform 6", 3, 6, 6, 6, 6),
    ("uniform 4", 3, 4, 4, 4, 4),
    ("system prompt at 4", 3, 4, 8, 8, 8),
    ("observations at 4", 3, 8, 4, 8, 8),
    ("read/update calls at 4", 3, 8, 8, 4, 8),
    ("submit call at 4", 3, 8, 8, 8, 4),
    ("everything but submit at 4", 3, 4, 4, 4, 8),
]


def parse_cell(text: str) -> tuple:
    """'NAME:HOPS:SYSTEM,READ,ACT,STOP' -> (name, hops, system, read, act, stop)."""
    try:
        name, hops, bits = text.rsplit(":", 2)
        values = [int(b) for b in bits.split(",")]
        if not name.strip() or len(values) != 4 or int(hops) < 1:
            raise ValueError
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected NAME:HOPS:SYSTEM,READ,ACT,STOP, got {text!r}") from None
    return (name.strip(), int(hops), *values)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--prereg", help="also run each pre-registered adaptive table at its own horizon")
    ap.add_argument("--cell", type=parse_cell, action="append", default=[], metavar="NAME:HOPS:S,R,A,T",
                    help="an extra cell: bits for system, read, act and stop at HOPS hops (repeatable)")
    ap.add_argument("--no-defaults", action="store_true", help="skip the built-in cells")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed0", type=int, default=200000)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    seeds = range(a.seed0, a.seed0 + a.n)
    for section, count in (("atlas", "n_tasks"), ("horizon", "n_episodes")):
        start = cfg[section]["seed0"]
        if any(start <= s < start + cfg[section][count] for s in seeds):
            ap.error(f"seeds overlap the {section} split")
    cells = ([] if a.no_defaults else list(CELLS)) + a.cell
    if a.prereg:
        for row in json.loads(Path(a.prereg).read_text())["per_hops"]:
            t = row["table"]
            cells.append((f"pre-registered {row['hops']}-hop table", int(row["hops"]),
                          t["system"], t["read"], t["act"], t["stop"]))
    if not cells:
        ap.error("no cells to run: drop --no-defaults or add --cell/--prereg")
    if len({c[0] for c in cells}) != len(cells):
        ap.error("cell names must be unique")
    levels = set(cfg["backend"]["levels"])
    for name, _, *bits in cells:
        if not set(bits) <= levels:
            ap.error(f"{name}: bits {bits} not in levels {sorted(levels)}")

    out = Path(a.out or f"runs/interaction-{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=False)
    be = make_backend(cfg["backend"], cfg["seed"])
    rows = []
    print(f"{'cell':<32}{'hops':>5}  sys/read/act/stop  {'success':>9}  95% CI          avg bits  failures by phase / kind")
    for name, hops, sys_b, read_b, act_b, stop_b in cells:
        t0 = time.perf_counter()
        policy = PhaseTablePolicy({"system": sys_b, "read": read_b, "act": act_b, "stop": stop_b})
        h = AgentHarness(be, policy, think=cfg["agent"]["think"], **harness_kwargs(cfg))
        env = HorizonGymEnv(env_config(cfg, n_hops=hops))
        episodes = []
        for seed in seeds:
            tr = h.run_episode(env, seed)
            fail = next((t for t in tr.turns if not t.correct), None)
            episodes.append({
                "seed": seed, "success": tr.success, "turns": len(tr.turns), "avg_bits": tr.avg_bits,
                "fail_phase": None if fail is None else fail.expected_phase.value,
                "fail_reason": None if fail is None else fail.reason,
                "fail_call": None if fail is None or fail.call is None else fail.call.to_json(),
            })
        k = sum(e["success"] for e in episodes)
        lo, hi = wilson_interval(k, len(episodes))
        phases = Counter(e["fail_phase"] for e in episodes if e["fail_phase"])
        kinds = Counter(e["fail_reason"] for e in episodes if e["fail_reason"])
        avg_bits = sum(e["avg_bits"] for e in episodes) / len(episodes)
        row = {"cell": name, "hops": hops, "bits": {"system": sys_b, "read": read_b, "act": act_b, "stop": stop_b},
               "success": k, "n": len(episodes), "ci95": [lo, hi], "avg_bits": avg_bits,
               "fail_phase": dict(phases), "fail_reason": dict(kinds), "seconds": time.perf_counter() - t0,
               "episodes": episodes}
        rows.append(row)
        print(f"{name:<32}{hops:>5}  {sys_b}/{read_b}/{act_b}/{stop_b:<12}  {k:>3}/{len(episodes):<5}  "
              f"[{lo:.2f}, {hi:.2f}]   {avg_bits:6.2f}    {dict(phases)} {dict(kinds.most_common(2))}", flush=True)
        example = next((e for e in episodes if e["fail_call"]), None)
        if example:
            print(f"{'':<37}e.g. seed {example['seed']}: {example['fail_reason']}; called {example['fail_call']}", flush=True)
    report = {"config": str(Path(a.config).resolve()), "source_sha256": identity(cfg)["source_sha256"],
              "seeds": [a.seed0, a.seed0 + a.n - 1], "cells": rows}
    (out / "interaction.json").write_text(json.dumps(report, indent=2))
    print(f"\nwritten to {out / 'interaction.json'}")


if __name__ == "__main__":
    main()
