"""Command-line entry point: ``hazprec <command> [options]``.

    play         run and print one episode
    atlas        measure the sensitivity atlas (resumable JSONL)
    fit          fit per-phase excess-hazard curves from an atlas
    preregister  derive the adaptive schedule and write write-once predictions
    horizon      run the paired horizon sweep (resumable) and test the predictions
    analyze      re-analyze a finished horizon sweep against a pre-registration
    demo         full pipeline on the simulated backend in a few minutes
    smoke        load a real backend and run one short episode per bit-width
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from hazprec.agent.harness import AgentHarness
from hazprec.config import env_config, harness_kwargs, load_config
from hazprec.envs.horizongym import HorizonGymEnv
from hazprec.eval.horizon import analyze_horizon, format_horizon, run_horizon_sweep
from hazprec.eval.survival import fit_hazard_model, hazard_by_turn
from hazprec.pipeline import adaptive_policy_for, build_predictions
from hazprec.quant.backends import make_backend
from hazprec.scheduler import PhaseTablePolicy, UniformPolicy, make_policy
from hazprec.sensitivity import fit_atlas, format_fit, run_atlas
from hazprec.sensitivity.replay import validate_atlas_records
from hazprec.theory import file_sha256, load_preregistration, preregister
from hazprec.utils.io import read_jsonl
from hazprec.utils.experiment import identity, ensure_experiment, verify_experiment


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", "-c", default=None, help="YAML config (defaults to the simulated backend)")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set env.n_hops=8")


def _parse_policy(spec: str):
    """'uniform:4' | 'table:read=3,think=4,act=6,stop=8' | path to a pre-registration JSON."""
    if spec.startswith("uniform:"):
        return UniformPolicy(int(spec.split(":", 1)[1]))
    if spec.startswith("table:"):
        table = dict(kv.split("=") for kv in spec.split(":", 1)[1].split(","))
        return PhaseTablePolicy({k: int(v) for k, v in table.items()})
    if spec.endswith(".json"):
        return PhaseTablePolicy(load_preregistration(spec)["adaptive_table"])
    return make_policy(json.loads(spec))


# --------------------------------------------------------------------------------------
def cmd_play(a) -> None:
    cfg = load_config(a.config, a.overrides)
    be = make_backend(cfg["backend"], cfg["seed"])
    env = HorizonGymEnv(env_config(cfg))
    h = AgentHarness(be, _parse_policy(a.policy), think=cfg["agent"]["think"], **harness_kwargs(cfg))
    tr = h.run_episode(env, a.seed)
    print(env.messages()[1]["content"], "\n")
    for t in tr.turns:
        segs = " ".join(f"{s.phase.value}@{s.bits}b/{s.tokens}t" for s in t.segments)
        mark = "ok " if t.correct else "ERR"
        call = t.call.to_json() if t.call else "<unparseable>"
        print(f"[{mark}] turn {t.turn:>2} step {t.step_index:>2} ({t.expected_phase.value:<4}) {segs}\n      {call}")
        if not t.correct:
            print(f"      -> {t.reason}")
        if a.verbose:
            for s in t.segments:
                if s.text:
                    print(f"      {s.phase.value}: {s.text.strip()[:300]!r}")
    print(f"\nsuccess={tr.success} turns={len(tr.turns)} tokens={tr.tokens} avg_bits={tr.avg_bits:.2f}")


def cmd_atlas(a) -> None:
    cfg = load_config(a.config, a.overrides)
    ensure_experiment(a.out, identity(cfg), "atlas")
    be = make_backend(cfg["backend"], cfg["seed"])
    at = cfg["atlas"]
    seeds = list(range(int(at["seed0"]), int(at["seed0"]) + int(at["n_tasks"])))
    n = run_atlas(be, env_config(cfg), seeds, cfg["backend"]["levels"], int(cfg["backend"]["ref_bits"]),
                  int(at["n_samples"]), a.out, think=cfg["agent"]["think"], max_turns=at.get("max_turns"),
                  harness_kwargs=harness_kwargs(cfg), experiment=identity(cfg))
    print(f"wrote {n} records to {a.out}")


def cmd_fit(a) -> None:
    fit = fit_atlas(read_jsonl(a.atlas))
    print(format_fit(fit))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(fit.to_dict(), indent=2))


def cmd_preregister(a) -> None:
    cfg = load_config(a.config, a.overrides)
    experiment = identity(cfg)
    verify_experiment(a.atlas, experiment, "atlas")
    records = read_jsonl(a.atlas)
    validate_atlas_records(records, cfg)
    fit = fit_atlas(records)
    th = cfg["theory"]
    pred = build_predictions(fit, float(th["bbar"]), cfg["backend"]["levels"], int(th["alloc_hops"]),
                             cfg["agent"]["think"], th.get("curve_mode", "parametric"),
                             sweep_hops=cfg["horizon"]["hops"])
    body = preregister(a.out, {
        "experiment": experiment, "config": cfg, "atlas_path": str(Path(a.atlas).resolve()), "atlas_sha256": file_sha256(a.atlas),
        "fit": fit.to_dict(), **pred,
    })
    p = body["prediction"]
    print(format_fit(fit))
    print(f"\nadaptive table: {body['adaptive_table']}  (avg {body['avg_bits_at_alloc_hops']:.2f} bits"
          f" at {body['alloc_hops']} hops; uniform = {body['uniform_bits']})")
    for r in body["per_hops"]:
        print(f"  hops={r['hops']:>3}: table {r['table']} avg {r['avg_bits']:.2f} bits,"
              f" predicted log ratio {r['log_ratio']:.4f}")
    diag = body.get("diagnostics") or {}
    if "continuous_excess_ratio" in diag:
        print(f"continuous water-filling excess hazard / uniform (Prop. 1 quantity): "
              f"{diag['continuous_excess_ratio']:.3f}")
    hstar = p["h_star_hops"]
    print(f"pre-registered log-ratio slope per hop: {p['log_ratio_slope_per_hop']:.4f};"
          f" peak absolute gain H* = {'n/a' if hstar is None else round(hstar, 1)} hops")
    print(f"written to {a.out} (sha256 {body['sha256'][:12]}...)")


def cmd_horizon(a) -> None:
    cfg = load_config(a.config, a.overrides)
    pre = load_preregistration(a.prereg)
    experiment = identity(cfg)
    if pre.get("experiment") != experiment or pre["config"] != cfg:
        raise ValueError("config/source/dependencies differ from preregistration; create a new experiment")
    if file_sha256(pre["atlas_path"]) != pre["atlas_sha256"]:
        raise ValueError("atlas changed after preregistration")
    ensure_experiment(a.out, experiment, "horizon", pre["sha256"])
    be = make_backend(cfg["backend"], cfg["seed"])
    hz = cfg["horizon"]
    policies = {"uniform": UniformPolicy(pre["uniform_bits"]), "adaptive": adaptive_policy_for(pre)}
    run_horizon_sweep(be, env_config(cfg), hz["hops"], policies, int(hz["n_episodes"]), a.out,
                      int(hz["seed0"]), cfg["agent"]["think"], harness_kwargs(cfg),
                      experiment=experiment, prereg_sha=pre["sha256"])
    _report(a.out, pre)


def _report(runs: str, pre: dict) -> None:
    verify_experiment(runs, pre["experiment"], "horizon", pre["sha256"])
    records = read_jsonl(runs)
    hz = pre["config"]["horizon"]
    expected = {(h, policy, seed) for h in hz["hops"] for policy in ("uniform", "adaptive")
                for seed in range(hz["seed0"], hz["seed0"] + hz["n_episodes"])}
    actual = {(r["hops"], r["policy_label"], r["seed"]) for r in records}
    if actual != expected or len(records) != len(expected):
        raise ValueError("incomplete or duplicate sweep; finish the registered run before analysis")
    an = analyze_horizon(records, prediction=pre["prediction"], per_hops=pre.get("per_hops"))
    print(format_horizon(an))
    Path(runs).with_suffix(".summary.json").write_text(json.dumps(an, indent=2))


def cmd_analyze(a) -> None:
    _report(a.runs, load_preregistration(a.prereg))


def cmd_demo(a) -> None:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    over = list(a.overrides)
    base = ["atlas.n_tasks=30", "atlas.n_samples=40", "env.n_hops=4", "horizon.n_episodes=1500",
            "horizon.hops=[2,8,16,32,48,64]"] + over
    cfg_path = a.config
    atlas = out / "atlas.jsonl"
    prereg = out / "prereg.json"
    runs = out / "horizon.jsonl"
    print("== 1/3 measuring the sensitivity atlas (simulated backend) ==")
    cmd_atlas(argparse.Namespace(config=cfg_path, overrides=base, out=str(atlas)))
    print("\n== 2/3 fitting curves and pre-registering predictions ==")
    if not prereg.exists():
        cmd_preregister(argparse.Namespace(config=cfg_path, overrides=base, atlas=str(atlas), out=str(prereg)))
    print("\n== 3/3 horizon sweep: adaptive vs uniform on paired seeds ==")
    cmd_horizon(argparse.Namespace(config=cfg_path, overrides=base, prereg=str(prereg), out=str(runs)))
    trajs_note = "per-turn hazard and logistic hazard model are available via hazprec.eval.survival"
    print(f"\n{trajs_note}")


def cmd_smoke(a) -> None:
    cfg = load_config(a.config, a.overrides)
    be = make_backend(cfg["backend"], cfg["seed"])
    env = HorizonGymEnv(env_config(cfg, n_hops=a.hops))
    for b in cfg["backend"]["levels"]:
        h = AgentHarness(be, UniformPolicy(b), think=cfg["agent"]["think"], **harness_kwargs(cfg))
        tr = h.run_episode(env, a.seed)
        calls = [t.call.to_json() if t.call else "<unparseable>" for t in tr.turns]
        print(f"{b:>2} bits: success={tr.success} turns={len(tr.turns)} tokens={tr.tokens}")
        for t, c in zip(tr.turns, calls):
            print(f"    {'ok ' if t.correct else 'ERR'} {c[:160]}")
        hz = hazard_by_turn([tr.to_dict()])
        print(f"    per-turn failures: {[r['failures'] for r in hz]}")
    _ = fit_hazard_model  # available for notebooks


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="hazprec", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("play")
    _common(p)
    p.add_argument("--policy", default="uniform:8")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", "-v", action="store_true")
    p.set_defaults(fn=cmd_play)

    p = sub.add_parser("atlas")
    _common(p)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_atlas)

    p = sub.add_parser("fit")
    p.add_argument("--atlas", required=True)
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("preregister")
    _common(p)
    p.add_argument("--atlas", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_preregister)

    p = sub.add_parser("horizon")
    _common(p)
    p.add_argument("--prereg", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_horizon)

    p = sub.add_parser("analyze")
    p.add_argument("--runs", required=True)
    p.add_argument("--prereg", required=True)
    p.set_defaults(fn=cmd_analyze)

    p = sub.add_parser("demo")
    _common(p)
    p.add_argument("--out", default="runs/demo")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("smoke")
    _common(p)
    p.add_argument("--hops", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(fn=cmd_smoke)

    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main(sys.argv[1:])
