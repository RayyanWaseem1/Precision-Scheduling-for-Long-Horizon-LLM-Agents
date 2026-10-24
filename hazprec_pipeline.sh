#!/usr/bin/env bash
# HazPrec pipeline driver for the Mac. Save it in the repo root and run it with bash.
#
#   bash hazprec_pipeline.sh verify
#       compile, run the tests, and check the revised gate on the simulator (~1 min)
#   bash hazprec_pipeline.sh diagnose
#       readiness gates for every model in DIAG_CONFIGS, THINK off then on, bits 8/6/4/3,
#       then a summary to paste back (~1-2 h)
#   bash hazprec_pipeline.sh pilot NAME CONFIG [KEY=VALUE ...]
#       freeze CONFIG plus overrides into runs/pilot_NAME/config.yaml, run the final gate,
#       and only if it passes: atlas, fit, pre-registration to prereg/pilot_NAME.json.
#       Stops before the sweep. Rerun the same command to resume.
#   bash hazprec_pipeline.sh sweep NAME
#       horizon sweep and analysis; refuses until prereg/pilot_NAME.json is committed
#   bash hazprec_pipeline.sh summary RUN_DIR [RUN_DIR ...]
#       compact text report of every gate, atlas, fit, pre-registration and sweep result
#
# The Mac stays awake while the script runs (caffeinate waits on this process), so no
# command inside needs its own caffeinate. The script lives outside src/ and scripts/,
# so it does not change the experiment identity. Written for macOS's bash 3.2.
set -eo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
DIAG_CONFIGS="${DIAG_CONFIGS:-q17=configs/mac_pilot.yaml q3b=configs/mac_pilot_3b.yaml}"
DIAG_N="${DIAG_N:-30}"

[ -x "$PY" ] || { echo "no virtualenv at $ROOT/.venv; create it and pip install -e '.[mlx,dev]'" >&2; exit 1; }
if command -v caffeinate >/dev/null 2>&1; then caffeinate -i -w $$ & fi

stamp() { date +%Y%m%d-%H%M%S; }

require_change_set() {
  if ! grep -q "operating_point" scripts/phase1_gate.py || ! grep -q "think_end" src/hazprec/agent/harness.py; then
    echo "the 2026-10-22 change set is not applied (see APPLY.md)" >&2
    exit 1
  fi
}

record_provenance() {  # dir: dependencies and source state next to the results
  "$PY" -m pip freeze > "$1/requirements.txt"
  git rev-parse HEAD > "$1/git_commit.txt" 2>/dev/null || true
  git status --porcelain > "$1/git_status.txt" 2>/dev/null || true
}

cmd_verify() {
  require_change_set
  local out="$ROOT/runs/verify-$(stamp)"
  mkdir -p "$out"
  "$PY" -m compileall -q src tests scripts
  "$PY" -m pytest -q 2>&1 | tee "$out/tests.log"
  # The simulator can never be certified, so this gate exits nonzero by design.
  "$PY" scripts/phase1_gate.py -c configs/sim.yaml --n 10 --cost-samples 1 \
    --set atlas.max_turns=1 --out "$out/gate_sim.json" > "$out/gate_sim.log" 2>&1 || true
  "$PY" - "$out/gate_sim.json" <<'PY'
import json, sys
body = json.load(open(sys.argv[1]))
names = [c["name"] for c in body["checks"]]
assert "operating_point" in names and all("think_end" in cell for cell in body["table"].values()), names
print("revised gate runs on the simulator; checks:", ", ".join(names))
PY
  echo "[verify] OK; logs in $out"
}

cmd_diagnose() {
  require_change_set
  local run entry model config think tag t0
  mkdir -p "$ROOT/runs"
  run="$(mktemp -d "$ROOT/runs/diag-$(stamp).XXXX")"
  record_provenance "$run"
  for entry in $DIAG_CONFIGS; do
    cp "${entry#*=}" "$run/${entry%%=*}.yaml"
  done
  echo "[diagnose] results go to $run"
  for think in false true; do
    for entry in $DIAG_CONFIGS; do
      model="${entry%%=*}"
      tag="${model}_think-${think}"
      t0=$SECONDS
      echo "[diagnose] gate $tag started $(date +%H:%M)"
      # A failed gate is a result here, not an error: keep going and report it.
      if "$PY" -u scripts/phase1_gate.py -c "$run/$model.yaml" --set "agent.think=$think" \
          --bits 8 6 4 3 --n "$DIAG_N" --out "$run/gate_$tag.json" 2>&1 | tee "$run/gate_$tag.log"; then
        echo "[diagnose] $tag: ready"
      else
        echo "[diagnose] $tag: not ready (see the summary)"
      fi
      echo "[diagnose] $tag took $(( (SECONDS - t0) / 60 )) min"
    done
  done
  cmd_summary "$run" | tee "$run/summary.txt"
  echo
  echo "[diagnose] done. Paste $run/summary.txt back, or commit and push $run."
}

cmd_pilot() {
  require_change_set
  [ $# -ge 2 ] || { echo "usage: pilot NAME CONFIG [KEY=VALUE ...]" >&2; exit 2; }
  local name="$1" config="$2"
  shift 2
  local run="$ROOT/runs/pilot_$name" prereg="$ROOT/prereg/pilot_$name.json"
  mkdir -p "$run" "$ROOT/prereg"

  # Freeze the resolved config once. atlas, preregister and horizon all read this file
  # with no overrides, so they share one experiment identity.
  "$PY" - "$config" "$run/config.yaml" "$@" <<'PY'
import sys
from pathlib import Path
import yaml
from hazprec.config import load_config
src, dst, overrides = sys.argv[1], Path(sys.argv[2]), sys.argv[3:]
cfg = load_config(src, overrides)
if dst.exists():
    if load_config(str(dst)) != cfg:
        sys.exit(f"{dst} holds a different config; use a new NAME for a new experiment")
else:
    dst.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"[pilot] config {dst}:")
print(dst.read_text())
PY
  [ -f "$run/requirements.txt" ] || record_provenance "$run"

  if ! "$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["ready"] else 1)' "$run/gate.json" 2>/dev/null; then
    echo "[pilot] final gate started $(date +%H:%M)"
    if ! "$PY" -u scripts/phase1_gate.py -c "$run/config.yaml" --out "$run/gate.json" 2>&1 | tee "$run/gate.log"; then
      echo "[pilot] gate not passed: no atlas collected. Report: bash hazprec_pipeline.sh summary $run" >&2
      exit 1
    fi
  fi

  echo "[pilot] atlas started $(date +%H:%M) (resumable: rerun this command after an interruption)"
  "$PY" -u -m hazprec.cli atlas -c "$run/config.yaml" --out "$run/atlas.jsonl" 2>&1 | tee -a "$run/atlas.log"
  "$PY" - "$run/atlas.jsonl" "$run/config.yaml" <<'PY'
import sys
from collections import Counter
from hazprec.config import load_config
from hazprec.sensitivity.fit import phase_key
from hazprec.utils.io import read_jsonl
rows, cfg = read_jsonl(sys.argv[1]), load_config(sys.argv[2])
refs = [r for r in rows if r["type"] == "reference"]
coverage = Counter(phase_key(r) for r in rows if r["type"] == "probe" and r["probe"] != "ref")
print(f"[pilot] reference successes {sum(r['success'] for r in refs)}/{len(refs)}; phase coverage {dict(coverage)}")
missing = {"system", "read", "act", "stop"} | ({"think"} if cfg["agent"]["think"] else set())
missing -= set(coverage)
if missing:
    sys.exit(f"[pilot] no probes for {sorted(missing)}; allocation needs every phase measured")
PY
  "$PY" -m hazprec.cli fit --atlas "$run/atlas.jsonl" --out "$run/fit.json" 2>&1 | tee "$run/fit.log"

  if [ -f "$prereg" ]; then
    echo "[pilot] $prereg already exists (write-once); keeping it"
  else
    "$PY" -m hazprec.cli preregister -c "$run/config.yaml" --atlas "$run/atlas.jsonl" \
      --out "$prereg" 2>&1 | tee "$run/preregister.log"
  fi
  cmd_summary "$run" | tee "$run/summary.txt"
  cat <<EOF

[pilot] Pre-registration written. Before the sweep:
  1. Paste $run/summary.txt back for review (do not edit code from here on).
  2. Commit the pre-registration and its evidence:
       git add prereg/pilot_$name.json runs/pilot_$name/{config.yaml,gate.json,gate.log,fit.json,fit.log,preregister.log,requirements.txt,summary.txt}
       git commit -m "Pre-register pilot_$name"
  3. bash hazprec_pipeline.sh sweep $name
EOF
}

cmd_sweep() {
  require_change_set
  [ $# -eq 1 ] || { echo "usage: sweep NAME" >&2; exit 2; }
  local name="$1"
  local run="$ROOT/runs/pilot_$name" prereg="prereg/pilot_$name.json"
  [ -f "$run/config.yaml" ] && [ -f "$prereg" ] || { echo "run: bash hazprec_pipeline.sh pilot $name ... first" >&2; exit 1; }
  # Pre-registrations are committed before their sweep (CLAUDE.md invariant).
  if ! git ls-files --error-unmatch "$prereg" >/dev/null 2>&1 || ! git diff --quiet HEAD -- "$prereg"; then
    echo "commit $prereg before running its sweep" >&2
    exit 1
  fi
  echo "[sweep] started $(date +%H:%M) (resumable: rerun this command after an interruption)"
  "$PY" -u -m hazprec.cli horizon -c "$run/config.yaml" --prereg "$prereg" \
    --out "$run/horizon.jsonl" 2>&1 | tee -a "$run/horizon.log"
  "$PY" -m hazprec.cli analyze --runs "$run/horizon.jsonl" --prereg "$prereg" 2>&1 | tee "$run/analyze.log"
  cmd_summary "$run" | tee "$run/summary.txt"
  echo
  echo "[sweep] done. Paste $run/summary.txt back, or commit and push runs/pilot_$name."
}

cmd_summary() {
  [ $# -ge 1 ] || { echo "usage: summary RUN_DIR [RUN_DIR ...]" >&2; exit 2; }
  "$PY" - "$@" <<'PY'
import json, sys
from collections import Counter
from pathlib import Path

def pct(x):
    return "n/a" if x is None else f"{x:.3f}"

def ci(pair):
    return f"[{pair[0]:.3f}, {pair[1]:.3f}]"

def gate(path):
    try:
        body = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        print(f"\n## {path.name}: unreadable ({exc}); see {path.with_suffix('.log').name}")
        return
    cfg, args = body["config"], body["arguments"]
    model = cfg["backend"].get(cfg["backend"]["kind"], {}).get("model", cfg["backend"]["kind"])
    print(f"\n## {path.name}  ready={body['ready']}  mode={body['mode']}")
    print(f"model={model} think={cfg['agent']['think']} bbar={cfg['theory']['bbar']} "
          f"hops={args['hops']} gate_hops={args['gate_hops']} n={args['n']} "
          f"source={body['experiment']['source_sha256'][:12]}")
    print("cell      success          turn hazard [95% CI]       parse  think tok/turn  think_end / act_end      failures")
    for key, cell in body["table"].items():
        k = round(cell["success"] * cell["episodes"])
        print(f"{key:<8} {k:>3}/{cell['episodes']:<3} {pct(cell['success']):>6}  "
              f"{cell['turn_hazard']:.3f} {ci(cell['turn_hazard_ci95'])}  {cell['parse_rate']:.2f}  "
              f"{cell.get('think_tokens_per_turn', float('nan')):>8.1f}  "
              f"{cell.get('think_end', {})} / {cell.get('act_end', {})}  {cell['failure_kinds']}")
    for check in body["checks"]:
        value = check["value"]
        if check["name"] == "operating_point":
            value = {k: value[k] for k in ("bbar", "gate_hops", "turn_hazard")} | {
                "predicted_success": {h: f"{s:.3g}" for h, s in value["predicted_success"].items()}}
        elif check["name"] == "cost":
            value = {k: value[k] for k in ("hours_total", "coverage_scale")}
        print(f"  [{'PASS' if check['pass'] else 'FAIL'}] {check['name']}: {value}")
    for key, episodes in body.get("episodes", {}).items():
        failed = [e for e in episodes if e.get("failure_detail")]
        if failed:
            e = failed[0]
            d = e["failure_detail"]
            print(f"  e.g. {key} seed {e['seed']} turn {d['turn']}: {d['reason']} | expected {d['expected']}\n"
                  f"       think[{d['think_end']}] {d['think'][:200]!r}\n       act[{d['act_end']}] {d['act'][:120]!r}")

def atlas(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    refs = [r for r in rows if r["type"] == "reference"]
    probes = [r for r in rows if r["type"] == "probe"]
    print(f"\n## {path.name}: {len(refs)} reference tasks, {sum(r['success'] for r in refs)} successful, "
          f"{len(probes)} probe records")
    ends = Counter(t.get("think_end") for r in refs for t in r["turns"])
    print(f"reference THINK end reasons: {dict(ends)}")

for arg in sys.argv[1:]:
    run = Path(arg)
    print(f"# {run}")
    for path in sorted(run.glob("gate*.json")):
        gate(path)
    if (run / "atlas.jsonl").exists():
        atlas(run / "atlas.jsonl")
    for name in ("fit.log", "preregister.log", "analyze.log"):
        if (run / name).exists():
            print(f"\n## {name}\n" + (run / name).read_text().rstrip())
PY
}

case "${1:-}" in
  verify)   shift; cmd_verify "$@" ;;
  diagnose) shift; cmd_diagnose "$@" ;;
  pilot)    shift; cmd_pilot "$@" ;;
  sweep)    shift; cmd_sweep "$@" ;;
  summary)  shift; cmd_summary "$@" ;;
  *) sed -n '2,20p' "$0"; exit 2 ;;
esac
