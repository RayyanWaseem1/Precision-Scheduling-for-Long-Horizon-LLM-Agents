# hazprec — Hazard-Optimal Precision Scheduling for Long-Horizon LLM Agents

Phase 1 codebase for the research project described in the proposal doc
*Hazard-Optimal Precision Scheduling for Long-Horizon LLM Agents*.

The question: given a fixed precision budget, how should an agent spend bits across the
steps of a trajectory so it fails least often? Compression is treated as a per-step
**hazard**; the theory says bits should go where sensitivity per token is highest, and
predicts that the advantage over uniform precision grows with horizon at a slope you can
compute (and pre-register) before running the experiment.

## What is in Phase 1

| Piece | Where | Status |
| --- | --- | --- |
| HorizonGym: controllable-horizon tool world with step-level verification | `src/hazprec/envs/horizongym/` | tested |
| Agent harness with per-segment precision (READ / THINK / ACT / STOP) | `src/hazprec/agent/harness.py` | tested |
| `PrecisionBackend` interface + shared token-level logic (append-only history, precision provenance, snapshot/restore) | `src/hazprec/quant/backends/base.py` | tested |
| Simulated oracle backend with a known hazard model | `quant/backends/simulated.py` | tested |
| PyTorch/transformers backend with fake-quantized weights (Colab GPU, or CPU) | `quant/backends/hf_fakequant.py` | tested on a tiny random Llama |
| MLX backend (MacBook): one resident copy per bit-width, shared KV cache | `quant/backends/mlx_backend.py` | GPU unit tests; validate your model with smoke/gate |
| Allocation: water-filling, exact DP, greedy, phase tables | `src/hazprec/theory/allocation.py` | tested |
| Predictions: AM–GM gain, κ, H*, HorizonGym survival lines | `src/hazprec/theory/predictions.py` | tested |
| Write-once, hash-checked pre-registration | `src/hazprec/theory/preregister.py` | tested |
| Replay-based sensitivity atlas (resumable JSONL) | `src/hazprec/sensitivity/` | tested |
| Survival model, bootstrap, paired horizon analysis | `src/hazprec/eval/` | tested |
| Policies P0 uniform, P1 phase table, P2 paced, P3 oracle phase | `src/hazprec/scheduler/policies.py` | P0/P1 tested; P2 implemented |
| CLI, configs, Colab notebook | `src/hazprec/cli.py`, `configs/`, `colab/` | |

Follow [the Phase 1 runbook](docs/PHASE1_RUNBOOK.md) for installation, verification,
real-model readiness checks, and complete experiments. The current code includes
append-only KV history, per-token precision provenance, matched atlas estimation,
strict configuration validation, and experiment manifests. Tests cover CPU/PyTorch
and no-download MLX models; MLX tests require Metal access.


## Install

```bash
# MacBook (Apple Silicon)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[mlx,dev]"

# Colab / any NVIDIA machine
pip install -e ".[cuda,dev]"

pytest -q            # HF tests are skipped automatically if torch is missing
```

## Quick start (no GPU, about a minute)

```bash
hazprec demo --out runs/demo
```

This runs the whole Phase 1–3 loop on the simulated backend:

1. **Atlas** — reference episodes at 8 bits; every turn on the correct prefix is replayed with
   one segment at 3/4/5/6 bits and the rest at 8; failure rates are recorded.
2. **Pre-registration** — per-phase curves are fitted, the adaptive schedule is allocated at a
   matched average budget for every horizon, and the predicted log-success-ratio slope and
   H* are written to a write-once JSON file with a SHA-256.
3. **Horizon sweep** — uniform vs adaptive on the same task seeds at each horizon; the measured
   slope (paired bootstrap CI) is compared with the pre-registered one.

## Command reference

```bash
hazprec play       -c CONFIG --policy uniform:4 --seed 0 [-v]
hazprec play       -c CONFIG --policy table:read=3,think=4,act=6,stop=8
hazprec atlas      -c CONFIG --out runs/atlas.jsonl            # resumable
hazprec fit        --atlas runs/atlas.jsonl
hazprec preregister -c CONFIG --atlas runs/atlas.jsonl --out prereg/NAME.json
hazprec horizon    -c CONFIG --prereg prereg/NAME.json --out runs/horizon.jsonl   # resumable
hazprec analyze    --runs runs/horizon.jsonl --prereg prereg/NAME.json
hazprec smoke      -c CONFIG [--hops 2]                        # load a real model, 1 episode per bit-width
```

Any config value can be overridden: `--set env.n_hops=8 --set atlas.n_samples=16`.

## Configs

| Config | Backend | Use |
| --- | --- | --- |
| `configs/sim.yaml` | simulated | pipeline and theory validation |
| `configs/mac_pilot.yaml` | MLX, Qwen3-1.7B, 3/4/6/8 bits | development on the MacBook |
| `configs/mac_pilot_3b.yaml` | MLX, Qwen2.5-3B-Instruct, 3/4/6/8 bits | pilot sensitivity atlas (≈8–9 GB) |
| `configs/colab_8b.yaml` | HF fake-quant, Qwen3-8B, 3–8 bits + BF16 | headline numbers on A100 80 GB / H100 |

## First run on the Mac

Run these checks on the Mac before collecting an atlas:

```bash
pip install -e ".[mlx,dev]"
pytest -q
hazprec smoke -c configs/mac_pilot.yaml               # converts the model to 3/4/6/8-bit MLX copies once
hazprec play  -c configs/mac_pilot.yaml --policy uniform:8 -v
```

What to look for: each bit-width loads; the agent emits `<tool_call>{...}</tool_call>`;
at 8 bits it usually completes a 2-hop task. Do not judge a model on 1-hop tasks: Qwen3-1.7B
failed 30 of 30 at 8 bits, re-reading the only record instead of submitting. If the model ignores the
tool format, try `--set agent.think=false` or a different model before changing code.
Results of the first real-model runs are in [docs/FINDINGS.md](docs/FINDINGS.md).

## Colab

Upload the repo zip to `MyDrive/hazprec/` and open `colab/hazprec_colab.ipynb`. Every
atlas and sweep record is appended to Drive. Resume with the identical source/config
and dependencies; experiment manifests reject incompatible reuse.

## Design notes

- **Precision semantics.** Each segment runs at one weight precision, and the KV cache is
  shared across precisions, so a cached token keeps the precision it was computed at.
  The HF backend swaps fake-quantized weights in place; the MLX backend switches between
  resident quantized copies that share one cache.
- **Fake quantization** (group-wise asymmetric RTN) gives the numbers a real low-bit kernel
  would, up to accumulation order. That is what hazard measurement needs. Wall-clock and
  bytes-moved numbers come later from real kernels (Any-Precision LLM, Triton bitplane KV).
- **Budget currency.** Bits are weighted by tokens processed per segment (prefill and decode).
  The adaptive schedule is allocated per horizon so its expected average never exceeds the
  uniform b̄. Realized averages can drift by about ±0.1 bit because failed episodes end early;
  the analysis reports them for every horizon.
- **Replay protocol.** Only turns on the reference trajectory's correct prefix are replayed.
  Each sample restores the exact reference prefix and its original precision provenance
  before processing the observation at the probe precision.
- **Phase hinter.** Policies see whether the next call will end the episode through a
  deployable hinter that uses only the agent's own observations, never the oracle.
  `OraclePhasePolicy` (P3) gets the true phase and is an upper bound only.

## Repository layout

```
configs/                 YAML configs (sim, Mac pilot, Colab 8B)
colab/hazprec_colab.ipynb
docs/demo_output.txt     historical simulated output (before estimator repairs)
scripts/phase1_gate.py   real-model readiness and measured replay cost
src/hazprec/
  phases.py              phase taxonomy and segment records
  config.py              config loading and overrides
  pipeline.py            atlas fit -> adaptive schedules -> predictions
  cli.py
  agent/                 tool-call format, harness
  envs/horizongym/       generator, environment, checker, phase hinter
  quant/                 fake quantization, backends
  scheduler/             precision policies
  sensitivity/           replay atlas and curve fitting
  theory/                allocation, predictions, pre-registration
  eval/                  statistics, survival model, horizon analysis
tests/
```

## Next (Phase 2 onward)

1. Mac: run the smoke test, then a small atlas with `configs/mac_pilot.yaml` to check
   tool-call reliability at 8 bits and the cost per task.
2. Colab: `configs/colab_8b.yaml` atlas on 100 tasks; measure throughput; scale up.
3. Add the KL proxy (teacher-forced token KL per step) to the atlas for the E2 alignment
   analysis, and the entropy-router baseline.
4. Train the sensitivity probe (P2) on atlas labels.
5. Joint weight + KV scheduling (bitplane KV, precision decay) in Phase 5.
