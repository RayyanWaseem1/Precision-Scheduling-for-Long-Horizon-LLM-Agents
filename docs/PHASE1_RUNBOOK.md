# Running Phase 1 and preparing for Phase 2

Updated October 22, 2026. Results and interpretation of the first real-model runs
(Qwen3-1.7B, Qwen2.5-3B) are kept in `docs/FINDINGS.md`. The prerequisite fixes and the reviewed correctness
repairs are implemented in this checkout. Do not apply either patch to this source
again. `docs/phase1-prerequisites.patch` is historical; the root KV patch is a
replacement diff against the source immediately before the latest repairs.

The route is: install → tests → simulation → real-model smoke → readiness gate →
small atlas → full pilot → preregistration → paired evaluation. The gate is an
engineering checkpoint; its thresholds do not prove the research hypothesis.

1. **Install in the project environment.**

   ```bash
   cd /Users/rayyanwaseem/Desktop/Projects/Precision-Scheduling-for-Long-Horizon-LLM-Agents
   source .venv/bin/activate
   python -m pip install -e ".[mlx,dev]"
   ```

   Create `.venv` with `python3 -m venv .venv` first on a fresh checkout. On an
   NVIDIA machine install `.[cuda,dev]`. To also run PyTorch CPU tests on your Mac,
   install `.[mlx,cuda,dev]`; the extra's name does not require an NVIDIA GPU.
   The package requires Python 3.10+ and Transformers 4.51+ for the configured
   Qwen3 models ([Qwen model card](https://huggingface.co/Qwen/Qwen3-1.7B#quickstart)).

2. **Compile and test.**

   ```bash
   python -m compileall -q src tests scripts
   python -m pytest -q
   ```

   PyTorch tests use a tiny random Llama without model downloads. MLX tests use a
   tiny random Qwen3 and require a Metal device. A skipped backend module means
   that backend was not validated by that run. A sandbox may block Metal even on
   a supported Mac; run those tests from a normal terminal.

3. **Run the simulated end-to-end experiment.**

   ```bash
   hazprec demo -c configs/sim.yaml --out runs/sim_phase1_v2
   hazprec analyze --runs runs/sim_phase1_v2/horizon.jsonl \
     --prereg runs/sim_phase1_v2/prereg.json
   ```

   The full demo uses 30 atlas tasks, 40 samples per setting, and 18,000 evaluation
   episodes across six horizons and two policies. It creates an atlas, a hashed
   preregistration, a sweep, a summary, and experiment manifests (`*.meta.json`).
   Re-running the identical command resumes it and preserves its preregistration.
   Changing config, source, or dependencies requires a fresh output path.

   For a smaller infrastructure check:

   ```bash
   hazprec demo -c configs/sim.yaml --out runs/sim_check_v2 \
     --set atlas.n_tasks=6 --set atlas.n_samples=10 \
     --set 'horizon.hops=[1,2,4]' --set horizon.n_episodes=100
   ```

   Inspect complete paired counts, the measured slope/interval, and realized bits.
   `docs/demo_output.txt` is historical and predates the new estimator; do not expect
   identical numbers. A prediction outside its interval is a result to investigate,
   not a reason to rewrite predictions after evaluation.

4. **Smoke-test the Mac model.**

   ```bash
   hazprec smoke -c configs/mac_pilot.yaml
   hazprec play -c configs/mac_pilot.yaml --policy uniform:8 \
     --set env.n_hops=2 --seed 0 -v
   ```

   The first run downloads Qwen3-1.7B and converts 3/4/6/8-bit copies in
   `~/.cache/hazprec/mlx`. Check that every level loads and executes, calls parse,
   and the 8-bit reference can finish read → update → read → update → submit. Smoke
   defaults to 2 hops: Qwen3-1.7B failed 30 of 30 1-hop tasks at 8 bits (29 by
   re-reading the record instead of submitting), so 1 hop says nothing about readiness.
   Low-bit task failures can be legitimate measurements. Repeated reference failures
   need investigation.

   For malformed/truncated calls, inspect `-v`, try a separate diagnostic with
   `--set agent.think=false`, or raise `agent.max_act_tokens`. The harness's THINK
   segment is distinct from Qwen's `enable_thinking` chat-template option. Save
   any adopted settings before collecting the atlas; they are part of its identity.

5. **Run the real-model readiness gate.**

   ```bash
   mkdir -p reports
   set -o pipefail
   caffeinate -i "$PWD/.venv/bin/python" scripts/phase1_gate.py -c configs/mac_pilot.yaml \
     --out reports/gate_mac_v2.json 2>&1 | tee reports/gate_mac_v2.txt
   ```

   `pipefail` preserves the gate's nonzero exit status through `tee`. Pass the venv's
   interpreter to `caffeinate` by path: `caffeinate -i python ...` failed with
   `ModuleNotFoundError: No module named 'hazprec'` on the Mac while plain `python`
   worked. Alternatively run `caffeinate -i -w $$ &` once to keep the Mac awake for
   the life of the shell. The gate:

   | Check | Required behavior |
   | --- | --- |
   | Format | Reference tool-call parse rate at least 0.98 |
   | Reliability | Reference success at least 0.80 at every tested horizon |
   | Signal | Lowest-bit per-turn hazard interval above the reference interval at the gate horizon |
   | Operating point | Uniform precision at `theory.bbar`, at its measured per-turn hazard, keeps predicted success of at least 0.10 at every sweep horizon (`--min-sweep-survival`) |
   | Cache | Every tested episode preserves precision counts, appends history, and has valid observation overhead |
   | Cost | Timed mixed-precision replay estimates an atlas within the configured budget |

   It also executes one episode at every configured bit width. Missing checks
   prevent approval. Reference, `theory.bbar` and lowest precision must be requested; gate
   seeds are checked against atlas/evaluation seeds. Cost uses actual replay,
   including boundary forwards and cache reconstruction, conservatively scaled
   to full reference-path coverage. It remains an estimate, not a runtime guarantee.

   By default the gate runs 30 episodes at the shortest sweep horizon and the atlas
   horizon (`env.n_hops`, also the default `--gate-hops`), at the reference,
   `theory.bbar` and lowest precisions. The operating-point check exists because a
   uniform arm with zero successes makes the smoothed log ratio, and so the measured
   slope, an artifact of the smoothing constant. Each cell prints up to two failing
   turns: what the model wrote, what the checker expected, and how THINK and ACT ended
   (`tag`, `eos` for an end-of-turn token, `cap` for the token budget, or `skipped`
   when the model ended its turn before a tool call). The JSON report keeps every
   failing turn and per-level THINK-length and end-reason counts. Cost calibration
   uses one held-out task and two samples per setting; increase `--cost-tasks` for
   a more representative estimate. `--atlas-budget-hours` defaults to 24.

   `--quick` runs a shorter diagnostic and always returns nonzero because it does
   not certify full readiness. Simulated gates likewise cannot certify a real
   model. A failed check is not permission to skip it. Investigate failures and
   choose any revised task, model, thresholds, or budget before the final gate.
   Preserve the JSON report, which records all thresholds and resolved settings.

   The optional `configs/mac_pilot_3b.yaml` uses Qwen2.5-3B. Smoke-test it before
   running its gate. Use separate output paths for each model/configuration.

6. **Freeze a config and collect a small atlas.**

   ```bash
   mkdir -p runs/mac_check_v2
   python - <<'PY'
   from pathlib import Path
   import yaml
   from hazprec.config import load_config
   cfg = load_config('configs/mac_pilot.yaml', ['atlas.n_tasks=5', 'atlas.n_samples=2'])
   Path('runs/mac_check_v2/config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
   PY
   time hazprec atlas -c runs/mac_check_v2/config.yaml --out runs/mac_check_v2/atlas.jsonl
   hazprec fit --atlas runs/mac_check_v2/atlas.jsonl --out runs/mac_check_v2/fit.json
   ```

   Include any gate-approved overrides in `load_config` before collecting data.
   At three hops and four levels this check uses at most 700 replayed turns plus
   reference work. With `agent.think=false`, THINK probes are omitted.

   Inspect reference success and phase coverage:

   ```bash
   python - <<'PY'
   from collections import Counter
   from hazprec.utils.io import read_jsonl
   from hazprec.sensitivity.fit import phase_key
   rows = read_jsonl('runs/mac_check_v2/atlas.jsonl')
   refs = [r for r in rows if r['type'] == 'reference']
   probes = [r for r in rows if r['type'] == 'probe' and r['probe'] != 'ref']
   print('Reference successes:', sum(r['success'] for r in refs), '/', len(refs))
   print('Phase coverage:', dict(Counter(phase_key(r) for r in probes)))
   PY
   ```

   With thinking enabled, require system/read/think/act/stop coverage. Allocation
   rejects missing measurements instead of inventing sensitivities or token costs.
   Excess hazard is estimated against the reference at the same task and turn.
   Reported intervals include reference sampling uncertainty and are conservative,
   conditional on the observed contexts; they are not population-level task CIs.

7. **Collect the full pilot in a separate run.**

   ```bash
   mkdir -p runs/mac_phase1_v2
   python - <<'PY'
   from pathlib import Path
   import yaml
   from hazprec.config import load_config
   cfg = load_config('configs/mac_pilot.yaml')
   Path('runs/mac_phase1_v2/config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
   PY
   python -m pip freeze > runs/mac_phase1_v2/requirements.txt
   time hazprec atlas -c runs/mac_phase1_v2/config.yaml --out runs/mac_phase1_v2/atlas.jsonl
   hazprec fit --atlas runs/mac_phase1_v2/atlas.jsonl --out runs/mac_phase1_v2/fit.json
   ```

   The default corrected Mac config uses 20 tasks × up to seven turns × ten
   settings × four samples = at most 5,600 replayed turns. Preserve a copy/commit
   of the exact source and model artifacts along with the config and dependencies.
   Keep the same model revision/files throughout; a remote model name alone is
   not an immutable weights identifier.

8. **Pre-register, preserve the predictions, and evaluate.**

   ```bash
   hazprec preregister -c runs/mac_phase1_v2/config.yaml \
     --atlas runs/mac_phase1_v2/atlas.jsonl --out prereg/mac_phase1_v2.json
   ```

   Preregistration requires a complete atlas with the configured tasks, probe
   settings, and sample counts. Inspect predicted tables, expected average bits,
   slope, and H*. Preserve/commit this file before running evaluation. It is
   write-once. No adaptive win or particular slope is required for valid science.

   ```bash
   time hazprec horizon -c runs/mac_phase1_v2/config.yaml \
     --prereg prereg/mac_phase1_v2.json --out runs/mac_phase1_v2/horizon.jsonl
   hazprec analyze --runs runs/mac_phase1_v2/horizon.jsonl --prereg prereg/mac_phase1_v2.json
   ```

   The default Mac sweep has four horizons × 100 seeds × two policies = 800 records.
   Evaluation verifies source/dependencies/config and the atlas hash against the
   preregistration. Analysis requires every registered record exactly once and
   identical paired seeds across horizons. Its bootstrap resamples seed IDs jointly
   across horizons and policies. Finish partial sweeps before final analysis.

9. **Resume the identical experiment safely.**

   Re-run its original atlas or horizon command. Keep the `.meta.json` sidecars
   with the data. The CLI rejects legacy data without manifests and rejects changed
   source, dependencies, configuration, or preregistration identity. Atlas resume
   also verifies regenerated reference trajectories against stored text/prefix
   hashes. If a reference changes, start a new run rather than mixing trajectories.

   Incomplete final JSONL fragments are preserved in `.torn` files before the
   writer resumes. Corrupt interior lines and duplicate keys cause errors. Use one
   writer per output path. Do not change manifests to bypass a mismatch. Lower-level
   `run_atlas` and `run_horizon_sweep` callers must supply
   `experiment=identity(resolved_config)`; sweeps also receive `prereg_sha`.

10. **Understand the corrected precision accounting.**

    Generated tokens retain their original KV precision. The next observation is
    appended using a collision-checked template marker; unsupported templates fail
    instead of silently re-prefilling history. Snapshots include IDs, precisions,
    conversation state, and logits, and can reconstruct a diverged prefix.

    A model that ends its turn during THINK gets no tool call: nothing is decoded
    after its end-of-turn token, and the turn fails as unparseable with
    `think_end: eos` and `act_end: skipped`. Before this rule the harness forced
    `<tool_call>` after the end-of-turn token, which put off-template text into the
    KV history of every later turn.

    At a precision boundary, a one-token forward on a disposable cache copy produces
    the first sampling distribution at the requested precision. Existing KV is
    preserved. `boundary_tokens` records this extra work and is included in the
    segment's token/bit cost. Copying cache state has memory/time overhead; actual
    replay timing includes it. The scheduler still allocates using reference token
    costs, so inspect realized budgets rather than assuming exact equality.

11. **Hand off to Colab/NVIDIA and Phase 2.**

    `configs/colab_8b.yaml` uses Qwen3-8B, BF16 reference, and five lower precisions.
    Its rough weight-storage estimate is 72 GB before runtime overhead. Inspect
    actual GPU VRAM with `nvidia-smi`; high system RAM does not imply GPU capacity.
    Choose a smaller model or fewer levels in a fresh config if it does not fit.

    Package this checkout with the directory name the notebook expects:

    ```bash
    python - <<'PY'
    from pathlib import Path
    from zipfile import ZipFile, ZIP_DEFLATED
    roots = ['src', 'tests', 'scripts', 'configs', 'colab', 'docs', 'README.md', 'pyproject.toml']
    with ZipFile('hazard-precision.zip', 'w', ZIP_DEFLATED) as z:
        for name in roots:
            p = Path(name)
            for f in ([p] if p.is_file() else p.rglob('*')):
                if f.is_file() and '__pycache__' not in f.parts:
                    z.write(f, 'hazard-precision/' + f.as_posix())
    PY
    ```

    Upload the zip to `MyDrive/hazprec/` and open `colab/hazprec_colab.ipynb`.
    Its subprocess calls propagate failures, and the atlas cell requires a successful
    gate report for the same experiment. On a reconnect, rerun setup with the same
    source/dependency versions; retained reports are verified before reuse.

    The full 8B atlas can require 166,400 replayed turns plus references; the sweep
    has 3,000 episodes. Measure cost before scaling. KL/entropy logging, learned
    sensitivity probes, and batched replay remain Phase 2 engineering, not features
    supplied by these repairs. Joint weight/KV scheduling comes later.

    Keep the atlas, fit, preregistration, sweep, summary, manifests, gate report,
    model/source snapshot, and dependency list. Sweep files contain episode summaries;
    survival helpers requiring individual turns need separately saved trajectories.
