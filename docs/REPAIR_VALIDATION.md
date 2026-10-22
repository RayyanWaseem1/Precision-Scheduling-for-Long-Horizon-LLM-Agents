# Repair validation — October 21, 2026

All repairs identified in the repository review are implemented in this checkout.
The replacement root patch contains these changes against the pre-repair snapshot;
it should not be applied again. The superseded patch is preserved in
`docs/archive/original-kv-history.patch`.

## Implemented repairs

- Preserve generated history and per-token KV precision when appending observations,
  including chat-template marker collisions. Restore divergent cache snapshots at
  their original precision and verify the actual continuation, not just saved logits.
- Recompute the first-token distribution at a precision boundary without changing
  the persistent prefix cache, and charge the extra forward to that segment.
- Compare sensitivity probes with matching task/turn reference samples, include
  reference uncertainty, and require measured phase coverage before allocation.
- Correct constant and one-sided quantization ranges and prevent half-precision
  scale underflow.
- Bind resumable outputs and preregistrations to resolved configuration, source,
  dependency versions, and data hashes. Reject incomplete or incompatible inputs.
- Recover torn final JSONL writes without swallowing subsequent records; reject
  interior corruption and duplicate completion keys.
- Require complete paired evaluation data and bootstrap paired seeds jointly across
  policies and horizons.
- Validate configuration keys, precision levels, counts, horizons, and gate arguments.
  Require every readiness check, prevent simulated/quick runs from certifying
  readiness, and estimate cost using timed mixed-precision replay.
- Update backend tests, installation requirements, the Phase 1 runbook, and Colab
  execution so failures propagate and preregistrations are preserved.

## Validation performed

- Full test suite with Metal access: **81 passed**. Backend tests use tiny random
  PyTorch Llama and MLX Qwen3 models without downloads. MLX tests exercise 3/4/6/8-bit
  execution, boundary cache preservation, and snapshot continuation.
- Full simulation: `hazprec demo -c configs/sim.yaml --out runs/validation_sim_v2`
  completed atlas collection, preregistration, and all **18,000 evaluation episodes**.
  The atlas has 3,436 records. The fitted log-ratio slope was approximately 0.0078
  per hop, with a 95% interval of [0.0067, 0.0090]; the preregistered slope was 0.0076.
  These are simulated validation results, not evidence for a real-model benefit.
- Real Qwen3-1.7B on MLX: all four configured precisions loaded and executed. A
  one-task, one-sample, three-turn atlas at 4/8 bits completed mixed-precision replay
  and fitting (`runs/validation_mlx_v2/atlas.jsonl`, 13 records). This checks the
  real-model execution path; it is too small for research conclusions.
- Validation environment included Torch 2.14.1, Transformers 5.20.2, MLX 0.32.4,
  and mlx-lm 0.32.2. Run manifests retain the source/dependency identities.

## Remaining readiness requirement

The default Qwen3-1.7B reference failed the one-hop task in the real smoke test:
after reading and updating, it issued another read instead of submitting. A
diagnostic with harness thinking disabled also failed. No passing real-model
readiness gate is claimed, and no full real-model atlas or horizon evaluation has
been certified.

Phase 2 therefore remains blocked on reference-model reliability and a passing
real-model gate. Investigate the model/task settings, freeze any adopted changes,
and follow `PHASE1_RUNBOOK.md` before collecting the full pilot. The repaired gate
must not be bypassed. Model files/revisions must also be preserved: a model name
alone is not an immutable weights identifier.
