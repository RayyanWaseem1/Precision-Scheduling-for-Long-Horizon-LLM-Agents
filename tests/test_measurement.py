import numpy as np

from hazprec.envs.horizongym import HorizonGymConfig
from hazprec.eval.horizon import analyze_horizon
from hazprec.eval.survival import fit_hazard_model, hazard_by_turn
from hazprec.quant.backends import HazardModel, SimulatedBackend
from hazprec.quant.fake_quant import fake_quant_np
from hazprec.sensitivity import fit_atlas, measure_task, phase_key


def test_fake_quant_error_falls_about_4x_per_bit():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(256, 512)).astype(np.float32)
    bits = [3, 4, 5, 6, 8]
    mse = [float(np.mean((fake_quant_np(w, b, 128) - w) ** 2)) for b in bits]
    slope = np.polyfit(bits, np.log2(mse), 1)[0]
    assert -2.3 < slope < -1.7  # theory: -2


def test_fit_recovers_known_sensitivities_from_synthetic_records():
    rng = np.random.default_rng(0)
    truth = {"read": 0.3, "think": 0.5, "act": 2.0, "stop": 6.0}
    base = 0.003
    recs = []
    for seed, (probe, exp_phase, key) in enumerate([
        ("read", "act", "read"), ("think", "act", "think"),
        ("act", "act", "act"), ("act", "stop", "stop")]):
        recs.append({"type": "probe", "task_seed": seed, "probe": "ref", "bits": 8,
                     "ref_bits": 8, "turn": 3, "expected_phase": exp_phase,
                     "n": 200000, "failures": int(rng.binomial(200000, base))})
        for b in (3, 4, 5, 6):
            p = base + truth[key] * 2.0 ** (-2 * b)
            n = 200000
            recs.append({"type": "probe", "task_seed": seed, "probe": probe, "bits": b,
                         "ref_bits": 8, "turn": 3, "expected_phase": exp_phase,
                         "n": n, "failures": int(rng.binomial(n, p))})
    fit = fit_atlas(recs)
    for k, s in truth.items():
        assert abs(fit.curves[k].s - s) / s < 0.1, k
    slope = fit.curves["act"].log2_slope
    assert slope is not None, "Expected a fitted slope for the synthetic ACT records"
    assert -2.4 < slope < -1.6
    assert phase_key({"probe": "read", "turn": 0}) == "system"


def test_replay_atlas_on_simulated_backend_recovers_act_sensitivity():
    hz = HazardModel({"system": 0.3, "read": 0.3, "think": 0.5, "act": 2.0, "stop": 6.0}, 0.002)
    be = SimulatedBackend(hz)
    cfg = HorizonGymConfig(n_hops=3)
    recs = []
    for seed in range(12):
        recs += list(measure_task(be, cfg, seed, (3, 4, 8), 8, n_samples=60))
    assert any(r["type"] == "reference" for r in recs)
    fit = fit_atlas(recs)
    assert 1.4 < fit.curves["act"].s < 2.6
    assert fit.curves["stop"].s > fit.curves["act"].s
    assert fit.tokens["act"] == 24 and abs(fit.freq_per_hop["act"] - 2.0) < 0.3


def test_survival_model_recovers_bit_effect():
    rng = np.random.default_rng(3)
    trajs = []
    for seed in range(4000):
        b = int(rng.choice([3, 4, 8]))
        turns = []
        for t in range(10):
            p = 0.002 + 2.0 * 2.0 ** (-2 * b)
            fail = rng.random() < p
            turns.append({"turn": t, "step_index": t, "expected_phase": "act", "correct": not fail,
                          "segments": [{"phase": "read", "bits": 8, "tokens": 50},
                                       {"phase": "act", "bits": b, "tokens": 20}]})
            if fail:
                break
        trajs.append({"seed": seed, "turns": turns, "policy": {"name": "x"}})
    m = fit_hazard_model(trajs, features=("turn", "act_bits"))
    # logit(p) ~ log(p) for small p: coefficient on 2^-2b should be near 2/0.002-ish scale > 0
    assert m["coef"]["act_2^-2b"] > 0 and m["coef"]["act_2^-2b"] / m["se"]["act_2^-2b"] > 5
    table = hazard_by_turn(trajs)
    assert table[0]["at_risk"] == 4000


def test_horizon_analysis_recovers_known_slope():
    rng = np.random.default_rng(4)
    recs = []
    for hops in (2, 8, 16, 32):
        for seed in range(3000):
            u = rng.random()
            su = np.exp(-0.02 * hops)
            sa = np.exp(-0.012 * hops)
            recs.append({"hops": hops, "policy_label": "uniform", "seed": seed, "success": u < su, "avg_bits": 4})
            recs.append({"hops": hops, "policy_label": "adaptive", "seed": seed, "success": u < sa, "avg_bits": 4})
    an = analyze_horizon(recs, prediction={"log_ratio_slope_per_hop": 0.008, "h_star_hops": None}, n_boot=400)
    lo, hi = an["fit"]["slope_ci95"]
    assert lo < 0.008 < hi and an["prediction"]["slope_inside_ci"]
