import math

import numpy as np
import pytest

from hazprec.theory import (
    allocate_dp,
    allocate_exact,
    allocate_greedy,
    am_gm,
    delta_table,
    excess_exp2,
    h_star_from_hazards,
    h_star_from_lines,
    kappa,
    load_preregistration,
    phase_table,
    preregister,
    waterfill,
)


def test_waterfill_equal_costs_gives_am_gm_ratio():
    rng = np.random.default_rng(0)
    s = np.exp(rng.normal(0, 1.0, 12))
    c = np.ones_like(s)
    bbar = 4.0
    b = waterfill(s, c, bbar * c.sum(), bmin=-50, bmax=50)  # no clipping
    assert np.isclose(b.mean(), bbar)
    ratio = np.sum(s * 2.0 ** (-2 * b)) / (np.sum(s) * 2.0 ** (-2 * bbar))
    am, gm = am_gm(s)
    assert np.isclose(ratio, gm / am, rtol=1e-8)
    # Optimality: excess hazard per unit cost is equalized.
    marg = s * 2.0 ** (-2 * b) / c
    assert np.allclose(marg, marg[0])


def test_waterfill_matches_closed_form_with_costs():
    s = np.array([4.0, 1.0, 0.25])
    c = np.array([1.0, 2.0, 4.0])
    b = waterfill(s, c, budget=4.0 * c.sum(), bmin=-50, bmax=50)
    beta = b - 0.5 * np.log2(s / c)
    assert np.allclose(beta, beta[0])


def test_uniform_is_optimal_when_sensitivity_per_cost_is_constant():
    s = np.array([2.0, 2.0, 2.0])
    c = np.array([1.0, 1.0, 1.0])
    assert np.allclose(waterfill(s, c, 12.0), 4.0)


def test_greedy_close_to_exact_and_within_budget():
    rng = np.random.default_rng(1)
    levels = [3, 4, 5, 6, 8]
    worst = 1.0
    for _ in range(60):
        n = rng.integers(2, 6)
        s = np.exp(rng.normal(0, 1.5, n))
        c = rng.integers(5, 200, n).astype(float)
        budget = 4.5 * c.sum()
        d = delta_table(s, levels)
        g = allocate_greedy(d, c, levels, budget)
        e = allocate_exact(d, c, levels, budget)
        p = allocate_dp(d, c, levels, budget)
        for alloc in (g, e, p):
            assert np.dot(c, alloc) <= budget + 1e-9
        val = lambda a: sum(excess_exp2(si, bi) for si, bi in zip(s, a))  # noqa: E731
        assert val(e) <= val(g) + 1e-12
        assert np.isclose(val(p), val(e))  # DP is exact
        worst = max(worst, val(g) / val(e))
    assert worst < 2.0  # greedy is a heuristic with heterogeneous costs; DP is the exact path


def test_dp_scales_to_long_per_step_schedules():
    rng = np.random.default_rng(5)
    levels = [3, 4, 5, 6, 8]
    s = np.exp(rng.normal(0, 1.2, 200))
    c = rng.integers(10, 120, 200).astype(float)
    budget = 4.0 * c.sum()
    p = allocate_dp(delta_table(s, levels), c, levels, budget)
    g = allocate_greedy(delta_table(s, levels), c, levels, budget)
    val = lambda a: sum(excess_exp2(si, bi) for si, bi in zip(s, a))  # noqa: E731
    assert np.dot(c, p) <= budget and val(p) <= val(g) + 1e-12


def test_greedy_is_exact_with_equal_costs():
    rng = np.random.default_rng(2)
    levels = [3, 4, 5, 6, 7, 8]
    for _ in range(30):
        s = np.exp(rng.normal(0, 1.0, 5))
        c = np.ones(5)
        d = delta_table(s, levels)
        g = allocate_greedy(d, c, levels, 4.6 * 5)
        e = allocate_exact(d, c, levels, 4.6 * 5)
        assert np.isclose(sum(excess_exp2(a, b) for a, b in zip(s, g)), sum(excess_exp2(a, b) for a, b in zip(s, e)))


def test_phase_table_puts_bits_on_sensitive_cheap_phases():
    sens = {"read": 0.3, "think": 0.5, "act": 2.0, "stop": 6.0}
    tokens = {"read": 80, "think": 40, "act": 24, "stop": 24}
    freq = {"read": 2, "think": 2, "act": 2, "stop": 0.1}
    t = phase_table(sens, tokens, freq, [3, 4, 5, 6, 8], bbar=4)
    assert t["act"] > t["read"] and t["stop"] >= t["act"]


def test_h_star_formula_matches_numeric_peak():
    h0, A, G = 0.004, 0.01, 0.004
    hs = np.linspace(0, 500, 200001)
    gap = np.exp(-hs * (h0 + G)) - np.exp(-hs * (h0 + A))
    assert abs(hs[np.argmax(gap)] - h_star_from_hazards(h0, A, G)) < 0.01
    # The log-linear form agrees with the hazard form.
    assert math.isclose(h_star_from_lines(0, -(h0 + G), 0, -(h0 + A)), h_star_from_hazards(h0, A, G))


def test_kappa_and_am_gm():
    am, gm = am_gm([1.0, 4.0])
    assert am == 2.5 and math.isclose(gm, 2.0)
    assert math.isclose(kappa([1.0, 4.0], 2.0), 2 ** -4 * 0.5)


def test_preregistration_is_write_once_and_tamper_evident(tmp_path):
    p = tmp_path / "pre.json"
    preregister(p, {"slope": 0.007})
    with pytest.raises(FileExistsError):
        preregister(p, {"slope": 0.009})
    assert load_preregistration(p)["slope"] == 0.007
    p.write_text(p.read_text().replace("0.007", "0.009"))
    with pytest.raises(ValueError):
        load_preregistration(p)
