from hazprec.agent.harness import AgentHarness
from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
from hazprec.phases import Phase
from hazprec.quant.backends import HazardModel, SimulatedBackend
from hazprec.scheduler import PhaseTablePolicy, UniformPolicy


def success_rate(bits_policy, n=300, hops=6, hazard=None):
    env = HorizonGymEnv(HorizonGymConfig(n_hops=hops))
    h = AgentHarness(SimulatedBackend(hazard or HazardModel()), bits_policy)
    trajs = [h.run_episode(env, seed) for seed in range(n)]
    return sum(t.success for t in trajs) / n, trajs


def test_reference_precision_is_reliable():
    rate, trajs = success_rate(UniformPolicy(8))
    assert rate > 0.9
    t = trajs[0]
    assert len(t.turns) == 13 and t.success
    phases = [s.phase for s in t.turns[0].segments]
    assert phases == [Phase.SYSTEM, Phase.THINK, Phase.ACT]
    assert t.turns[-1].sched_phase is Phase.STOP  # hinter predicts the submit turn
    assert t.avg_bits == 8
    assert all((u.think_end, u.act_end) == ("tag", "tag") for u in t.turns)
    assert t.to_dict()["turns"][0]["act_end"] == "tag"


def test_low_precision_fails_more_and_failures_are_logged():
    hi, _ = success_rate(UniformPolicy(8))
    lo, trajs = success_rate(UniformPolicy(3))
    assert lo < hi - 0.2
    failed = [t for t in trajs if not t.success]
    assert failed and all(t.first_failure_step is not None for t in failed)
    assert all(not t.turns[-1].correct for t in failed)


def test_phase_table_routes_bits_by_phase():
    pol = PhaseTablePolicy({"read": 3, "think": 4, "act": 6, "stop": 8})
    _, trajs = success_rate(pol, n=5)
    t = next(t for t in trajs if t.success)
    first, last = t.turns[0], t.turns[-1]
    assert [s.bits for s in first.segments] == [3, 4, 6]  # SYSTEM falls back to READ bits
    assert last.segments[-1].bits == 8 and last.expected_phase is Phase.STOP


def test_paired_seeds_give_identical_trajectories():
    a, ta = success_rate(UniformPolicy(4), n=50)
    b, tb = success_rate(UniformPolicy(4), n=50)
    assert [t.success for t in ta] == [t.success for t in tb]
