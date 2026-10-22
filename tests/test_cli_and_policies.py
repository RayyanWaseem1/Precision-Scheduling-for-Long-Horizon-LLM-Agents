import json

from hazprec.agent.harness import AgentHarness
from hazprec.cli import main
from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
from hazprec.quant.backends import SimulatedBackend
from hazprec.scheduler import PacedPolicy
from hazprec.theory import load_preregistration


def test_paced_policy_tracks_target_average_bits():
    env = HorizonGymEnv(HorizonGymConfig(n_hops=8))
    pol = PacedPolicy([3, 4, 5, 6, 8], 4.5, {"system": 0.3, "read": 0.3, "think": 0.5, "act": 2.0, "stop": 6.0},
                      eta=0.5, ref_ratio=0.02)
    h = AgentHarness(SimulatedBackend(), pol)
    trajs = [h.run_episode(env, s) for s in range(40)]
    avg = sum(t.bit_tokens for t in trajs) / sum(t.tokens for t in trajs)
    assert abs(avg - 4.5) < 0.6
    # Sensitive, cheap tool calls get more bits than long observation prefills.
    segs = [s for t in trajs for turn in t.turns for s in turn.segments]
    act = [s.bits for s in segs if s.phase.value == "act"]
    read = [s.bits for s in segs if s.phase.value == "read"]
    assert sum(act) / len(act) > sum(read) / len(read)


def test_cli_end_to_end_small(tmp_path, capsys):
    over = ["--set", "atlas.n_tasks=4", "--set", "atlas.n_samples=5", "--set", "env.n_hops=2",
            "--set", "horizon.hops=[1,3]", "--set", "horizon.n_episodes=60"]
    atlas, pre, runs = tmp_path / "a.jsonl", tmp_path / "p.json", tmp_path / "h.jsonl"
    main(["atlas", "--out", str(atlas), *over])
    main(["preregister", "--atlas", str(atlas), "--out", str(pre), *over])
    main(["horizon", "--prereg", str(pre), "--out", str(runs), *over])
    body = load_preregistration(pre)
    assert {r["hops"] for r in body["per_hops"]} == {1, 3}
    summary = json.loads((tmp_path / "h.summary.json").read_text())
    assert len(summary["rows"]) == 2
    # Re-running is a no-op thanks to resume keys.
    n_lines = len(runs.read_text().splitlines())
    main(["horizon", "--prereg", str(pre), "--out", str(runs), *over])
    assert len(runs.read_text().splitlines()) == n_lines
    main(["play", "--policy", "uniform:8", "--seed", "1"])
    assert "success=" in capsys.readouterr().out


def test_cli_enforces_identity_completeness_and_preserves_demo_prereg(tmp_path):
    import pytest
    over=['--set','atlas.n_tasks=4','--set','atlas.n_samples=4','--set','env.n_hops=1',
          '--set','horizon.hops=[1,2]','--set','horizon.n_episodes=10']
    out=tmp_path/'demo'
    main(['demo','--out',str(out),*over])
    pre=(out/'prereg.json').read_bytes()
    main(['demo','--out',str(out),*over])
    assert (out/'prereg.json').read_bytes()==pre
    with pytest.raises(ValueError,match='incompatible'):
        main(['atlas','--out',str(out/'atlas.jsonl'),*over,'--set','atlas.n_samples=5'])
    with pytest.raises(ValueError,match='differ from preregistration'):
        main(['horizon','--out',str(tmp_path/'new.jsonl'),'--prereg',str(out/'prereg.json'),
              *over,'--set','horizon.n_episodes=12'])
    sweep=out/'horizon.jsonl'
    sweep.write_text('\n'.join(sweep.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='incomplete'):
        main(['analyze','--runs',str(sweep),'--prereg',str(out/'prereg.json')])
    atlas=out/'atlas.jsonl'
    atlas.write_text('\n'.join(atlas.read_text().splitlines()[:-1])+'\n')
    with pytest.raises(ValueError,match='incomplete'):
        main(['preregister','--atlas',str(atlas),'--out',str(tmp_path/'new-pre.json'),*over])
