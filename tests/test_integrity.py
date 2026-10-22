import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from hazprec.config import load_config
from hazprec.eval.horizon import analyze_horizon
from hazprec.pipeline import adaptive_schedule
from hazprec.quant.fake_quant import fake_quant_np
from hazprec.sensitivity.fit import fit_atlas
from hazprec.utils.experiment import ensure_experiment, identity, verify_experiment
from hazprec.utils.io import JsonlWriter, read_jsonl


@pytest.mark.parametrize('value', [0.0, 1.0, -1.0, 1e-7, -1e-7])
def test_constant_weight_groups(value):
    weights = np.full((2,128),value,dtype=np.float32)
    np.testing.assert_allclose(fake_quant_np(weights,4),weights,atol=1e-9,rtol=1e-5)


@pytest.mark.parametrize('sign',[1,-1])
def test_one_sided_groups(sign):
    weights = sign * np.linspace(.1,2,128,dtype=np.float32)[None,:]
    assert np.max(np.abs(fake_quant_np(weights,4)-weights)) <= 2/15 + 1e-6


def paired_records():
    rows=[]
    for turn,phase,fail in [(0,'act',0),(1,'stop',50)]:
        for probe,bits in [('ref',8),('act',4)]:
            rows.append(dict(type='probe',task_seed=0,turn=turn,expected_phase=phase,
                             probe=probe,bits=bits,ref_bits=8,n=100,failures=fail))
    return rows


def test_no_spurious_phase_sensitivity_from_baseline_difficulty():
    fit=fit_atlas(paired_records())
    for c in fit.curves.values():
        assert c.s == 0
        assert c.table[4]['delta'] == 0
        assert c.table[4]['lo'] < 0 < c.table[4]['hi']


def test_reference_uncertainty_is_included():
    rows=paired_records()[2:]
    wide=fit_atlas(rows).curves['stop'].table[4]
    rows[0]['n']=10000; rows[0]['failures']=5000
    narrow=fit_atlas(rows).curves['stop'].table[4]
    assert narrow['hi']-narrow['lo'] < wide['hi']-wide['lo']


def test_unmatched_or_duplicate_probes_rejected():
    rows=paired_records()
    with pytest.raises(ValueError,match='matching'):
        fit_atlas(rows[1:])
    with pytest.raises(ValueError,match='duplicate'):
        fit_atlas(rows+rows[:1])
    with pytest.raises(ValueError,match='coverage'):
        adaptive_schedule(fit_atlas(rows),4,[3,4,8],3)


@pytest.mark.parametrize('override',[
    'atlas.n_task=5','agent.think=1','atlas.n_samples=0','backend.levels=[8,4]',
    'backend.ref_bits=16','theory.bbar=7','horizon.hops=[1]',
    'env.n_hops=9001','backend.hf.temperatur=1','seed=-1',
])
def test_config_rejects_invalid_settings(override):
    with pytest.raises(ValueError): load_config(overrides=[override])


def test_supported_configs_and_nested_overrides():
    for path in Path('configs').glob('*.yaml'): load_config(path)
    assert load_config(overrides=['backend.sim.base_hazard=0'])['backend']['sim']['base_hazard']==0


def test_torn_tail_quarantined_before_valid_append(tmp_path):
    path=tmp_path/'records.jsonl'
    path.write_bytes(b'{"first":1}\n{"broken":\xe2')
    with pytest.warns(RuntimeWarning,match='quarantined'):
        with JsonlWriter(path) as writer: writer.write({'second':2})
    assert read_jsonl(path)==[{'first':1},{'second':2}]
    assert path.with_name(path.name+'.torn').read_bytes()==b'{"broken":\xe2'


def test_missing_final_newline_preserves_record(tmp_path):
    path=tmp_path/'records.jsonl'; path.write_text('{"first":1}')
    with JsonlWriter(path) as w: w.write({'second':2})
    assert read_jsonl(path)==[{'first':1},{'second':2}]


def test_interior_corruption_is_not_silently_skipped(tmp_path):
    path=tmp_path/'records.jsonl'; text='{"broken":\n{"good":2}\n'; path.write_text(text)
    for action in (read_jsonl,JsonlWriter):
        with pytest.raises(ValueError,match='corrupt'): action(path)
    assert path.read_text()==text


def test_resume_identity_enforced(tmp_path):
    path=tmp_path/'records.jsonl'; cfg=load_config(); original=identity(cfg)
    ensure_experiment(path,original,'atlas')
    ensure_experiment(path,original,'atlas')
    changed=copy.deepcopy(cfg); changed['atlas']['n_samples']+=1
    with pytest.raises(ValueError,match='incompatible'):
        ensure_experiment(path,identity(changed),'atlas')
    with pytest.raises(ValueError,match='mismatch'):
        verify_experiment(path,original,'horizon','different-prereg')
    bare=tmp_path/'bare.jsonl'; bare.write_text('{"old":1}\n')
    with pytest.raises(ValueError,match='unversioned'):
        ensure_experiment(bare,original,'atlas')


def test_unequal_or_unpaired_sweep_rejected():
    records=[dict(hops=h,seed=s,policy_label=p,success=(s%2==0),avg_bits=4)
             for h,n in [(1,2),(2,4)] for s in range(n) for p in ['uniform','adaptive']]
    with pytest.raises(ValueError,match='identical paired seed'):
        analyze_horizon(records,n_boot=10)
    with pytest.raises(ValueError,match='unpaired'):
        analyze_horizon(records[:-1],n_boot=10)


@pytest.fixture
def gate():
    path=Path(__file__).parents[1]/'scripts/phase1_gate.py'
    spec=importlib.util.spec_from_file_location('phase1_gate',path)
    assert spec is not None and spec.loader is not None, f'Cannot load gate module from {path}'
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('args',[
    ['--bits','8'],['--bits','3'],['--bits','8','3'],['--n','0'],['--hops','1'],['--cost-samples','0'],
    ['--seed0','0'],['--min-ref-success','nan'],['--min-sweep-survival','1'],
])
def test_gate_rejects_missing_checks_before_model_loading(gate,args,monkeypatch):
    monkeypatch.setattr(gate,'make_backend',lambda *a: pytest.fail('loaded model before validating input'))
    with pytest.raises(SystemExit) as error: gate.main(['-c','configs/sim.yaml',*args])
    assert error.value.code==2


def test_simulated_gate_cannot_certify_real_model_ready(gate,tmp_path):
    report=tmp_path/'gate.json'
    result=gate.main(['-c','configs/sim.yaml','--quick','--out',str(report),
                      '--cost-samples','1','--set','atlas.max_turns=1'])
    body=json.loads(report.read_text())
    assert result==1 and body['ready'] is False
    assert {c['name'] for c in body['checks']}=={'format','reliable','signal','operating_point','cache','cost'}
    assert body['arguments']['hops']==[4] and body['arguments']['gate_hops']==4  # quick uses the atlas horizon
    assert next(c for c in body['checks'] if c['name']=='cache')['pass'] is False
    assert body['atlas_cost']['measured_probe_samples']>0


def test_gate_defaults_follow_the_experiment_horizons_and_budget(gate,tmp_path):
    report=tmp_path/'gate.json'
    gate.main(['-c','configs/sim.yaml','--n','3','--out',str(report),
               '--cost-samples','1','--set','atlas.max_turns=1'])
    body=json.loads(report.read_text())
    # sim.yaml: shortest sweep horizon 2, atlas horizon 4; reference 8, bbar 4, lowest 3
    assert body['arguments']['hops']==[2,4] and body['arguments']['gate_hops']==4
    assert set(body['table'])=={f'{b}b_{h}h' for b in (8,4,3) for h in (2,4)}
    checks={c['name']:c for c in body['checks']}
    assert set(checks['reliable']['value'])=={'2','4'}
    point=checks['operating_point']['value']
    assert point['bbar']==4 and set(point['predicted_success'])=={'2','8','16','32','48','64'}
    assert all(lo<=point['predicted_success'][h]<=hi for h,(lo,hi) in point['predicted_success_ci95'].items())
    assert checks['operating_point']['pass']==(point['predicted_success']['64']>=0.1)


def test_sweep_survival_compounds_the_turn_hazard(gate):
    assert gate.sweep_survival(0.1,[2,8])=={2:0.9**5,8:0.9**17}


def test_gate_episode_records_failing_turn_text(gate):
    from hazprec.agent.harness import AgentHarness
    from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
    from hazprec.quant.backends import SimulatedBackend
    from hazprec.scheduler import UniformPolicy
    env=HorizonGymEnv(HorizonGymConfig(n_hops=4))
    h=AgentHarness(SimulatedBackend(),UniformPolicy(3))
    episodes=[gate.run_episode(h,env,seed) for seed in range(40)]
    failed=[e for e in episodes if not e['success']]
    assert failed and all(e['failure_detail'] is None for e in episodes if e['success'])
    detail=failed[0]['failure_detail']
    assert detail['reason']!='ok' and json.loads(detail['expected'])['name'] in {'read','update','submit'}
    assert (detail['think_end'],detail['act_end'])==('tag','tag') and detail['act']
    assert sum(failed[0]['think_end'].values())==failed[0]['turns']
