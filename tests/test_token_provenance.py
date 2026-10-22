import json

import pytest

from hazprec.agent.harness import AgentHarness
from hazprec.quant.backends.base import TokenBackend
from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
from hazprec.scheduler import UniformPolicy
from hazprec.sensitivity.replay import measure_task


class ToyBackend(TokenBackend):
    levels=(4,8)
    ref_bits=8
    eos_ids={0}
    def _render_text(self,messages):
        return ''.join(f"[{m['role']}]{m['content']}\n" for m in messages)+'[assistant]<think></think>'
    def _render(self,messages): return self._encode(self._render_text(messages))
    def _encode(self,text): return list(map(ord,text))
    def _decode(self,ids): return ''.join(map(chr,ids))
    def _reset_cache(self): self.cache=[]
    def _trim(self,n_keep): self.cache=self.cache[:n_keep]
    def _feed(self,ids,bits):
        self.cache.extend((i,bits) for i in ids)
        return bits
    def _sample(self,logits): return 64+logits
    def _reseed(self,seed): pass


def test_precision_boundary_uses_new_logits_and_preserves_previous_kv():
    be=ToyBackend(); be.reset(0)
    be.ingest([{'role':'user','content':'hello'}],4)
    previous=list(be.cache)
    result=be.generate(8,1)
    assert result.text=='H'  # 8-bit logits, rather than D from the previous 4-bit pass
    assert result.n_tokens==2 and result.boundary_tokens==1
    assert be.cache[:-1]==previous
    assert be.cached_bits[:-1]==[4]*len(previous)
    assert be.cache==list(zip(be.cached_ids,be.cached_bits))


def test_marker_collision_preserves_full_observation_and_provenance():
    be=ToyBackend(); be.reset(0)
    messages=[{'role':'user','content':'hello'}]
    be.ingest(messages,4); be.generate(8,2)
    previous=list(be.cache)
    obs='prefix '+be.SENTINEL+' suffix'
    n=be.ingest(messages+[{'role':'assistant','content':'rewritten'}, {'role':'tool','content':obs}],4)
    assert be.cache[:len(previous)]==previous
    assert obs in be._decode(be.cached_ids[len(previous):])
    assert n==len(be.cache)-len(previous)
    assert be.stats=={'append':1,'full':1}


def test_restore_rebuilds_actual_kv_after_divergence():
    be=ToyBackend(); be.reset(0)
    be.ingest([{'role':'user','content':'hello'}],4)
    early=be.snapshot(); be.generate(8,3); late=be.snapshot()
    expected=list(be.cache)
    be.restore(early); be.generate(4,4); be.restore(late)
    assert be.cache==expected
    be.reset(0); be.restore(late)
    assert be.cache==expected
    assert be.cache==list(zip(be.cached_ids,be.cached_bits))


def test_unsupported_template_fails_instead_of_reprefilling_history():
    be=ToyBackend(); be.reset(0)
    messages=[{'role':'user','content':'hello'}]; be.ingest(messages,4); be.generate(8,1)
    be._render_text=lambda messages:'template dropped assistant content'
    with pytest.raises(RuntimeError,match='exactly one'):
        be.ingest(messages+[{'role':'assistant','content':'x'},{'role':'tool','content':'y'}],4)


class OracleTokenBackend(ToyBackend):
    def __init__(self):
        super().__init__(); self.prefixes=[]
    def bind_env(self,env): self.env=env
    def ingest(self,messages,bits):
        self.prefixes.append((len(messages),list(self.cache)))
        return super().ingest(messages,bits)
    def generate(self,bits,max_tokens,stop=(),prefix=''):
        text='<tool_call>' if '<tool_call>' in stop else self.env.expected.to_json()+'</tool_call>'
        self.pending=iter(map(ord,text))
        return super().generate(bits,max_tokens,stop,prefix)
    def _sample(self,logits): return next(self.pending,0)


def test_replay_restores_reference_prefix_for_every_probe():
    be=OracleTokenBackend()
    rows=list(measure_task(be,HorizonGymConfig(n_hops=1),0,(4,8),8,n_samples=2,
                           harness_kwargs={'max_think_tokens':32,'max_act_tokens':256}))
    assert rows[0]['success'] is True
    reference=dict(be.prefixes[:3])
    assert len(reference)==3
    for size,actual in be.prefixes[3:]: assert actual==reference[size]
    assert all(r['failures']==0 for r in rows if r['type']=='probe')
    assert all('text' in seg for turn in rows[0]['turns'] for seg in turn['segments'])


class ScriptedTokenBackend(ToyBackend):
    """Decodes a fixed script per generate() call; an exhausted script emits end of turn (0)."""
    def __init__(self,scripts):
        super().__init__(); self.scripts=list(scripts)
    def generate(self,bits,max_tokens,stop=(),prefix=''):
        self.pending=iter(map(ord,self.scripts.pop(0)))
        return super().generate(bits,max_tokens,stop,prefix)
    def _sample(self,logits): return next(self.pending,0)


def _scripted_turns(scripts,n_turns,error_mode='fatal',max_think_tokens=64):
    env=HorizonGymEnv(HorizonGymConfig(n_hops=1,error_mode=error_mode)); env.reset(0)
    be=ScriptedTokenBackend(scripts); be.reset(0)
    h=AgentHarness(be,UniformPolicy(8),max_think_tokens=max_think_tokens,max_act_tokens=128,hinter_factory=None)
    messages=env.messages()
    return be,env,[h.run_turn(env,messages,t) for t in range(n_turns)]


def test_end_of_turn_during_think_is_a_format_failure_without_a_forced_tool_call():
    env=HorizonGymEnv(HorizonGymConfig(n_hops=1)); env.reset(0)
    good=env.expected.to_json()+'</tool_call>'
    be,env,(failed,recovered)=_scripted_turns(['I will read it.','<tool_call>',good],2,'recoverable')
    assert failed.call is None and not failed.correct and failed.reason=='could not parse a tool call'
    assert (failed.think_end,failed.act_end)==('eos','skipped')
    assert [s.tokens for s in failed.segments[1:]]==[16,0]  # 15 characters + end of turn; no ACT decode
    after=be._decode(be.cached_ids).split('I will read it.\x00',1)[1]  # history after the failed turn
    assert after.count('<tool_call>')==1 and after.index('ERROR')<after.index('<tool_call>')
    assert recovered.correct and (recovered.think_end,recovered.act_end)==('tag','tag')
    assert be.cache==list(zip(be.cached_ids,be.cached_bits))


def test_think_budget_exhaustion_is_recorded_and_forces_the_tool_call():
    env=HorizonGymEnv(HorizonGymConfig(n_hops=1)); env.reset(0)
    be,env,(turn,)=_scripted_turns(['x'*10,env.expected.to_json()+'</tool_call>'],1,max_think_tokens=5)
    assert (turn.think_end,turn.act_end)==('cap','tag') and turn.correct
    assert env.task is not None, 'Expected an initialized task after running the scripted turn'
    assert be._decode(be.cached_ids).endswith('xxxxx<tool_call>'+env.task.oracle[0].to_json()+'</tool_call>')
    assert turn.to_dict()['think_end']=='cap'
