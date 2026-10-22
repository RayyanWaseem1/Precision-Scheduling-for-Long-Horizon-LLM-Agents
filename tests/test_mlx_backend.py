"""No-download MLX tests on a tiny Qwen3; requires access to a Metal device."""
import copy

import pytest

try:
    import mlx.core as mx
except ImportError as exc:
    pytest.skip(f'MLX is unavailable: {exc}', allow_module_level=True)
nn=pytest.importorskip('mlx.nn')
from mlx_lm.models.qwen3 import Model,ModelArgs
from hazprec.quant.backends.base import TokenBackend
from hazprec.quant.backends.mlx_backend import MLXBackend


class Tokenizer:
    def encode(self,text,**kwargs): return [ord(c)%128 for c in text]
    def decode(self,ids): return ''.join(chr(i) for i in ids)
    def apply_chat_template(self,messages,tokenize=True,**kwargs):
        text=''.join(f"[{m['role']}]{m['content']}\n" for m in messages)+'[assistant]<think></think>'
        return self.encode(text) if tokenize else text


@pytest.fixture
def make():
    mx.random.seed(0)
    model=Model(ModelArgs(model_type='qwen3',hidden_size=64,num_hidden_layers=2,
                         intermediate_size=128,num_attention_heads=4,rms_norm_eps=1e-6,
                         vocab_size=128,num_key_value_heads=2,max_position_embeddings=2048,
                         head_dim=16,tie_word_embeddings=False,rope_theta=10000))
    mx.eval(model.parameters())
    models={}
    for bits in (3,4,6,8):
        m=copy.deepcopy(model)
        nn.quantize(m,group_size=64,bits=bits)
        mx.eval(m.parameters()); models[bits]=m
    def build():
        be=MLXBackend.__new__(MLXBackend); TokenBackend.__init__(be)
        be.mx=mx; be.models=models; be.levels=(3,4,6,8); be.ref_bits=8
        be.tokenizer=Tokenizer(); be.tool_role='tool'; be.chat_template_kwargs={}
        be.eos_ids=set(); be.temperature=0.; be.cache=None
        be.reset(0)
        return be
    return build


def test_all_mlx_precisions_and_boundary_cache(make):
    be=make(); messages=[{'role':'user','content':'hello'}]
    for bits in be.levels:
        be.reset(0); be.ingest(messages,bits); assert be.generate(bits,1).n_tokens==1
    be.reset(0); be.ingest(messages,4)
    prefix=[(mx.array(c.keys),mx.array(c.values),c.offset) for c in be.cache]
    be.generate(8,1)
    for cache,(keys,values,n) in zip(be.cache,prefix):
        assert bool(mx.allclose(cache.keys[...,:n,:],keys[...,:n,:]).item())
        assert bool(mx.allclose(cache.values[...,:n,:],values[...,:n,:]).item())
    assert be.cached_bits[-1]==8


def test_mlx_restored_cache_supports_same_continuation(make):
    be=make(); ref=make(); messages=[{'role':'user','content':'snapshot'}]
    be.ingest(messages,4); early=be.snapshot(); be.generate(8,3); late=be.snapshot()
    ref.ingest(messages,4); ref._push(be.cached_ids[len(ref.cached_ids):],8)
    expected=ref._feed([10],8)
    be.restore(early); be.generate(3,5); be.restore(late)
    assert bool(mx.allclose(be._feed([10],8),expected,atol=1e-5).item())
    rebuilt=make(); rebuilt.restore(late)
    assert bool(mx.allclose(rebuilt._feed([10],8),expected,atol=1e-5).item())


def test_mlx_append_preserves_template_tail_and_precisions(make):
    be=make(); msgs=[{'role':'user','content':'hello'}]
    be.ingest(msgs,4); be.generate(8,2)
    ids=list(be.cached_ids); bits=list(be.cached_bits)
    obs='collision '+be.SENTINEL+' end'
    be.ingest(msgs+[{'role':'assistant','content':'normalized'}, {'role':'tool','content':obs}],4)
    assert be.cached_ids[:len(ids)]==ids and be.cached_bits[:len(bits)]==bits
    assert obs in be._decode(be.cached_ids[len(ids):])
