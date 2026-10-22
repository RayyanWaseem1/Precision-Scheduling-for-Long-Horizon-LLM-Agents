"""Cache-consistency tests for the token-level backend on a tiny random Llama (CPU).

Skipped when torch/transformers are not installed (e.g. on a MLX-only Mac setup).
"""

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from hazprec.quant.backends.hf_fakequant import HFFakeQuantBackend  # noqa: E402

CHARS = ["<pad>", "<eos>"] + [chr(c) for c in range(32, 127)] + ["\n"]


class CharTokenizer:
    eos_token_id = 1

    def __init__(self):
        self.stoi = {c: i for i, c in enumerate(CHARS)}

    def encode(self, text, add_special_tokens=False):
        return [self.stoi.get(c, 0) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(CHARS[i] if i > 1 else "" for i in ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kw):
        text = "".join(f"[{m['role']}]{m['content']}\n" for m in messages)
        if add_generation_prompt:
            text += "[assistant]<think></think>"
        return self.encode(text) if tokenize else text


@pytest.fixture
def tiny():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(
        vocab_size=len(CHARS), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
        eos_token_id=1, pad_token_id=0, bos_token_id=1,
    )
    model = transformers.LlamaForCausalLM(cfg).eval()
    return model, CharTokenizer()


def full_logits(model, ids):
    with torch.no_grad():
        return model(input_ids=torch.tensor([ids])).logits[0, -1].float()


def make(tiny, **kw):
    model, tok = tiny
    return HFFakeQuantBackend(model, tok, levels=(4, 8, 16), ref_bits=16, group_size=32, temperature=0.0, **kw)


def test_incremental_cache_matches_full_forward(tiny):
    be = make(tiny)
    be.reset(0)
    msgs = [{"role": "system", "content": "you are terse"}, {"role": "user", "content": "say hi"}]
    be.ingest(msgs, 16)
    be.generate(16, max_tokens=6)
    assert torch.allclose(be._logits, full_logits(tiny[0], be.cached_ids), atol=1e-4)


def test_lcp_trim_on_divergent_history(tiny):
    be = make(tiny)
    be.reset(0)
    be.ingest([{"role": "user", "content": "abcdef"}], 16)
    n = be.ingest([{"role": "user", "content": "abcXYZ"}], 16)
    shared = len("[user]abc")
    assert n == len(be.cached_ids) - shared  # only the divergent suffix was prefilled
    assert torch.allclose(be._logits, full_logits(tiny[0], be.cached_ids), atol=1e-4)


def test_snapshot_restore_reproduces_greedy_continuation(tiny):
    be = make(tiny)
    be.reset(0)
    be.ingest([{"role": "user", "content": "count"}], 16)
    snap = be.snapshot()
    first = be.generate(16, 8).text
    be.restore(snap)
    assert be.generate(16, 8).text == first


def test_stop_string_and_overflow(tiny):
    be = make(tiny)
    be.reset(0)
    be.ingest([{"role": "user", "content": "x"}], 16)
    g = be.generate(16, 30, prefix="<tool_call>{}</tool_call>tail", stop=("</tool_call>",))
    # The prefix is fed, not generated, so the stop is detected only in sampled text.
    assert g.n_tokens >= len("<tool_call>{}</tool_call>tail")


def test_weight_switcher_restores_master_and_error_falls_with_bits(tiny):
    be = make(tiny)
    lin = be.switcher.linears[0][1]
    master = be.switcher.master[be.switcher.linears[0][0]].clone()
    errs = {}
    for b in (4, 8):
        be.switcher.set_bits(b)
        errs[b] = float(((lin.weight.detach() - master) ** 2).mean())
    be.switcher.set_bits(16)
    assert torch.equal(lin.weight, master)
    assert errs[8] < errs[4] / 50  # about 4x per bit -> ~256x for 4 extra bits


def test_full_episode_and_replay_run_on_token_backend(tiny):
    from hazprec.agent.harness import AgentHarness
    from hazprec.envs.horizongym import HorizonGymConfig, HorizonGymEnv
    from hazprec.scheduler import PhaseTablePolicy
    from hazprec.sensitivity import measure_task

    model, tok = tiny
    be = HFFakeQuantBackend(model, tok, levels=(4, 8, 16), ref_bits=16, group_size=32, temperature=0.8)
    cfg = HorizonGymConfig(n_hops=1)
    h = AgentHarness(be, PhaseTablePolicy({"read": 4, "think": 8, "act": 16}), max_think_tokens=8, max_act_tokens=8)
    tr = h.run_episode(HorizonGymEnv(cfg), 0)
    assert tr.turns and not tr.success  # a random model cannot solve the task
    assert [s.bits for s in tr.turns[0].segments] == [4, 8, 16]
    recs = list(measure_task(be, cfg, 0, (4, 8, 16), 16, n_samples=2,
                             harness_kwargs={"max_think_tokens": 6, "max_act_tokens": 6}))
    assert recs[0]["type"] == "reference"
    probes = [r for r in recs if r["type"] == "probe"]
    assert {r["probe"] for r in probes} == {"ref", "read", "think", "act"}
    assert all(r["n"] == 2 for r in probes)


def test_mixed_precision_segments_change_outputs(tiny):
    be = make(tiny)
    be.reset(0)
    msgs = [{"role": "user", "content": "hello there"}]
    be.ingest(msgs, 16)
    hi = be._logits.clone()
    be.reset(0)
    be.ingest(msgs, 4)
    assert not torch.allclose(hi, be._logits)


def test_boundary_forward_preserves_kv_and_uses_new_precision(tiny):
    import copy
    model,tok=tiny
    be=HFFakeQuantBackend(copy.deepcopy(model),tok,levels=(4,8,16),ref_bits=16,group_size=32,temperature=0)
    expected=HFFakeQuantBackend(copy.deepcopy(model),tok,levels=(4,8,16),ref_bits=16,group_size=32,temperature=0)
    msgs=[{'role':'user','content':'boundary'}]
    for obj in (be,expected): obj.reset(0); obj.ingest(msgs,4)
    prefix_ids=list(be.cached_ids); prefix_bits=list(be.cached_bits)
    # Independent reference: rerun only the last token at 8 bits using a separate cache.
    expected._trim(len(prefix_ids)-1)
    expected_logits=expected._feed(prefix_ids[-1:],8)
    token=int(torch.argmax(expected_logits))
    result=be.generate(8,1)
    assert be.cached_ids[-1]==token and result.boundary_tokens==1
    assert be.cached_ids[:-1]==prefix_ids and be.cached_bits[:-1]==prefix_bits
    # Rebuild original KV on another model, then feed the same new token at 8 bits.
    expected.reset(0); expected.ingest(msgs,4)
    continuation=expected._feed([token],8)
    assert torch.allclose(be._logits,continuation,atol=1e-5)


def test_restore_is_valid_for_subsequent_computation(tiny):
    import copy
    model,tok=tiny
    def fresh():
        obj=HFFakeQuantBackend(copy.deepcopy(model),tok,levels=(4,8,16),ref_bits=16,group_size=32,temperature=0)
        obj.reset(0)
        return obj
    be=fresh()
    msgs=[{'role':'user','content':'snapshot'}]
    be.ingest(msgs,4); early=be.snapshot(); be.generate(8,4); late=be.snapshot()
    ids=list(be.cached_ids)
    # Independently replay the known forward sequence; do not restore snapshot logits.
    reference=fresh(); reference.ingest(msgs,4)
    reference._push(ids[len(reference.cached_ids):],8)
    expected=reference._feed([10],8)
    be.restore(early); be.generate(4,7); be.restore(late)
    assert torch.allclose(be._feed([10],8),expected,atol=1e-5)
    rebuilt=fresh(); rebuilt.restore(late)
    assert torch.allclose(rebuilt._feed([10],8),expected,atol=1e-5)


def test_torch_quantization_constant_and_tiny_scales():
    from hazprec.quant.fake_quant import _torch_quantize,_torch_dequantize
    for value in (0.,1.,-1.,1e-7,-1e-7):
        w=torch.full((2,128),value)
        q=_torch_quantize(w,4,128)
        assert torch.all(q.scale>0)
        assert torch.allclose(_torch_dequantize(q,w.dtype),w,atol=1e-9,rtol=1e-5)
