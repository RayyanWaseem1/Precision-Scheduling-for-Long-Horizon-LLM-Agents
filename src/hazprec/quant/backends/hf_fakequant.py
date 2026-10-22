"""
PyTorch + transformers backend with fake-quantized weights. 

One model object; ``TorchWeightSwitcher`` swaps every Linear layer's weights to the 
requested bit-width before each forward pass. The KV cache is shared across 
precisions, so cached entries keep the precision they were computed at. 

Runs on CUDA (Colab A100/H100) for real models and on CPU for tiny test models. 
For wall-clock speed numbers use real low-bit kernels (Any-Precision LLM) in Phase 4;
the numerics here are what the hazard measurements need.
"""

from __future__ import annotations 

from typing import Any, Sequence 

from hazprec.quant.fake_quant import TorchWeightSwitcher

from .base import TokenBackend

def _as_id_list(rendered: Any) -> list[int]:
    """apply_chat_tempalte returns a list, a list of lists, a tensor or a BatchEncoding
    depending on the transformers version; normalize to a flat list of ints."""
    if hasattr(rendered, "keys") and "input_ids" in rendered:
        rendered = rendered["input_ids"]
    if hasattr(rendered, "tolist"):
        rendered = rendered.tolist()
    if rendered and isinstance(rendered[0], list):
        rendered = rendered[0]
    return [int(t) for t in rendered]

class HFFakeQuantBackend(TokenBackend):
    def __init__(
        self,
        model: Any,
        tokenizer: Any = None,
        levels: Sequence[int] = (3, 4, 5, 6, 8, 16),
        ref_bits: int = 16,
        group_size: int = 128,
        device: str | None = None, 
        dtype: str = "bfloat16",
        temperature: float = 0.7,
        top_p: float = 1.0,
        skip_modules: Sequence[str] = ("lm_head",),
        chat_template_kwargs: dict | None = None, 
        tool_role: str = "tool",
        seed: int = 0,
    ):
        super().__init__()
        import torch 

        self.torch = torch 
        if isinstance(model, str):
            from transformers import AutoModelForCausalLM, AutoTokenizer

            device = device or ("cuda" if torch.cuda.is_available() else "cpu")
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model)
            model = AutoModelForCausalLM.from_pretrained(model, torch_dtype = getattr(torch, dtype)).to(device)
        if tokenizer is None:
            raise ValueError("a tokenizer is required when passing a model object")
        self.model = model.eval()
        self.tokenizer = tokenizer 
        self.device = device or next(model.parameters()).device 
        self.levels = tuple(sorted(set(levels)))
        self.ref_bits = ref_bits 
        self.temperature = temperature 
        self.top_p = top_p 
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tool_role = tool_role 
        self.switcher = TorchWeightSwitcher(model, self.levels, group_size, tuple(skip_modules))
        eos = getattr(tokenizer, "eos_token_id", None)
        cfg_eos = getattr(getattr(model, "generation_config", None), "eos_token_id", None)
        self.eos_ids = {e for e in _flatten(eos) + _flatten(cfg_eos) if e is not None}
        self.gen = torch.Generator(device = "cpu")
        self.gen.manual_seed(seed)
        self.cache: Any = None 

    # primitives #
    def _messages_for_template(self, messages: list[dict]) -> list[dict]:
        if self.tool_role == "tool":
            return messages 
        out = []
        for m in messages:
            if m["role"] == "tool":
                out.append({"role": self.tool_role, "content": f"TOOL RESULT: \n{m['content']}"})
            else:
                out.append(m)
        return out 

    def _render(self, messages: list[dict]) -> list[int]:
        rendered = self.tokenizer.apply_chat_template(
            self._messages_for_template(messages), add_generation_prompt = True, tokenize = True, 
            **self.chat_template_kwargs,
        )
        return _as_id_list(rendered)

    def _render_text(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(
            self._messages_for_template(messages), add_generation_prompt=True,
            tokenize=False, **self.chat_template_kwargs,
        )

    def _new_cache(self):
        from transformers import DynamicCache 

        try: 
            return DynamicCache(config = self.model.config)
        except TypeError:
            return DynamicCache()

    def _reset_cache(self) -> None:
        self.cache = self._new_cache()

    def _feed(self, ids: list[int], bits: int):
        torch = self.torch
        if self.cache is None:
            self.cache = self._new_cache()
        self.switcher.set_bits(bits)
        with torch.no_grad():
            out = self.model(
                input_ids = torch.tensor([ids], device = self.device), past_key_values = self.cache, use_cache = True
            )
        self.cache = out.past_key_values
        return out.logits[0, -1].float().cpu()

    def _trim(self, n_keep: int) -> None:
        n_drop = len(self.cached_ids) - n_keep 
        if n_drop > 0:
            # negative crop = drop tokens from the end (works across transformers 4.4x-5.x).
            self.cache.crop(-n_drop)

    def _sample(self, logits) -> int:
        torch = self.torch
        if self.temperature <= 0:
            return int(torch.argmax(logits))
        probs = torch.softmax(logits / self.temperature, dim = -1)
        if self.top_p < 1.0:
            sp, si = torch.sort(probs, descending = True)
            keep = torch.cumsum(sp, - 1) - sp < self.top_p
            sp = sp * keep 
            idx = torch.multinomial(sp / sp.sum(), 1, generator = self.gen)
            return int(si[idx])
        return int(torch.multinomial(probs, 1, generator = self.gen))

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens = False)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens = False))

    def _reseed(self, seed: int) -> None:
        self.gen.manual_seed(seed)

def _flatten(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple, set)):
        return list(x)
    return [x]
