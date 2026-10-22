""" Apple Silicon backend (MacBook) built on MLX / mlx-lm

MLX cannot switch one model's precison in place, so we keep one quantized copy per
bit-width resident in unified memory and let all copies share a single KV cache. The 
copies have the same architecture, so their cache tensors are interchangeable, and each 
segment's KV entries are produced by the copy at that segment's precision.

Memory on an 18 GB M3 Pro: a 3B model at 3, 4, 6, and 8 bits need roughly 8-9 GB for 
the coopies together; a 1.7B model is comfortable at every level. Do not add a BF16
copy of a 3B model: treat the 8-bit copy as the reference on the Mac.

NOTE: written against mlx-lm >= 0.20. It could not be executed on the Linux machine
this repo was built on; run ``hazprec smoke`` on the Mac first
"""

from __future__ import annotations 

from pathlib import Path 
from typing import Any, Sequence 

from .base import TokenBackend
from .hf_fakequant import _as_id_list, _flatten 

def _mlx_path(cache_dir: Path, model: str, bits: int, group_size: int) -> Path:
    slug = model.replace("/", "--")
    tag = "bf16" if bits >= 16 else f"q{bits}-g{group_size}"
    return cache_dir / f"{slug}-{tag}"

def _convert(model: str, out: Path, bits: int, group_size: int) -> None:
    try:
        from mlx_lm import convert # mlx-lm >= 0.19
    except ImportError: # pragma: no cover 
        from mlx_lm.utils import convert 
    if bits >= 16:
        convert(hf_path = model, mlx_path=str(out), dtype = "bfloat16")
    else:
        convert(hf_path = model, mlx_path=str(out), quantize=True, q_bits=bits, q_group_size=group_size)

class MLXBackend(TokenBackend):
    def __init__(
        self,
        model: str, 
        levels: Sequence[int] = (3, 4, 6, 8),
        ref_bits: int = 8,
        group_size: int = 64,
        cache_dir: str = "~/.cache/hazprec/mlx",
        temperature: float = 0.7,
        chat_template_kwargs: dict | None = None,
        tool_role: str = "tool",
        seed: int = 0,
    ):
        super().__init__()
        import mlx.core as mx 
        from mlx_lm import load 

        self.mx = mx
        self.levels = tuple(sorted(set(levels)))
        self.ref_bits = ref_bits
        self.temperature = temperature 
        self.chat_template_kwargs = dict(chat_template_kwargs or {})
        self.tool_role = tool_role 
        root = Path(cache_dir).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        self.models: dict[int, Any] = {}
        self.tokenizer: Any = None 
        for b in self.levels:
            path = _mlx_path(root, model, b, group_size)
            if not path.exists():
                print(f"[hazprec] converting {model} to MLX at {b} bits -> {path}")
                _convert(model, path, b, group_size)
            m, tok = load(str(path))
            self.models[b] = m 
            if self.tokenizer is None:
                self.tokenizer = tok 
            eos = getattr(self.tokenizer, "eos_token_ids", None) or getattr(self.tokenizer, "eos_token_id", None)
            self.eos_ids = {int(e) for e in _flatten(eos) if e is not None}
            mx.random.seed(seed)
            self.cache: Any = None 

    # primitives #
    def _render(self, messages: list[dict]) -> list[int]:
        msgs = messages 
        if self.tool_role != "tool":
            msgs = [
                {"role": self.tool_role, "content": f"TOOL RESULT: \n{m['content']}"} if m["role"] == "tool" else m
                for m in messages
            ]
        rendered = self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt = True, tokenize = True, **self.chat_template_kwargs
        )
        return _as_id_list(rendered)

    def _render_text(self, messages: list[dict]) -> str:
        msgs = messages if self.tool_role == "tool" else [
            {"role": self.tool_role, "content": f"TOOL RESULT: \n{m['content']}"}
            if m["role"] == "tool" else m for m in messages
        ]
        return self.tokenizer.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False, **self.chat_template_kwargs,
        )

    def _reset_cache(self) -> None:
        from mlx_lm.models.cache import make_prompt_cache

        self.cache = make_prompt_cache(self.models[self.ref_bits])

    def _feed(self, ids: list[int], bits: int):
        mx = self.mx 
        if self.cache is None:
            self._reset_cache()
        logits = self.models[bits](mx.array(ids)[None], cache = self.cache)
        last = logits[0, -1, :].astype(mx.float32)
        mx.eval(last)
        return last 

    def _trim(self,  n_keep: int) -> None:
        from mlx_lm.models.cache import can_trim_prompt_cache, trim_prompt_cache

        n_drop = len(self.cached_ids) - n_keep 
        if n_drop <= 0:
            return 
        if not can_trim_prompt_cache(self.cache):
            raise RuntimeError("this model's KV cache cannot be trimmed")
        trim_prompt_cache(self.cache, n_drop)

    def _sample(self, logits) -> int:
        mx = self.mx 
        if self.temperature <= 0:
            return int(mx.argmax(logits).item())
        return int(mx.random.categorical(logits / self.temperature).item())

    def _decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def _encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens = False))

    def _reseed(self, seed: int) -> None:
        self.mx.random.seed(seed)