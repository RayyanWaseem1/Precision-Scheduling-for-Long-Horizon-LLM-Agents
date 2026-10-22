"""The precision-backend interface.

A backend holds one conversation's KV state and lets every segment run at its own
weight precision. KV entries therefore carry the precision of the weights that
produced them, which is exactly the semantics of an any-precision model.

Implementations:
    SimulatedBackend   oracle agent with a known hazard model (theory validation, tests)
    HFFakeQuantBackend PyTorch + transformers, fake-quantized weights (CUDA, or CPU for tiny models)
    MLXBackend         Apple Silicon, one resident MLX copy per bit-width sharing one KV cache
"""

from __future__ import annotations

import copy
import uuid

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class GenResult:
    text: str  # generated text, cut before the stop string if one was hit
    n_tokens: int
    stop_hit: str | None = None
    overflow: str = ""  # text generated after the stop string inside the same token(s)
    eos: bool = False
    boundary_tokens: int = 0  # extra forward work; included in n_tokens


class PrecisionBackend(ABC):
    levels: tuple[int, ...]
    ref_bits: int

    @abstractmethod
    def reset(self, seed: int | None = None) -> None:
        """Start a new conversation (clears the KV cache, reseeds sampling)."""

    @abstractmethod
    def ingest(self, messages: list[dict], bits: int) -> int:
        """Bring the KV cache up to date with ``messages`` (plus the assistant
        generation prompt), prefilling only the new tokens at ``bits``. Returns the
        number of tokens prefilled."""

    @abstractmethod
    def generate(self, bits: int, max_tokens: int, stop: Sequence[str] = (), prefix: str = "") -> GenResult:
        """Continue the current assistant turn at ``bits``. ``prefix`` is forced text fed
        before sampling (e.g. an opening tool-call tag)."""

    def snapshot(self) -> Any:
        """Opaque handle to the current state; ``restore`` returns to it, even after the
        conversation has diverged or moved past the snapshot point."""
        raise NotImplementedError

    def restore(self, snap: Any) -> None:
        raise NotImplementedError

    def bind_env(self, env: Any) -> None:
        """Hook for oracle-driven backends; real model backends ignore it."""

    def reseed(self, seed: int) -> None:
        """Reseed sampling without touching the cache."""

    def check_bits(self, bits: int) -> int:
        if bits not in self.levels:
            raise ValueError(f"bits={bits} not available; levels={self.levels}")
        return bits


def longest_common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def find_stop(text: str, stops: Sequence[str]) -> tuple[int, str] | None:
    best: tuple[int, str] | None = None
    for s in stops:
        if not s:
            continue
        i = text.find(s)
        if i >= 0 and (best is None or i < best[0]):
            best = (i, s)
    return best


class TokenBackend(PrecisionBackend):
    """Shared token-level logic for real models.

    * **Append-only history.** After a turn, the cache holds exactly the tokens the model
      generated, at the precision it generated them. The next ``ingest`` appends only the
      template's tail (end of the assistant turn, the new tool message, the generation
      prompt), found by rendering the conversation with a sentinel in place of the
      assistant content. Re-rendering the assistant text instead would re-tokenize it,
      diverge from the generated tokens, and re-prefill it at READ precision.
    * **Precision provenance.** ``cached_bits[i]`` is the precision token ``i`` was
      computed at. ``restore`` cuts back to the longest matching (token, bits) prefix
      and re-feeds the rest of the snapshot at each token's original precision, so a
      replay starts from exactly the state the reference run had.

    Subclasses implement ``_render``, ``_render_text``, ``_feed``, ``_trim``,
    ``_reset_cache``, ``_sample``, ``_decode``, ``_encode``, ``_reseed`` and ``eos_ids``.
    """

    SENTINEL = "<<HZP_ASSISTANT_SENTINEL_7f3a>>"
    eos_ids: set[int]

    def __init__(self) -> None:
        self.cached_ids: list[int] = []
        self.cached_bits: list[int] = []
        self._logits: Any = None
        self._logits_bits: int | None = None
        self._msgs: list[dict] | None = None  # conversation as of the last ingest
        self.stats = {"append": 0, "full": 0}

    # -- primitives ------------------------------------------------------------------
    @abstractmethod
    def _render(self, messages: list[dict]) -> list[int]:
        """Token ids of the chat template with the generation prompt appended."""

    def _render_text(self, messages: list[dict]) -> str:
        """The same rendering as text. Backends that cannot provide it fall back to
        full re-rendering (correct prefixes only, with the caveats above)."""
        raise NotImplementedError

    @abstractmethod
    def _feed(self, ids: list[int], bits: int) -> Any:
        """Run ``ids`` through the model at ``bits``, extending the cache; return the
        logits for the last position."""

    @abstractmethod
    def _trim(self, n_keep: int) -> None: ...

    @abstractmethod
    def _reset_cache(self) -> None: ...

    @abstractmethod
    def _sample(self, logits: Any) -> int: ...

    @abstractmethod
    def _decode(self, ids: list[int]) -> str: ...

    @abstractmethod
    def _encode(self, text: str) -> list[int]: ...

    @abstractmethod
    def _reseed(self, seed: int) -> None: ...

    # -- cache bookkeeping -----------------------------------------------------------
    def _push(self, ids: list[int], bits: int) -> None:
        self._logits = self._feed(ids, bits)
        self._logits_bits = bits
        self.cached_ids.extend(ids)
        self.cached_bits.extend([bits] * len(ids))

    def _cut_to(self, n_keep: int) -> None:
        if n_keep < len(self.cached_ids):
            if n_keep == 0:
                self._reset_cache()
            else:
                self._trim(n_keep)
            self.cached_ids = self.cached_ids[:n_keep]
            self.cached_bits = self.cached_bits[:n_keep]
            self._logits = None

    # -- interface -------------------------------------------------------------------
    def reset(self, seed: int | None = None) -> None:
        self._reset_cache()
        self.cached_ids = []
        self.cached_bits = []
        self._logits = None
        self._logits_bits = None
        self._msgs = None
        if seed is not None:
            self._reseed(seed)

    def reseed(self, seed: int) -> None:
        self._reseed(seed)

    def _is_assistant_extension(self, messages: list[dict]) -> bool:
        prev = self._msgs
        return (
            prev is not None
            and bool(self.cached_ids)
            and len(messages) > len(prev) + 1
            and messages[: len(prev)] == prev
            and messages[len(prev)].get("role") == "assistant"
        )

    def _tail_ids(self, messages: list[dict]) -> list[int] | None:
        """Template tokens that follow the just-generated assistant message."""
        assert self._msgs is not None
        k = len(self._msgs)
        marker = self.SENTINEL
        while any(marker in str(m) for m in messages):
            marker = "<<HZP_" + uuid.uuid4().hex + ">>"
        probe = self._msgs + [{**messages[k], "content": marker}] + messages[k + 1:]
        try:
            text = self._render_text(probe)
        except NotImplementedError as exc:
            raise RuntimeError("append-only history requires a text chat template") from exc
        if text.count(marker) != 1:
            raise RuntimeError("chat template must preserve exactly one assistant marker")
        i = text.index(marker)
        tail = self._encode(text[i + len(marker):])
        # The model may already have emitted the end-of-turn token itself.
        if tail and self.cached_ids and self.cached_ids[-1] in self.eos_ids and tail[0] == self.cached_ids[-1]:
            tail = tail[1:]
        if not tail:
            raise RuntimeError("chat template produced an empty observation tail")
        return tail

    def ingest(self, messages: list[dict], bits: int) -> int:
        self.check_bits(bits)
        tail = self._tail_ids(messages) if self._is_assistant_extension(messages) else None
        if tail is not None:
            self._push(tail, bits)
            self.stats["append"] += 1
            self._msgs = [dict(m) for m in messages]
            return len(tail)
        # Full render: the first turn, or a conversation the cache cannot extend.
        ids = self._render(messages)
        if not ids:
            raise ValueError("chat template rendered no tokens")
        k = longest_common_prefix(self.cached_ids, ids)
        if k == len(ids):
            k -= 1  # nothing new: re-feed the last token so we hold fresh logits
        self._cut_to(k)
        delta = ids[k:]
        self._push(delta, bits)
        self.stats["full"] += 1
        self._msgs = [dict(m) for m in messages]
        return len(delta)

    def _boundary_logits(self, bits: int):
        """Evaluate the last context token at new bits on a disposable cache copy.

        The persistent KV stays at its original precisions. This extra token forward
        is charged to the new segment. Cache copying is deliberately conservative;
        benchmark actual replay cost before scaling up.
        """
        original = self.cache
        try:
            self.cache = copy.deepcopy(original)
            self._trim(len(self.cached_ids) - 1)
            return self._feed(self.cached_ids[-1:], bits)
        finally:
            self.cache = original

    def generate(self, bits: int, max_tokens: int, stop: Sequence[str] = (), prefix: str = "") -> GenResult:
        self.check_bits(bits)
        n_prefix = 0
        boundary = 0
        if prefix:
            pids = self._encode(prefix)
            self._push(pids, bits)
            n_prefix = len(pids)
        if self._logits is None:
            raise RuntimeError("generate() called without context; call ingest() first")
        if max_tokens > 0 and self._logits_bits != bits:
            self._logits = self._boundary_logits(bits)
            self._logits_bits = bits
            boundary = 1
        out: list[int] = []
        text = ""
        for _ in range(max_tokens):
            tok = self._sample(self._logits)
            out.append(tok)
            self._push([tok], bits)
            if tok in self.eos_ids:
                return GenResult(self._decode(out[:-1]), len(out) + n_prefix + boundary, None, "", True, boundary)
            text = self._decode(out)
            hit = find_stop(text, stop)
            if hit is not None:
                i, s = hit
                return GenResult(text[:i], len(out) + n_prefix + boundary, s, text[i + len(s):], False, boundary)
        return GenResult(text, len(out) + n_prefix + boundary, None, "", False, boundary)

    def snapshot(self) -> Any:
        msgs = None if self._msgs is None else [dict(m) for m in self._msgs]
        return (list(self.cached_ids), list(self.cached_bits), self._logits, msgs, self._logits_bits)

    def restore(self, snap: Any) -> None:
        ids, bits, logits, msgs, logits_bits = snap
        n = min(len(self.cached_ids), len(ids))
        k = 0
        while k < n and self.cached_ids[k] == ids[k] and self.cached_bits[k] == bits[k]:
            k += 1
        self._cut_to(k)
        i = k
        while i < len(ids):  # re-feed each run of tokens at the precision it had
            j = i
            while j < len(ids) and bits[j] == bits[i]:
                j += 1
            self._push(list(ids[i:j]), bits[i])
            i = j
        self._logits = logits
        self._logits_bits = logits_bits
        self._msgs = None if msgs is None else [dict(m) for m in msgs]
