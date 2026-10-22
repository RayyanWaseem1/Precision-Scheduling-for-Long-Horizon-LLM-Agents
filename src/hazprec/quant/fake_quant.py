"""
Group-wise asymmetric round-to-nearest (RNT) weight quantization

Fake quantization (quantize then dequantize into the original dtype) gives the same 
numbers a real low-bit kernel would compute with, up to accumulation order. That is 
all the hazard measurements need; speed numbers come from real kernels later. 

Mean-squared error falls by about 4x per extra bit (MSA ~ 2^(-2b)), which is the 
basis of the excess-hazard model in the proposal (Section 5.1)
"""

from __future__ import annotations 

from dataclasses import dataclass 
from typing import Any 

import numpy as np 

def _group_view(w: np.ndarray, group_size: int) -> tuple[np.ndarray, int]:
    out, inp = w.shape
    g = group_size if group_size and inp % group_size == 0 else inp 
    return w.reshape(out, inp // g, g), g 

def quantize_np(w: np.ndarray, bits: int, group_size: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ Return (codes uint8, scale, zero) for a 2-D weight matrix"""
    if not 1 <= bits <= 8:
        raise ValueError("bits must be in 1..8")
    wg, _ = _group_view(w.astype(np.float32), group_size)
    qmax = 2**bits - 1
    wmin = np.minimum(wg.min(axis = -1, keepdims = True), 0.0)
    wmax = np.maximum(wg.max(axis = -1, keepdims = True), 0.0)
    scale = np.maximum(wmax - wmin, 1e-8) / qmax 
    zero = np.clip(np.round(-wmin / scale), 0, qmax)
    codes = np.clip(np.round(wg / scale) + zero, 0, qmax).astype(np.uint8)
    return codes, scale.astype(np.float32), zero.astype(np.float32)

def dequantize_np(codes: np.ndarray, scale: np.ndarray, zero: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return ((codes.astype(np.float32) - zero) * scale).reshape(shape)

def fake_quant_np(w: np.ndarray, bits: int, group_size: int = 128) -> np.ndarray:
    codes, scale, zero = quantize_np(w, bits, group_size)
    return dequantize_np(codes, scale, zero, w.shape).astype(w.dtype)

### PyTorch: switch a model's Linear weights between precisions in place ###

@dataclass 
class _TorchCodes:
    codes: Any # unit8 tensor (out, n_groups, g)
    scale: Any # float32: prevent small-scale underflow (out, n_groups, 1)
    zero: Any # uint8 (out, n_groups, 1)
    shape: tuple[int, int]

def _torch_quantize(w, bits: int, group_size: int) -> _TorchCodes:
    import torch 

    out, inp = w.shape
    g = group_size if group_size and inp % group_size == 0 else inp 
    wg = w.detach().float().reshape(out, inp // g, g)
    qmax = 2 ** bits - 1
    wmin = wg.amin(-1, keepdim = True).clamp(max=0)
    wmax = wg.amax(-1, keepdim = True).clamp(min=0)
    scale = (wmax - wmin).clamp(min = 1e-8) / qmax 
    zero = torch.round(-wmin / scale).clamp(0, qmax)
    codes = torch.clamp(torch.round(wg / scale) + zero, 0, qmax).to(torch.uint8)
    return _TorchCodes(codes, scale, zero.to(torch.uint8), (out, inp))

def _torch_dequantize(c: _TorchCodes, dtype):
    return ((c.codes.float() - c.zero.float()) * c.scale.float()).reshape(c.shape).to(dtype)

class TorchWeightSwitcher:
    """ Holds per-bit-width codes for every ``nn.Linear`` and waps them in on demand
    
    ``16`` means the original (master) weights. Memory: one uint8 code tensor per 
    level (about half the BF16 model size each) plus the master copy if 16 is a level
    """

    def __init__(self, model, levels: tuple[int, ...], group_size: int = 128, skip: tuple[str, ...] = ("lm_head",)):
        import torch.nn as nn 

        self.levels = tuple(sorted(set(levels)))
        self.group_size = group_size 
        self.linears = [
            (name, mod)
            for name, mod in model.named_modules()
            if isinstance(mod, nn.Linear) and not any(s in name for s in skip)
        ]
        if not self.linears:
            raise ValueError("no nn.Linear modules found to quantize")
        self.keep_master = 16 in self.levels 
        self.master = {name: mod.weight.detach().clone() for name, mod in self.linears} if self.keep_master else {}
        self.codes: dict[int, dict[str, _TorchCodes]] = {}
        for b in self.levels:
            if b >= 16:
                continue 
            self.codes[b] = {name: _torch_quantize(mod.weight, b, group_size) for name, mod in self.linears}
        self.current = 16
        # start from the highest real level if master weights are not kept 
        if not self.keep_master:
            self.set_bits(max(self.codes))

    def set_bits(self, bits: int) -> None:
        import torch 

        if bits == self.current:
            return 
        if bits not in self.levels:
            raise ValueError(f"bits = {bits} not in {self.levels}")
        with torch.no_grad():
            for name, mod in self.linears:
                if bits >= 16:
                    mod.weight.data.copy_(self.master[name])
                else:
                    mod.weight.data.copy_(_torch_dequantize(self.codes[bits][name], mod.weight.dtype))
        self.current = bits 