"""YAML configs with defaults and ``--set key.path=value`` overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Sequence

import yaml

from hazprec.envs.horizongym import HorizonGymConfig

DEFAULTS: dict[str, Any] = {
    "seed": 0,
    "backend": {"kind": "sim", "levels": [3, 4, 5, 6, 8], "ref_bits": 8},
    "env": {"n_hops": 5, "n_fields": 2, "filler_lines": 4, "error_mode": "fatal", "max_retries": 2},
    "agent": {"think": True, "max_think_tokens": 160, "max_act_tokens": 96},
    "atlas": {"n_tasks": 40, "n_samples": 8, "seed0": 0, "max_turns": None},
    "theory": {"bbar": 4, "alloc_hops": 8, "curve_mode": "parametric"},
    "horizon": {"hops": [2, 4, 8, 12, 16], "n_episodes": 400, "seed0": 100000},
}


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _set(cfg: dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        node.setdefault(k, {})
        if not isinstance(node[k], dict):
            raise ValueError(f"unknown or non-mapping override path: {dotted}")
        node = node[k]
    node[keys[-1]] = yaml.safe_load(raw)


def load_config(path: str | Path | None = None, overrides: Sequence[str] = ()) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        extra = yaml.safe_load(Path(path).read_text())
        _known_keys(extra, _SCHEMA)
        cfg = deep_merge(cfg, extra)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key.path=value, got {item!r}")
        k, v = item.split("=", 1)
        _set(cfg, k.strip(), v.strip())
    validate_config(cfg)
    return cfg


def env_config(cfg: dict, **override: Any) -> HorizonGymConfig:
    return HorizonGymConfig(**{**cfg["env"], **override})


def harness_kwargs(cfg: dict) -> dict:
    a = cfg["agent"]
    return {"max_think_tokens": int(a["max_think_tokens"]), "max_act_tokens": int(a["max_act_tokens"])}

# Templates may accept model-specific options; only that nested mapping is open.
_SCHEMA = {
    'seed': None,
    'backend': {
        'kind': None, 'levels': None, 'ref_bits': None,
        'sim': {'sensitivity': {p: None for p in ('system','read','think','act','stop')},
                'base_hazard': None, 'think_tokens': None, 'act_tokens': None},
        'hf': {k: ('open' if k == 'chat_template_kwargs' else None) for k in
               ('model','group_size','device','dtype','temperature','chat_template_kwargs','tool_role')},
        'mlx': {k: ('open' if k == 'chat_template_kwargs' else None) for k in
                ('model','group_size','cache_dir','temperature','chat_template_kwargs','tool_role')},
    },
    'env': {k: None for k in ('n_hops','n_fields','filler_lines','error_mode','max_retries','token_len','decoy_field')},
    'agent': {k: None for k in ('think','max_think_tokens','max_act_tokens')},
    'atlas': {k: None for k in ('n_tasks','n_samples','seed0','max_turns')},
    'theory': {k: None for k in ('bbar','alloc_hops','curve_mode')},
    'horizon': {k: None for k in ('hops','n_episodes','seed0')},
}


def _known_keys(value, schema, prefix=''):
    if not isinstance(value, dict):
        raise ValueError(f'{prefix or "config"} must be a mapping')
    for k, v in value.items():
        path = f'{prefix}.{k}' if prefix else str(k)
        if k not in schema:
            raise ValueError(f'unknown config key: {path}')
        if isinstance(schema[k], dict):
            _known_keys(v, schema[k], path)
        elif schema[k] == 'open' and not isinstance(v, dict):
            raise ValueError(f'{path} must be a mapping')


def validate_config(cfg: dict) -> None:
    import math
    _known_keys(cfg, _SCHEMA)

    def integer(value, name, minimum=1):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')

    def number(value, name, minimum=0):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < minimum:
            raise ValueError(f'{name} must be finite and >= {minimum}')

    integer(cfg['seed'], 'seed', 0)
    be = cfg['backend']
    if be['kind'] not in ('sim', 'hf', 'mlx'):
        raise ValueError('backend.kind must be sim, hf, or mlx')
    levels = be['levels']
    allowed = {2,3,4,6,8,16} if be['kind'] == 'mlx' else set(range(1,9)) | {16}
    if not isinstance(levels, list) or not levels or any(type(b) is not int or b not in allowed for b in levels):
        raise ValueError(f'backend.levels must be a nonempty list drawn from {sorted(allowed)}')
    if levels != sorted(set(levels)):
        raise ValueError('backend.levels must be unique and ascending')
    integer(be['ref_bits'], 'backend.ref_bits')
    if be['ref_bits'] != max(levels):
        raise ValueError('backend.ref_bits must be the highest available precision')
    if be['kind'] in ('mlx', 'hf'):
        sub = be.get(be['kind'], {})
        if not isinstance(sub.get('model'), str) or not sub['model'].strip():
            raise ValueError('real backend requires a model name/path')
        integer(sub.get('group_size', 64 if be['kind']=='mlx' else 128), 'group_size')
        number(sub.get('temperature', .7), 'temperature')
        if sub.get('tool_role', 'tool') not in ('tool','user'):
            raise ValueError('tool_role must be tool or user')
        if be['kind']=='hf' and sub.get('dtype','bfloat16') not in ('bfloat16','float16','float32'):
            raise ValueError('unsupported hf dtype')
    sim = be.get('sim', {})
    number(sim.get('base_hazard', .002), 'base_hazard')
    if sim.get('base_hazard', .002) >= 1:
        raise ValueError('base_hazard must be < 1')
    for k,v in sim.get('sensitivity', {}).items(): number(v, f'sensitivity.{k}')
    for k in ('think_tokens','act_tokens'):
        if k in sim: integer(sim[k], k)
    for k in ('n_hops','n_fields','token_len'):
        integer(cfg['env'].get(k, 5), 'env.'+k)
    for k in ('filler_lines','max_retries'):
        integer(cfg['env'][k], 'env.'+k, 0)
    if cfg['env']['n_hops'] > 9000:
        raise ValueError('n_hops exceeds the record ID space')
    env_config(cfg)  # validates fields and error mode
    if type(cfg['agent']['think']) is not bool or type(cfg['env'].get('decoy_field',True)) is not bool:
        raise ValueError('think and decoy_field must be booleans')
    for k in ('max_think_tokens','max_act_tokens'): integer(cfg['agent'][k], 'agent.'+k)
    for k in ('n_tasks','n_samples'): integer(cfg['atlas'][k], 'atlas.'+k)
    integer(cfg['atlas']['seed0'], 'atlas.seed0', 0)
    if cfg['atlas']['max_turns'] is not None: integer(cfg['atlas']['max_turns'], 'atlas.max_turns')
    integer(cfg['theory']['alloc_hops'], 'theory.alloc_hops')
    number(cfg['theory']['bbar'], 'theory.bbar', 1)
    if cfg['theory']['bbar'] not in levels:
        raise ValueError('theory.bbar must be an available precision for the uniform baseline')
    if cfg['theory']['curve_mode'] not in ('parametric','empirical'):
        raise ValueError('curve_mode must be parametric or empirical')
    hops = cfg['horizon']['hops']
    if not isinstance(hops, list) or len(hops)<2:
        raise ValueError('horizon.hops requires at least two horizons')
    for h in hops: integer(h, 'horizon.hops entry')
    if len(set(hops)) != len(hops) or max(hops)>9000:
        raise ValueError('horizon.hops must be unique and <= 9000')
    integer(cfg['horizon']['n_episodes'], 'horizon.n_episodes', 2)
    integer(cfg['horizon']['seed0'], 'horizon.seed0', 0)
