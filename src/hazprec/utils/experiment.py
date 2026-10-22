"""Experiment identity independent of Git tracking and working-tree cleanliness."""
from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def identity(config: dict) -> dict:
    package = Path(__file__).resolve().parents[1]
    source = hashlib.sha256()
    project = package.parents[1]
    paths = list(package.rglob('*.py')) + list((project / 'scripts').glob('*.py'))
    if (project / 'pyproject.toml').exists(): paths.append(project / 'pyproject.toml')
    for path in sorted(paths):
        source.update(str(path.relative_to(project)).encode())
        source.update(path.read_bytes())
    dependencies = {}
    for name in ('numpy','pyyaml','torch','transformers','mlx','mlx-lm','tokenizers'):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    body = {'schema': 2, 'config': config, 'source_sha256': source.hexdigest(),
            'dependencies': dependencies, 'python': platform.python_version()}
    return {**body, 'sha256': digest(body)}


def manifest_path(path) -> Path:
    return Path(str(path) + '.meta.json')


def read_manifest(path) -> dict:
    p = manifest_path(path)
    if not p.exists():
        raise ValueError(f'{path} has no experiment manifest; use a fresh run path')
    body = json.loads(p.read_text())
    claimed = body.pop('sha256', None)
    if claimed != digest(body):
        raise ValueError(f'modified experiment manifest: {p}')
    return body


def ensure_experiment(path, experiment: dict, stage: str, prereg_sha: str | None = None) -> None:
    expected = {'experiment': experiment, 'stage': stage, 'prereg_sha256': prereg_sha}
    p = manifest_path(path)
    if p.exists():
        if read_manifest(path) != expected:
            raise ValueError(f'incompatible experiment at {path}; use a new output path')
        return
    data = Path(path)
    if data.exists() and data.stat().st_size:
        raise ValueError(f'refusing to resume unversioned data at {path}')
    p.parent.mkdir(parents=True, exist_ok=True)
    # Single writer per experiment; exclusive creation prevents accidental replacement.
    with p.open('x') as f:
        json.dump({**expected, 'sha256': digest(expected)}, f, indent=2, sort_keys=True)


def verify_experiment(path, experiment, stage, prereg_sha=None):
    expected = {'experiment': experiment, 'stage': stage, 'prereg_sha256': prereg_sha}
    if read_manifest(path) != expected:
        raise ValueError(f'experiment identity mismatch for {path}')
