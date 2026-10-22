"""
Pre-registration: write predictions to a hashed, write-once JSON file *before* the 
experiemtn that tests them, and record the git commit if there is one
"""

from __future__ import annotations 

import datetime as _dt 
import hashlib 
import json 
import subprocess 
from pathlib import Path 
from typing import Any 

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr = subprocess.DEVNULL, text = True).strip()
    except Exception:
        return None 

def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def preregister(path: str | Path, payload: dict[str, Any], overwrite: bool = False) -> dict[str, Any]:
    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pre-registration are write-once (use a new file name)")
    body = {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec = "seconds"),
        "git_commit": _git_commit(),
        **payload,
    }
    canonical = json.dumps(body, sort_keys = True, default = str)
    body["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    path.parent.mkdir(parents = True, exist_ok = True)
    path.write_text(json.dumps(body, indent = 2, sort_keys = True, default = str))
    return body 

def load_preregistration(path: str | Path, verify: bool = True) -> dict[str, Any]:
    body = json.loads(Path(path).read_text())
    if verify:
        claimed = body.pop("sha256", None)
        actual = hashlib.sha256(json.dumps(body, sort_keys = True, default = str).encode()).hexdigest()
        if claimed != actual:
            raise ValueError(f"pre-registration {path} was modified after it was written")
        body["sha256"] = claimed 
    return body