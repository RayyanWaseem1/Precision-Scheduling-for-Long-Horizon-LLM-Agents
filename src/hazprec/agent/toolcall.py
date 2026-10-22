"""
Tool-call text format: ``<tool_call>{"name": ..., "arguments": {...}} </tool_call>``

This is the Qwen/Hermes convention, which most open models already use
"""

from __future__ import annotations 

import json 
import re 
from dataclasses import dataclass, field 
from typing import Any 

OPEN_TAG = "<tool_call>"
CLOSE_TAG = "</tool_call>"

_TAGGED = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)

@dataclass(frozen = True)
class ToolCall:
    name: str 
    arguments: dict[str, Any] = field(default_factory = dict)

    def to_json(self) -> str:
        return json.dumps({"name": self.name, "arguments": self.arguments}, sort_keys = True)

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": dict(self.arguments)}


def format_tool_call(call: ToolCall) -> str:
    return f"{OPEN_TAG}{call.to_json()}{CLOSE_TAG}"


def _from_obj(obj: Any) -> ToolCall | None:
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
        return None 
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str): # some models double-encode arguemtns
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            return None 
    if not isinstance(args, dict):
        return None 
    return ToolCall(obj["name"].strip(), args)


def _first_json_object(text: str) -> Any:
    """ Decode the first balanced JSON object in ``text`` (tolerates trailing junk)"""
    start = text.find("{")
    if start < 0:
        return None 
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        return obj 
    except json.JSONDecodeError:
        return None 


def parse_tool_call(text: str) -> ToolCall | None:
    """ Parse the first tool call in a model response; `` None `` if there is none"""
    m = _TAGGED.search(text)
    if m:
        return _from_obj(_first_json_object(m.group(1)))
    # Fallback: a bare JSON object with a "name" key 
    return _from_obj(_first_json_object(text))