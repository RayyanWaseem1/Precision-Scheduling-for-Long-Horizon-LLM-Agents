"""
A deployable phase hinter: predicts whether the next tool call will end the episode.

It uses only what the agent has seen (observations and its own calls), never the 
oracle, so a policy that uses it is deployable. A learned probe replaces it later
"""

from __future__ import annotations 

from hazprec.agent.toolcall import ToolCall 
from hazprec.phases import Phase 

class HorizonGymStopHinter:
    def __init__(self) -> None:
        self.reset() 

    def reset(self) -> None:
        self.saw_end = False 
        self.last_correct_update = False 

    def update(self, call: ToolCall | None, observation: str, correct: bool) -> None:
        if call is not None and call.name == "read" and "next: END" in observation:
            self.saw_end = True 
        self.last_correct_update = bool(call is not None and call.name == "update" and correct)

    def hint(self) -> Phase:
        return Phase.STOP if (self.saw_end and self.last_correct_update) else Phase.ACT