from __future__ import annotations 

from dataclasses import asdict, dataclass 

@dataclass
class HorizonGymConfig:
    """
    Knobs of the controllable-horizon environment.

    The episode follows a chain of ``n_hops`` records. Each hop needs two dependent
    tool calls (``read`` then ``update``), and the episode ends with ``submite``, so an 
    episode has ``2 * n_hops + 1`` checkable actions
    """

    n_hops: int = 5
    # Arguments per update call, in order (ack, seq, src): 1 = read-heavy, 3 = argument-heavy
    n_fields: int = 2
    # Filler lines per record: controls observation length (and KV share of bytes)
    filler_lines: int = 4
    # "fatal": first wrong action ends the episode. "recoverable": the action is rejected
    # with an error message and may be retried up to ``max_retries`` times
    error_mode: str = "fatal"
    max_retries: int = 2 
    token_len: int = 5
    # Include a decoy ``old_token`` field next to the real ``token`` (makes misreads possible)
    decoy_field: bool = True 

    def __post_init__(self) -> None:
        if self.n_hops < 1:
            raise ValueError("n_hops must be >= 1")
        if not 1 <= self.n_fields <= 3:
            raise ValueError("n_fields must be 1, 2, or 3")
        if self.error_mode not in ("fatal", "recoverable"):
            raise ValueError("error_mode must be 'fatal' or 'recoverable'")

    @property
    def horizon(self) -> int:
        return 2 * self.n_hops + 1 

    def to_dict(self) -> dict:
        return asdict(self)