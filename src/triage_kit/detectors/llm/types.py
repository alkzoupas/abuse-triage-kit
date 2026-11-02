"""Provider-neutral types for the LLM reviewer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMResponse:
    text: str
    tokens_in: int
    tokens_out: int


class LLMProvider(Protocol):
    name: str

    def complete(self, prompt: str) -> LLMResponse: ...


def approx_tokens(text: str) -> int:
    """Rough token estimate (4 characters per token), used where no tokenizer is available."""
    return max(1, math.ceil(len(text) / 4))
