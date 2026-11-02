"""MockLLM: a fixed, trivial, family-agnostic stand-in for a real model.

It sees ONLY the rendered prompt string. Its score is a hash of that string, so
it carries no knowledge of any pattern and never receives labels or metadata.
Its numbers are plumbing checks, never results.
"""

from __future__ import annotations

import hashlib
import json

from triage_kit.detectors.llm.types import LLMResponse, approx_tokens


class MockLLM:
    name = "mock"

    def complete(self, prompt: str) -> LLMResponse:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        score = int(digest[:16], 16) / 16**16
        text = json.dumps(
            {
                "verdict": "flag" if score >= 0.5 else "pass",
                "score": round(score, 6),
                "rationale": "MOCK: score is a hash of the prompt text, not a judgement.",
            }
        )
        return LLMResponse(
            text=text, tokens_in=approx_tokens(prompt), tokens_out=approx_tokens(text)
        )
