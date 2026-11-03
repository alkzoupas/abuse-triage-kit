"""LLMReviewer: the detector wrapper around any provider."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from triage_kit.detectors.llm.render import render_prompt
from triage_kit.detectors.llm.types import LLMProvider
from triage_kit.schema import AccountEvents


@dataclass(frozen=True)
class CallRecord:
    account_id: str
    tokens_in: int
    tokens_out: int


def parse_verdict(text: str) -> tuple[float, str]:
    """Extract (score in [0, 1], rationale) from a model reply. Unparseable -> 0.5."""
    m = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not m:
        return 0.5, "unparseable reply"
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return 0.5, "unparseable reply"
    if not isinstance(d, dict):
        return 0.5, "unparseable reply"
    rationale = str(d.get("rationale", ""))
    score = d.get("score")
    if isinstance(score, int | float):
        return min(1.0, max(0.0, float(score))), rationale
    verdict = str(d.get("verdict", "")).lower()
    if verdict in ("abuse", "flag"):
        return 0.9, rationale
    if verdict in ("benign", "pass"):
        return 0.1, rationale
    return 0.5, "unparseable reply"


class LLMReviewer:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.name = f"llm:{provider.name}"
        self.calls: list[CallRecord] = []

    def score(self, account_events: AccountEvents) -> tuple[float, str]:
        resp = self.provider.complete(render_prompt(account_events))
        self.calls.append(CallRecord(account_events.account_id, resp.tokens_in, resp.tokens_out))
        return parse_verdict(resp.text)
