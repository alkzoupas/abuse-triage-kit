"""Optional real-model provider (Anthropic Messages API).

Used only by `python -m triage_kit.eval.llm_pass` when the owner runs it with
ANTHROPIC_API_KEY set and --yes. Tests use a stub client only; every real client
is created through `_make_real_client`, which the test suite blocks.

Server-side refusal fallbacks are on by default (`fallbacks="default"`); pass
fallbacks=False to call the plain Messages endpoint instead.
"""

from __future__ import annotations

from typing import Any

from triage_kit.detectors.llm.types import LLMResponse

DEFAULT_MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


def _make_real_client() -> Any:  # pragma: no cover - never called in tests
    import anthropic  # optional dependency: pip install "abuse-triage-kit[anthropic]"

    return anthropic.Anthropic()  # reads credentials from the environment itself


class AnthropicProvider:
    name = "anthropic"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        client: Any | None = None,
        max_tokens: int = 4000,
        effort: str | None = None,
        fallbacks: bool = True,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.effort = effort
        self.fallbacks = fallbacks
        self.client = client if client is not None else _make_real_client()

    def complete(self, prompt: str) -> LLMResponse:
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.effort:
            kw["output_config"] = {"effort": self.effort}
        if self.fallbacks:
            resp = self.client.beta.messages.create(
                betas=[FALLBACK_BETA], fallbacks="default", **kw
            )
        else:
            resp = self.client.messages.create(**kw)
        usage = resp.usage
        if getattr(resp, "stop_reason", None) == "refusal":
            text = ""  # declined: parse_verdict maps this to a neutral 0.5
        else:
            text = "".join(
                getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text"
            )
        return LLMResponse(text, int(usage.input_tokens), int(usage.output_tokens))
