import ast
import json
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.conftest import Dataset
from triage_kit.detectors.llm.anthropic_provider import AnthropicProvider
from triage_kit.detectors.llm.mock import MockLLM
from triage_kit.detectors.llm.render import MAX_LINES, render_prompt
from triage_kit.detectors.llm.reviewer import LLMReviewer, parse_verdict
from triage_kit.generator import FAMILIES
from triage_kit.schema import EVENT_TYPES

MOCK_SRC = Path(__file__).resolve().parents[1] / "src/triage_kit/detectors/llm/mock.py"


class StubMessages:
    def __init__(self, reply: str, stop_reason: str = "end_turn") -> None:
        self.reply, self.stop_reason, self.calls = reply, stop_reason, []  # type: ignore[var-annotated]

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        return SimpleNamespace(
            content=[
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text=self.reply),
            ],
            usage=SimpleNamespace(input_tokens=123, output_tokens=45),
            stop_reason=self.stop_reason,
        )


class StubClient:
    def __init__(self, reply: str, stop_reason: str = "end_turn") -> None:
        self.beta = SimpleNamespace(messages=StubMessages(reply, stop_reason))
        self.messages = StubMessages(reply, stop_reason)


def test_render_prompt_is_bounded_and_mentions_events(dev_s1: Dataset) -> None:
    big = max(dev_s1.data, key=lambda ae: len(ae.events))
    p = render_prompt(big)
    assert len(big.events) > MAX_LINES
    assert "omitted" in p
    assert p.count("\n") < MAX_LINES + 60
    assert '"score"' in p  # asks for a JSON verdict


def test_prompts_carry_no_generator_metadata(dev_s1: Dataset) -> None:
    labels = dev_s1.labels
    bad_words = set(FAMILIES) | {"is_abuse", "group_id", "template_id", "benign"}
    for ae, _lab in list(zip(dev_s1.data, labels, strict=True))[::7]:
        p = render_prompt(ae)
        for w in bad_words:
            assert not re.search(rf"\b{w}\b", p), w
        assert "grp-" not in p and "tpl-" not in p


def test_mock_is_deterministic_and_bounded() -> None:
    m = MockLLM()
    a, b = m.complete("hello"), m.complete("hello")
    assert a == b
    score, why = parse_verdict(a.text)
    assert 0 <= score <= 1 and why.startswith("MOCK")
    assert a.tokens_in > 0 and a.tokens_out > 0


def test_mock_source_has_no_generator_vocabulary() -> None:
    tree = ast.parse(MOCK_SRC.read_text())
    words = (
        set(FAMILIES)
        | set(EVENT_TYPES)
        | {
            "ring",
            "template",
            "promo",
            "scrap",
            "card",
            "login",
            "review",
            "disposable",
            "fail",
            "velocity",
            "burst",
            "fraud",
            "abuse_family",
        }
    )
    imported = [n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
    imported += [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert all(
        m
        in (
            "__future__",
            "hashlib",
            "json",
            "math",
            "dataclasses",
            "triage_kit.detectors.llm.types",
        )
        for m in imported
    ), imported
    strings = [
        n.value.lower()
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    names = [n.id.lower() for n in ast.walk(tree) if isinstance(n, ast.Name)]
    for w in words:  # word-start match, so "string" does not trip "ring"
        for s in strings + names:
            assert not re.search(rf"(?<![a-z]){re.escape(w)}", s), (w, s)


def test_reviewer_records_tokens_per_call(dev_s1: Dataset) -> None:
    r = LLMReviewer(MockLLM())
    for ae in dev_s1.data[:5]:
        s, why = r.score(ae)
        assert 0 <= s <= 1
    assert len(r.calls) == 5
    assert all(c.tokens_in > 0 and c.tokens_out > 0 for c in r.calls)
    assert [c.account_id for c in r.calls] == [ae.account_id for ae in dev_s1.data[:5]]


@pytest.mark.parametrize(
    "text,score",
    [
        ('{"verdict": "abuse", "score": 0.8, "rationale": "x"}', 0.8),
        ('Sure. {"verdict": "benign", "score": 1.7, "rationale": "x"} done', 1.0),
        ('{"verdict": "abuse", "rationale": "no score"}', 0.9),
        ("not json at all", 0.5),
    ],
)
def test_parse_verdict(text: str, score: float) -> None:
    assert parse_verdict(text)[0] == pytest.approx(score)


def test_anthropic_provider_with_stub_client(dev_s1: Dataset) -> None:
    reply = json.dumps({"verdict": "abuse", "score": 0.7, "rationale": "stub"})
    client = StubClient(reply)
    prov = AnthropicProvider(model="some-model", client=client)
    r = LLMReviewer(prov)
    s, why = r.score(dev_s1.data[0])
    assert s == pytest.approx(0.7)
    assert r.calls[0].tokens_in == 123 and r.calls[0].tokens_out == 45
    call = client.beta.messages.calls[0]
    assert call["model"] == "some-model"
    assert call["messages"][0]["role"] == "user"
    assert call["fallbacks"] == "default"
    nofb = StubClient(reply)
    AnthropicProvider(model="m", client=nofb, fallbacks=False).complete("hi")
    assert nofb.messages.calls and not nofb.beta.messages.calls


def test_anthropic_refusal_is_neutral() -> None:
    prov = AnthropicProvider(model="m", client=StubClient("", stop_reason="refusal"))
    s, why = parse_verdict(prov.complete("hi").text)
    assert s == 0.5 and "unparseable" in why


def test_constructing_without_client_is_blocked_in_tests() -> None:
    from tests.conftest import REAL_CLIENT_ATTEMPTS, RealClientForbidden

    before = len(REAL_CLIENT_ATTEMPTS)
    with pytest.raises(RealClientForbidden):
        AnthropicProvider(model="m")
    assert len(REAL_CLIENT_ATTEMPTS) == before + 1
    REAL_CLIENT_ATTEMPTS.pop()  # this deliberate attempt is the guard's own self-test


def test_prompt_v2_names_no_abuse_type_and_v1_is_unchanged(dev_s1: Dataset) -> None:
    ae = dev_s1.data[0]
    v1, v2 = render_prompt(ae), render_prompt(ae, version="v2")
    assert render_prompt(ae, version="v1") == v1
    assert "hijacked accounts" in v1  # v1 still names the planted abuse types
    # v2 differs only in the instructions: the account block and timeline are the same text
    assert v1.split("ACCOUNT\n", 1)[1] == v2.split("ACCOUNT\n", 1)[1]
    head = v2.split("ACCOUNT\n", 1)[0].lower()
    for w in (
        "hijack", "takeover", "fake", "reviews", "promo", "harvest", "scrap", "card", "payment",
        "script", "automat", "template", r"\bring", "coupon", r"\bbot",
    ):  # fmt: skip
        assert not re.search(w, head), w
    assert '"score"' in head and "human investigator" in head
    with pytest.raises(ValueError):
        render_prompt(ae, version="v9")
