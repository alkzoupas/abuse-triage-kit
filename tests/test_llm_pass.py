import json
from pathlib import Path

import pytest

from tests.test_llm import StubClient
from triage_kit.eval import llm_pass

FAKE_KEY = "sk-test-not-a-real-key-123"


def _args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--seed",
        "5",
        "--s",
        "1.0",
        "--n",
        "1500",
        "--model",
        "example-llm",
        "--limit",
        "20",
        "--data-root",
        str(tmp_path / "data"),
        "--results-root",
        str(tmp_path / "res"),
        *extra,
    ]


def test_refuses_without_api_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        llm_pass.main(
            _args(tmp_path, "--yes"), env={}, client_factory=lambda: pytest.fail("no client")
        )
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().out


def test_prints_estimate_and_stops_without_yes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = llm_pass.main(
        _args(tmp_path),
        env={"ANTHROPIC_API_KEY": FAKE_KEY},
        client_factory=lambda: pytest.fail("must not create a client without --yes"),
    )
    out = capsys.readouterr().out
    assert rc == 0 and "Estimated cost" in out and "--yes" in out and "PLACEHOLDER" in out
    assert FAKE_KEY not in out
    assert not (tmp_path / "res").exists() or not list((tmp_path / "res").rglob("llm_bundle*"))


def test_writes_scores_only_bundle_with_stub(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    reply = json.dumps({"verdict": "flag", "score": 0.25, "rationale": "secret reasoning"})
    stub = StubClient(reply)
    rc = llm_pass.main(
        _args(tmp_path, "--yes"), env={"ANTHROPIC_API_KEY": FAKE_KEY}, client_factory=lambda: stub
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert FAKE_KEY not in out
    bundles = list((tmp_path / "res").rglob("llm_bundle*.json"))
    assert len(bundles) == 1
    b = json.loads(bundles[0].read_text())
    assert b["model"] == "example-llm" and b["provider"] == "anthropic"
    assert len(b["scores"]) == 20 and set(b["scores"].values()) == {0.25}
    assert b["tokens_in_total"] == 20 * 123 and b["tokens_out_total"] == 20 * 45
    text = bundles[0].read_text()
    assert "secret reasoning" not in text and "TIMELINE" not in text  # scores only
    assert len(stub.beta.messages.calls) == 20


def test_default_path_goes_through_the_guarded_factory(tmp_path: Path) -> None:
    from tests.conftest import REAL_CLIENT_ATTEMPTS, RealClientForbidden

    with pytest.raises(RealClientForbidden):
        llm_pass.main(_args(tmp_path, "--yes"), env={"ANTHROPIC_API_KEY": FAKE_KEY})
    assert REAL_CLIENT_ATTEMPTS.pop() == "attempt"  # guard's own self-test
