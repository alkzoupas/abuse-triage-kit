"""Acceptance: the commit adding PREREGISTRATION.md comes before the commit adding the
full-eval entry point and before any detector module. Needs full git history
(CI checks out with fetch-depth: 0)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHORS = ["src/triage_kit/eval/run_eval.py", "src/triage_kit/detectors"]


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()


def _first_commit_adding(path: str) -> str:
    out = _git("log", "--diff-filter=A", "--format=%H", "--reverse", "--", path)
    return out.splitlines()[0] if out else ""


def test_history_is_complete() -> None:
    assert _git("rev-parse", "--is-shallow-repository") == "false", "fetch full history"


def test_preregistration_precedes_detectors_and_eval_entry_point() -> None:
    prereg = _first_commit_adding("PREREGISTRATION.md")
    assert prereg, "PREREGISTRATION.md was never committed"
    for anchor in ANCHORS:
        if not (ROOT / anchor).exists():
            continue  # not written yet; checked again once it exists
        first = _first_commit_adding(anchor)
        if not first:
            continue  # exists in the working tree but is not committed yet
        assert first != prereg, f"{anchor} added in the same commit as PREREGISTRATION.md"
        r = subprocess.run(["git", "merge-base", "--is-ancestor", prereg, first], cwd=ROOT)
        assert r.returncode == 0, f"{anchor} was added before PREREGISTRATION.md"
