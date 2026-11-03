"""Acceptance: no test creates a real LLM client."""

import ast
from pathlib import Path

TESTS = Path(__file__).resolve().parent


def test_guard_is_installed() -> None:
    import triage_kit.detectors.llm.anthropic_provider as ap

    assert ap._make_real_client.__name__ == "_blocked"


def test_no_test_file_imports_the_sdk_or_builds_a_client_directly() -> None:
    for f in TESTS.glob("*.py"):
        tree = ast.parse(f.read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                assert all(a.name.split(".")[0] != "anthropic" for a in n.names), f.name
            if isinstance(n, ast.ImportFrom):
                assert (n.module or "").split(".")[0] != "anthropic", f.name
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
                assert n.func.attr not in ("Anthropic", "AsyncAnthropic"), f.name
