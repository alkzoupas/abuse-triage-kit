"""Acceptance: `make demo` runs offline (network disabled) in under 10 minutes and writes
report/index.html with every required section heading."""

import subprocess
import sys
import time
from pathlib import Path

from triage_kit.report import MOCK_TAG, REQUIRED_SECTIONS

BLOCK_NETWORK_AND_RUN = """
import socket
def _blocked(*a, **k):
    raise OSError("network disabled for this test")
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
import runpy
runpy.run_module("triage_kit.demo", run_name="__main__")
"""


def test_demo_offline_under_ten_minutes(tmp_path: Path) -> None:
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-c", BLOCK_NETWORK_AND_RUN],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=600,
    )
    elapsed = time.time() - t0
    assert r.returncode == 0, r.stderr[-3000:]
    assert elapsed < 600
    page = (tmp_path / "report" / "index.html").read_text()
    for h in REQUIRED_SECTIONS:
        assert f"<h2>{h.replace(chr(39), '&#x27;')}</h2>" in page or f"<h2>{h}</h2>" in page, h
    assert MOCK_TAG in page
    assert "rules encode human" in page and "designed knowing this mechanism" in page
    assert "<script" not in page  # static: no JS
