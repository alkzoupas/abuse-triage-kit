"""Loads labels and generator metadata. Only `triage_kit.eval` may import this."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triage_kit.generator import FAMILIES

COORDINATED = tuple(n for n, f in FAMILIES.items() if f.coordinated)


@dataclass
class Truth:
    labels: dict[str, dict[str, Any]]
    meta: dict[str, Any]


def load_truth(directory: Path) -> Truth:
    labels = {}
    for line in (directory / "truth" / "labels.jsonl").read_text().splitlines():
        r = json.loads(line)
        labels[r["account_id"]] = r
    meta = json.loads((directory / "truth" / "meta.json").read_text())
    return Truth(labels, meta)
