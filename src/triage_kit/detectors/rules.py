"""Rules detector: evaluates rules.yaml over the engineered features."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from triage_kit.detectors.features import FEATURE_NAMES, account_features, feature_matrix
from triage_kit.schema import AccountEvents

OPS = ("at_least", "at_most", "is_true")


class RuleError(ValueError):
    pass


@dataclass(frozen=True)
class Condition:
    feature: str
    op: str
    value: float

    def holds(self, x: float) -> bool:
        if self.op == "at_least":
            return x >= self.value
        if self.op == "at_most":
            return x <= self.value
        return bool(x) == bool(self.value)


@dataclass(frozen=True)
class Rule:
    id: str
    family: str
    says: str
    conditions: tuple[Condition, ...]
    weight: float

    def fires(self, f: dict[str, float]) -> bool:
        return all(c.holds(f[c.feature]) for c in self.conditions)


def _cond(d: Any, rid: str) -> Condition:
    if not isinstance(d, dict) or "feature" not in d:
        raise RuleError(f"rule {rid}: each condition needs a 'feature'")
    feat = d["feature"]
    if feat not in FEATURE_NAMES:
        raise RuleError(f"rule {rid}: unknown feature {feat!r}")
    ops = [k for k in d if k != "feature"]
    if len(ops) != 1 or ops[0] not in OPS:
        raise RuleError(f"rule {rid}: condition needs exactly one of {OPS}")
    return Condition(feat, ops[0], float(d[ops[0]]))


def load_rules(path: Path | None = None) -> list[Rule]:
    text = (
        path.read_text()
        if path is not None
        else resources.files("triage_kit.detectors").joinpath("rules.yaml").read_text()
    )
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or not isinstance(doc.get("rules"), list):
        raise RuleError("rules file must contain a top-level 'rules:' list")
    out = []
    for r in doc["rules"]:
        rid = str(r.get("id", "?"))
        for key in ("id", "family", "says", "if", "weight"):
            if key not in r:
                raise RuleError(f"rule {rid}: missing {key!r}")
        cond = r["if"]
        conds = [_cond(c, rid) for c in cond["all"]] if "all" in cond else [_cond(cond, rid)]
        w = float(r["weight"])
        if not 0 < w <= 1:
            raise RuleError(f"rule {rid}: weight must be in (0, 1]")
        out.append(Rule(rid, str(r["family"]), str(r["says"]), tuple(conds), w))
    return out


class RulesDetector:
    name = "rules"

    def __init__(self, seen_families: Iterable[str], path: Path | None = None) -> None:
        seen = set(seen_families)
        self.rules = [r for r in load_rules(path) if r.family in seen]

    def _combine(self, f: dict[str, float]) -> tuple[float, list[Rule]]:
        fired = [r for r in self.rules if r.fires(f)]
        return 1.0 - float(np.prod([1.0 - r.weight for r in fired])), fired

    def score(self, account_events: AccountEvents) -> tuple[float, str]:
        s, fired = self._combine(account_features(account_events))
        if not fired:
            return 0.0, "no rule fired"
        return s, "; ".join(f"{r.id} [{r.family}]: {r.says}" for r in fired)

    def score_many(self, data: Sequence[AccountEvents], X: np.ndarray | None = None) -> np.ndarray:
        X = feature_matrix(data) if X is None else X
        out = np.empty(len(X))
        for i, row in enumerate(X):
            out[i] = self._combine(dict(zip(FEATURE_NAMES, row, strict=False)))[0]
        return out
