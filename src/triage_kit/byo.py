"""Bring your own events: load accounts + events (CSV or JSONL) in the kit's schema,
validate them, and score them with the rules and the feature model.

    python -m triage_kit.byo --accounts my_accounts.csv --events my_events.jsonl \
        --out byo_scores.csv

The feature model is trained on SYNTHETIC data from this kit (no labels exist for
your events), so its scores are a demonstration, not a calibrated probability.
See docs/byo-events.md.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from triage_kit.detectors.feature_model import FeatureModel
from triage_kit.detectors.features import feature_table
from triage_kit.detectors.rules import RulesDetector, load_rules
from triage_kit.schema import (
    Account,
    AccountEvents,
    Event,
    SchemaError,
    account_schema,
    event_schema,
    group_events,
    validate_account,
    validate_event,
)


class BYOError(ValueError):
    """Input does not match the schema. The message names the file, line and field."""


def _coerce(value: str, types: list[str], field: str, where: str) -> Any:
    if value == "" and "null" in types:
        return None
    try:
        if "integer" in types:
            return int(value)
        if "number" in types:
            return float(value)
        if "boolean" in types:
            v = value.strip().lower()
            if v in ("true", "1", "yes"):
                return True
            if v in ("false", "0", "no"):
                return False
            raise ValueError(value)
    except ValueError:
        raise BYOError(f"{where}field {field}: cannot read {value!r} as {types[0]}") from None
    return value


def _rows(path: Path, schema: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    props = schema["properties"]
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            missing = [c for c in schema["required"] if c not in cols]
            if missing:
                raise BYOError(f"{path.name}: missing column(s) {', '.join(missing)}")
            unknown = [c for c in cols if c not in props]
            if unknown:
                raise BYOError(f"{path.name}: unknown column(s) {', '.join(unknown)}")
            for i, row in enumerate(reader, 2):  # line 1 is the header
                where = f"{path.name} line {i}: "
                out = {}
                for k, v in row.items():
                    p = props[k]
                    t = p.get("type", ["string"])
                    types = [t] if isinstance(t, str) else list(t)
                    if "enum" in p:
                        types = ["string"]
                    out[k] = _coerce(v or "", types, k, where)
                yield where, out
    elif path.suffix in (".jsonl", ".ndjson"):
        for i, line in enumerate(path.read_text().splitlines(), 1):
            if not line.strip():
                continue
            where = f"{path.name} line {i}: "
            try:
                yield where, json.loads(line)
            except json.JSONDecodeError as e:
                raise BYOError(f"{where}not valid JSON ({e.msg})") from None
    else:
        raise BYOError(f"{path.name}: unsupported file type {path.suffix!r} (use .csv or .jsonl)")


def load(accounts_path: Path, events_path: Path) -> list[AccountEvents]:
    """Load and validate. Raises BYOError with a precise message on the first problem."""
    accounts: list[Account] = []
    seen: set[str] = set()
    for where, d in _rows(accounts_path, account_schema()):
        try:
            validate_account(d, where=where)
        except SchemaError as e:
            raise BYOError(str(e)) from None
        if d["account_id"] in seen:
            raise BYOError(f"{where}duplicate account_id {d['account_id']!r}")
        seen.add(d["account_id"])
        accounts.append(Account.from_dict(d))
    events: list[Event] = []
    for where, d in _rows(events_path, event_schema()):
        full = {k: d.get(k) for k in event_schema()["properties"]} | d
        try:
            validate_event(full, where=where)
        except SchemaError as e:
            raise BYOError(str(e)) from None
        if full["account_id"] not in seen:
            raise BYOError(
                f"{where}unknown account_id {full['account_id']!r} (not in accounts file)"
            )
        events.append(Event.from_dict(full))
    return group_events(accounts, events)


def train_synthetic_model(n: int = 5000, seed: int = 101, s: float = 0.5) -> FeatureModel:
    """gb+sim trained on the kit's synthetic data (dev seed), all six families."""
    from triage_kit.generator import GeneratorConfig, generate

    g = generate(GeneratorConfig(seed=seed, s=s, n_accounts=n))
    data = group_events(g.accounts, g.events)
    y = np.array([int(r["is_abuse"]) for r in g.labels])
    _, X, names = feature_table(data, reference=data)
    return FeatureModel("gb+sim").fit_table(X, y, names)


def score(data: list[AccountEvents], model: FeatureModel) -> list[dict[str, Any]]:
    rules = RulesDetector(seen_families={r.family for r in load_rules()})
    _, X, names = feature_table(data, reference=data)  # similarity within YOUR population
    from triage_kit.detectors.similarity import SimilarityIndex

    model.index = SimilarityIndex(data)
    p = model.predict_table(X)
    out: list[dict[str, Any]] = []
    for ae, pm in zip(data, p, strict=True):
        rs, rw = rules.score(ae)
        _, mw = model.score(ae)
        out.append(
            {
                "account_id": ae.account_id,
                "rules_score": round(rs, 6),
                "rules_rationale": rw,
                "model_score": round(float(pm), 6),
                "model_rationale": mw,
            }
        )
    out.sort(key=lambda r: (-float(r["model_score"]), str(r["account_id"])))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--accounts", type=Path, required=True)
    ap.add_argument("--events", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("byo_scores.csv"))
    ap.add_argument("--train-n", type=int, default=5000, help="synthetic accounts to train on")
    a = ap.parse_args(argv)
    try:
        data = load(a.accounts, a.events)
    except BYOError as e:
        print(f"input error: {e}")
        return 2
    print(f"validated {len(data)} accounts, {sum(len(x.events) for x in data)} events")
    model = train_synthetic_model(n=a.train_n)
    rows = score(data, model)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["account_id"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {a.out} (ranked by model_score; model trained on synthetic data, see docs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
