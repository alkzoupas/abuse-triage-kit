import csv
import json
from pathlib import Path

import pytest

from triage_kit import byo

ROOT = Path(__file__).resolve().parents[1]
EX = ROOT / "examples" / "byo"


def _write(p: Path, text: str) -> Path:
    p.write_text(text)
    return p


ACCTS = "account_id,created_at,country,email_domain\nu1,1767225600,DE,mail.example\n"


def test_example_files_load_and_validate() -> None:
    data = byo.load(EX / "accounts.csv", EX / "events.jsonl")
    assert len(data) >= 10 and sum(len(ae.events) for ae in data) > 100


@pytest.mark.parametrize(
    "events,msg",
    [
        ("account_id,ts,type,ip,device_id\nu1,1767229200,teleport,10.0.0.1,d1\n", "line 2.*type"),
        ("account_id,ts,type,ip,device_id\nu1,yesterday,login,10.0.0.1,d1\n", "line 2.*ts"),
        ("account_id,ts,type,ip\nu1,1767229200,login,10.0.0.1\n", "missing column.*device_id"),
        (
            "account_id,ts,type,ip,device_id\nzz,1767229200,login,10.0.0.1,d1\n",
            "unknown account_id 'zz'",
        ),
        (
            "account_id,ts,type,ip,device_id,rating\nu1,1767229200,review,1.1.1.1,d1,9\n",
            "line 2.*rating",
        ),
    ],
)
def test_malformed_input_is_rejected_with_clear_error(
    tmp_path: Path, events: str, msg: str
) -> None:
    a = _write(tmp_path / "a.csv", ACCTS)
    e = _write(tmp_path / "e.csv", events)
    with pytest.raises(byo.BYOError, match=msg):
        byo.load(a, e)


def test_bad_json_line_and_duplicate_accounts(tmp_path: Path) -> None:
    a = _write(tmp_path / "a.csv", ACCTS + "u1,1767225600,DE,mail.example\n")
    e = _write(tmp_path / "e.jsonl", "")
    with pytest.raises(byo.BYOError, match="duplicate account_id 'u1'"):
        byo.load(a, e)
    a2 = _write(tmp_path / "a2.csv", ACCTS)
    e2 = _write(tmp_path / "e2.jsonl", '{"account_id": "u1", "ts": 1767229200\n')
    with pytest.raises(byo.BYOError, match="e2.jsonl line 1: not valid JSON"):
        byo.load(a2, e2)
    with pytest.raises(byo.BYOError, match="unsupported file type"):
        byo.load(a2, _write(tmp_path / "e.txt", ""))


def test_end_to_end_on_example(tmp_path: Path) -> None:
    out = tmp_path / "scores.csv"
    rc = byo.main(
        [
            "--accounts",
            str(EX / "accounts.csv"),
            "--events",
            str(EX / "events.jsonl"),
            "--out",
            str(out),
            "--train-n",
            "1500",
        ]
    )
    assert rc == 0
    rows = list(csv.DictReader(out.open()))
    data = byo.load(EX / "accounts.csv", EX / "events.jsonl")
    assert {r["account_id"] for r in rows} == {ae.account_id for ae in data}
    for r in rows:
        assert 0 <= float(r["rules_score"]) <= 1 and 0 <= float(r["model_score"]) <= 1
        assert r["rules_rationale"] and r["model_rationale"]
    scores = [float(r["model_score"]) for r in rows]
    assert scores == sorted(scores, reverse=True)
    assert not (EX / "truth").exists()  # example ships without labels
    json.dumps(rows)
