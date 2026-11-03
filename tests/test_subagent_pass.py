import re
from collections import Counter

from tests.conftest import Dataset
from triage_kit.eval.subagent_pass import FORBIDDEN, assign_batches, batch_text, parse_reply


def test_batches_cover_every_account_once_and_split_groups(dev_s1: Dataset) -> None:
    labels = {r["account_id"]: r for r in dev_s1.labels}
    ids = sorted(labels)[:900]
    batches = assign_batches(ids, labels, 48, seed=1)
    flat = [a for b in batches for a in b]
    assert sorted(flat) == ids  # every account exactly once
    assert max(len(b) for b in batches) - min(len(b) for b in batches) <= 2
    for b in batches:  # never two members of one group in a batch
        g = Counter(labels[a]["group_id"] for a in b if labels[a].get("group_id"))
        assert all(c == 1 for c in g.values())
    assert batches == assign_batches(ids, labels, 48, seed=1)  # deterministic


def test_batch_text_has_no_label_vocabulary(dev_s1: Dataset) -> None:
    text = batch_text(dev_s1.data[:48]).lower()
    assert not [w for w in FORBIDDEN if w in text]
    assert len(re.findall(r"===account \d+===", text)) == 48


def test_parse_reply_scores_known_ids_and_defaults_missing_to_half() -> None:
    reply = "\n".join(
        [
            '{"id": "a1", "verdict": "flag", "score": 0.9, "rationale": "x"}',
            "some chatter",
            '{"id": "a2", "verdict": "pass", "rationale": "no score"}',
            '{"id": "zz", "verdict": "flag", "score": 1.0}',  # not expected: ignored
            '{"id": "a1", "verdict": "pass", "score": 0.0}',  # duplicate: first wins
        ]
    )
    got = parse_reply(reply, ["a1", "a2", "a3"])
    assert got == {"a1": 0.9, "a2": 0.1, "a3": 0.5}
