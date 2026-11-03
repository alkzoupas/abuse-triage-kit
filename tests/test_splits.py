import pytest

from tests.conftest import Dataset
from triage_kit.eval.splits import (
    HELDOUT_FOLDS,
    LeakageError,
    assert_split_ok,
    make_splits,
    unit_of,
    unit_times,
)
from triage_kit.eval.truth import COORDINATED
from triage_kit.generator import FAMILIES


def _splits(d: Dataset):  # type: ignore[no-untyped-def]
    return make_splits([ae.account for ae in d.data], {r["account_id"]: r for r in d.labels})


def test_one_in_family_split_and_three_rotated_heldout_folds(dev_s1: Dataset) -> None:
    sp = _splits(dev_s1)
    assert [s.name for s in sp] == ["in_family", "heldout_1", "heldout_2", "heldout_3"]
    held = [f for fold in HELDOUT_FOLDS for f in fold]
    assert sorted(held) == sorted(FAMILIES)  # each family held out exactly once
    for s in sp[1:]:
        assert len(s.train_families) == 4 and len(s.test_families) == 2


def test_no_account_or_group_in_both_train_and_test(dev_s1: Dataset) -> None:
    labels = {r["account_id"]: r for r in dev_s1.labels}
    for s in _splits(dev_s1):
        assert not (s.train_ids & s.test_ids)
        tr = {labels[a]["group_id"] for a in s.train_ids} - {None}
        te = {labels[a]["group_id"] for a in s.test_ids} - {None}
        assert not (tr & te)
        assert_split_ok(s, labels)


def test_heldout_never_trains_on_a_heldout_family(dev_s1: Dataset) -> None:
    labels = {r["account_id"]: r for r in dev_s1.labels}
    for s in _splits(dev_s1)[1:]:
        fams = {labels[a]["family"] for a in s.train_ids}
        assert not (fams & set(s.test_families))


def test_assertion_fires_on_a_leaky_split(dev_s1: Dataset) -> None:
    labels = {r["account_id"]: r for r in dev_s1.labels}
    s = _splits(dev_s1)[1]
    victim = next(a for a in s.test_ids if labels[a]["family"] in s.test_families)
    bad = s.__class__(
        s.name, s.train_ids | {victim}, s.test_ids - {victim}, s.train_families, s.test_families
    )
    with pytest.raises(LeakageError):
        assert_split_ok(bad, labels)
    group_member = next(a for a in s.test_ids if labels[a]["group_id"])
    bad2 = s.__class__(
        s.name, s.train_ids | {group_member}, s.test_ids, s.train_families, s.test_families
    )
    with pytest.raises(LeakageError):
        assert_split_ok(bad2, labels)


def test_split_is_by_time(dev_s1: Dataset) -> None:
    accts = {ae.account_id: ae.account for ae in dev_s1.data}
    labels = {r["account_id"]: r for r in dev_s1.labels}
    s = _splits(dev_s1)[0]
    unit_time = unit_times(list(accts.values()), labels)
    tr = max(unit_time[unit_of(labels[a])] for a in s.train_ids)
    te = min(unit_time[unit_of(labels[a])] for a in s.test_ids)
    assert tr < te
    assert max(accts[a].created_at for a in s.train_ids) < min(
        accts[a].created_at for a in s.test_ids
    )


def test_every_family_and_coordinated_group_count_reaches_test(dev_s1: Dataset) -> None:
    labels = {r["account_id"]: r for r in dev_s1.labels}
    s = _splits(dev_s1)[0]
    for fam in FAMILIES:
        assert any(labels[a]["family"] == fam for a in s.test_ids), fam
    for fam in COORDINATED:
        groups = {labels[a]["group_id"] for a in s.test_ids if labels[a]["family"] == fam}
        assert len(groups) >= 3, fam


def test_coordinated_groups_reach_test_even_at_null(dev_s0: Dataset) -> None:
    labels = {r["account_id"]: r for r in dev_s0.labels}
    s = _splits(dev_s0)[0]
    for fam in COORDINATED:
        groups = {labels[a]["group_id"] for a in s.test_ids if labels[a]["family"] == fam}
        assert len(groups) >= 3, fam


def test_strict_time_split_at_account_level_and_age_parity_at_null(dev_s0: Dataset) -> None:
    """Every train account is created before every test account. Without this, at s = 0
    whole groups of independently-aged accounts land in test and are older than the
    benign test accounts, which leaks group membership through account age."""
    import numpy as np

    accts = {ae.account_id: ae.account for ae in dev_s0.data}
    labels = {r["account_id"]: r for r in dev_s0.labels}
    s = _splits(dev_s0)[0]
    assert max(accts[a].created_at for a in s.train_ids) < min(
        accts[a].created_at for a in s.test_ids
    )
    ben = np.mean([accts[a].created_at for a in s.test_ids if labels[a]["family"] == "benign"])
    for fam in COORDINATED:
        grp = np.mean([accts[a].created_at for a in s.test_ids if labels[a]["family"] == fam])
        assert abs(grp - ben) < 4 * 86400, fam  # within 4 days on average
    assert s.excluded_ids and not (s.excluded_ids & (s.train_ids | s.test_ids))
    assert all(labels[a]["group_id"] for a in s.excluded_ids)
