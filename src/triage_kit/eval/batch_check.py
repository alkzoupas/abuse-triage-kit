"""Does a reviewer's score depend on its batch-mates? (exploratory)

See PREREGISTRATION.md, AMENDMENTS (2026-11-12). The second-stage numbers reuse one `llm:v2`
score per account from a full-test pass. This check rescores the accounts that decide the
held-out added recall (the pool) among different batch-mates, and reruns two of the new
batches unchanged as a noise floor.

    python -m triage_kit.eval.batch_check prepare --seed 9001 --out results/subagent_v2_rebatch
        writes OUT/batches/NN.txt and OUT/manifest.json
    # arm B: one fresh subagent per batch writes OUT/replies/NN.txt
    # arm B': batches 00 and 01 again, fresh subagents, OUT/replies_rerun/NN.txt
    python -m triage_kit.eval.batch_check collect --seed 9001 --out OUT --model M
        writes results/<run>/batch_check.json

The pool is abuse-dense (every family account stage 1 lets through, plus the benign accounts
that compete with them for stage 2's flags), so arm B is a stress test, not the composition of
a survivor-only review. Accounts outside the pool are not rescored and rank last.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Collection, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from scipy import stats

from triage_kit.eval.second_stage import (
    Stage1,
    bundle_path,
    fit_stage1,
    heldout_split_of,
    take_top_k,
    two_stage,
)
from triage_kit.eval.splits import cell_ids
from triage_kit.eval.subagent_pass import (
    FORBIDDEN,
    _ids_in,
    assign_batches,
    batch_text,
    parse_reply,
)
from triage_kit.generator import FAMILIES

POOL_SIZE = 188
BATCH_SIZE = 38  # five batches: one crew has five members in the pool
SHUFFLE_SEED = 20261112
RERUN_BATCHES = (0, 1)
MAX_RECALL_CHANGE = 0.10
HEADLINE = ("ato", "card_testing")
MOVED = 0.1
VERSION = "v2"
EPS = 1e-9


def select_pool(
    family_survivors: Collection[str], benign_scores: Mapping[str, float], size: int, seed: int
) -> list[str]:
    """Every surviving family account, then benign accounts by descending score until the pool
    has `size`. Ties in that order are broken by a seeded permutation. Sorted ids."""
    if len(family_survivors) > size:
        raise ValueError("the surviving family accounts alone exceed the pool size")
    names = sorted(benign_scores)
    tie = np.random.default_rng(seed).permutation(len(names))
    order = sorted(range(len(names)), key=lambda i: (-benign_scores[names[i]], int(tie[i])))
    fill = [names[i] for i in order[: size - len(family_survivors)]]
    return sorted([*family_survivors, *fill])


def pool_recall(
    y: npt.ArrayLike, s1: npt.ArrayLike, s2: npt.ArrayLike, in_pool: npt.ArrayLike, k: float
) -> float:
    """Added recall when stage 2 ranks the pool only: accounts outside it rank last."""
    s2a = np.where(np.asarray(in_pool, dtype=bool), np.asarray(s2, dtype=float), -1.0)
    return two_stage(y, s1, s2a, np.ones(len(s2a)), k)["added_recall"]


def score_changes(old: Sequence[float], new: Sequence[float]) -> dict[str, Any]:
    """How far scores moved between two passes over the same accounts."""
    o, n = np.asarray(old, dtype=float), np.asarray(new, dtype=float)
    if len(o) == 0:
        return {
            "n": 0,
            "share_moved": None,
            "mean_abs": None,
            "mean_signed": None,
            "spearman": None,
        }
    d = n - o
    constant = len(o) < 2 or np.ptp(o) == 0 or np.ptp(n) == 0
    rho = None if constant else round(float(stats.spearmanr(o, n).statistic), 6)
    return {
        "n": len(o),
        "share_moved": round(float(np.mean(np.abs(d) > MOVED + EPS)), 6),
        "mean_abs": round(float(np.mean(np.abs(d))), 6),
        "mean_signed": round(float(np.mean(d)), 6),
        "spearman": rho,
    }


def verdict(changes: Mapping[str, float], limit: float = MAX_RECALL_CHANGE) -> str:
    """The amendment's three outcomes from each family's absolute change in added recall."""
    if any(f not in changes for f in HEADLINE):
        raise ValueError("a headline family is missing")
    if all(d <= limit + EPS for d in changes.values()):
        return "supported"
    if all(changes[f] <= limit + EPS for f in HEADLINE):
        return "headline_only"
    return "not_supported"


def read_replies(
    replies: Path, batches: Sequence[Sequence[str]], which: Sequence[int]
) -> dict[str, float]:
    """Scores from the reply files of batches `which`. A missing file or an account with no
    parseable line stops the collection: nothing is scored 0.5 here."""
    scores: dict[str, float] = {}
    for i in which:
        p = replies / f"{i:02d}.txt"
        if not p.exists():
            raise FileNotFoundError(f"no reply for batch {i:02d} in {replies}")
        text = p.read_text()
        absent = [a for a in batches[i] if a not in _ids_in(text)]
        if absent:
            raise ValueError(f"batch {i:02d}: no parseable line for {absent}")
        scores.update(parse_reply(text, batches[i]))
    return scores


def _cells(st: Stage1, v2: Mapping[str, float]) -> list[dict[str, Any]]:
    """Held-out cells with stage 1's flags (share taken, top k) per account."""
    out = []
    for fam in FAMILIES:
        sp = st.splits[heldout_split_of()[fam]]
        ids = cell_ids(sp, fam, st.labels)
        y = np.array([int(st.labels[a]["family"] == fam) for a in ids])
        s1 = np.array([st.scores[sp.name][a] for a in ids])
        taken = take_top_k(s1, np.ones(len(ids)), st.cfg.k)
        s2 = np.array([float(v2.get(a, 0.5)) for a in ids])
        out.append({"family": fam, "ids": ids, "y": y, "s1": s1, "taken": taken, "s2": s2})
    return out


def _pool(st: Stage1, v2: Mapping[str, float]) -> tuple[list[str], list[dict[str, Any]]]:
    """The declared pool: family accounts that survive stage 1 in their own held-out cell, and
    benign accounts that survive it in at least one held-out split, by descending score."""
    cells = _cells(st, v2)
    survivors: set[str] = set()
    benign: dict[str, float] = {}
    for c in cells:
        for a, yy, t in zip(c["ids"], c["y"], c["taken"], strict=True):
            if t >= 1:
                continue
            if yy:
                survivors.add(a)
            else:
                benign[a] = float(v2[a])
    return select_pool(survivors, benign, POOL_SIZE, SHUFFLE_SEED), cells


def _original(st: Stage1, model: str) -> tuple[dict[str, float], dict[str, float]]:
    """(original v2 score per account, reported held-out added recall per family)."""
    res = st.cfg.results_root / st.cfg.name
    v2 = json.loads(bundle_path(res, model, VERSION).read_text())["scores"]
    reported = {
        c["family"]: float(c["stage2"]["llm:v2"]["added_recall"])
        for c in json.loads((res / "second_stage.json").read_text())["cells"]
        if c["condition"] == "heldout"
    }
    return v2, reported


def _check_reproduces(
    cells: Sequence[Mapping[str, Any]], pool: Collection[str], reported: Mapping[str, float], k: int
) -> dict[str, float]:
    got = {}
    for c in cells:
        in_pool = np.array([a in pool for a in c["ids"]])
        got[c["family"]] = pool_recall(c["y"], c["s1"], c["s2"], in_pool, k)
        if abs(got[c["family"]] - reported[c["family"]]) > 1e-5:
            raise RuntimeError(
                f"{c['family']}: pool gives {got[c['family']]:.6f} with the original scores, "
                f"reported {reported[c['family']]:.6f}; the pool rule is wrong"
            )
    return got


def prepare(seed: int, s: float, n: int, out: Path, model: str) -> dict[str, Any]:
    st = fit_stage1(seed, s, n)
    v2, reported = _original(st, model)
    pool, cells = _pool(st, v2)
    _check_reproduces(cells, set(pool), reported, st.cfg.k)
    by_id = {ae.account_id: ae for ae in st.data}
    batches = assign_batches(pool, st.labels, BATCH_SIZE, SHUFFLE_SEED)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "replies").mkdir(parents=True, exist_ok=True)
    (out / "replies_rerun").mkdir(parents=True, exist_ok=True)
    for i, ids in enumerate(batches):
        text = batch_text([by_id[a] for a in ids], VERSION)
        bad = [w for w in FORBIDDEN if w in text.lower()]
        if bad:
            raise RuntimeError(f"batch {i} contains label vocabulary: {bad}")
        (out / "batches" / f"{i:02d}.txt").write_text(text)
    manifest = {
        "seed": seed,
        "s": s,
        "n": n,
        "prompt_version": VERSION,
        "shuffle_seed": SHUFFLE_SEED,
        "rerun_batches": list(RERUN_BATCHES),
        "batches": batches,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def collect(seed: int, s: float, n: int, out: Path, model: str) -> Path:
    st = fit_stage1(seed, s, n)
    v2, reported = _original(st, model)
    pool, cells = _pool(st, v2)
    manifest = json.loads((out / "manifest.json").read_text())
    batches: list[list[str]] = manifest["batches"]
    if sorted(a for b in batches for a in b) != pool:
        raise RuntimeError("the manifest does not hold the declared pool")
    new = read_replies(out / "replies", batches, range(len(batches)))
    rerun = read_replies(out / "replies_rerun", batches, RERUN_BATCHES)

    pool_set = set(pool)
    original = _check_reproduces(cells, pool_set, reported, st.cfg.k)
    families = []
    for c in cells:
        in_pool = np.array([a in pool_set for a in c["ids"]])
        s2_new = np.array([new.get(a, -1.0) for a in c["ids"]])
        got = pool_recall(c["y"], c["s1"], s2_new, in_pool, st.cfg.k)
        families.append(
            {
                "family": c["family"],
                "n_pos": int(c["y"].sum()),
                "added_recall_original": round(original[c["family"]], 6),
                "added_recall_rebatched": round(got, 6),
                "change": round(got - original[c["family"]], 6),
            }
        )

    def moves(ids: Sequence[str], a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, Any]:
        groups = {
            "all": list(ids),
            "family": [x for x in ids if st.labels[x]["family"] != "benign"],
            "benign": [x for x in ids if st.labels[x]["family"] == "benign"],
        }
        return {
            g: score_changes([a[x] for x in members], [b[x] for x in members])
            for g, members in groups.items()
        }

    rerun_ids = sorted(rerun)
    result = {
        "run": st.cfg.name,
        "model": model,
        "prompt_version": VERSION,
        "k": st.cfg.k,
        "pool": len(pool),
        "pool_family": sum(1 for a in pool if st.labels[a]["family"] != "benign"),
        "batches": len(batches),
        "rerun_batches": list(RERUN_BATCHES),
        "max_recall_change": MAX_RECALL_CHANGE,
        "moved_threshold": MOVED,
        "families": families,
        "verdict": verdict({f["family"]: abs(f["change"]) for f in families}),
        "original_vs_rebatched": moves(pool, v2, new),
        "rebatched_vs_rerun": moves(rerun_ids, new, rerun),
        "original_vs_rebatched_on_rerun_accounts": moves(rerun_ids, v2, new),
        "scores": {
            a: {
                "original": v2[a],
                "rebatched": new[a],
                **({"rerun": rerun[a]} if a in rerun else {}),
            }
            for a in pool
        },
    }
    dest = st.cfg.results_root / st.cfg.name
    path = dest / "batch_check.json"
    path.write_text(json.dumps(result, sort_keys=True, indent=1) + "\n")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=("prepare", "collect"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="claude-opus-5-5")
    a = ap.parse_args(argv)
    if a.step == "prepare":
        m = prepare(a.seed, a.s, a.n, a.out, a.model)
        sizes = [len(b) for b in m["batches"]]
        print(f"wrote {len(sizes)} batches {sizes} to {a.out / 'batches'}")
    else:
        print(f"wrote {collect(a.seed, a.s, a.n, a.out, a.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
