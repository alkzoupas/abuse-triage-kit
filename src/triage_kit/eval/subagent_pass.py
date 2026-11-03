"""Exploratory LLM pass with Claude Code subagents as the reviewer (no API key needed).

See PREREGISTRATION.md, AMENDMENTS (2026-11-03, night). This is NOT the API reviewer:
accounts are reviewed in batches by subagents inside Claude Code, with no token counts.

    python -m triage_kit.eval.subagent_pass prepare --seed 9001 --s 0.5 --out DIR
        writes DIR/batches/NN.txt (one prompt per subagent) and DIR/manifest.json
    # one fresh subagent per batch: it reads DIR/batches/NN.txt and writes DIR/replies/NN.txt
    python -m triage_kit.eval.subagent_pass collect --seed 9001 --s 0.5 --out DIR --model M
        parses the replies and writes results/<run>/llm_bundle_subagent_<M>.json

Labels are used only to keep members of the same group in different batches. No label,
family name or group id appears in any batch file (checked when the batches are written).
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from triage_kit.detectors.llm.render import PROMPT_VERSION, render_prompt
from triage_kit.detectors.llm.reviewer import parse_verdict
from triage_kit.eval.llm_bundle import bundle_metrics
from triage_kit.eval.run_eval import RunConfig
from triage_kit.eval.splits import make_splits
from triage_kit.eval.truth import load_truth
from triage_kit.generator import GeneratorConfig, generate, write_generated
from triage_kit.schema import AccountEvents, read_dataset

BATCH_SIZE = 48
SHUFFLE_SEED = 20261103
WRAPPER = """You are helping score an evaluation. Judge only from the text in this file. Do not \
read, search, open or run anything else: no other files, no code, no web.

Below are {n} separate account reviews, each between ===ACCOUNT k=== and ===END k===. Each has \
its own instructions. Judge every account on its own, as if it were the only one you had seen.

Your reply is exactly {n} lines and nothing else: one JSON object per line, in the order given, \
each of the form
{{"id": "<the account id shown as id=...>", "verdict": "flag" or "pass", "score": <0 to 1>, \
"rationale": "<one sentence>"}}

"""
FORBIDDEN = ("family", "group_id", "grp-", "is_abuse", "template_operator", "review_ring")


def _load(seed: float | int, s: float, n: int) -> tuple[RunConfig, list[AccountEvents], Any]:
    cfg = RunConfig(seed=int(seed), s=s, n_accounts=n, n_boot=1000)
    data_dir = cfg.data_root / cfg.name
    if not (data_dir / "events.jsonl").exists():
        write_generated(generate(GeneratorConfig(seed=int(seed), s=s, n_accounts=n)), data_dir)
    return cfg, read_dataset(data_dir), load_truth(data_dir).labels


def assign_batches(
    test_ids: Sequence[str], labels: Mapping[str, Mapping[str, Any]], size: int, seed: int
) -> list[list[str]]:
    """Shuffle test accounts into batches of about `size`, never putting two members of the
    same group in one batch. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    ids = [test_ids[i] for i in rng.permutation(len(test_ids))]
    n_b = max(1, -(-len(ids) // size))
    batches: list[list[str]] = [[] for _ in range(n_b)]
    groups: dict[str, list[str]] = {}
    singles: list[str] = []
    for a in ids:
        g = labels[a].get("group_id")
        (groups.setdefault(str(g), []) if g else singles).append(a)
    for _, members in sorted(groups.items()):
        if len(members) > n_b:
            raise ValueError("a group has more test members than there are batches")
        order = sorted(range(n_b), key=lambda b: (len(batches[b]), rng.random()))
        for b, a in zip(order, members, strict=False):
            batches[b].append(a)
    for a in singles:
        b = min(range(n_b), key=lambda i: (len(batches[i]), i))
        batches[b].append(a)
    for batch in batches:
        rng.shuffle(batch)
    return batches


def batch_text(accounts: Sequence[AccountEvents]) -> str:
    parts = [WRAPPER.format(n=len(accounts))]
    for k, ae in enumerate(accounts, 1):
        parts.append(f"===ACCOUNT {k}===\n{render_prompt(ae)}===END {k}===\n")
    return "\n".join(parts)


def parse_reply(text: str, expected_ids: Sequence[str]) -> dict[str, float]:
    """Score each expected id from its JSON line; missing or unparseable -> 0.5."""
    found: dict[str, float] = {}
    for line in (text or "").splitlines():
        m = re.search(r"\{.*\}", line)
        if not m:
            continue
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and str(d.get("id")) in expected_ids and str(d["id"]) not in found:
            found[str(d["id"])] = parse_verdict(m.group(0))[0]
    return {a: found.get(a, 0.5) for a in expected_ids}


def prepare(seed: int, s: float, n: int, out: Path) -> dict[str, Any]:
    _, data, labels = _load(seed, s, n)
    split = make_splits([ae.account for ae in data], labels)[0]
    by_id = {ae.account_id: ae for ae in data}
    batches = assign_batches(sorted(split.test_ids), labels, BATCH_SIZE, SHUFFLE_SEED)
    (out / "batches").mkdir(parents=True, exist_ok=True)
    (out / "replies").mkdir(parents=True, exist_ok=True)
    for i, ids in enumerate(batches):
        text = batch_text([by_id[a] for a in ids])
        low = text.lower()
        bad = [w for w in FORBIDDEN if w in low]
        if bad:
            raise RuntimeError(f"batch {i} contains label vocabulary: {bad}")
        (out / "batches" / f"{i:02d}.txt").write_text(text)
    manifest = {"seed": seed, "s": s, "n": n, "prompt_version": PROMPT_VERSION, "batches": batches}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def collect(seed: int, s: float, n: int, out: Path, model: str) -> Path:
    cfg, data, labels = _load(seed, s, n)
    split = make_splits([ae.account for ae in data], labels)[0]
    manifest = json.loads((out / "manifest.json").read_text())
    scores: dict[str, float] = {}
    missing_batches, unparsed = [], 0
    for i, ids in enumerate(manifest["batches"]):
        p = out / "replies" / f"{i:02d}.txt"
        if not p.exists():
            missing_batches.append(i)
            continue
        got = parse_reply(p.read_text(), ids)
        unparsed += sum(1 for a in ids if a not in _ids_in(p.read_text()))
        scores.update(got)
    bundle = {
        "provider": "claude-code-subagent",
        "model": model,
        "run": cfg.name,
        "prompt_version": f"{PROMPT_VERSION}+batched{BATCH_SIZE}",
        "n_scored": len(scores),
        "n_unparsed_scored_0.5": unparsed,
        "missing_batches": missing_batches,
        "scores": {a: round(v, 6) for a, v in sorted(scores.items())},
        "tokens_in_total": None,
        "tokens_out_total": None,
        "note": (
            "Exploratory: Claude Code subagents, about 48 accounts per subagent, inside the "
            "Claude Code system prompt; no token counts; not the API reviewer."
        ),
        "metrics": bundle_metrics(cfg, split, labels, scores),
    }
    dest = cfg.results_root / cfg.name
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"llm_bundle_subagent_{model.replace('/', '_')}.json"
    path.write_text(json.dumps(bundle, sort_keys=True, indent=1) + "\n")
    return path


def _ids_in(text: str) -> set[str]:
    out = set()
    for m in re.finditer(r'"id"\s*:\s*"([^"]+)"', text or ""):
        out.add(m.group(1))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("step", choices=("prepare", "collect"))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="unknown")
    a = ap.parse_args(argv)
    if a.step == "prepare":
        m = prepare(a.seed, a.s, a.n, a.out)
        print(f"wrote {len(m['batches'])} batches to {a.out / 'batches'}")
    else:
        print(f"wrote {collect(a.seed, a.s, a.n, a.out, a.model)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
