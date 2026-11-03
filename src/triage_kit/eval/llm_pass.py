"""Optional real-LLM pass (owner-run, costs money).

    ANTHROPIC_API_KEY=... python -m triage_kit.eval.llm_pass --seed 9001 --s 0.5 \
        --model claude-opus-5-5 [--limit 200] --yes

Scores the test accounts of one run with a real model and writes a scores-only
bundle (results/<run>/llm_bundle_<model>.json) that the report loads. Prints a
cost estimate first, from config/prices.yaml (placeholder values unless you edit
them), and does nothing without --yes. Never prints the key or any env var.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from triage_kit.detectors.llm import anthropic_provider
from triage_kit.detectors.llm.anthropic_provider import AnthropicProvider
from triage_kit.detectors.llm.render import PROMPT_VERSION, render_prompt
from triage_kit.detectors.llm.reviewer import LLMReviewer
from triage_kit.detectors.llm.types import approx_tokens
from triage_kit.eval.llm_bundle import bundle_metrics
from triage_kit.eval.metrics import load_prices
from triage_kit.eval.run_eval import ROOT, RunConfig
from triage_kit.eval.splits import make_splits
from triage_kit.eval.truth import load_truth
from triage_kit.generator import GeneratorConfig, generate, write_generated
from triage_kit.schema import read_dataset


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    client_factory: Callable[[], Any] | None = None,
) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, default=0.5)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--model", default="claude-opus-5-5")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N test accounts")
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--effort", default=None, help="optional output_config effort level")
    ap.add_argument(
        "--no-fallbacks", action="store_true", help="disable server-side refusal fallbacks"
    )
    ap.add_argument("--prices", type=Path, default=ROOT / "config" / "prices.yaml")
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--results-root", type=Path, default=Path("results"))
    ap.add_argument("--yes", action="store_true", help="actually call the API (costs money)")
    a = ap.parse_args(argv)
    env = os.environ if env is None else env

    if not env.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set; nothing to do. (Only its presence is checked.)")
        raise SystemExit(2)

    cfg = RunConfig(
        seed=a.seed, s=a.s, n_accounts=a.n, data_root=a.data_root, results_root=a.results_root
    )
    data_dir = cfg.data_root / cfg.name
    if not (data_dir / "events.jsonl").exists():
        write_generated(generate(GeneratorConfig(seed=a.seed, s=a.s, n_accounts=a.n)), data_dir)
    data = read_dataset(data_dir)
    labels = load_truth(data_dir).labels  # used ONLY to pick the test split, never sent
    split = make_splits([ae.account for ae in data], labels)[0]
    test_ids = sorted(split.test_ids)
    if a.limit:
        test_ids = test_ids[: a.limit]
    by_id = {ae.account_id: ae for ae in data}
    todo = [by_id[x] for x in test_ids]

    prices = load_prices(a.prices)
    if a.model not in prices["models"]:
        print(f"No price for {a.model!r} in {a.prices}; add one before running.")
        raise SystemExit(2)
    p = prices["models"][a.model]
    est_in = sum(approx_tokens(render_prompt(ae)) for ae in todo)
    est_out = a.max_tokens * len(todo)  # upper bound
    est = (est_in * p["input_per_mtok"] + est_out * p["output_per_mtok"]) / 1e6
    tag = (
        " (PLACEHOLDER prices from config; edit config/prices.yaml)"
        if prices["placeholder"]
        else ""
    )
    print(f"{len(todo)} accounts, ~{est_in:,} input tokens, <= {est_out:,} output tokens.")
    print(f"Estimated cost: up to {est:.2f} {prices['currency']}{tag}")
    if not a.yes:
        print("Dry run. Re-run with --yes to call the API.")
        return 0

    # look up through the module so the test-suite guard can intercept it
    client = (client_factory or anthropic_provider._make_real_client)()
    prov = AnthropicProvider(
        model=a.model,
        client=client,
        max_tokens=a.max_tokens,
        effort=a.effort,
        fallbacks=not a.no_fallbacks,
    )
    rev = LLMReviewer(prov)
    scores = {ae.account_id: round(rev.score(ae)[0], 6) for ae in todo}
    bundle = {
        "provider": "anthropic",
        "model": a.model,
        "run": cfg.name,
        "prompt_version": PROMPT_VERSION,
        "n_scored": len(scores),
        "scores": scores,
        "tokens_in_total": sum(c.tokens_in for c in rev.calls),
        "tokens_out_total": sum(c.tokens_out for c in rev.calls),
        # per-family metrics on the in-family test cells (scored accounts only)
        "metrics": bundle_metrics(
            RunConfig(seed=a.seed, s=a.s, n_accounts=a.n, n_boot=1000), split, labels, scores
        ),
    }
    out = cfg.results_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"llm_bundle_{a.model.replace('/', '_')}.json"
    path.write_text(json.dumps(bundle, sort_keys=True, indent=1) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
