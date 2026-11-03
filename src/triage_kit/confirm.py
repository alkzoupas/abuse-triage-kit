"""`make confirm`: frozen confirmation of similarity v3 on fresh seeds (PREREGISTRATION.md
AMENDMENTS, 2026-11-03 later). Per confirmation seed: standard s = 0 (null control),
standard s = 0.5 and ordinary-mode s = 0.5, each with bootstrap CIs. Then rebuilds the
report, which gains a confirmation section. Needs `make demo` results for the rest of the
report. Separate from `make demo` so the demo stays under its 10-minute limit.
"""

from __future__ import annotations

import time
from pathlib import Path

from triage_kit.demo import ROOT, _commit
from triage_kit.eval.run_eval import RunConfig, run_eval
from triage_kit.report import build_report, load_prereg


def plan(prereg: dict[str, object]) -> list[tuple[int, float, str]]:
    s = float(prereg["primary_s"])  # type: ignore[arg-type]
    seeds = [int(x) for x in prereg["confirmation_seeds"]]  # type: ignore[attr-defined]
    return [
        (sd, sv, mode)
        for sd in seeds
        for sv, mode in ((0.0, "standard"), (s, "standard"), (s, "ordinary"))
    ]


def main(out_dir: Path = Path(".")) -> Path:
    prereg = load_prereg(ROOT / "PREREGISTRATION.md")
    t0 = time.time()
    for sd, sv, mode in plan(prereg):
        cfg = RunConfig(
            seed=sd,
            s=sv,
            n_boot=prereg["bootstrap_samples"],
            k=prereg["review_budget_k"],
            data_root=out_dir / "data",
            results_root=out_dir / "results",
            template_mode=mode,
        )
        run_eval(cfg)
        print(f"[{time.time() - t0:5.0f}s] evaluated {cfg.name}", flush=True)
    path = build_report(
        out_dir / "results",
        ROOT / "PREREGISTRATION.md",
        out_dir / "report" / "index.html",
        commit=_commit(),
    )
    print(f"wrote {path} in {time.time() - t0:.0f}s")
    return path


if __name__ == "__main__":
    main()
