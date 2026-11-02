"""`make demo`: run the pre-registered evaluation on the EVAL seeds and write report/index.html.

Runs (all from PREREGISTRATION.md): the primary eval seed at s = 0 (null control),
the primary s and s = 1, each with bootstrap CIs; then the replication seeds at the
primary s (point estimates). Everything is offline.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

from triage_kit.eval.run_eval import RunConfig, run_eval
from triage_kit.report import build_report, load_prereg

ROOT = Path(__file__).resolve().parents[2]


def _commit() -> str:
    try:
        rev = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main(out_dir: Path = Path(".")) -> Path:
    prereg = load_prereg(ROOT / "PREREGISTRATION.md")
    seed, s = prereg["primary_seed"], prereg["primary_s"]
    n_boot = prereg["bootstrap_samples"]
    plan = [(seed, 0.0, n_boot), (seed, s, n_boot), (seed, 1.0, n_boot)]
    plan += [(seed, float(x), n_boot) for x in prereg["extra_s"]]  # post-hoc levels (amendments)
    plan += [(r, s, 0) for r in prereg["replication_seeds"]]
    t0 = time.time()
    for sd, sv, nb in plan:
        cfg = RunConfig(
            seed=sd,
            s=sv,
            n_boot=nb,
            k=prereg["review_budget_k"],
            data_root=out_dir / "data",
            results_root=out_dir / "results",
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
