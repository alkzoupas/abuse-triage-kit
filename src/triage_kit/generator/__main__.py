"""python -m triage_kit.generator --seed 101 --s 1.0 --out data/dev-101"""

import argparse
from pathlib import Path

from triage_kit.generator import GeneratorConfig, generate, health_check, write_generated


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--s", type=float, default=1.0, help="signal strength in [0, 1]")
    ap.add_argument("--n", type=int, default=5000, help="number of accounts")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    g = generate(GeneratorConfig(seed=a.seed, s=a.s, n_accounts=a.n))
    problems = health_check(g)
    if problems:
        raise SystemExit("health check failed: " + "; ".join(problems))
    write_generated(g, a.out)
    print(f"wrote {len(g.accounts)} accounts, {len(g.events)} events to {a.out}")


if __name__ == "__main__":
    main()
