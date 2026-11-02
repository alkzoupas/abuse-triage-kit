# PROGRESS

## Stories
- [x] 1 Repo skeleton (pyproject, Makefile, CI, MIT)
- [x] 2 Event schema: dataclasses + JSON Schema (src/triage_kit/schema.py, schemas/*.json)
- [x] 3 Generator: benign + 6 families, knob s, truth/ kept separate (src/triage_kit/generator/)
- [x] 4 Oracle per family (src/triage_kit/eval/oracle.py) + AP/ROC-AUC with known-answer tests. Dev seed 101: oracle AP = 1.0 for all 6 families at s=1, ~= base rate at s=0.
- [x] 5 Splits: time-ordered 60/40 by unit (account, or group for coordinated), in_family + 3 held-out folds, leakage asserted

## Decisions
- src/ layout, package `triage_kit`; Makefile uses .venv if present so local and CI share targets.
- Schema has fixed optional fields (target_id, amount, success, rating, text) instead of a free-form dict: easier to validate and document.
- Knob s scales every mechanism (Binomial(n,s), Poisson(r*s), prob ~ s, lerp of created_at); at s=0 nothing is added or removed, so abuse = benign draw.
- review_ring and template_operator are coordinated; 30 groups each per 5,000 accounts (raised from 20 before any detector existed: only 4 ring groups reached test). Health floor min_groups=12.
- Train fraction 0.6 (not 0.7) so coordinated test cells get ~6-14 groups. Held-out folds test on the SAME test accounts as in_family, for a like-for-like comparison.
- Group counts (structure only, no metrics) were printed once for eval seed 9001; nothing was tuned on it.
- Listing popularity sampled from a precomputed CDF (perf: generation 11s -> 5s for 5,000 accounts). No detector exists yet, so this is not a post-detector generator change.

## Questions for owner
- (none yet)

## Last check
- `make test`: 35 passed in 47.39s (exit 0); lint and typecheck exit 0.
