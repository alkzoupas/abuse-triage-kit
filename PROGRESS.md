# PROGRESS

## Stories
- [x] 1 Skeleton (pyproject, Makefile, CI, MIT) · [x] 2 Schema (dataclasses + JSON Schema)
- [x] 3 Generator: benign + 6 families, knob s, truth/ kept separate
- [x] 4 Oracle (eval/oracle.py). Dev 101: oracle AP = 1.0 all families at s=1, ~base rate at s=0
- [x] 5 Splits: time-ordered 60/40 by unit (account, or group), in_family + 3 held-out folds
- [x] PREREGISTRATION.md committed before any detector / eval entry point (machine-checked)
- [x] 6a rules + features + template_sim · [x] 6b lr/gb ± sim · [x] 6c LLM reviewer (mock + stub-tested Anthropic)
- [x] 7 Evaluation: run_eval.py (84-cell grid), metrics + bootstrap + cost (known-answer tests), llm_pass CLI
- [ ] 8 Report + `make demo`: written, first run pending
- [ ] 9 BYO adapter + docs/byo-events.md + examples/byo: written, tests green
- [ ] 10 README: not started

## Null-control failures and fixes (stop rule)
- #1 dev 101 s=0: review_ring cells had 0 test positives. Cause: groups timed by EARLIEST member,
  so at s=0 every group fell into train. Fix: time a group by its lowest-id member. Bootstrap now
  returns NaN (not 0) when a class is missing; null test asserts every cell has positives.
- #2 dev 101 s=0: rules ROC Bonferroni CI < 0.5 in 5 cells. Cause: percentile bootstrap cannot
  vary a class where no member fired (0/21 abuse vs 3% benign is likely under the null). Fix:
  continuity correction (2 pseudo-units per class) for ROC-AUC only. Calibrated on dev
  101-103 at s=0 (numbers in PREREGISTRATION.md AMENDMENTS): 0 Bonferroni misses on all seeds.
- Neither fix touched the generator or a detector. Both logged as a dated prereg amendment.

## Decisions
- Knob s scales every mechanism (Binomial(n,s), Poisson(r*s), prob ~ s, lerp of created_at); at s=0 nothing is added or removed.
- 30 groups per coordinated family per 5,000 accounts (raised from 20 before any detector existed). Health floor 12.
- Train fraction 0.6 so coordinated test cells get ~6-14 groups; held-out folds share in_family's test accounts.
- Eval seed 9001 group counts (structure only) were printed once while choosing the split fraction; no metrics.
- Generator change (precomputed CDF, perf) happened before any detector existed.
- Rules use a generic disposable-email substring heuristic and no IP-range knowledge (avoid encoding generator vocab).
- Anthropic provider: default model claude-opus-5-5, server-side refusal fallbacks on (`--no-fallbacks`); refusal -> 0.5.
- Prompt avoids family names so the leakage guard can forbid them outright.
- `report.py` exempt from line length only (HTML templates).
- BYO: feature model is trained on synthetic dev-seed data; output is a ranking, documented as such.

## Questions for owner
- Q1 Pre-registered template_sim is weak (dev 101: template mean 0.068 vs benign 0.050). Not a bug:
  exact Jaccard to crew-mates 0.02-0.09 vs <=0.03 benign pairs, MinHash agrees; 64 hashes + max over
  5,000 neighbours drowns it. Kept as declared. Run a v2 (more hashes / containment) as labelled post-hoc?
- Q2 PR-AUC intervals are judged against the base rate, but a random ranking's expected AP is slightly
  above the base rate with few positives. Fine under Bonferroni here; worth a note if cells get smaller.

## Suggestions (not done)
- Paired bootstrap for detector differences; an s-sweep curve; real-tokenizer token counts.

## Last check
- Full tree incl. stories 7-9: `make test` 93 passed in 387.94s (0:06:27) (exit 0); lint, typecheck exit 0.
