# PROGRESS

## Stories
- [x] 1 Skeleton (pyproject, Makefile, CI, MIT) · [x] 2 Schema (dataclasses + JSON Schema)
- [x] 3 Generator: benign + 6 families, knob s, truth/ kept separate
- [x] 4 Oracle (eval/oracle.py). Dev 101: oracle AP = 1.0 all families at s=1, ~base rate at s=0
- [x] 5 Splits: time-ordered 60/40 by unit (account, or group), in_family + 3 held-out folds
- [x] PREREGISTRATION.md committed before any detector / eval entry point (machine-checked)
- [x] 6a rules + features + template_sim · [x] 6b lr/gb ± sim · [x] 6c LLM reviewer (mock + stub-tested Anthropic)
- [x] 7 Evaluation: run_eval.py (84-cell grid), metrics + bootstrap + cost (known-answer tests), llm_pass CLI
- [x] 8 Report + `make demo` (eval seeds; 3m33s here); all required sections; disclosure of eval re-run
- [x] 9 BYO adapter + docs/byo-events.md + examples/byo
- [x] 10 README (numbers copied from report at commit 4c8dce0)
- [x] 11 Owner follow-ups (post-hoc, dated amendments): few-cluster CI widening (Q3), tie-aware null
  PR-AUC + degenerate flag (Q2), template_sim_v2 + lr/gb+sim2 (Q1). Eval seeds run a 3rd time, disclosed.
- [x] 12 Post-hoc s=0.25 level + signal-strength sweep section (amendment declared before the run).
- [x] 13 Owner-approved generator addition: template_mode="cross_only" (default output byte-identical,
  digest checked). Result: gb 0.993 without similarity; expectations (a),(b) MISS, (c) hit.
- Headline (seed 9001, commit af90a8e): null 0/108; winner calls 8/12 (pre-reg CIs), 7/12 (widened);
  H4 MISS, H5 MISS; gb+sim2 vs gb on template: +0.016 at s=0.5, -0.002 at s=0.25 (expectation MISS).

## Null-control failures and fixes (stop rule)
- #1 dev 101 s=0: review_ring cells had 0 test positives. Cause: groups timed by EARLIEST member,
  so at s=0 every group fell into train. Fix: time a group by its lowest-id member. Bootstrap now
  returns NaN (not 0) when a class is missing; null test asserts every cell has positives.
- #2 dev 101 s=0: rules ROC Bonferroni CI < 0.5 in 5 cells. Cause: percentile bootstrap cannot
  vary a class where no member fired (0/21 abuse vs 3% benign is likely under the null). Fix:
  continuity correction (2 pseudo-units per class) for ROC-AUC only. Calibrated on dev
  101-103 at s=0 (numbers in PREREGISTRATION.md AMENDMENTS): 0 Bonferroni misses on all seeds.
- #3 EVAL 9001 s=0 (first make demo): 4/84 cells failed. Cause (confirmed on dev 101/102): whole
  groups at s=0 hold independently-aged members, so test group members were 10-14 days older than
  benign test accounts. Fix: exclude group members on the far side of the cutoff. Eval re-run once.
- After #3, dev 101: 0, dev 102: 0, dev 103: 1 borderline miss (12-group cell; hash mock also 0.41).
- No fix touched the generator or a detector. All logged as dated prereg amendments.
- Generator changes after detectors existed: only #13 (owner-approved, opt-in, default byte-identical).

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
- Q1-Q3 answered by owner (v2 feature, better null reference, few-cluster correction); done.
- Q4 answered (add s=0.25): done; v2 still adds nothing there (-0.002).
- Q5 answered (cross-only mode): done. It did not isolate the cross-account channel: script steps are
  unusual per account (dev 101: reviews without purchase ROC 0.915, listing creation 0.889).
- Q6 Fair test of similarity needs scripts drawn from ordinary actions (another generator change). Do it?

## Suggestions (not done)
- Paired bootstrap for detector differences; an s-sweep curve; real-tokenizer token counts.

## Last check
- `make test`: 104 passed in 662.57s (0:11:02) (exit 0); `make lint`, `make typecheck` exit 0; `make demo` 7m09s, null 0/108.
