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
- [x] 14 Owner-approved generator addition: template_mode="ordinary" (scripts of login/search/view
  only; earlier modes byte-identical, digest checked). Dev acceptance check failed as declared
  (oracle 0.144), one revision (uniform targets), narrow miss (oracle 0.899, login ROC 0.665), logged.
  Result: gb 0.010 (a hit), gb+sim2 +0.001 (b MISS), v2 vs v1 (c MISS), oracle 0.570 (d MISS).
- [x] 15 Post-hoc similarity v3 (rare ordered view pairs + queries, shared counts, top-3) + lr/gb+sim3.
  Dev 101 ordinary, feature alone PR: v2 0.032, v3 0.635. Eval 9001 ordinary in-family: gb+sim3 0.607
  vs gb 0.010 (a, b hit); held-out no gain; standard s=0.5 +0.038 (d hit); null 1/132 strict miss
  (lr+sim3 card_testing, CI just below null PR-AUC) -> (c) MISS. Seventh eval-seed run, disclosed.
- [x] 16 Frozen v3 confirmation, fresh seeds 9004-9006 (`make confirm`, commit b524623, 13m08s).
  NOT CONFIRMED: ordinary in-family gain +0.504 / +0.055 / +0.691 (9005 < 0.10). (b), (c), (e) hit.
  (d) MISS: seed 9006 null 2/132 Bonferroni misses (gb, gb+sim on scraping in-family; borderline,
  pre-registered detectors). Nothing fixed or re-run, per the amendment.
- [x] 17 Second frozen v3 confirmation: seeds 9007-9009 at 20,000 accounts (~49 test template groups
  each), `make confirm-large`, commit 6dfa5fd, 38m50s. CONFIRMED: gains +0.498 / +0.376 / +0.371.
  (b), (c) hit. (d) MISS: seed 9008 null 5/132 Bonferroni misses (lr x4 variants on scraping
  heldout_2, gb+sim2 on ato; all borderline, none template). Nothing fixed or re-run.
- [x] 18 Widening check at 20,000 accounts (dev 101-103, s=0, decision rule fixed in advance): ADEQUATE.
  Pooled uncorrected 95% misses vs the right null: PR-AUC 3.6%, ROC 3.5% (5,000: 6.7%, 3.5%);
  0 Bonferroni misses on all three dev seeds at 20k. Confirmation-seed misses: 5 of 7 contain their
  null PR-AUC (base-rate reference artefact), 2 marginal. Also fixed: confirmation sections' count of
  cells excluding the null PR-AUC (was strict-only, showed 0 where it was 2).
- [x] 19 Null control's primary PR-AUC reference is now the null PR-AUC (owner-approved, amendment);
  pre-registered base-rate test kept, shown, and still scores H1 and the confirmations' (d). Re-scored
  existing null runs, primary / pre-registered fails: 9001 1/0 (lr+sim3 card_testing, below chance),
  9004 0/0, 9005 0/0, 9006 0/2, 9007 0/0, 9008 2/5, 9009 0/0.
- [x] 20 Exploratory LLM pass with Claude Code subagents (no API key in the environment; owner-requested,
  declared before scoring). 41 fresh general-purpose subagents (claude-opus-5-5), ~47 accounts each,
  group members split across batches; each may only Read its batch and Write its reply. Seed 9001
  s=0.5, 1,922/1,922 scored, 0 unparsed, ~8M subagent tokens. PR-AUC: ato 0.988, card 0.951,
  scraping 0.962, template 0.868, promo 0.809, ring 0.581. H6 ranking part MISS (ring weakest).
  Also fixed: the report's Real LLM pass table read metrics nothing computed (now computed when a
  bundle is written, for the API pass too).
- [x] 21 Second-stage recall (owner-requested, amendment 2026-11-09; branch claude/repo-review-linkedin-04cea0).
  Outcome: the report shows what an LLM reviewer adds after `gb+sim` flags its top 50, in-family and
  held out, for a generic prompt (v2, primary) and the type-naming prompt (v1), against random,
  rules-all-on and more-stage-1. Acceptance: tests/test_second_stage.py (known-answer two-stage
  recall, tie rule, random expectation, bootstrap determinism), tests/test_llm.py (v2 prompt names
  no abuse type; v1 byte-identical), tests/test_demo.py (report section appears when both bundles
  exist). Status: done 2026-11-14.
  Result (seed 9001 s=0.5, claude-opus-5-5 subagents, v2 1,922/1,922 parsed, v1 1 unparsed):
  held out, llm:v2 added recall ato 0.879, card 0.718, ring 0.263, scraping 0.231, promo 0.120,
  template 0.118. Expectations: (a) HIT 4/6 (criterion), (b) MISS 3/6, (c) HIT 5/6 (largest v1
  gain 0.023, two ties), (d) HIT 6/6, (e) MISS (template_operator smallest, not review_ring).
  Procedure change 2026-11-11 for the last 15 batches (paged reads in one message, waves of 3):
  about 0.4M input-side tokens per reviewer against about 0.8M before. Results are not committed;
  the full report needs `make demo` in this worktree. Last check: `make check` exit 0, 124 passed.
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
- Generator changes after detectors existed: only #13 and #14 (owner-approved, opt-in, earlier modes byte-identical).

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
- Q6 answered (ordinary-action scripts): done. Per-account signal gone, but v1 and v2 find nothing either.
  Diagnosis (dev 101, post-run): sim2 alone ROC 0.447; best crew-mate Jaccard 0.032 < best benign 0.042,
  because one ~10-event replay is diluted by ~100 benign tokens.
- Q7 answered (v3): done. Finds ordinary-script crews in-family (0.607), not held out.
- Q8 answered (confirm on 9004-9006): done, v3 not confirmed (one seed of three below +0.10).
- Q9 answered (diagnose 9005, now a dev seed): v3 tracks script length (Spearman 0.68-0.77); scripts
  >= 9 views ROC 0.93-0.96, <= 6 views 0.64-0.66, same on all three seeds. 9005 drew short scripts
  (median 7 views). Benign noise floor ~1.8 shared rare view pairs; queries contribute nothing.
- Q10 answered (option 2, more template groups): done, v3 confirmed at 20,000 accounts per seed.
- Q11 answered (check widening on dev seeds at 20k): widening adequate, see story 18.
- Q12 answered (switch to null PR-AUC): done, story 19. The primary eval seed now shows FAIL on one
  below-chance cell; reported as is.
- Q13 The subagent pass is batched and inside Claude Code. An API pass (one account per call, real
  token counts) would test H6 properly and give a cost; it needs an ANTHROPIC_API_KEY in the
  environment's settings. Worth doing once a key is available?

## Suggestions (not done)
- Paired bootstrap for detector differences; an s-sweep curve; real-tokenizer token counts.

## Last check
- `make test`: 113 passed in 665.97s (0:11:05) (exit 0); lint, typecheck exit 0; `make demo` 9m55s, null 1/132 primary (null PR-AUC), 0/132 pre-registered (base rate); `make confirm` 13m08s; `make confirm-large` 38m50s.
