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
  exist). Status: done 2026-11-11.
  Result (seed 9001 s=0.5, claude-opus-5-5 subagents, v2 1,922/1,922 parsed, v1 1 unparsed):
  held out, llm:v2 added recall ato 0.879, card 0.718, ring 0.263, scraping 0.231, promo 0.120,
  template 0.118. Expectations: (a) HIT 4/6 (criterion), (b) MISS 3/6, (c) HIT 5/6 (largest v1
  gain 0.023, two ties), (d) HIT 6/6, (e) MISS (template_operator smallest, not review_ring).
  Procedure change 2026-11-11 for the last 15 batches (paged reads in one message, waves of 3):
  about 0.4M input-side tokens per reviewer against about 0.8M before. Results are not committed.
  Last check: `make check` exit 0, 124 passed.
- [x] 22 Fixes from the round-1 design review (Codex gpt-6.1-sol, `reviews/round-1/`, verdict
  FIX-THEN-SHIP; owner asked for all four; branch claude/story22-review-fixes).
  Outcome: the report, README, code docstrings and pre-registration describe the second stage as
  what it is: a retrospective calculation on fixed full-test scores (1), with intervals
  conditional on those scores (2), a fractional tie rule that is a surrogate when stage 1 ties at
  its budget (3), and stage-2 precision shown beside added recall (4). No number changed.
  Acceptance (tests/test_second_stage.py): test_report_describes_a_retrospective_calculation_on_full_test_scores,
  test_report_says_what_the_intervals_are_conditional_on,
  test_fractional_stage_one_tie_is_a_surrogate_not_the_sequential_expectation (documents the
  limit; passed before the change), test_cell_records_whether_stage_one_ties_at_the_budget,
  test_report_states_the_tie_rule_as_a_surrogate_and_whether_stage_one_tied,
  test_report_shows_stage_two_precision_beside_added_recall. Five of six failed first.
  `second_stage.json` regenerated for the new `stage1_boundary_tie` field: false in all 12
  cells, every other value identical to the previous file. Status: done in `db772cb`, merged
  to `main` as `1a917f4` (PR #5).
  Not done: the tie implementation is unchanged (surrogate kept, as the review allowed); no
  survivor-only pass; the old "needs no training data" README sentence still stands with its
  dated correction (the reviewer's low-priority remark, not one of the four fixes).
- [x] 23 Batch-independence check (owner said go 2026-11-12 after reading the earlier posts;
  amendment 2026-11-12; branch claude/story23-batch-independence, off story 24).
  Outcome: a measured answer to how much `llm:v2` scores, and the held-out added recall built on
  them, move when the deciding accounts are scored among different batch-mates, shown beside the
  story 21 result with a rerun-noise floor.
  Acceptance (tests/test_batch_check.py): test_pool_holds_every_surviving_family_account_and_fills_with_top_benign,
  test_pool_is_deterministic_and_breaks_score_ties_by_seed,
  test_pool_recall_with_original_scores_reproduces_full_ranking_recall,
  test_pool_recall_ranks_accounts_outside_the_pool_last,
  test_score_changes_known_answer, test_verdict_three_outcomes,
  test_collect_refuses_when_a_reply_is_missing; then, with results,
  test_report_shows_batch_check_beside_second_stage.
  Status: run and collected 2026-11-12 on branch `claude/story23-batch-independence` (7 reviewers,
  all `claude-opus-5-5`, no reply missing). Verdict by the fixed criterion: **supported**. Added
  recall, original to rebatched: ato 0.879 to 0.860, review_ring 0.263 to 0.316, promo_abuse 0.120
  to 0.139, scraping 0.231 and card_testing 0.718 unchanged, template_operator 0.118 to 0.147
  (largest change 0.053, limit 0.10). Scores did move: benign mean change -0.090, 34% moved by
  more than 0.1; family -0.037, 14%; unchanged rerun (76 accounts) 1% moved, mean absolute 0.014.
  Result file `results/seed9001_s0.50_n5000/batch_check.json` (gitignored, like all results);
  README caveat updated. `report/index.html` rebuilt at `3d88167` with `build_report` (gitignored).
- [x] 24 Pre-publication housekeeping (owner said go, 2026-11-12; branch claude/story24-prepublish).
  Outcome: the tree can be made public without dead references, future dates or private paths.
  Acceptance (document checks, no new tests): (1) `reviews/round-1/` is tracked, so the pointer in
  PREREGISTRATION.md resolves; (2) no date later than 2026-11-12 in PROGRESS.md or
  `reviews/round-1/TRIAGE.md`; (3) `gitleaks git` over the history and `gitleaks dir` over the tree
  report no leaks; (4) no absolute home path in a tracked file; (5) stories 23 and Q13 read as
  open; (6) `make check` exit 0. Changes: `reviews/`, `.gitignore` (ignores `/graft/`) and
  `.ignore` committed; the dates that were two days in the future corrected to 2026-11-11 (commit dates of `173b457`
  and the review files); the reviewer's scratch script `audit_openai.py` now finds the repo root
  from its own location (it had an absolute checkout path; rerun after the edit: exit 0, same summary line as the review records).
  Not done: local `main` was not synced (the git guard leaves pulls on `main` to the owner);
  repo visibility is unchanged (private).
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
- Q13 OPEN, NOT DONE. The subagent pass is batched and inside Claude Code. An API pass (one
  account per call, real token counts) would test H6 properly and give a cost; it needs an
  ANTHROPIC_API_KEY in the environment's settings. Until then there is no cost figure.

## Resume here (2026-11-12)
- **Branch:** `claude/story23-batch-independence` (head `75c7006` plus this note), which sits on
  top of `claude/story24-prepublish` (`5f38de8`), so merging it brings stories 23 and 24 together.
  Neither is merged or pushed. Merge and push are the owner's; so is `git pull` on local `main`, which was 2 commits behind
  `origin/main` on 2026-11-12.
- **Environment:** `.venv` in the main checkout (Python 3.12, `uv pip install -e ".[dev]"`); plain
  `make check` works. The old story worktree (`.claude/worktrees/repo-review-linkedin-04cea0`) is
  not needed for its venv; its `results/` is the only second copy of the reviewer replies.
  Deleting it is the owner's call.
- **Very next step (owner):** merge the story 23 branch, make the repo public, post the write-up.
  The post draft (final wording agreed 2026-11-12, kept outside the repo) states: synthetic data,
  one seed, Claude-written generator; batch-mates shifted scores but not the catches (story 23:
  0.879 to 0.860 account takeover, 0.718 unchanged card testing); no cost figure (Q13).
- **Not done, in case work resumes:** Q13, a second eval seed. Story 23 left two things untested:
  accounts outside the pool of 188 were not rescored, and the check is one run on one seed.

### Story 23 (run 2026-11-12, supported; result in the story list above; sketch kept for the record): does a reviewer's score depend on its batch?
Why: story 21's second-stage numbers reuse one score per account from a full-test pass (round-1
review, finding 1). They describe a survivor-only reviewer only if a score does not depend on
the other accounts in the batch. The reviewer called this the riskiest assumption.
- **Outcome:** a measured answer, on existing accounts, to how much `llm:v2` scores move when the
  same accounts are scored in different batch compositions, reported next to the story 21 result.
- **Sketch (to settle with the owner before the amendment):** take the accounts of about 4
  existing v2 batches (about 188 accounts), put them into new batches with a different shuffle
  seed and different batch-mates, score with the same v2 prompt, model and tool limits, and
  compare each account's new score with its old one. About 4 reviewers, about 1.6M input-side
  tokens at about 0.4M each.
- **Open design choices, with my recommendation:**
  1. Which accounts: 4 whole existing batches reshuffled among themselves (cheapest), or a sample
     stratified by family that includes every stage-1 survivor near the stage-2 budget
     (recommended: those are the accounts whose scores decide the reported recall).
  2. New batch-mates: other full-test accounts (measures plain rerun plus rebatch noise), or
     stage-1 survivors only (recommended: that is the composition the claim is about).
  3. A rerun of the same batches unchanged as a noise floor (recommended, about 2 reviewers more):
     without it a score change cannot be split into batch effect and rerun noise.
  4. Criterion, fixed before scoring: for example, recompute the held-out v2 added recall with the
     new scores swapped in and call the assumption supported if no family moves by more than a
     declared amount, plus the share of accounts whose score moves by more than 0.1.
- **Needed before any scoring:** a dated amendment in PREREGISTRATION.md (exploratory, further
  use of seed 9001), acceptance tests named in this story, then the story's status line.
- **Inputs that exist:** `results/subagent_v2/` (`manifest.json`, `batches/`, `replies/`; 41 each),
  `results/seed9001_s0.50_n5000/llm_bundle_subagent_claude-opus-5-5_v2.json`, `second_stage.json`;
  code in `src/triage_kit/eval/subagent_pass.py` (`assign_batches`, `prepare`, `collect`) and
  `src/triage_kit/eval/second_stage.py`.
- **Cost rules (owner's standing instruction):** pilot one reviewer, measure it from its
  transcript, give the estimate, get approval, then waves of 3. See "Running a subagent pass"
  below. One model for the whole pass.

## Where things stand (2026-11-11)
Everything is on `main` (PR #3 merged, `ec474c0`); `main` is the remote's default branch. Reviewer
replies, bundles and `second_stage.json` are in `results/` of the main checkout (git-ignored, so
local only; the only other copy is the old story worktree).

What the LLM work showed (stories 20 and 21; synthetic data, one seed, Claude-written generator):
- A reviewer after the feature model pays off where that model is blind. Held out, `gb+sim` alone
  catches 0.119 of account takeover and 0.231 of card testing in its top 50; a generic reviewer's
  next 50 flags add 0.879 and 0.718.
- It adds little where stage 1 already works (0.12 to 0.26 on the other four families), and for
  review ring and template operator the gain is not distinguishable from random at this size.
  Those are the families whose evidence is across accounts; the reviewer sees one at a time.
- Naming the abuse types in the prompt barely changes anything (largest gain 0.023), so the result
  does not rest on the prompt cue the 2026-11-09 review found.
- A second opinion beats a bigger budget: v2 beats 50 more `gb+sim` flags in 6 of 6 families.
- All hand-written rules switched on match the reviewer on card testing and scraping.
- Still unknown: cost (no token counts in a subagent pass), a second seed, and real data.

Running a subagent pass without burning the usage limit:
- A reviewer costs about 0.4M input-side tokens per batch of about 47 accounts on Opus, most of it
  the batch text billed on each of 3 turns plus about 31k of Claude Code overhead per turn.
- The read guard rejects a whole-file read of a 200 to 270 KB batch. Tell each reviewer the line
  count and ask for all 500-line page reads in one message, then one Write. Without that they take
  6 to 12 turns and about twice the tokens.
- Pilot one batch, measure it from its transcript, then waves of 3. A wave of 20 hit the limit and
  lost 15 reviewers' work. The weekly headroom cannot be read in advance.
- Never mix models inside one pass, and never collect a pass with batches missing (they score 0.5).

Open, in the order suggested (updated 2026-11-11):
1. Story 23 is done on its branch (supported); owner to merge. The write-up can now say the
   batch assumption was checked, with the score-shift caveat, instead of "untested".
2. Decide whether the repo goes public for a write-up; it is private now. The round-1 review's
   fixes are in (story 22). `report/index.html` was rebuilt at `db772cb` with `build_report`.
3. Q13, the API pass: the only way to get a cost figure and test H6's cost part.
4. Optional: the second stage on one more eval seed (v2 only is 41 reviewers, about 16M tokens).
   Only worth it if story 23 supports the method.
5. Owner decisions left: delete the old story worktree. (`reviews/`, `.gitignore`, `.ignore` and
   the date corrections were done in story 24.)

## Suggestions (not done)
- Paired bootstrap for detector differences; an s-sweep curve; real-tokenizer token counts.

## Last check
- 2026-11-12, branch `claude/story24-prepublish` (tree of `origin/main` `8086c52` plus story 24),
  main checkout's `.venv`: `make check` exit 0, lint and typecheck clean,
  `130 passed in 313.66s (0:05:13)`. `gitleaks git` (59 commits) and `gitleaks dir reviews`:
  no leaks found. `make demo` not rerun since `173b457` (the report was rebuilt at `db772cb`
  from existing results with `build_report`). `make confirm` and `make confirm-large` not rerun
  (last: 13m08s and 38m50s, before story 20).
