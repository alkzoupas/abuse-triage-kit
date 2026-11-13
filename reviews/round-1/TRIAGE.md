# Triage: round 1, design review of the report and second-stage findings

- **Date:** 2026-11-11. **Candidate:** `main` at `2162af1`; report built at `173b457` (same tree).
- **Reviewer:** Codex, `gpt-6.1-sol`, effort medium (as reported in the Orca launch receipt). Run `run_89498755ecf3`, dispatch `ctx_6311b72a0a2b`.
- **Review file:** `openai-gpt-6.1-sol-design.md` (the record; not edited). First pass saved as `first-pass.md`, sha256 `9eda35584a1bc30b93ca961e88095988a29c3548b024fd80bc4b59d9f51d7b19`; the final file keeps it intact as its prefix.
- **Verdict:** FIX-THEN-SHIP, unchanged after disclosure.
- **Stakes:** routine (the repo has no AGENTS.md requiring a review). **Mode:** design.

## Where the reviewer worked and what it saw early

In place in the main checkout, not a clean worktree: the report and results are git-ignored, so a new worktree would not have contained them. Blinding therefore relied on the reviewer complying with the don't-open list. Its "Context seen" section reports the shared AGENTS instructions, the injected graft repo map and Trail banner, the adversarial-review skill and the Orca guide, and `git status --short` path names. It reports no memory recall and no opening of PROGRESS.md, `.claude/`, other reviews or commit messages. After disclosure it chose not to open PROGRESS.md or commit messages either, so its reconciliation rests on the coordinator's summary only.

## Guard check

Snapshot of 333 files before and after (`guard-before.txt`, `guard-after.txt`). Source, docs, report and results are identical. Differences: `graft/.graph/wiring.json`, `graft/INDEX.md` and a new `graft/reviews/round-1/scratch/audit_openai.md`. These are graft's own index files, and the new one is a card for the reviewer's scratch script, so this looks like graft re-indexing after the new files appeared, not a reviewer write. I did not confirm that from the reviewer's transcript. `graft/` is git-ignored. Two other terminals were open in the workspace during the review ("Terminal 1", "Terminal 3"); nothing suggests they wrote anything.

Worker state: succeeded; its terminal was listed as reclaimable with an automatic release time, with no `next` action.

## Findings

| # | Finding (severity) | After disclosure | New or raised before | Valid? | Proposed action |
|---|---|---|---|---|---|
| 1 | The report says each second stage "sees only what stage 1 allowed", but the LLM scored all 1,922 accounts in mixed batches and the stage-1 filter is applied afterwards in the ranking (High) | Kept | New | **Valid.** `subagent_pass.py` `prepare` batches every test account; `second_stage.py` `run` reuses one score per account in all 12 cells; the wording is at `report.py` `_second_stage`, the `second_stage.py` docstring and PREREGISTRATION.md line 550. The README wording ("flags 50 more accounts among those stage 1 let through") is closer to what was done but does not say the scores came from a full-population pass. | Reword the report, docstring and README to "a retrospective calculation on fixed scores from a full-test batched review", and say that independence from batch composition was not tested. Add a dated clarification to PREREGISTRATION.md without rewriting the amendment. No rerun needed. Optional follow-up below. |
| 2 | The 95% intervals resample accounts with the reviewer's scores held fixed; the report does not say so (Medium) | Kept | New | **Valid.** `cell_second_stage` resamples truth units and reuses the same score arrays in every replicate. This is the bootstrap the amendment declared, so the scored criterion stands; the gap is in what readers are told. | Add one sentence to the report and README: the intervals are conditional on this split, these fitted models, this batch assignment and these replies, and do not cover a rerun or a rebatching. |
| 3 | Carrying fractional weights from stage 1 into stage 2 is not the expected value under sequential random tie-breaking (Medium) | Kept | New | **Valid as a method flaw; no effect on the reported point estimates.** I reproduced the counterexample: `two_stage` returns 0.333333 where the exact expectation is 0.25. I then refit stage 1 and checked all 12 cells: none has a tie at the stage-1 budget boundary (`scratch/coordinator_verify_ties.py`, exit 0), so stage 1 is deterministic there and the fractional rule is exact for every point estimate. Bootstrap replicates were not checked, so a small effect on the intervals is not ruled out. | Reword the tie rule as a defined fractional-weight surrogate that is exact when stage 1 has no boundary tie, state that none occurred in this run, and add the counterexample as a test documenting the limit. Changing the implementation is optional. |
| 4 | Stage-2 precision was promised in the amendment and is stored in the JSON, but the report and README show added recall only (Medium) | Kept | New | **Valid.** `_second_stage` in `report.py` renders no precision. Held out, `llm:v2` precision is 0.738 ato, 0.560 card_testing, 0.120 scraping, 0.120 promo_abuse, 0.100 review_ring, 0.080 template_operator. | Add precision (or true flags out of 50) beside added recall in the report table and the README bullet. Values exist; no rerun. Any public post should carry it: on four families the gain is 4 to 6 real accounts per 50 flags. |

Smaller points the reviewer raised without making them findings:
- The README keeps the old "because it needs no training data" sentence and corrects it afterwards. Replacing the sentence would read more clearly. Valid; low priority.
- The model name in the bundles is a string passed on the command line. Transcripts record `claude-opus-5-5` for the 62 reviewers checked (40 on 2026-11-10 and 22 after); the earlier batches (about 00–20 of each prompt) were not checked. Limitation, not a finding.

## What the review confirmed

The reviewer's read-only audit (`scratch/audit_openai.py`, exit 0) found both manifests complete and unique, every batch matching its manifest, all 3,844 bundled scores reproduced from the 82 reply files, the one missing v1 account disclosed, and expectation counts 4/3/5/6 with template_operator weakest, matching the README and report. I did not rerun its script.

## Blinding

All four findings were raised in the first pass and all four were kept after disclosure; none was withdrawn or changed. Disclosure added no finding.

## Optional follow-up for finding 1 (not proposed as required)

A direct test of batch independence without 12 full passes: re-score a few existing batches' accounts in new batch compositions and compare the per-account scores. At about 0.4M input-side tokens per reviewer, four reshuffled batches is about 1.6M tokens. It needs its own amendment before any scoring.

## Status

Review-only request: findings and proposed fixes above; nothing in the report, README, pre-registration, code or results was changed. Awaiting the owner's choice.
