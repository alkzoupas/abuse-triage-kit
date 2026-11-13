# Scores behind the LLM results (eval seed 9001, s = 0.5)

Everything else in the kit is rebuilt from seeds. The LLM reviewer's scores are not, because
they came from Claude Code subagents, so the files that hold them are kept here.

| file | what it holds |
|---|---|
| `llm_bundle_subagent_claude-opus-5-5.json` | prompt v1 (names the abuse types in paraphrase): one score per test account, 1,922 accounts, and the metrics computed from them |
| `llm_bundle_subagent_claude-opus-5-5_v2.json` | prompt v2 (names no abuse type): the same |
| `second_stage.json` | added recall of each second stage after `gb+sim` flags its top 50, per abuse type, in-family and held out |
| `batch_check.json` | the batch-independence check: old and new scores for the 188 rescored accounts, and the recall recomputed from them |

The files hold scores and metrics only: no prompts, no reviewer replies and no labels.
Account ids are synthetic.

## Recomputing the second-stage numbers

```bash
make demo
cp evidence/seed9001_s0.50_n5000/llm_bundle_subagent_*.json results/seed9001_s0.50_n5000/
python -m triage_kit.eval.second_stage --seed 9001 --model claude-opus-5-5
```

`make demo` regenerates the data and the first-stage scores from the seed. The last command
rewrites `results/seed9001_s0.50_n5000/second_stage.json`, which should match the copy here.
