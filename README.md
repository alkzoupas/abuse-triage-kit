# abuse-triage-kit

An offline starter kit for small marketplaces that want to compare three ways of finding
abusive accounts, on synthetic data where the right answers are known:

1. **Hand-written rules**: a short YAML file a non-engineer can read and edit.
2. **A feature model**: logistic regression and gradient boosting on engineered features,
   with and without a template-similarity feature.
3. **An LLM reviewer**: renders an account's timeline into a prompt and parses a verdict.
   Offline it runs a **mock** whose score is a hash of the prompt. Plug in a real model
   yourself (see below).

The kit's point is the *comparison method*, and honesty about it: hypotheses
pre-registered before any detector existed ([PREREGISTRATION.md](PREREGISTRATION.md)), a
null control, held-out families, group-aware splits, bootstrap confidence intervals, and a
report that lists its misses. A detector losing is a fine result if the report shows it truthfully.

Everything is synthetic. Nothing calls a network service unless you run the optional
real-LLM pass with your own key.

## Quickstart

Requires Python 3.12.

```bash
python -m venv .venv && .venv/bin/pip install -e ".[dev]"
```

```bash
make test typecheck lint
```

```bash
make demo
```

`make demo` generates the data, runs the pre-registered evaluation on the eval seeds and writes **`report/index.html`**: one static page, no server, no
JavaScript. Generated data (`data/`), results (`results/`) and the report are gitignored and
rebuilt from seeds; the same seed gives byte-identical data and metrics.

## What's in the box

| piece | where |
|---|---|
| Event schema (accounts + timestamped events), dataclasses + JSON Schema | `src/triage_kit/schema.py`, `src/triage_kit/schemas/` |
| Generator: benign population + six abuse families, signal knob `s` | `src/triage_kit/generator/` |
| Families: account takeover, fake-review ring, promo abuse, scraping, card testing, template operator | `generator/families.py` (each documents its mechanism and evidence channel) |
| Oracle that knows the planted mechanism (proves the signal exists; not a detector) | `src/triage_kit/eval/oracle.py` |
| Splits: in-family and held-out family, by time, by account and by group | `src/triage_kit/eval/splits.py` |
| Detectors | `src/triage_kit/detectors/` (`rules.yaml`, `feature_model.py`, `llm/`) |
| Evaluation entry point | `python -m triage_kit.eval.run_eval --seed 101 --s 0.5` |
| Frozen v3 confirmation on fresh seeds (after `make demo`) | `make confirm` |
| Second v3 confirmation, 20,000 accounts per seed (~40 min) | `make confirm-large` |
| LLM pass without an API key: Claude Code subagents as the reviewer (exploratory) | `python -m triage_kit.eval.subagent_pass prepare` / `collect` |
| Second-stage recall: what the LLM adds after `gb+sim` flags its top 50 (exploratory) | `python -m triage_kit.eval.second_stage --seed 9001` |
| Report | `src/triage_kit/report.py` |
| Bring your own events | [docs/byo-events.md](docs/byo-events.md), `python -m triage_kit.byo` |

Labels and generator metadata go to a separate `truth/` folder that detectors never read.
Tests enforce this: an import scan of the detectors, a check of feature columns and rendered
prompts for metadata, and a check that mock verdicts don't change when labels are shuffled.

## Results in brief

A boosted model (`gb+sim`) flags its top 50 accounts. A second stage then flags 50 more from
the accounts it let through. The table shows the share of each abuse type caught when the
model **never trained on that type** (eval seed 9001, about 1,750 accounts per test, 19 to 50
abusive).

| abuse type | model's top 50 | LLM's next 50 add | rules' next 50 add |
|---|---|---|---|
| account takeover | 12% | 88% | 79% |
| card testing | 23% | 72% | 72% |
| fake-review ring | 53% | 26% | 3% |
| scraping | 73% | 23% | 23% |
| promo abuse | 78% | 12% | 6% |
| template operator | 76% | 12% | 0% |

- **The LLM helps most where the model is blind.** The reviewer is Claude Opus with a prompt
  that names no abuse type. On the two types the model misses, it recovers most of the rest.
- **Hand-written rules do about as well there.** The rules were written knowing what was
  planted, so they stand for the rules a team would write once it understood the attack. The
  LLM gets there without that knowledge.
- **Where the model trained on the type, the LLM adds little.** Nothing on account takeover,
  card testing and scraping, which the model already catches at 95 to 100%; 12 to 16% on the
  other three.
- **The result survives a change of batch.** The reviewer scores about 47 accounts at a time.
  Rescoring the 188 deciding accounts among different batch-mates moved individual scores but
  barely moved the catches (88% became 86%, 72% stayed 72%).

Limits: synthetic data, one eval seed, small counts per type. Claude wrote the generator and
did the reviewing, so the planted abuse may be easy for it. Each test is one abuse type
against benign accounts, not a mixed queue, so precision will not transfer. There is no cost
figure. The second-stage analysis is exploratory: declared before scoring, but added after the
original pre-registration.

All findings, with intervals and every miss: [docs/FINDINGS.md](docs/FINDINGS.md). The scores
behind the table: [evidence/](evidence/).

## Who did what

Alkis Zoupas set the questions, the pass bars and every decision; he is the "owner" in the
notes. Claude Code wrote the code and the generator and ran the LLM reviews. A model from a
second vendor reviewed the second-stage write-up ([reviews/round-1](reviews/round-1/)).

Status: paused in November 2026. Open next steps are an API pass with real token counts, for
a cost figure, and a second eval seed.

## How to read the report

- **Setup** describes the data, the six mechanisms and the detectors. The rules encode human
  priors written by someone who knew what was planted, an advantage no real rule writer has.
- **Null control** must pass before anything else means anything.
- **Signal check** shows the oracle finds every family at s = 1, so a detector that misses
  is missing a signal that is there.
- **In-family vs held-out** has one chart per family. Filled dots are in-family, hollow are
  held-out, whiskers are 95% CIs, and the dashed line is the base rate. A whisker crossing
  the null PR-AUC (shown per detector in the table) means "no detectable signal".
  Overlapping whiskers between two detectors mean "no detectable difference". Families with fewer than 10 test groups are flagged as
  having wide CIs.
- **Template operator** compares the models with the pre-registered similarity feature
  (`+sim`) and the post-hoc v2 (`+sim2`, exploratory) against neither.
- **Cost** is tokens × placeholder prices. Replace `config/prices.yaml` with real prices first.
- **Expected vs actual** scores every pre-registered expectation as hit or MISS.
- **What this can't tell you** is written for a buyer deciding whether to run this on their own events.

## Plugging in a real LLM

The reviewer talks to any object with this interface
(`src/triage_kit/detectors/llm/types.py`):

```python
class LLMProvider(Protocol):
    name: str
    def complete(self, prompt: str) -> LLMResponse: ...   # LLMResponse(text, tokens_in, tokens_out)
```

`LLMReviewer(provider)` renders each account with `render_prompt`, calls `complete`, parses
the reply (`{"verdict": ..., "score": 0-1, "rationale": ...}`), and logs tokens per call.
To use another vendor, write a provider of a few lines and pass it in.

An Anthropic provider is included (`anthropic_provider.py`; `pip install -e ".[anthropic]"`).
The pass is opt-in at every step: it runs only when `ANTHROPIC_API_KEY` is set in your
environment, prints a cost estimate from `config/prices.yaml` first, and does nothing
without `--yes`:

```bash
python -m triage_kit.eval.llm_pass --seed 9001 --s 0.5 --model claude-opus-5-5 --limit 200
```

Add `--yes` to actually call the API. Server-side refusal fallbacks are on by default
(`--no-fallbacks` turns them off); a refused account scores a neutral 0.5. The pass writes a
scores-only bundle (`results/<run>/llm_bundle_<model>.json`, no prompts or rationales), and
the next `make demo` adds a "Real LLM pass" section from it. Tests exercise this provider
only against a stub client, and the test suite fails if any test tries to build a real one.

## Bring your own events

Export your accounts and events into the schema (CSV or JSONL), then:

```bash
python -m triage_kit.byo --accounts examples/byo/accounts.csv --events examples/byo/events.jsonl --out byo_scores.csv
```

Bad input is rejected with the file, line and field. You get a ranking from the rules and
the feature model, not an accuracy figure: without labels, nothing can measure precision on
your data. See [docs/byo-events.md](docs/byo-events.md).

## Working notes

[PROGRESS.md](PROGRESS.md) is the working log: stories, decisions, open questions and the last
check results, kept as the audit trail. [PREREGISTRATION.md](PREREGISTRATION.md) is append-only; changes are dated amendments.

MIT licensed.
