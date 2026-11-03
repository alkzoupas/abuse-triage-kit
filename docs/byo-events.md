# Bring your own events

You can run the rules and the feature model on your own marketplace events, as long as
they're in the kit's schema. This page covers the format, the command, and what the
output does and doesn't mean.

## The schema

Two files: one row per **account**, one row per **event**. Times are integer Unix
seconds (UTC). The machine-readable definitions are
`src/triage_kit/schemas/account.schema.json` and `event.schema.json`; the dataclasses are
in `src/triage_kit/schema.py`.

**Accounts**

| column | type | required | notes |
|---|---|---|---|
| `account_id` | string | yes | unique |
| `created_at` | integer | yes | signup time, Unix seconds |
| `country` | string | yes | two capital letters, e.g. `DE` |
| `email_domain` | string | yes | domain only, never the full address |

**Events**

| column | type | required | notes |
|---|---|---|---|
| `account_id` | string | yes | must exist in the accounts file |
| `ts` | integer | yes | Unix seconds |
| `type` | string | yes | one of `login`, `login_fail`, `view`, `search`, `payment_attempt`, `purchase`, `review`, `listing_create`, `promo_redeem`, `profile_update`, `payout_change` |
| `ip` | string | yes | any stable identifier works (hash it if you prefer) |
| `device_id` | string | yes | any stable identifier |
| `target_id` | string | no | listing id, promo code, etc. |
| `amount` | number | no | payment / purchase amount, listing price |
| `success` | boolean | no | `true`/`false` (CSV also accepts `1`/`0`, `yes`/`no`) |
| `rating` | integer | no | 1–5, for reviews |
| `text` | string | no | review text, listing text, search query |

Map your own event names onto these types. If an action has no counterpart, leave it out.

**Formats.** Use `.csv` with a header row (empty cell = missing), or `.jsonl` with one JSON
object per line (missing keys = missing). Accounts and events can use different formats.

A small synthetic example lives in `examples/byo/` (`accounts.csv`, `events.jsonl`).

## Run it

```bash
python -m triage_kit.byo --accounts examples/byo/accounts.csv --events examples/byo/events.jsonl --out byo_scores.csv
```

The adapter validates every row first and stops at the first problem with a message
naming the file, line and field, for example:

```
input error: events.csv line 2: event field type: 'teleport' is not one of ['login', ...]
input error: events.csv: missing column(s) device_id
input error: events.jsonl line 7: unknown account_id 'zz' (not in accounts file)
```

It then writes `byo_scores.csv`, one row per account, ranked by `model_score`:

| column | meaning |
|---|---|
| `rules_score`, `rules_rationale` | all rules in `rules.yaml`; the rationale lists the rules that fired |
| `model_score`, `model_rationale` | gradient boosting + template-similarity feature; the rationale lists the most unusual features |

## How to read the output

- **These are rankings, not verdicts.** Without labels for your accounts, nothing here
  can tell you precision or recall. Use the ranking to pick a review queue, then label
  the top-k (and a random sample, so you can estimate what you miss).
- **The model was trained on the kit's synthetic data**, not yours. Its scores show what
  the approach looks like on your schema; they are not calibrated probabilities. To use
  it for real, retrain on your own labelled history (`FeatureModel.fit_table`).
- **The rules are starting priors.** Edit `rules.yaml` (plain YAML, readable without
  code) to match your policy and thresholds.
- **Template similarity is computed within the population you load.** Load a meaningful
  population (weeks of accounts, not a handful), or the feature has nothing to compare against.
- The LLM reviewer isn't part of this command. To try a real model on your accounts,
  see the README section on plugging in an LLM.
