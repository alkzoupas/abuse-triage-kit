# PROGRESS

## Stories
- [x] 1 Repo skeleton (pyproject, Makefile, CI, MIT)
- [x] 2 Event schema: dataclasses + JSON Schema (src/triage_kit/schema.py, schemas/*.json)
- [x] 3 Generator: benign + 6 families, knob s, truth/ kept separate (src/triage_kit/generator/)

## Decisions
- src/ layout, package `triage_kit`; Makefile uses .venv if present so local and CI share targets.
- Schema has fixed optional fields (target_id, amount, success, rating, text) instead of a free-form dict: easier to validate and document.
- Knob s scales every mechanism (Binomial(n,s), Poisson(r*s), prob ~ s, lerp of created_at); at s=0 nothing is added or removed, so abuse = benign draw.
- review_ring and template_operator are the coordinated (grouped) families; >=12 groups each (min_groups).
- Listing popularity sampled from a precomputed CDF (perf: generation 11s -> 5s for 5,000 accounts). No detector exists yet, so this is not a post-detector generator change.

## Questions for owner
- (none yet)

## Last check
- `make test`: 17 passed in 42.70s (exit 0); lint and typecheck exit 0.
