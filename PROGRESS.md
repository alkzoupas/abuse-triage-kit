# PROGRESS

## Stories
- [x] 1 Repo skeleton (pyproject, Makefile, CI, MIT)
- [x] 2 Event schema: dataclasses + JSON Schema (src/triage_kit/schema.py, schemas/*.json)

## Decisions
- src/ layout, package `triage_kit`; Makefile uses .venv if present so local and CI share targets.
- Schema has fixed optional fields (target_id, amount, success, rating, text) instead of a free-form dict: easier to validate and document.

## Questions for owner
- (none yet)

## Last check
- `make test`: 7 passed (exit 0); lint and typecheck exit 0.
