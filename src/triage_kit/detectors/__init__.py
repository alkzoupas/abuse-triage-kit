"""Detectors. All share one interface:

    score(account_events: AccountEvents) -> (score in [0, 1], rationale: str)

Detectors see only the event schema. They must never import `triage_kit.eval`
(labels, oracle, splits) or `triage_kit.generator` (enforced by tests).
"""
