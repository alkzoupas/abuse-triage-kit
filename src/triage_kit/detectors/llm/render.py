"""Render one account's timeline into a reviewer prompt. Uses only schema fields."""

from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from triage_kit.schema import AccountEvents, Event

MAX_LINES = 150  # events shown; longer timelines show the first and last halves
PROMPT_VERSION = "v1"

INSTRUCTIONS = """You review marketplace accounts for abuse of any kind: hijacked accounts, \
fake reviews, promotion farming, bulk content harvesting, testing stolen payment cards, \
scripted or automated operation, or anything else. Read the account timeline below and judge \
whether this account should be sent to a human investigator.

Answer with ONE JSON object and nothing else:
{"verdict": "flag" or "pass", "score": <number from 0 to 1, likelihood of abuse>, \
"rationale": "<one or two sentences citing specific events>"}
"""

# v2 names no abuse type (PREREGISTRATION.md, AMENDMENTS, 2026-11-09). v1 names the six planted
# ones in paraphrase, so a reviewer using it has been told what to look for.
_ANSWER = INSTRUCTIONS[INSTRUCTIONS.index("Answer with ONE JSON object") :]
INSTRUCTIONS_V2 = (
    "You review marketplace accounts. Read the account timeline below and judge whether this "
    "account is abusing the marketplace and should be sent to a human investigator.\n\n" + _ANSWER
)
PROMPTS = {"v1": INSTRUCTIONS, "v2": INSTRUCTIONS_V2}


def _fmt_ts(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M:%S")


def _line(e: Event, prev_ts: int | None) -> str:
    gap = "" if prev_ts is None else f" (+{e.ts - prev_ts}s)"
    parts = [f"{_fmt_ts(e.ts)}{gap} {e.type}", f"ip={e.ip}", f"device={e.device_id}"]
    if e.target_id is not None:
        parts.append(f"target={e.target_id}")
    if e.amount is not None:
        parts.append(f"amount={e.amount:.2f}")
    if e.success is not None:
        parts.append("ok" if e.success else "FAILED")
    if e.rating is not None:
        parts.append(f"rating={e.rating}")
    if e.text:
        parts.append(f'text="{e.text[:120]}"')
    return " ".join(parts)


def render_prompt(ae: AccountEvents, version: str = PROMPT_VERSION) -> str:
    if version not in PROMPTS:
        raise ValueError(f"unknown prompt version {version!r}")
    a = ae.account
    ev = list(ae.events)
    counts = Counter(e.type for e in ev)
    head = [
        PROMPTS[version],
        "ACCOUNT",
        f"id={a.account_id} created={_fmt_ts(a.created_at)} country={a.country} "
        f"email_domain={a.email_domain}",
        f"{len(ev)} events; by type: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        f"{len({e.device_id for e in ev})} devices, {len({e.ip for e in ev})} IPs",
        "",
        "TIMELINE (UTC)",
    ]
    if len(ev) <= MAX_LINES:
        shown: list[Event | None] = list(ev)
    else:
        half = MAX_LINES // 2
        shown = [*ev[:half], None, *ev[-half:]]
    lines = []
    prev: int | None = None
    for e in shown:
        if e is None:
            lines.append(f"... {len(ev) - MAX_LINES} events omitted ...")
            prev = None
            continue
        lines.append(_line(e, prev))
        prev = e.ts
    return "\n".join(head + lines) + "\n"
