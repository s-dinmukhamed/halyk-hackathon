"""Step 3 — RESOLVE: current vs outdated terms.

The core trap of this challenge: an amendment (допсоглашение) can change a
covenant's threshold. The agent must evaluate against the *latest effective*
version, not the original loan agreement.

Strategy: group covenants by (borrower_id, covenant_id); within a group, the
one with the latest effective_date wins; older ones are marked superseded.
Covenants with no date keep the original agreement's value unless a dated
amendment overrides them.
"""
from __future__ import annotations

from datetime import date

from .schemas import Covenant


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
            from datetime import datetime
            return datetime.strptime(s.strip(), fmt).date()
        except (ValueError, AttributeError):
            continue
    return None


def _key(c: Covenant) -> tuple[str, str]:
    return (c.borrower_id, c.covenant_id)


def resolve_covenants(covenants: list[Covenant]) -> list[Covenant]:
    """Return the effective covenant per (borrower, covenant_id); mark the rest superseded."""
    groups: dict[tuple[str, str], list[Covenant]] = {}
    for c in covenants:
        groups.setdefault(_key(c), []).append(c)

    resolved: list[Covenant] = []
    for group in groups.values():
        if len(group) == 1:
            resolved.append(group[0])
            continue
        # Latest effective date wins; None dates sort earliest (original terms).
        def sort_key(c: Covenant):
            d = _parse_date(c.source.effective_date)
            return (d is not None, d or date.min)

        ordered = sorted(group, key=sort_key)
        winner = ordered[-1]
        for c in ordered[:-1]:
            c.superseded = True
        resolved.append(winner)
    return resolved
