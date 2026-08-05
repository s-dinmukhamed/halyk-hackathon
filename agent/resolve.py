"""Pick the covenant version in force, discarding superseded ones.

The trap of this challenge: an amendment (допсоглашение) can change a covenant's
threshold, so we must evaluate against the latest effective version, not the
original agreement. Covenants are grouped by (borrower, covenant); the latest
effective_date wins, undated ones sort as original terms.
"""
from __future__ import annotations

from datetime import date, datetime

from .schemas import Covenant


def _parse_date(s: str | None) -> date | None:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d"):
        try:
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

        def sort_key(c: Covenant):
            d = _parse_date(c.source.effective_date)
            return (d is not None, d or date.min)

        ordered = sorted(group, key=sort_key)
        winner = ordered[-1]
        for c in ordered[:-1]:
            c.superseded = True
        resolved.append(winner)
    return resolved
