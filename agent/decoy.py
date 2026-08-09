"""Decoy-transaction detection.

The ledger is seeded with fabricated "decoy" transactions whose counterparty is a
recurring Anglo brand name (Bridgeport, Hartley, Juniper, ...) and whose description
deliberately contradicts the counterparty's business. Genuine transactions come from
real Kazakh entities (KEGOC, Halyk Bank, Ural Rig Manufacturing, Pavlodar Regional
Grid, ...), each appearing only once or twice. We separate the two by the frequency
of the counterparty's leading brand word: fabricated brands recur many times across
the whole ledger, real entities do not.
"""
from __future__ import annotations

import re
from collections import Counter

# Real Kazakh geographic / company tokens that legitimately recur across scenarios
# (e.g. "Ural Rig Manufacturing", "Ural Grinding Works") and must never be treated
# as fabricated brands even though they appear more than a few times.
_REAL_TOKENS = {
    "Ural", "KEGOC", "KazMunayGas", "Kazakhmys", "Halyk", "Kaspi", "Kazyna",
    "KazTransOil", "Magnum", "Metro", "Eurasia", "Rheinland", "Development",
    "Kazakhstan", "National", "Caspian", "Atyrau", "Pavlodar", "Irtysh", "Ertis",
    "Almaty", "Aktau", "Aktobe", "Kostanay", "Karaganda", "Ekibastuz", "Uralsk",
    "Semey", "Kyzylorda", "Turkistan", "Zhezkazgan", "Oskemen", "Taraz", "Shymkent",
    "Astana", "Aral", "Tengiz", "Mangystau", "Zhambyl", "Ulytau", "Zhetysu",
    "Zhaiyk", "Syrdarya", "Ilek", "Sarybel", "Saryarka", "Altyn", "Turan", "Tien",
    "City",
}


def _lead_word(counterparty: str) -> str:
    """The entity's leading brand word, ignoring any parenthetical city suffix."""
    name = re.sub(r"\(.*?\)", "", counterparty or "").strip()
    words = re.findall(r"[A-Za-z]+", name)
    return words[0] if words else ""


def build_fake_brands(counterparties: list[str], threshold: int = 4) -> set[str]:
    """Leading brand words that recur at least `threshold` times are fabricated."""
    freq: Counter[str] = Counter()
    for cp in counterparties:
        w = _lead_word(cp)
        if w:
            freq[w] += 1
    return {w for w, c in freq.items() if c >= threshold and w not in _REAL_TOKENS}


def is_decoy(counterparty: str, fake_brands: set[str]) -> bool:
    return _lead_word(counterparty) in fake_brands
