"""Bounded, order-preserving parallelism for the I/O-bound pipeline stages.

The finals window is ~3 hours and a tie on points is broken by *submission time*,
so wall-clock matters. The slow stages are network-bound (an LLM call per
document) and file-bound (parsing many PDFs) — both benefit from threads despite
the GIL, because they spend most of their time waiting.

Two guarantees this helper adds over a raw ThreadPoolExecutor:
  * order preserved — output[i] corresponds to input[i], so the run stays
    deterministic regardless of which thread finishes first.
  * bounded workers — free-tier LLM APIs rate-limit aggressively; a modest,
    configurable pool avoids turning speed-ups into 429 storms.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Iterable, TypeVar

from .config import settings

T = TypeVar("T")
R = TypeVar("R")


def pmap(fn: Callable[[T], R], items: Iterable[T],
         max_workers: int | None = None) -> list[R]:
    """Parallel map that preserves input order.

    Falls back to a plain sequential map when there is nothing to gain (0/1 item
    or a single worker), which keeps stack traces clean during debugging.
    """
    items = list(items)
    workers = max_workers if max_workers is not None else settings.max_workers
    workers = max(1, min(workers, len(items)))
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        # executor.map preserves input order and re-raises exceptions on consume.
        return list(pool.map(fn, items))


def pflatmap(fn: Callable[[T], list[R]], items: Iterable[T],
             max_workers: int | None = None) -> list[R]:
    """Like pmap but concatenates list results, preserving item order."""
    out: list[R] = []
    for sublist in pmap(fn, items, max_workers=max_workers):
        out.extend(sublist)
    return out
