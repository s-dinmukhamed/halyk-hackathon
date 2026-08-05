"""Bounded, order-preserving parallel map for the I/O-bound stages.

The slow stages wait on the network (an LLM call per document) or disk (parsing
many PDFs), so threads help despite the GIL. Over a raw ThreadPoolExecutor this
keeps output order aligned with input (so runs stay deterministic) and caps the
worker count (so free-tier APIs don't rate-limit us).
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

    Runs sequentially when there's nothing to gain (0/1 item or a single
    worker), which keeps stack traces clean while debugging.
    """
    items = list(items)
    workers = max_workers if max_workers is not None else settings.max_workers
    workers = max(1, min(workers, len(items)))
    if workers == 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(fn, items))  # map keeps order and re-raises errors


def pflatmap(fn: Callable[[T], list[R]], items: Iterable[T],
             max_workers: int | None = None) -> list[R]:
    """Like pmap but concatenates list results, preserving item order."""
    out: list[R] = []
    for sublist in pmap(fn, items, max_workers=max_workers):
        out.extend(sublist)
    return out
