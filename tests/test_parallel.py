"""Tests for the parallel helpers — correctness and order preservation."""
from __future__ import annotations

import time

from agent.parallel import pflatmap, pmap


def test_pmap_preserves_order():
    # Reverse-sleep so later items finish first if order were not enforced.
    def work(i):
        time.sleep((10 - i) * 0.005)
        return i * i

    assert pmap(work, range(10), max_workers=8) == [i * i for i in range(10)]


def test_pmap_empty_and_single():
    assert pmap(str, []) == []
    assert pmap(str, [7]) == ["7"]


def test_pmap_actually_concurrent():
    def slow(_):
        time.sleep(0.05)
        return 1

    t0 = time.time()
    pmap(slow, range(8), max_workers=8)
    # 8 x 50ms sequential = 400ms; concurrent should be well under half that.
    assert time.time() - t0 < 0.2


def test_pmap_reraises():
    def boom(i):
        if i == 3:
            raise ValueError("boom")
        return i

    try:
        pmap(boom, range(6), max_workers=4)
        assert False, "expected ValueError"
    except ValueError as e:
        assert "boom" in str(e)


def test_pflatmap_concatenates_in_order():
    assert pflatmap(lambda n: [n, n], [1, 2, 3], max_workers=3) == [1, 1, 2, 2, 3, 3]
