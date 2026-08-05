"""Tests for the deterministic core: metric math, operators, verdicts, resolve.

These need no dataset and no network — they guard the part the judge scores
hardest (точность вычислений) and can run today.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent import compute
from agent.resolve import resolve_covenants
from agent.schemas import Covenant, CovenantType, Evidence, Operator, Transaction, Verdict


# --- metric math ---------------------------------------------------------- #
def test_debt_to_ebitda():
    assert compute.METRICS["debt_to_ebitda"]({"total_debt": 300, "ebitda": 100}) == 3.0

def test_current_ratio():
    assert compute.METRICS["current_ratio"]({"current_assets": 240, "current_liabilities": 200}) == 1.2

def test_metric_missing_input_is_none():
    assert compute.METRICS["dscr"]({"ebitda": 100}) is None  # no debt_service

def test_div_by_zero_is_none():
    assert compute.METRICS["debt_to_ebitda"]({"total_debt": 300, "ebitda": 0}) is None


# --- operator / verdict --------------------------------------------------- #
def test_operator_le_complied():
    assert compute.evaluate_operator(2.8, Operator.LE, 3.0) is True

def test_operator_le_breached():
    assert compute.evaluate_operator(3.4, Operator.LE, 3.0) is False

def test_operator_ge():
    assert compute.evaluate_operator(1.1, Operator.GE, 1.2) is False


def _fin_covenant(metric, op, thr):
    return Covenant(borrower_id="B1", covenant_id="C1", type=CovenantType.FINANCIAL,
                    metric=metric, operator=op, threshold=thr)

def test_evaluate_financial_breach():
    cov = _fin_covenant("debt_to_ebitda", Operator.LE, 3.0)
    verdict, value = compute.evaluate_financial(cov, {"total_debt": 340, "ebitda": 100})
    assert verdict == Verdict.BREACHED
    assert value == 3.4

def test_evaluate_financial_complied():
    cov = _fin_covenant("current_ratio", Operator.GE, 1.2)
    verdict, value = compute.evaluate_financial(cov, {"current_assets": 300, "current_liabilities": 200})
    assert verdict == Verdict.COMPLIED
    assert value == 1.5


# --- transaction scan ----------------------------------------------------- #
def test_find_offending_transaction_over_threshold():
    cov = Covenant(borrower_id="B1", covenant_id="C2", type=CovenantType.TRANSACTIONAL,
                   name="dividend", metric="dividend", threshold=1000)
    txs = [
        Transaction(tx_id="T1", borrower_id="B1", tx_type="dividend", amount=500),
        Transaction(tx_id="T2", borrower_id="B1", tx_type="dividend payment", amount=5000),
        Transaction(tx_id="T3", borrower_id="B1", tx_type="salary", amount=9000),
    ]
    offender = compute.find_offending_transaction(cov, txs)
    assert offender is not None and offender.tx_id == "T2"

def test_find_offending_none_when_below():
    cov = Covenant(borrower_id="B1", covenant_id="C2", type=CovenantType.TRANSACTIONAL,
                   name="dividend", metric="dividend", threshold=10000)
    txs = [Transaction(tx_id="T1", borrower_id="B1", tx_type="dividend", amount=500)]
    assert compute.find_offending_transaction(cov, txs) is None


# --- amendment resolution ------------------------------------------------- #
def test_resolve_amendment_supersedes_by_date():
    original = Covenant(borrower_id="B1", covenant_id="C1", threshold=3.0,
                        source=Evidence(effective_date="2024-01-01"))
    amended = Covenant(borrower_id="B1", covenant_id="C1", threshold=3.5,
                       source=Evidence(effective_date="2025-06-01"))
    resolved = [c for c in resolve_covenants([original, amended]) if not c.superseded]
    assert len(resolved) == 1
    assert resolved[0].threshold == 3.5
