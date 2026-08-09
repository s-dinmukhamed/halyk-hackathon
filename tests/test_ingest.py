"""Tests for ledger loading, scenario mapping, and document routing."""
from __future__ import annotations

from pathlib import Path

from agent import ingest

_LEDGER = """txn_id,date,account_id,counterparty,description,amount,currency
TXN-P5-0004,2025-01-02,ACC-7805,Sarybel Capital LLP,Management advisory retainer,-273418.66,USD
TXN-P5-0013,2025-03-01,ACC-7805,KEGOC Grid Operations JSC,Capacity sales,8214663.28,USD
TXN-9001-0007,2025-01-02,ACC-9001,Decoy Co,Noise,-100.00,USD
"""


def _write_ledger(tmp_path: Path) -> Path:
    p = tmp_path / "master_ledger_2025.csv"
    p.write_text(_LEDGER, encoding="utf-8")
    return p


def test_load_ledger_tags_scenario_from_prefix(tmp_path):
    txs = ingest.load_ledger(_write_ledger(tmp_path))
    by_id = {t.tx_id: t for t in txs}
    assert by_id["TXN-P5-0004"].scenario_id == "P5"
    assert by_id["TXN-P5-0004"].amount == -273418.66
    assert by_id["TXN-9001-0007"].scenario_id == "9001"


def test_scenario_index_maps_account_to_scenario(tmp_path):
    idx = ingest.load_scenario_index(_write_ledger(tmp_path))
    assert idx["ACC-7805"] == "P5"
    assert idx["ACC-9001"] == "9001"


def test_classify_document():
    assert ingest.classify_document("ДОГОВОР БАНКОВСКОГО ЗАЙМА ...") == ingest.CREDIT_AGREEMENT
    assert ingest.classify_document("АУДИТОРСКОЕ ДЕЛО ... примечания аудитора") == ingest.AUDIT
    assert ingest.classify_document("Досье «Знай своего клиента» (KYC)") == ingest.KYC
    assert ingest.classify_document("Бренд-гайд и политика отпусков") == ingest.OTHER
