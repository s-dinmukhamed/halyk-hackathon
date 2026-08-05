"""Tests for the submission validator."""
from __future__ import annotations

from agent.validate import validate_payload


def _answer(**over):
    base = {
        "borrower_id": "B1",
        "covenant_id": "C1",
        "verdict": "complied",
        "value": 2.5,
        "evidence_transaction_id": None,
    }
    base.update(over)
    return base


def test_clean_submission_passes():
    rep = validate_payload({"answers": [_answer()]})
    assert rep.ok
    assert rep.stats["answers"] == 1


def test_top_level_list_accepted():
    rep = validate_payload([_answer()])
    assert rep.ok
    assert rep.stats["wrapper_key"] == "(top-level list)"


def test_empty_is_error():
    rep = validate_payload({"answers": []})
    assert not rep.ok


def test_missing_ids_error():
    rep = validate_payload({"answers": [_answer(borrower_id="", covenant_id="")]})
    assert not rep.ok
    assert any("borrower_id" in e for e in rep.errors)
    assert any("covenant_id" in e for e in rep.errors)


def test_missing_component_key_is_error():
    a = _answer()
    del a["value"]
    rep = validate_payload({"answers": [a]})
    assert not rep.ok
    assert any("value" in e for e in rep.errors)


def test_bad_verdict_label_warns():
    rep = validate_payload({"answers": [_answer(verdict="OK")]})
    assert rep.ok  # structural pass
    assert any("not in" in w for w in rep.warnings)


def test_breach_without_evidence_warns():
    rep = validate_payload({"answers": [_answer(verdict="breached", evidence_transaction_id=None)]})
    assert any("evidence" in w.lower() for w in rep.warnings)


def test_null_value_warns():
    rep = validate_payload({"answers": [_answer(value=None)]})
    assert any("null" in w.lower() for w in rep.warnings)


def test_duplicate_warns():
    rep = validate_payload({"answers": [_answer(), _answer()]})
    assert any("duplicate" in w.lower() for w in rep.warnings)
