"""Tests for the nested-template submission validator."""
from __future__ import annotations

from agent.validate import validate_payload


def _cell(status="COMPLIANT", actual=2.5, evidence_txn_id=None):
    return {"status": status, "actual": actual, "evidence_txn_id": evidence_txn_id}


def _payload(**cell_over):
    return {
        "team": "hustle",
        "contact_email": "x@example.com",
        "model": "gemini-2.0-flash",
        "answers": {"P1": {"6.1": _cell(**cell_over), "6.2": _cell(), "6.3": _cell()}},
    }


def test_clean_submission_passes():
    rep = validate_payload(_payload())
    assert rep.ok
    assert rep.stats["cells"] == 3


def test_missing_status_key_is_error():
    p = _payload()
    del p["answers"]["P1"]["6.1"]["status"]
    rep = validate_payload(p)
    assert not rep.ok
    assert any("status" in e for e in rep.errors)


def test_null_status_is_error():
    rep = validate_payload(_payload(status=None))
    assert not rep.ok


def test_bad_status_label_is_error():
    rep = validate_payload(_payload(status="complied"))
    assert not rep.ok
    assert any("status" in e for e in rep.errors)


def test_non_numeric_actual_is_error():
    rep = validate_payload(_payload(actual=None))
    assert not rep.ok


def test_negative_actual_warns():
    rep = validate_payload(_payload(actual=-3.0))
    assert rep.ok
    assert any("negative" in w for w in rep.warnings)


def test_empty_team_warns():
    p = _payload()
    p["team"] = ""
    rep = validate_payload(p)
    assert any("team" in w for w in rep.warnings)


def test_template_missing_cell_is_error():
    template = {"answers": {"P1": {"6.1": {}, "6.2": {}, "6.3": {}},
                            "P2": {"6.1": {}, "6.2": {}, "6.3": {}}}}
    rep = validate_payload(_payload(), template)
    assert not rep.ok
    assert any("P2" in e for e in rep.errors)
