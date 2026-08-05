"""Step 6 — EMIT: internal answers -> submission.json in the required template.

The exact template is unknown until the public dataset drops (6 Aug). We keep
the mapping isolated here so adapting to the real key names is a one-file change.
Guiding rule: ALWAYS fill all three components (verdict / value / evidence tx),
never emit null where the scorer expects a value — partial credit is per-component.
"""
from __future__ import annotations

import json
from pathlib import Path

from .schemas import CovenantAnswer, Verdict

# TODO(6 Aug): replace with the organizers' exact field names from the template.
FIELD_VERDICT = "verdict"
FIELD_VALUE = "value"
FIELD_EVIDENCE_TX = "evidence_transaction_id"
VERDICT_LABELS = {Verdict.COMPLIED: "complied", Verdict.BREACHED: "breached"}


def answer_to_record(ans: CovenantAnswer) -> dict:
    return {
        "borrower_id": ans.borrower_id,
        "covenant_id": ans.covenant_id,
        FIELD_VERDICT: VERDICT_LABELS[ans.verdict],
        FIELD_VALUE: ans.value,
        FIELD_EVIDENCE_TX: ans.evidence_tx_id,
    }


def write_submission(answers: list[CovenantAnswer], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    payload = {"answers": [answer_to_record(a) for a in answers]}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
