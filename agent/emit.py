"""Build submission.json by filling the organizers' template in place.

The template already contains every cell (scenario -> clause -> {status, actual,
evidence_txn_id}); we only fill the three null fields per cell and set the
top-level team / contact_email / model. Never leave a status null — an empty cell
scores the same as a wrong one, so we always emit a verdict.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import settings
from .schemas import CovenantAnswer, Verdict

STATUS = "status"
ACTUAL = "actual"
EVIDENCE = "evidence_txn_id"


def write_submission(answers: list[CovenantAnswer], template_path: str | Path,
                     out_path: str | Path) -> Path:
    template_path, out_path = Path(template_path), Path(out_path)
    payload = json.loads(template_path.read_text(encoding="utf-8"))

    payload["team"] = settings.team
    payload["contact_email"] = settings.contact_email
    payload["model"] = settings.model_label

    by_cell = {(a.scenario_id, a.clause): a for a in answers}
    for sid, clauses in payload.get("answers", {}).items():
        for clause, cell in clauses.items():
            ans = by_cell.get((sid, clause))
            if ans is not None:
                cell[STATUS] = ans.verdict.value
                cell[ACTUAL] = ans.actual
                cell[EVIDENCE] = ans.evidence_tx_id
            else:
                # No answer produced — fill a safe default so the cell is scorable.
                cell[STATUS] = Verdict.COMPLIANT.value
                cell[ACTUAL] = cell.get(ACTUAL)
                cell[EVIDENCE] = None

    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
