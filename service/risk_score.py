"""
One 0 to 1000 display score, derived from RuseTect's decision so the two can never disagree:
  0 to 299    approve silently
  300 to 599  verify: pause screen, then voice or face check
  600 to 1000 hold for an analyst (1000 = the payment model's block level, which still needs two analysts)
Inside each band the position shows how strong the evidence is (payment model risk against its thresholds, and
context points out of 100). It is a ranking aid for people, not a probability.
"""
from __future__ import annotations

RANK = {"Approve": 0, "Step-up verification": 1, "Hold for analyst": 2, "Block + STR draft": 3}


def display_score(final_decision: str, payment_risk: float | None, total_points: int, cfg: dict) -> dict:
    r = RANK.get(final_decision, 0); risk = payment_risk or 0.0
    vt, ht, bt = cfg.get("verify_threshold", 0.1), cfg.get("hold_threshold", 0.375), cfg.get("block_threshold", 0.83)
    pts = min(1.0, total_points / 100)
    if r == 0:
        s = 299 * max(min(1.0, risk / vt), min(1.0, total_points / 40))
    elif r == 1:
        s = 300 + 299 * max(min(1.0, (risk - vt) / max(ht - vt, 1e-9)), pts)
    elif r == 2:
        s = 600 + 349 * max(min(1.0, (risk - ht) / max(bt - ht, 1e-9)), pts)
    else:
        s = 950 + 50 * min(1.0, (risk - bt) / max(1 - bt, 1e-9))
    s = int(round(max(0, min(1000, s))))
    band = "approve" if s < 300 else "verify" if s < 600 else "analyst hold"
    return {"score": s, "band": band, "note": "display score derived from the decision; not a probability"}
