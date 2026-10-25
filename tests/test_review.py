"""Alerts, four-eyes review by authenticated analysts, hash-chained audit log, signed evidence packs."""
import sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "service"))
from fastapi.testclient import TestClient
from index import app
from review import ReviewDesk
import uuid
H1 = {"X-Analyst-Key": "demo-analyst-1"}; H2 = {"X-Analyst-Key": "demo-analyst-2"}


def act(c, aid, headers, action, nonce=None):
    return c.post(f"/api/v2/alerts/{aid}/review", json={"action": action}, headers={**headers, "X-Nonce": nonce or uuid.uuid4().hex})

S09 = "The finance office has revised the vendor details. Please remit the balance to the replacement beneficiary ahead of the afternoon cutoff. Call 9876543210 or pay vendor.new@ybl."


def test_alert_review_audit_evidence():
    c = TestClient(app); c.post("/api/v2/demo/reset", params={"warm": False})
    r = c.post("/api/v2/assess", json={"timestamp": "2022-09-09 10:00", "from_bank": "070", "from_account": "RV1", "to_bank": "011",
                                       "to_account": "RM1", "amount_paid": 9300, "payment_format": "ACH", "message": S09, "consent": True}).json()
    aid = r["alert_id"]; assert aid and r["shown_decision"] != "Block + STR draft"
    assert c.post(f"/api/v2/alerts/{aid}/review", json={"action": "approve_block"}).status_code == 401                    # no key
    assert act(c, aid, {"X-Analyst-Key": "demo-viewer"}, "approve_block").status_code == 403                                # viewer role
    assert act(c, aid, H1, "approve_block", nonce="fixed-nonce-123").json()["status"] == "block_pending_second_approval"
    assert act(c, aid, H2, "approve_block", nonce="fixed-nonce-123").status_code == 409                                    # replayed nonce
    assert act(c, aid, H1, "approve_block").status_code == 400                                                              # four-eyes
    assert act(c, aid, H2, "approve_block").json()["status"] == "blocked"
    au = c.get("/api/v2/audit").json(); assert au["chain_valid"] and len(au["entries"]) >= 3
    ev = c.get(f"/api/v2/alerts/{aid}/evidence").json()
    assert ev["pack_hash"] and "message" not in ev["alert"]["payment"]
    pub = c.get("/api/v2/keys/public").json()["public_key_b64"]
    from security import Signer
    assert Signer.verify({"pack_hash": ev["pack_hash"], "alert_id": aid}, ev["signature"]["signature"], pub)            # signed pack
    assert c.get("/api/v2/consortium").json()["entries"][-1]["registry"] == "mule_account"                                   # block shared, hashed
    assert "9876543210" not in str(ev) and "vendor.new@ybl" not in str(ev)          # identifiers are redacted in the pack


def test_tampering_breaks_the_chain():
    d = ReviewDesk()
    d._append({"type": "x"}); d._append({"type": "y"})
    assert d.verify_chain()
    d.log[0]["type"] = "edited"
    assert not d.verify_chain()
