"""L-1 lure detector, the alert-fatigue policy and the message -> payment link."""
import sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "service"))
from fastapi.testclient import TestClient
from index import app
from scam_text import hard_signals, entities

c = TestClient(app)
SCAM = "Sir, CBI has registered an FIR against your number. To avoid digital arrest transfer Rs 2,50,000 to the RBI verification account rbi.verify@paytm. Do not tell anyone."
LEGIT_OTP = "482913 is your OTP for transaction of Rs 2,300 at MYNTRA on HDFC Bank card XX1234. Do not share this OTP with anyone."


def msg(text, account="T1", ts="2022-09-09 09:00"):
    return c.post("/api/v2/message", json=dict(text=text, timestamp=ts, bank="070", account=account)).json()


def test_scam_and_legit_messages():
    s = msg(SCAM, account="M1")
    assert s["band"] == "high" and s["workflow"]["id"] == "digital_arrest" and s["alert"]["notify"] is True
    assert "rbi.verify@paytm" in s["entities"]["upi_ids"]
    for t in (LEGIT_OTP, "Bhai 500 bhej de PhonePe pe, kal lauta dunga.", "Your Myntra order is out for delivery today."):
        r = msg(t, account="M1")
        assert r["band"] == "safe" and r["alert"]["notify"] is False, t


def test_hard_signals_skip_genuine_warnings():
    assert hard_signals(LEGIT_OTP) == []
    assert hard_signals("Flipkart: Share OTP 123456 only with the delivery agent at your door.") == []
    assert {h["id"] for h in hard_signals("Install AnyDesk and share the code with me")} >= {"remote_app"}
    assert entities("call 98765 43210 or pay to kyc.help@okaxis")["phones"] == ["9876543210"]


def test_alert_fatigue_dedup():
    first = msg(SCAM, account="F1", ts="2022-09-09 08:00")
    again = msg(SCAM, account="F1", ts="2022-09-09 09:00")
    assert first["alert"]["notify"] is True and again["alert"]["notify"] is False
    assert "Already warned" in again["alert"]["reason"]


def test_lure_escalates_following_payment():
    msg(SCAM, account="P1", ts="2022-09-09 09:00")
    pay = dict(timestamp="2022-09-09 10:00", from_bank="070", from_account="P1", to_bank="011", to_account="NEWPAYEE", amount_paid=50.0, payment_format="Cheque")
    r = c.post("/api/v2/payment", json=pay).json()
    assert r["decision"] != "Approve" and r["reasons"][0]["feature"] == "Payer received a scam-like message"
    other = dict(pay, from_account="Q7", to_bank="paytm", to_account="rbi.verify@paytm", timestamp="2022-09-09 10:05")
    r2 = c.post("/api/v2/payment", json=other).json()
    assert r2["decision"] in ("Hold for analyst", "Block + STR draft")
    s = c.get("/api/v2/alerts/summary").json()
    assert s["payments_escalated"] >= 2 and s["messages"] >= 2
