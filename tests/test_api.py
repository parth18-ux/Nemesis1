"""Smoke tests for the RuseTect API using only the bundled models and demo stream."""
import json, os, sys
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))
from fastapi.testclient import TestClient
from index import app

c = TestClient(app)
SAMPLES = json.loads((ROOT / "service" / "models" / "sample_applications.json").read_text())


def test_health_and_pages():
    assert c.get("/api/health").json()["engine_loaded"] is True
    for path in ("/", "/console", "/report", "/research", "/docs"):
        assert c.get(path).status_code == 200, path


def test_single_payment():
    r = c.post("/api/v2/payment", json=dict(timestamp="2022-09-09 10:00", from_bank="070", from_account="A1",
                                            to_bank="011", to_account="B2", amount_paid=120.0, payment_format="Cheque")).json()
    assert r["decision"] in ("Approve", "Step-up verification", "Hold for analyst", "Block + STR draft")


def test_onboarding_gate():
    mule = c.post("/api/v2/onboard", json=SAMPLES["mule"]).json()
    genuine = c.post("/api/v2/onboard", json=SAMPLES["genuine"]).json()
    assert mule["flag"] is True and genuine["flag"] is False


def test_demo_replay():
    r = c.post("/api/v2/demo/reset").json()
    assert r["history_ingested"] == 100_000
    total = {"high": 0}
    for _ in range(10):
        n = c.post("/api/v2/demo/next?n=2000").json()
        total["high"] += n["bands"]["high"]
    assert n["position"] == 20_000 and n["exits"]["L1"] > n["exits"]["L2"]


def test_payment_format_and_currency_are_validated():
    base = dict(timestamp="2022-09-09 10:00", from_bank="070", from_account="V1", to_bank="011", to_account="V2", amount_paid=100.0)
    r = c.post("/api/v2/payment", json=base)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "payment_format"]
    assert c.post("/api/v2/payment", json={**base, "payment_format": "UPI"}).status_code == 422
    r = c.post("/api/v2/payment", json={**base, "payment_format": "Wire", "payment_currency": "INR"})
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "payment_currency"]
    assert c.post("/api/v2/payment", json={**base, "payment_format": "Wire", "payment_currency": "Rupee"}).status_code == 200


def test_validation_lists_match_cascade_config():
    import typing
    from server import Currency, PaymentFormat
    cfg = json.loads((ROOT / "service" / "models" / "cascade_config.json").read_text())
    assert sorted(typing.get_args(PaymentFormat)) == sorted(cfg["fmt_codes"])
    assert sorted(typing.get_args(Currency)) == sorted(cfg["cur_codes"])


def test_production_mode_requires_secrets():
    from security import missing_production_secrets as miss
    assert miss({}) == []
    assert len(miss({"RUSETECT_ANALYST_KEYS": "{}"})) == 3
    assert miss({"RUSETECT_ANALYST_KEYS": "{}", "RUSETECT_SIGNING_KEY": "x", "RUSETECT_CONSORTIUM_SECRET": " "}) == \
        ["RUSETECT_CONSORTIUM_SECRET", "RUSETECT_HASH_SALT"]
    assert miss({"RUSETECT_ANALYST_KEYS": "{}", "RUSETECT_SIGNING_KEY": "a", "RUSETECT_CONSORTIUM_SECRET": "b", "RUSETECT_HASH_SALT": "c"}) == []
