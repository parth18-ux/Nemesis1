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
    for path in ("/", "/console", "/report", "/docs"):
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
