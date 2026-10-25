"""session layer features inside RuseTect: session and identity domain, attack chains, pause screen, duress PIN, campaigns,
consortium ledger, signed verdicts, STR drafts, display score, typing profile. Each test pins one behaviour."""
import sys, uuid
from pathlib import Path
ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "service"))
import pytest
from fastapi.testclient import TestClient
from index import app
import context_rules as CTX
from session_signals import SessionSignals, assess_session, identity_check, attack_chain, coercion_index
from intervention import DuressVault, plan, pick_question
from consortium import Consortium
from security import Signer, NonceCache, RateLimiter, allowed
from risk_score import display_score
from behaviour import TypingProfile
import campaigns as CAMPAIGNS
from coordination import CoordinationTracker

c = TestClient(app)
PAY = {"timestamp": "2022-09-09 10:00", "from_bank": "070", "from_account": "CV1", "to_bank": "011", "to_account": "CM1",
       "amount_paid": 48000, "payment_format": "ACH"}
ARREST = ("This is CBI cyber cell. You are under digital arrest for money laundering. Stay on the video call and transfer "
          "the amount to the safe account immediately. Do not tell anyone.")


def nonce():
    return uuid.uuid4().hex


# ---------------------------------------------------------------- session domain
def test_no_session_means_no_change():
    r0 = CTX.assess(None, CTX.PaymentContext(new_beneficiary=True), True)
    r1 = CTX.assess(None, CTX.PaymentContext(new_beneficiary=True), True, session=None)
    assert r0.band == r1.band and r1.session == 0


def test_coaching_cues_alone_ask_for_verification_never_a_hold():
    s = assess_session(SessionSignals(on_call=True, video_call=True, screen_share=True, remote_access_app=True, payee_pasted=True,
                                      hesitation_s=40, app_hidden_s=60))
    assert s.points == 40                                           # capped
    r = CTX.assess(None, CTX.PaymentContext(), True, session=s)
    assert r.band == "MEDIUM"                                       # one domain only: verify (pause screen), not escalate


def test_session_plus_payment_evidence_escalates():
    s = assess_session(SessionSignals(on_call=True, call_minutes=45, screen_share=True, payee_pasted=True))
    r = CTX.assess(None, CTX.PaymentContext(new_beneficiary=True, amount_deviation=True), True, session=s)
    assert r.band in ("HIGH", "CRITICAL") and any(x.startswith("session:") for x in r.reasons)


def test_coercion_index_forces_verification():
    s = assess_session(SessionSignals(screen_share=True, payee_pasted=True, hesitation_s=30))
    assert coercion_index(SessionSignals(screen_share=True, payee_pasted=True, hesitation_s=30)) == 0.7 and s.force_verify
    assert CTX.assess(None, CTX.PaymentContext(), True, session=s).band != "LOW"


def test_hesitation_alone_is_weak():
    s = assess_session(SessionSignals(hesitation_s=60))           # older and first time users pause too
    assert CTX.assess(None, CTX.PaymentContext(), True, session=s).band == "LOW"


# ---------------------------------------------------------------- identity and chains
def test_sim_swap_takeover_withholds_the_otp():
    r = identity_check(SessionSignals(device_hash="d2", enrolled_device_hashes=["d1"], sim_swap_hours=3))
    assert r["decision"] == "deny_otp_reverify"
    assert plan("Approve", [], r)["otp"] == "withheld"
    assert identity_check(SessionSignals(device_hash="d1", enrolled_device_hashes=["d1"]))["decision"] == "allow"
    assert identity_check(SessionSignals(sim_swap_hours=500))["decision"] == "allow"


def test_attack_chains_need_order_and_window():
    ev = [{"type": "password_reset", "minute": 0}, {"type": "new_device", "minute": 30}, {"type": "new_payee", "minute": 40},
          {"type": "transfer", "minute": 50}]
    assert attack_chain(ev)["chain"] == "full"
    assert attack_chain(list(reversed([dict(e, minute=-e["minute"]) for e in ev])))["chain"] is None        # wrong order
    assert attack_chain([{"type": "new_device", "minute": 0}, {"type": "new_payee", "minute": 20}, {"type": "transfer", "minute": 30}])["chain"] == "short"
    assert attack_chain([{"type": "new_device", "minute": 0}, {"type": "new_payee", "minute": 20}, {"type": "transfer", "minute": 300}])["chain"] is None


# ---------------------------------------------------------------- intervention and duress
def test_pause_screen_question_follows_the_evidence():
    assert pick_question(["message cue: authority", "message cue: pressure"]) == "arrest"
    assert pick_question(["session: screen_share"]) == "call"
    p = plan("Step-up verification", ["session: on_call_while_paying"])
    assert p["show"] == "pause" and p["cool_off_minutes"] == 30 and set(p["screen"]) == {"en", "hi"}
    assert plan("Approve", [])["show"] == "none"
    assert plan("Hold for analyst", [])["funds"] == "held for analyst review"


def test_duress_pin_looks_successful_but_holds():
    v = DuressVault(); v.register("A1", "1234", "4321", ["+91-90000-00000"])
    assert v.confirm("A1", "1234")["result"] == "ok"
    d = v.confirm("A1", "4321")
    assert d["shown_to_customer"] == "Payment successful" and d["funds"] == "held" and v.notifications
    assert v.confirm("A1", "0000")["result"] == "wrong_pin"
    with pytest.raises(ValueError):
        v.register("A2", "1111", "1111", [])
    assert b"1234" not in repr(v.users).encode()                     # stored as salted hashes only


# ---------------------------------------------------------------- consortium and signing
def test_consortium_ledger_is_signed_chained_and_hashed():
    con = Consortium(secret="t")
    con.publish("BANK-B", "mule_account", "011:MULE9", "confirmed")
    assert con.check("mule_account", "011:MULE9")["flagged"] and not con.check("mule_account", "011:OTHER")["flagged"]
    assert "MULE9" not in str(con.ledger) and con.verify()
    con.ledger[0]["reason"] = "edited"
    assert not con.verify()


def test_signer_round_trip_and_tamper():
    s = Signer(); sig = s.sign({"a": 1})
    assert Signer.verify({"a": 1}, sig["signature"], s.public_b64) and not Signer.verify({"a": 2}, sig["signature"], s.public_b64)


def test_nonce_rate_limit_roles():
    n = NonceCache(); assert n.use("abcdefgh1") and not n.use("abcdefgh1") and not n.use("short")
    r = RateLimiter(limit=2, per_s=60); assert r.allow("x") and r.allow("x") and not r.allow("x")
    assert allowed("analyst", "approve_block") and not allowed("viewer", "hold")


# ---------------------------------------------------------------- score, campaigns, typing, coordination
def test_display_score_bands_match_decisions():
    cfg = {"verify_threshold": 0.1, "hold_threshold": 0.375, "block_threshold": 0.83}
    assert display_score("Approve", 0.02, 0, cfg)["score"] < 300
    assert 300 <= display_score("Step-up verification", 0.2, 45, cfg)["score"] < 600
    assert 600 <= display_score("Hold for analyst", 0.5, 70, cfg)["score"] < 950
    assert display_score("Block + STR draft", 0.95, 0, cfg)["score"] >= 950


def test_campaigns_link_by_payee_identifier_and_script():
    def al(i, payee, text, ents):
        return {"alert_id": f"A{i}", "status": "open", "payment": {"to_bank": "011", "to_account": payee, "from_account": f"V{i}",
                "amount_paid": 1000, "timestamp": f"2022-09-0{i} 10:00"}, "assessed": {"context": {"redacted_text": text, "entity_hashes": ents}}}
    s = "Your KYC will expire today. Pay the verification fee to <UPI> now or your account will be blocked."
    cs = CAMPAIGNS.build([al(1, "P1", s, ["h1"]), al(2, "P2", s.replace("today", "tonight"), []), al(3, "P3", "hello", ["h1"]), al(4, "P9", "unrelated", [])])
    assert len(cs) == 1 and cs[0]["size"] == 3 and set(cs[0]["linked_by"]) >= {"near identical script", "same payee identifier in messages"}


def test_typing_profile_flags_a_different_typist():
    import random
    rnd = random.Random(0); keys = ["th", "he", "in", "er", "an", "re", "on", "at"]
    base = {k: 90 + 15 * i for i, k in enumerate(keys)}
    prof = TypingProfile()
    for _ in range(4):
        prof.enrol({k: v * rnd.uniform(0.95, 1.05) for k, v in base.items()})
    assert not prof.score({k: v * 1.02 for k, v in base.items()})["anomaly"]
    assert prof.score({k: v * 2.2 for k, v in base.items()})["anomaly"]


def test_coordination_rule_is_off_by_default_but_counts():
    t = CoordinationTracker(window_min=1000, flag_rank=1, k=2)
    t.record("s1", "P", 0, 2); t.record("s2", "P", 5, 2)
    assert t.flagged_payers("P", 10) == 2 and not t.coordinated("P", 10)
    t.enabled = True; assert t.coordinated("P", 10)


# ---------------------------------------------------------------- API end to end
def test_digital_arrest_end_to_end():
    c.post("/api/v2/demo/reset", params={"warm": False})
    r = c.post("/api/v2/assess", json={**PAY, "message": ARREST, "consent": True,
               "session": {"on_call": True, "video_call": True, "call_minutes": 50, "screen_share": True, "payee_pasted": True}}).json()
    assert r["final_decision"] in ("Hold for analyst", "Block + STR draft") or r["context"]["band"] in ("HIGH", "CRITICAL")
    assert r["risk_score"]["score"] >= 600 and r["intervention"]["show"] == "pause" and r["intervention"]["question_type"] == "arrest"
    pub = c.get("/api/v2/keys/public").json()["public_key_b64"]
    assert Signer.verify(r["signature"]["signed_fields"], r["signature"]["signature"], pub)
    aid = r["alert_id"]
    assert c.post(f"/api/v2/alerts/{aid}/customer", json={"answer": "stop"}).json()["status"] == "stopped_by_customer"
    s = c.get(f"/api/v2/alerts/{aid}/str").json()
    assert s["status"].startswith("DRAFT") and any("call" in g for g in s["sections"]["D_grounds_of_suspicion"])


def test_duress_and_identity_endpoints():
    c.post("/api/v2/demo/reset", params={"warm": False})
    assert c.post("/api/v2/duress/register", json={"account": "D1", "normal_pin": "2468", "duress_pin": "8642", "trusted_contacts": ["son"]}).status_code == 200
    assert c.post("/api/v2/payments/confirm", json={"account": "D1", "pin": "8642"}).json()["funds"] == "held"
    r = c.post("/api/v2/identity/check", json={"device_hash": "x", "enrolled_device_hashes": ["y"], "sim_swap_hours": 2}).json()
    assert r["decision"] == "deny_otp_reverify"


def test_cross_bank_flag_reaches_the_score():
    c.post("/api/v2/demo/reset", params={"warm": False})
    assert c.post("/api/v2/consortium/publish", json={"bank": "BANK-B", "identifier": "011:CM7"}).status_code == 401
    assert c.post("/api/v2/consortium/publish", json={"bank": "BANK-B", "identifier": "011:CM7"}, headers={"X-Analyst-Key": "demo-analyst-1"}).status_code == 200
    r = c.post("/api/v2/assess", json={**PAY, "to_account": "CM7"}).json()
    assert r["network"]["flagged_by_consortium"] == ["BANK-B"] and "account graph: coordinated_accounts" in r["context"]["reasons"]
