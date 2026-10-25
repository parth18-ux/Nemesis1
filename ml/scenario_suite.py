"""End to end replay suite (the "scam replay suite"): every scenario goes through the real API (/api/v2/assess,
/api/v2/identity/check, /api/v2/payments/confirm) with randomised parameters, and we count what the customer would
have experienced.

Scam scenarios count as STOPPED when the payment does not go straight through (verification with the pause screen,
an analyst hold, a withheld OTP, or a duress hold). Genuine scenarios count as FALSE ALARMS when they are not approved
silently, and as WRONGLY HELD when they reach an analyst hold. Time to decision comes from the API's own timer.
Fairness: the same genuine payment made by an older, slower customer (long hesitation) versus everyone else.

These scenarios are authored, not observed: they check that each signal does what it should end to end and that the
layers do not fight each other. They are not evidence of real world detection rates.
Writes results/scenario_suite.json.
"""
import json, os, random, sys, time, uuid
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "service"))
from fastapi.testclient import TestClient  # noqa: E402
from index import app  # noqa: E402

c = TestClient(app); rnd = random.Random(7); N = int(os.environ.get("SCENARIO_N", 40))
RANK = {"Approve": 0, "Step-up verification": 1, "Hold for analyst": 2, "Block + STR draft": 3}
MSG = {
    "arrest": "This is {who}. A parcel in your name has drugs and your Aadhaar is linked to money laundering. You are under digital arrest. "
              "Stay on the video call and transfer Rs {amt} to the RBI safe account immediately for verification. Do not tell anyone.",
    "kyc": "Dear customer your KYC will expire today and your account will be blocked. Install AnyDesk so our support executive can "
           "update it, then pay the verification fee to {upi} right now.",
    "invest": "Join our VIP trading group. Guaranteed returns of {pct}% weekly, double your money in a month. Pay the joining amount to {upi} today itself.",
    "family": "Hi beta, it's Papa. Please send {amt} for the electricity bill when you get a chance, thanks.",
}
WHO = ["CBI cyber cell", "Mumbai crime branch police", "customs officer from Delhi airport", "TRAI legal department"]
clock = [27_700_000]                                    # minutes since epoch, moves forward with every payment


def ts():
    clock[0] += rnd.randint(1, 30)
    t = time.gmtime(clock[0] * 60)
    return time.strftime("%Y-%m-%d %H:%M", t)


def pay(src, dst, amt, **kw):
    body = {"timestamp": ts(), "from_bank": "0900", "from_account": src, "to_bank": "0200", "to_account": dst, "amount_paid": amt,
            "payment_currency": "Rupee", "payment_format": "ACH", **kw}
    return c.post("/api/v2/assess", json=body).json()


def history(src, payee, n=5, amt=(800, 3000)):
    for _ in range(n):
        pay(src, payee, rnd.randint(*amt))


def run(name, fn):
    out = [fn(i) for i in range(N)]
    ranks = np.array([o["rank"] for o in out]); lat = [o["ms"] for o in out if o.get("ms") is not None]
    return {"n": N, "approved": int((ranks == 0).sum()), "verify": int((ranks == 1).sum()), "hold_or_more": int((ranks >= 2).sum()),
            "time_to_decision_ms_p50": round(float(np.percentile(lat, 50)), 1) if lat else None,
            "time_to_decision_ms_p95": round(float(np.percentile(lat, 95)), 1) if lat else None,
            "example": out[0].get("example")}


def assess_result(r, extra=None):
    return {"rank": RANK[r["final_decision"]], "ms": r.get("time_to_decision_ms"),
            "example": {"final": r["final_decision"], "context_band": r["context"]["band"], "score": r["risk_score"]["score"],
                        "pause_question": (r.get("intervention") or {}).get("question_type"), "reasons": r["context"]["reasons"][:6], **(extra or {})}}


# ---------------------------------------------------------------- genuine
def genuine_known(i, hesitation=0.0):
    src, payee = f"G{i}{'E' if hesitation else ''}", f"P{i}"
    history(src, payee)
    r = pay(src, payee, rnd.randint(900, 2800), session={"hesitation_s": hesitation, "device_hash": "d1", "enrolled_device_hashes": ["d1"]})
    return assess_result(r)


def genuine_new_payee(i):
    src = f"GN{i}"; history(src, f"PN{i}")
    return assess_result(pay(src, f"RENT{i}", rnd.randint(1500, 4000), session={"device_hash": "d1", "enrolled_device_hashes": ["d1"]}))


def genuine_family_call(i):
    src = f"GF{i}"; history(src, f"PF{i}")
    amt = rnd.randint(500, 2500)
    return assess_result(pay(src, f"PAPA{i}", amt, message=MSG["family"].format(amt=amt), consent=True,
                             session={"on_call": True, "call_minutes": rnd.uniform(1, 6), "device_hash": "d1", "enrolled_device_hashes": ["d1"]}))


def first_time_user(i):
    return assess_result(pay(f"FT{i}", f"SHOP{i}", rnd.randint(200, 1500), session={"hesitation_s": rnd.uniform(15, 45), "device_hash": "d1",
                                                                                    "enrolled_device_hashes": ["d1"]}))


# ---------------------------------------------------------------- scams
def digital_arrest(i):
    src = f"DA{i}"; history(src, f"PD{i}")
    amt = rnd.randint(40_000, 300_000)
    return assess_result(pay(src, f"SAFE{i % 7}", amt, message=MSG["arrest"].format(who=rnd.choice(WHO), amt=amt), consent=True,
                             session={"on_call": True, "video_call": True, "call_minutes": rnd.uniform(30, 180), "screen_share": rnd.random() < 0.6,
                                      "payee_pasted": True, "hesitation_s": rnd.uniform(5, 60), "device_hash": "d1", "enrolled_device_hashes": ["d1"]}))


def kyc_remote(i):
    src = f"KY{i}"; history(src, f"PK{i}")
    return assess_result(pay(src, f"KYCFEE{i % 5}", rnd.randint(2_000, 50_000), message=MSG["kyc"].format(upi=f"kyc.help{i % 5}@ybl"), consent=True,
                             session={"remote_access_app": True, "payee_pasted": True, "app_hidden_s": rnd.uniform(0, 90),
                                      "device_hash": "d1", "enrolled_device_hashes": ["d1"]}))


def investment(i):
    src = f"IN{i}"; history(src, f"PI{i}")
    return assess_result(pay(src, f"VIP{i % 4}", rnd.randint(25_000, 200_000), message=MSG["invest"].format(pct=rnd.choice([10, 20, 30]), upi=f"vip{i % 4}@okaxis"),
                             consent=True, session={"device_hash": "d1", "enrolled_device_hashes": ["d1"]}))


def attack_chain(i):
    src = f"AC{i}"; history(src, f"PA{i}"); t0 = clock[0]
    ev = [{"type": "password_reset", "minute": t0}, {"type": "new_device", "minute": t0 + 5}, {"type": "new_payee", "minute": t0 + 12},
          {"type": "limit_raise", "minute": t0 + 14}, {"type": "transfer", "minute": t0 + 15}]
    return assess_result(pay(src, f"DRAIN{i}", rnd.randint(50_000, 400_000), session={"device_hash": "dX", "enrolled_device_hashes": ["d1"], "events": ev}))


def typing_impostor(i):
    src = f"TI{i}"; history(src, f"PT{i}")
    return assess_result(pay(src, f"PT{i}", rnd.randint(900, 2800), session={"typing_anomaly": True, "device_hash": "dZ", "enrolled_device_hashes": ["d1"]}))


def sim_swap(i):
    t0 = time.perf_counter()
    r = c.post("/api/v2/identity/check", json={"device_hash": f"new{i}", "enrolled_device_hashes": ["d1"], "sim_swap_hours": rnd.uniform(0.5, 48)}).json()
    return {"rank": 2 if r["decision"] == "deny_otp_reverify" else 1 if r["decision"] == "step_up" else 0,
            "ms": round((time.perf_counter() - t0) * 1000, 2), "example": {"decision": r["decision"], "reason": r["reason"]}}


def duress(i):
    acct = f"DU{i}"; c.post("/api/v2/duress/register", json={"account": acct, "normal_pin": "2580", "duress_pin": "0852", "trusted_contacts": ["family"]})
    t0 = time.perf_counter(); r = c.post("/api/v2/payments/confirm", json={"account": acct, "pin": "0852"}).json()
    return {"rank": 2 if r.get("funds") == "held" else 0, "ms": round((time.perf_counter() - t0) * 1000, 2),
            "example": {"shown_to_customer": r.get("shown_to_customer"), "funds": r.get("funds")}}


def mule_ring():
    F = c.get("/api/v2/fixtures/mule").json(); res = []
    for k, amt in enumerate(F["deposits_inr"]):
        v = f"MV{k:02d}"; history(v, f"MP{k}", n=3)
        r1 = pay(v, F["mule"]["account"], amt)
        r2 = pay(F["mule"]["account"], F["collector"]["account"], round(amt * F["forward_share"], 2))
        res += [RANK[r1["final_decision"]], RANK[r2["final_decision"]]]
    r = np.array(res)
    return {"payments": len(r), "not_straight_through": int((r >= 1).sum()), "hold_or_more": int((r >= 2).sum()),
            "note": "victims with short history paying one mule that forwards 98% a minute later"}


T0 = time.time()
c.post("/api/v2/demo/reset", params={"warm": False})
out = {"genuine": {}, "scams": {}}
out["genuine"]["known_payee"] = run("known", lambda i: genuine_known(i))
out["genuine"]["known_payee_older_slow"] = run("elderly", lambda i: genuine_known(i, hesitation=rnd.uniform(25, 90)))
out["genuine"]["new_payee_rent"] = run("new", genuine_new_payee)
out["genuine"]["family_call_small_payment"] = run("family", genuine_family_call)
out["genuine"]["first_time_user"] = run("first", first_time_user)
for name, fn in [("digital_arrest", digital_arrest), ("kyc_remote_access", kyc_remote), ("investment_scam", investment),
                 ("attack_chain_takeover", attack_chain), ("typing_impostor_new_device", typing_impostor), ("sim_swap_takeover", sim_swap),
                 ("duress_pin", duress)]:
    out["scams"][name] = run(name, fn)
out["scams"]["mule_ring_fixture"] = mule_ring()
g, s = out["genuine"], {k: v for k, v in out["scams"].items() if "n" in v}
out["summary"] = {
    "scam_payments_stopped": f'{sum(v["n"] - v["approved"] for v in s.values())} of {sum(v["n"] for v in s.values())}',
    "genuine_false_alarms": f'{sum(v["n"] - v["approved"] for v in g.values())} of {sum(v["n"] for v in g.values())}',
    "genuine_wrongly_held": f'{sum(v["hold_or_more"] for v in g.values())} of {sum(v["n"] for v in g.values())}',
    "fairness_older_slow_vs_others_false_alarm_rate": [round((g["known_payee_older_slow"]["n"] - g["known_payee_older_slow"]["approved"]) / N, 3),
                                                       round((g["known_payee"]["n"] - g["known_payee"]["approved"]) / N, 3)],
}
out["caveat"] = "Authored scenarios (synthetic): they test mechanics end to end, not real world detection rates."
out["runtime_seconds"] = round(time.time() - T0, 1)
json.dump(out, open(os.path.join(ROOT, "results", "scenario_suite.json"), "w"), indent=1)
print(json.dumps({"summary": out["summary"], "genuine": {k: {x: v[x] for x in ("approved", "verify", "hold_or_more")} for k, v in g.items()},
                  "scams": {k: {x: v.get(x) for x in ("approved", "verify", "hold_or_more", "not_straight_through")} for k, v in out["scams"].items()}}, indent=1))
