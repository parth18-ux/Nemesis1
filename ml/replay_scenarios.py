"""Replay the fictional mule fixture through the live payment engine, then through the context layer.

1. payment engine alone (service/rusetect_engine.py, committed models, no account history)
2. context layer: transaction and graph checks derived from the engine's own live features; message context only
   where the fixture gives one and consent is True (victim V01)
3. combine(): the stricter of the two decisions (context veto), context alone never beyond an analyst hold
Writes results/scenario_results.json."""
import json, math, os, sys
from datetime import datetime, timedelta, timezone
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from rusetect_engine import Engine
from context_rules import assess, combine, PaymentContext, RULESET_VERSION

F = json.load(open(os.path.join(ROOT, "fixtures", "mule_replay.json")))
eng = Engine(); eng.reset()
t0 = datetime.strptime(F["start"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)


def ctx_from_features(f):
    hist = f["s_out_n"] >= 3
    usual = math.expm1(f["s_out_mean_usd"]) if hist else None
    usd = math.expm1(f["log_usd"])
    msi, ratio = f.get("s_mins_since_in"), f.get("s_out_over_last_in")
    rapid = (msi is not None and not (isinstance(msi, float) and math.isnan(msi)) and msi <= 10
             and ratio is not None and not (isinstance(ratio, float) and math.isnan(ratio)) and ratio >= 0.8)
    return PaymentContext(new_beneficiary=bool(f["new_pair"]), amount_deviation=bool(hist and usd > 3 * usual),
                          rapid_pass_through=bool(rapid), fan_in_24h=int(f["d_fanin_24h"]))


rows = []
def run(p, who, text=None, consent=False):
    out = eng.score_payment(p)
    f = out["features"]
    c = assess(text, ctx_from_features(f), consent=consent)
    comb = combine(out["decision"], c)
    rows.append({"who": who, "time": p["ts"], "from": p["from_account"], "to": p["to_account"], "amount_inr": p["amount_paid"],
                 "l1_score": out["l1_score"], "layer_exit": out.get("layer_exit"), "engine_risk": out.get("risk"), "engine_decision": out["decision"],
                 "context_points": {"context": c.context, "transaction": c.transaction, "graph": c.graph, "total": c.total},
                 "context_band": c.band, "context_reasons": c.reasons, "final_decision": comb["final_decision"], "raised_by_context": comb["raised_by_context"]})

M, C = F["mule"], F["collector"]
for i, amt in enumerate(F["deposits_inr"]):
    t = t0 + timedelta(minutes=i * F["gap_minutes"])
    v = f"V{i+1:02d}"
    base = {"payment_currency": "Rupee", "receiving_currency": "Rupee", "payment_format": "ACH"}
    p = {**base, "ts": t.strftime("%Y-%m-%d %H:%M"), "tmin": int(t.timestamp()) // 60, "from_bank": f"01{i+1:02d}", "from_account": v,
         "to_bank": M["bank"], "to_account": M["account"], "amount_paid": amt}
    run(p, "victim deposit", F["victim_message_for_V01"] if v == "V01" else None, consent=(v == "V01"))
    t2 = t + timedelta(minutes=1)
    q = {**base, "ts": t2.strftime("%Y-%m-%d %H:%M"), "tmin": int(t2.timestamp()) // 60, "from_bank": M["bank"], "from_account": M["account"],
         "to_bank": C["bank"], "to_account": C["account"], "amount_paid": round(amt * F["forward_share"], 2)}
    run(q, "mule onward transfer")

for r in rows:
    print(f"{r['time']} {r['who']:21s} {r['from']:6s}->{r['to']:6s} {r['amount_inr']:>8} L1 {r['l1_score']:.5f} engine {r['engine_decision']:20s} "
          f"ctx {r['context_points']['total']:3d} {r['context_band']:8s} final {r['final_decision']}")
from collections import Counter
summary = {"ruleset": RULESET_VERSION, "payments": len(rows),
           "engine_decisions": dict(Counter(r["engine_decision"] for r in rows)),
           "engine_l1_score_range": [min(r["l1_score"] for r in rows), max(r["l1_score"] for r in rows)],
           "context_bands": dict(Counter(r["context_band"] for r in rows)),
           "final_decisions": dict(Counter(r["final_decision"] for r in rows)),
           "raised_by_context": sum(r["raised_by_context"] for r in rows),
           "caveat": "One fictional fixture with no prior account history; it shows a gap and the veto mechanics, not a recall measurement."}
json.dump({"summary": summary, "rows": rows}, open(os.path.join(ROOT, "results", "scenario_results.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
