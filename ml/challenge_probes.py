"""Run post-freeze challenge probes through the frozen context rules, the 5-term keyword baseline and, for
comparison, the trained L-1 message model already in the repo. Reports every outcome, including failures.
Writes results/challenge_results.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from context_rules import assess, PaymentContext, RULESET_VERSION
try:
    from scam_text import MessageModel
    MM = MessageModel()
except Exception as e:          # the message model is optional for this script
    MM = None; print("message model not loaded:", e)

P = json.load(open(os.path.join(ROOT, "fixtures", "challenge_probes.json")))
pay = PaymentContext(**P["payment"]); rows = []
for p in P["probes"]:
    r = assess(p["text"], pay, consent=True)
    want_escalate = p["label"] == "scam"
    escalated = r.band in ("HIGH", "CRITICAL")
    m = MM.score(p["text"]) if MM else None
    rows.append({"id": p["id"], "label": p["label"], "kind": p["kind"], "context": r.context, "total": r.total, "band": r.band,
                 "families": [x.split(": ")[1] for x in r.reasons if x.startswith("message cue")],
                 "correct": escalated == want_escalate, "keyword_baseline": r.keyword_baseline["flag"],
                 "message_model": ({k: m[k] for k in ("risk", "band") if k in m} if m else None)})
    print(f"{p['id']} {p['label']:10s} {p['kind']:28s} ctx {r.context:2d} total {r.total:3d} {r.band:8s} {'ok ' if rows[-1]['correct'] else 'FAIL'} "
          f"kw={'Y' if r.keyword_baseline['flag'] else 'n'} model={rows[-1]['message_model']}")
sc = [r for r in rows if r["label"] == "scam"]; lg = [r for r in rows if r["label"] == "legitimate"]
summary = {"ruleset": RULESET_VERSION, "frozen_before_probes": True,
           "scams_escalated": f"{sum(r['band'] in ('HIGH','CRITICAL') for r in sc)}/{len(sc)}",
           "legitimate_wrongly_escalated": f"{sum(r['band'] in ('HIGH','CRITICAL') for r in lg)}/{len(lg)}",
           "keyword_baseline_scams": f"{sum(r['keyword_baseline'] for r in sc)}/{len(sc)}",
           "keyword_baseline_legitimate_flagged": f"{sum(r['keyword_baseline'] for r in lg)}/{len(lg)}"}
json.dump({"summary": summary, "rows": rows}, open(os.path.join(ROOT, "results", "challenge_results.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
