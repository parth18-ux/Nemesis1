"""Finale Part 5: unseen scam replay demo. One message from each scam workflow the 12 fixtures never covered, plus
legitimate controls, each followed by a payment from the recipient to a new payee, sent through POST /api/v2/assess
(payment cascade + context layer, intent model on). Prints a demo table; writes results/demo_unseen.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "ml"))
from fastapi.testclient import TestClient
from index import app
from scam_corpus import generate
UNSEEN = ["task_job", "sextortion", "utility_disconnect", "parcel_customs", "fake_loan", "lottery_prize", "govt_scheme", "reward_points", "family_emergency"]
LEGIT = ["bank_notice", "otp", "personal", "utility_bill", "delivery"]
msgs = generate(n_per_template=1, seed=21); pick = {}
for text, lab, wf, tid in msgs:
    if (wf in UNSEEN or wf in LEGIT) and wf not in pick: pick[wf] = (text, lab)
c = TestClient(app); c.post("/api/v2/demo/reset", params={"warm": False}); rows = []
for i, wf in enumerate(UNSEEN + LEGIT):
    text, lab = pick[wf]
    r = c.post("/api/v2/assess", json={"timestamp": f"2022-09-10 12:{i:02d}", "from_bank": "0500", "from_account": f"U{i:02d}", "to_bank": "0600",
                                       "to_account": f"NEWPAYEE{i:02d}", "amount_paid": 25000, "payment_currency": "Rupee", "payment_format": "ACH",
                                       "message": text, "consent": True}).json()
    rows.append({"workflow": wf, "label": "scam" if lab else "legitimate", "message": r["context"]["redacted_text"], "payment_decision": r["payment_decision"],
                 "context_band": r["context_band"], "context_points": r["context"]["total"], "final": r["shown_decision"], "reasons": r["context"]["reasons"]})
    print(f"{wf:20s} {rows[-1]['label']:10s} payment {r['payment_decision']:21s} context {r['context_band']:8s} ({r['context']['total']:3d}) final {r['shown_decision']}")
sc = [r for r in rows if r["label"] == "scam"]; lg = [r for r in rows if r["label"] == "legitimate"]
summary = {"scams_needing_a_human_or_customer_check": f"{sum(r['final'] != 'Approve' for r in sc)}/{len(sc)}",
           "legit_needing_a_check": f"{sum(r['final'] != 'Approve' for r in lg)}/{len(lg)}",
           "note": "One fill per workflow from the synthetic corpus (seed 21). The served message model was trained on these templates, so this is a demo of the flow, not a measurement; see results/fusion_oof.json for out-of-fold numbers."}
json.dump({"summary": summary, "rows": rows}, open(os.path.join(ROOT, "results", "demo_unseen.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
