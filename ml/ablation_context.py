"""Part 2 ablation of the context layer on the 12 fixtures and the 9 post-freeze probes.
Variants: A = ctx-1.0 (no negation, no model), B = + negation, C = + intent model, D = + negation + intent model.
NOTE: the probes were looked at while designing ctx-1.1, so they are now development data, not a clean holdout.
Writes results/context_ablation.json."""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from context_rules import assess, PaymentContext
from scam_text import MessageModel
MM = MessageModel()
FX = json.load(open(os.path.join(ROOT, "fixtures", "context_scenarios.json")))["scenarios"]
PR = json.load(open(os.path.join(ROOT, "fixtures", "challenge_probes.json")))
VAR = {"A ctx-1.0": dict(negation=False, use_model=False), "B + negation": dict(negation=True, use_model=False),
       "C + intent model": dict(negation=False, use_model=True), "D + negation + model": dict(negation=True, use_model=True)}
items = [(s["id"], "fixture", s["label"], s["text"], s["payment"], s["consent"]) for s in FX] + \
        [(p["id"], "probe", p["label"], p["text"], PR["payment"], True) for p in PR["probes"]]
out = {"note": "probes are development data after ctx-1.1 design; a fresh frozen holdout is Part 4", "variants": {}}
for name, v in VAR.items():
    rows = []
    for iid, kind, lab, text, pay, cons in items:
        mb = MM.score(text)["band"] if (v["use_model"] and text) else None
        r = assess(text, PaymentContext(**pay), cons, negation=v["negation"], model_band=mb)
        rows.append({"id": iid, "kind": kind, "label": lab, "total": r.total, "band": r.band})
    def stat(kind):
        sc = [r for r in rows if r["kind"] == kind and r["label"] == "scam"]; lg = [r for r in rows if r["kind"] == kind and r["label"] == "legitimate"]
        return {"scams_escalated": f"{sum(r['band'] in ('HIGH','CRITICAL') for r in sc)}/{len(sc)}",
                "scams_verify_or_more": f"{sum(r['band'] != 'LOW' for r in sc)}/{len(sc)}",
                "legit_escalated": f"{sum(r['band'] in ('HIGH','CRITICAL') for r in lg)}/{len(lg)}",
                "legit_low": f"{sum(r['band'] == 'LOW' for r in lg)}/{len(lg)}"}
    out["variants"][name] = {"fixtures": stat("fixture"), "probes": stat("probe"), "rows": rows}
    print(f"{name:22s} fixtures {stat('fixture')}\n{'':22s} probes   {stat('probe')}")
json.dump(out, open(os.path.join(ROOT, "results", "context_ablation.json"), "w"), indent=1)
