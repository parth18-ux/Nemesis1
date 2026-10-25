"""Holdout and ablation for the frozen context ruleset (ctx-1.1; no rule edited since).

A. Message holdout: the curated Indian scam / legitimate corpus from ml/scam_corpus.py (written earlier for the L-1
   message model, never used to design the context rules), re-filled with a NEW seed (seed 7, 3 fills per template).
   Each message is paired with three payment settings so the message side decides the difference:
     worst  = new beneficiary + amount outside usual range      (the PDF's fixture setting)
     new    = new beneficiary only
     none   = known payee, usual amount
B. Payment ablation on real-shaped traffic: the last 60,000 payments of the bundled IBM AMLworld sample (no messages),
   with transaction and graph checks from the engine's live features; full layer vs leave one domain out.
Writes results/holdout_eval.json."""
import csv, calendar, gzip, json, os, sys, time
from collections import Counter, defaultdict
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "ml"))
from context_rules import assess, PaymentContext, RULESET_VERSION, keyword_baseline, from_engine_features, sender_fan_in_24h, band_for
from scam_corpus import generate
from rusetect_engine import Engine

ESC = ("HIGH", "CRITICAL")
out = {"ruleset": RULESET_VERSION, "frozen_at_commit": "e8e00a0"}

# ---------------------------------------------------------------- A. message holdout
msgs = generate(n_per_template=3, seed=7)
settings = {"worst": PaymentContext(new_beneficiary=True, amount_deviation=True), "new": PaymentContext(new_beneficiary=True), "none": PaymentContext()}
A = {"messages": len(msgs), "scam": sum(m[1] for m in msgs), "legitimate": sum(1 - m[1] for m in msgs), "by_setting": {}}
for sname, pay in settings.items():
    rows = [(lab, wf, assess(text, pay, True)) for text, lab, wf, _ in msgs]
    sc = [r for l, w, r in rows if l == 1]; lg = [r for l, w, r in rows if l == 0]
    A["by_setting"][sname] = {
        "scam_escalated": round(sum(r.band in ESC for r in sc) / len(sc), 3), "scam_verify_or_more": round(sum(r.band != "LOW" for r in sc) / len(sc), 3),
        "legit_escalated": round(sum(r.band in ESC for r in lg) / len(lg), 3), "legit_verify_or_more": round(sum(r.band != "LOW" for r in lg) / len(lg), 3)}
# keyword baseline and per workflow detail (worst setting)
kw_s = sum(keyword_baseline(t)["flag"] for t, l, *_ in msgs if l == 1); kw_l = sum(keyword_baseline(t)["flag"] for t, l, *_ in msgs if l == 0)
A["keyword_baseline"] = {"scam_flagged": round(kw_s / A["scam"], 3), "legit_flagged": round(kw_l / A["legitimate"], 3)}
per = defaultdict(lambda: [0, 0])
for text, lab, wf, _ in msgs:
    r = assess(text, settings["worst"], True); per[(lab, wf)][0] += 1; per[(lab, wf)][1] += r.band in ESC
A["worst_by_workflow"] = sorted([{"label": "scam" if l else "legitimate", "workflow": w, "n": n, "escalated": e, "rate": round(e / n, 2)} for (l, w), (n, e) in per.items()],
                                key=lambda d: (d["label"], -d["rate"]))
out["A_message_holdout"] = A
print("A", json.dumps({k: v for k, v in A.items() if k != "worst_by_workflow"}, indent=1))

# ---------------------------------------------------------------- B. payment ablation on IBM sample
rows = list(csv.DictReader(gzip.open(os.path.join(ROOT, "service", "models", "demo_stream.csv.gz"), "rt")))
eng = Engine(load_models=False); eng.reset(); test_from = len(rows) - 60000
recs = []
for i, r in enumerate(rows):
    p = dict(from_bank=r["fb"], from_account=r["fa"], to_bank=r["tb"], to_account=r["ta"], amount_paid=float(r["amt_paid"]), payment_currency=r["cur_paid"],
             receiving_currency=r["cur_recv"], payment_format=r["fmt"], tmin=calendar.timegm(time.strptime(r["ts"], "%Y/%m/%d %H:%M")) // 60)
    sf = sender_fan_in_24h(eng, p) if i >= test_from else 0
    f = eng.features(p)
    if i >= test_from:
        recs.append((int(r["y"]), assess(None, from_engine_features(f, sf), True)))
def variant(drop):
    res = Counter()
    for y, r in recs:
        parts = {"transaction": r.transaction, "graph": r.graph}
        if drop: parts[drop] = 0
        tot = sum(parts.values()); nd = sum(v > 0 for v in parts.values())
        b = band_for(tot, nd)
        if r.band == "MEDIUM" and "account graph: sender_mule_signature" in r.reasons and drop != "graph": b = max(b, "MEDIUM", key=["LOW", "MEDIUM", "HIGH", "CRITICAL"].index)
        res[(y, b != "LOW")] += 1
    pos = res[(1, True)] + res[(1, False)]; flagged = res[(1, True)] + res[(0, True)]
    return {"flagged_verify_or_more": flagged, "flag_rate": round(flagged / len(recs), 4), "laundering_caught": res[(1, True)], "laundering": pos,
            "recall": round(res[(1, True)] / pos, 3) if pos else None, "precision": round(res[(1, True)] / flagged, 3) if flagged else None}
B = {"payments": len(recs), "full": variant(None), "without_transaction": variant("transaction"), "without_graph": variant("graph"),
     "note": "No messages in IBM data, so the context domain is empty; single-domain payments cap at MEDIUM by contract."}
out["B_payment_ablation_ibm"] = B
print("B", json.dumps(B, indent=1))
out["caveats"] = ["Message corpus is synthetic and template based; real reported messages are needed before trusting rates.",
                  "IBM AMLworld is synthetic; bundled sample has only 100,000 payments of history."]
json.dump(out, open(os.path.join(ROOT, "results", "holdout_eval.json"), "w"), indent=1)
