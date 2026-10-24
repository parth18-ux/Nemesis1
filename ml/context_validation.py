"""Run the 12 Round 1 context fixtures through service/context_rules.py and the 5-term keyword baseline.
Writes results/context_results.json and prints a table. Exit code 1 if any fixture differs from the PDF table."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "service"))
from context_rules import assess, PaymentContext, RULESET_VERSION

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fx = json.load(open(os.path.join(ROOT, "fixtures", "context_scenarios.json")))["scenarios"]
rows, mismatches = [], []
for s in fx:
    r = assess(s["text"], PaymentContext(**s["payment"]), consent=s["consent"])
    got = {"context": r.context, "transaction": r.transaction, "graph": r.graph, "total": r.total, "band": r.band}
    ok = got == s["expected"]
    if not ok:
        mismatches.append({"id": s["id"], "expected": s["expected"], "got": got})
    rows.append({"id": s["id"], "name": s["name"], "label": s["label"], **got, "action": r.action, "reasons": r.reasons,
                 "keyword_baseline_flag": r.keyword_baseline.get("flag", False), "redacted_text": r.redacted_text, "matches_pdf": ok})
    print(f"{s['id']:4s} {s['name']:30s} ctx {r.context:3d} txn {r.transaction:3d} graph {r.graph:3d} total {r.total:3d} {r.band:9s} "
          f"keyword={'Y' if r.keyword_baseline.get('flag') else 'n'} {'OK' if ok else 'DIFF'}")
scams = [r for r in rows if r["label"] == "scam"]; legit = [r for r in rows if r["label"] == "legitimate"]
summary = {"ruleset": RULESET_VERSION, "fixtures": len(rows), "matches_pdf_table": len(rows) - len(mismatches),
           "scams_escalated_high_or_critical": sum(r["band"] in ("HIGH", "CRITICAL") for r in scams),
           "scams_verification_medium": sum(r["band"] == "MEDIUM" for r in scams), "scams": len(scams),
           "legitimate_low": sum(r["band"] == "LOW" for r in legit), "legitimate": len(legit),
           "keyword_baseline_scams_flagged": sum(r["keyword_baseline_flag"] for r in scams),
           "keyword_baseline_legitimate_flagged": sum(r["keyword_baseline_flag"] for r in legit),
           "caveat": "Co-designed fictional fixtures: they show the mechanics of the contract, not generalisation."}
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
json.dump({"summary": summary, "rows": rows, "mismatches": mismatches}, open(os.path.join(ROOT, "results", "context_results.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
sys.exit(1 if mismatches else 0)
