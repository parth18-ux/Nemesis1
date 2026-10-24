"""How often does the ctx-1.1 sender mule signature fire on ordinary traffic, and how precise is it?
Streams the bundled IBM AMLworld sample (service/models/demo_stream.csv.gz: 100,000 history + 60,000 test-period payments)
through the engine's live feature state and counts the signature on the last 60,000. Writes results/mule_signature_check.json."""
import csv, calendar, gzip, json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from rusetect_engine import Engine
from context_rules import from_engine_features, sender_fan_in_24h, graph_points

rows = list(csv.DictReader(gzip.open(os.path.join(ROOT, "service", "models", "demo_stream.csv.gz"), "rt")))
eng = Engine(load_models=False); eng.reset()
test_from = len(rows) - 60000
n = flag = flag_pos = pos = rapid = 0
for i, r in enumerate(rows):
    p = dict(from_bank=r["fb"], from_account=r["fa"], to_bank=r["tb"], to_account=r["ta"], amount_paid=float(r["amt_paid"]),
             payment_currency=r["cur_paid"], receiving_currency=r["cur_recv"], payment_format=r["fmt"],
             tmin=calendar.timegm(time.strptime(r["ts"], "%Y/%m/%d %H:%M")) // 60)
    sf = sender_fan_in_24h(eng, p) if i >= test_from else 0
    f = eng.features(p)
    if i < test_from:
        continue
    n += 1; y = int(r["y"]); pos += y
    c = from_engine_features(f, sf); rapid += c.rapid_pass_through
    if "sender_mule_signature" in graph_points(c)["checks"]:
        flag += 1; flag_pos += y
out = {"test_payments": n, "laundering": pos, "base_rate": pos / n, "rapid_pass_through": rapid,
       "signature_flagged": flag, "signature_flag_rate": flag / n, "laundering_among_flagged": flag_pos,
       "precision": flag_pos / flag if flag else None, "recall": flag_pos / pos if pos else None,
       "caveat": "Bundled 100k-payment history only, so account history is shorter than in a bank; synthetic data."}
json.dump(out, open(os.path.join(ROOT, "results", "mule_signature_check.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
