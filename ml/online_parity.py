"""Warm the streaming engine with all pre-test history, then score the first 60K test payments
live and compare with the offline L2 scores (same models)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import glob, sys, time, calendar, json
import numpy as np, polars as pl, xgboost as xgb, lightgbm as lgb
sys.path.insert(0, os.path.join(REPO, "service"))
from rusetect_engine import Engine
from common import load, to_np, ALL
from sklearn.metrics import average_precision_score
d, split = load(); n = d.height; c80 = int(.8 * n); M = 60_000
cols = ["ts","fb","fa","tb","ta","amt_recv","cur_recv","amt_paid","cur_paid","fmt","y"]
raw = pl.concat([pl.read_csv(f, has_header=True, infer_schema=False, new_columns=cols) for f in
                 sorted(glob.glob(f"{DATA}/aml_HI-Small_part0*.csv.gz"))]).with_row_index("rid")
raw = d.select("rid", "order").head(c80 + M).join(raw, on="rid").sort("order")
eng = Engine(); t = time.time(); live = []
for i, r in enumerate(raw.iter_rows(named=True)):
    p = dict(from_bank=r["fb"], from_account=r["fa"], to_bank=r["tb"], to_account=r["ta"], amount_paid=float(r["amt_paid"]),
             payment_currency=r["cur_paid"], receiving_currency=r["cur_recv"], payment_format=r["fmt"],
             tmin=calendar.timegm(time.strptime(r["ts"], "%Y/%m/%d %H:%M")) // 60)
    if i < c80: eng.features(p)
    else: live.append(eng.score_payment(p))
    if i % 500000 == 0: print(i, f"{time.time()-t:.0f}s", flush=True)
    if i == c80: print(f"warmed on {c80} payments in {time.time()-t:.0f}s", flush=True)
lat = np.array([r["latency_ms"] for r in live])
risk_live = np.array([r["risk"] if r["layer_exit"] == "L2" else 0.0 for r in live])
# offline reference with identical models on offline features
X = to_np(d[c80:c80 + M], ALL); dm = xgb.DMatrix(X, feature_names=ALL)
g = eng.l1.predict(dm); off = np.where(g >= eng.cfg["l1_threshold"], (eng.l2x.predict(dm) + eng.l2l.predict(X)) / 2, 0.0)
y = d["y"].to_numpy()[c80:c80 + M]
out = dict(test_payments_scored=M, laundering_in_slice=int(y.sum()),
           max_abs_risk_diff=float(np.abs(risk_live - off).max()), mean_abs_risk_diff=float(np.abs(risk_live - off).mean()),
           same_decision_rate=float(((risk_live >= eng.cfg["block_threshold"]) == (off >= eng.cfg["block_threshold"])).mean()),
           pr_auc_live=float(average_precision_score(y, risk_live)), pr_auc_offline=float(average_precision_score(y, off)),
           latency_ms_p50=float(np.percentile(lat, 50)), latency_ms_p99=float(np.percentile(lat, 99)),
           l1_exit_rate=float(np.mean([r["layer_exit"] == "L1" for r in live])))
print(json.dumps(out, indent=1)); json.dump(out, open(f"{WORK}/results_online.json", "w"), indent=1)
