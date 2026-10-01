"""
RuseTect — layered filtration cascade, evaluated end-to-end on the held-out test period (last 20%).

  L1  Gatekeeper    tiny XGBoost (30 trees, depth 3) on causal graph features; drops the
                    obviously-clean traffic while keeping >= 99% of laundering (threshold set on val)
  L2  Graph-boost   XGBoost + LightGBM + RandomForest on the same features, averaged
  L3  Behaviour     LSTM + Transformer over sender/receiver event sequences, blended with L2
                    (blend weight chosen on val)
  L4  Ring linker   alerts joined into cases via shared accounts (union-find) -> one analyst case
                    per connected mule cluster; measured against the 370 ground-truth rings
Also saves the production artefacts to service/models/.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import glob, json, sys, time, pickle
import numpy as np, polars as pl
from common import load, to_np, train_sample, best_f1_threshold, evaluate, ALL, FMT_CODES, CUR_CODES
from sklearn.metrics import average_precision_score
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb, lightgbm as lgb

d, split = load()
y = d["y"].to_numpy().astype(np.int8)
tr = train_sample(y, split); va = np.where(split == 1)[0]; te = np.where(split == 2)[0]
Xtr, Xva, Xte = to_np(d[tr], ALL), to_np(d[va], ALL), to_np(d[te], ALL)
spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum()) ** 0.5
res = {}

# ---------------- L1 gatekeeper
t = time.time()
g = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.3, tree_method="hist", n_jobs=2, scale_pos_weight=spw).fit(Xtr, y[tr])
g_va, g_te = g.predict_proba(Xva)[:, 1], g.predict_proba(Xte)[:, 1]
pos_va = np.sort(g_va[y[va] == 1]); thr1 = pos_va[int(0.01 * len(pos_va))]   # keep 99% of val positives
keep_va, keep_te = g_va >= thr1, g_te >= thr1
res["L1"] = dict(threshold=float(thr1), pass_rate_test=float(keep_te.mean()), recall_test=float(y[te][keep_te].sum() / y[te].sum()),
                 dropped_txns=int((~keep_te).sum()))
print("L1", res["L1"], f"{time.time()-t:.0f}s", flush=True)

# ---------------- L2 graph-boost ensemble
t = time.time()
mx = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, scale_pos_weight=spw).fit(Xtr, y[tr])
ml = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, scale_pos_weight=spw).fit(Xtr, y[tr])
Xtr_i = np.nan_to_num(Xtr, nan=-1.0)
mr = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=2, class_weight="balanced_subsample", random_state=0).fit(Xtr_i, y[tr])
def rank01(p): return (np.argsort(np.argsort(p)) / (len(p) - 1)).astype(np.float32)
def l2(X):
    return np.vstack([mx.predict_proba(X)[:, 1], ml.predict_proba(X)[:, 1], mr.predict_proba(np.nan_to_num(X, nan=-1.0))[:, 1]])
P2_va, P2_te = l2(Xva), l2(Xte)
s2_va, s2_te = P2_va.mean(0), P2_te.mean(0)
print("L2 trained", f"{time.time()-t:.0f}s", flush=True)

# ---------------- L3 behaviour-sequence blend (preds from bench_deep.py, same val/test rows)
seq_va = np.mean([np.load(f"{WORK}/pred_{m}_va.npy") for m in ("LSTM", "Transformer")], 0)
seq_te = np.mean([np.load(f"{WORK}/pred_{m}_te.npy") for m in ("LSTM", "Transformer")], 0)
best = (-1, 0)
for w in np.linspace(0, 1, 11):
    s = (1 - w) * s2_va + w * seq_va
    ap = average_precision_score(y[va][keep_va], s[keep_va])
    if ap > best[0]: best = (ap, w)
w3 = float(best[1])
s3_va = np.where(keep_va, (1 - w3) * s2_va + w3 * seq_va, 0.0)
s3_te = np.where(keep_te, (1 - w3) * s2_te + w3 * seq_te, 0.0)
print("L3 blend weight on sequence models:", w3, flush=True)

# compare single models vs ensembles on the full test set
cmp = {}
for nm, pv, pt in [("XGBoost+graph", P2_va[0], P2_te[0]), ("L2 ensemble (XGB+LGBM+RF)", s2_va, s2_te),
                   ("L3 ensemble (L2 + LSTM/Transformer)", (1 - w3) * s2_va + w3 * seq_va, (1 - w3) * s2_te + w3 * seq_te),
                   ("Full cascade (L1 gate -> L2 -> L3)", s3_va, s3_te)]:
    r = evaluate(y[te], pt, best_f1_threshold(y[va], pv)); cmp[nm] = r
    print(f"{nm:40s} PR-AUC {r['pr_auc']:.3f} F1 {r['f1']:.3f} P {r['precision']:.3f} R {r['recall']:.3f} alerts {r['alerts']}", flush=True)
res["comparison"] = cmp
thr3 = best_f1_threshold(y[va], s3_va)

# decision bands (thresholds from validation): block+STR / hold / step-up verify / approve
pv = s3_va[y[va] == 1]
bands = {"block_str": float(thr3), "hold": float(np.quantile(s3_va[keep_va], 0.97)), "verify": float(thr1)}
res["bands"] = bands

# ---------------- L4 ring linker: connected components over alerted payments
alert = s3_te >= thr3
src = d["src_id"].to_numpy()[te]; dst = d["dst_id"].to_numpy()[te]
parent = {}
def find(a):
    while parent.setdefault(a, a) != a:
        parent[a] = parent[parent[a]]; a = parent[a]
    return a
for a, b in zip(src[alert], dst[alert]):
    ra, rb = find(a), find(b)
    if ra != rb: parent[ra] = rb
case_of_txn = np.array([find(a) for a in src[alert]])
n_alerts, n_cases = int(alert.sum()), int(len(np.unique(case_of_txn)))

# ground-truth rings from the Patterns file -> row ids
cols = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
raw = pl.concat([pl.read_csv(f, has_header=True, infer_schema=False, new_columns=cols) for f in
                 sorted(glob.glob(f"{DATA}/aml_HI-Small_part0*.csv.gz"))]).with_row_index("rid")
raw = raw.with_columns(pl.col("amt_recv").cast(pl.Float64).alias("amt")).select("rid", "ts", "fb", "fa", "tb", "ta", "amt")
pat = pl.read_parquet(f"{WORK}/patterns.parquet")
m = pat.join(raw, on=["ts", "fb", "fa", "tb", "ta", "amt"], how="inner")
print("pattern rows matched:", m.height, "of", pat.height, flush=True)
rid_te = d["rid"].to_numpy()[te]
pos_in_te = {int(r): i for i, r in enumerate(rid_te)}
ring_rows = {}
for ring, rid in zip(m["ring"].to_list(), m["rid"].to_list()):
    if rid in pos_in_te: ring_rows.setdefault(ring, []).append(pos_in_te[rid])
typ = dict(zip(pat["ring"].to_list(), pat["typ"].to_list()))
caught = {r: bool(alert[idx].any()) for r, idx in ring_rows.items()}
by_type = {}
for r, c in caught.items(): by_type.setdefault(typ[r], []).append(c)
res["L4"] = dict(alerts=n_alerts, cases=n_cases, alerts_per_case=round(n_alerts / max(n_cases, 1), 2),
                 rings_in_test=len(ring_rows), rings_caught=int(sum(caught.values())),
                 ring_recall=float(np.mean(list(caught.values()))) if caught else None,
                 ring_recall_by_type={k: round(float(np.mean(v)), 3) for k, v in sorted(by_type.items())},
                 true_positive_alerts=int((alert & (y[te] == 1)).sum()))
print("L4", json.dumps(res["L4"], indent=1), flush=True)

# funnel volumes
res["funnel"] = dict(test_txns=int(len(te)), after_L1=int(keep_te.sum()), alerts_after_L3=n_alerts, cases_after_L4=n_cases,
                     laundering_txns=int(y[te].sum()), laundering_after_L1=int(y[te][keep_te].sum()),
                     laundering_alerted=int((alert & (y[te] == 1)).sum()))

# ---------------- latency (single payment, CPU)
x1 = Xte[:1]
def lat(f, reps=200):
    t = time.perf_counter()
    for _ in range(reps): f()
    return (time.perf_counter() - t) / reps * 1000
res["latency_ms"] = dict(L1=round(lat(lambda: g.predict_proba(x1)), 2),
                         L2=round(lat(lambda: l2(x1), 50), 2))
print("latency", res["latency_ms"])
json.dump(res, open(f"{WORK}/results_cascade.json", "w"), indent=1)

# ---------------- export production artefacts
import os
os.makedirs(MODELS, exist_ok=True)
g.save_model(f"{MODELS}/l1_gatekeeper.json")
mx.save_model(f"{MODELS}/l2_xgb.json")
ml.booster_.save_model(f"{EXPORT}/l2_lgbm.txt")
json.dump(ml.booster_.dump_model(), open(f"{MODELS}/l2_lgbm.json", "w"), separators=(",", ":"))  # served by service/lgbm_lite.py
prev = json.load(open(f"{MODELS}/cascade_config.json"))
json.dump(dict(fmt_codes=FMT_CODES, cur_codes=CUR_CODES, usd_rates=prev["usd_rates"], features=ALL, l1_threshold=float(thr1), block_threshold=float(thr3), hold_threshold=bands["hold"],
               l3_sequence_weight=w3, note="RF and sequence models are batch-only; online path uses L1 + XGB + LGBM"),
          open(f"{MODELS}/cascade_config.json", "w"), indent=1)
print("saved models")
