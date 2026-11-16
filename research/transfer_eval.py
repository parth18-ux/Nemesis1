"""Cross-generator and scale transfer for the served model and the HI-Small-trained models from full_hi_eval.py.

    python3 -I research/transfer_eval.py <name> <dump.csv> <n_rows> [laundering_types_sidecar]

Reads the dump in chunks (works for HI-Medium, 32M payments). Chronological split by row: first 60% train, last 20%
test. Reports, on the test rows:
  * the served cascade zero-shot, and the HI-Small control (31 features) and no-format candidate zero-shot
  * the same two recipes retrained in-domain (all positives + a share of negatives from the train rows)
  * laundering rate by payment format (the shortcut check), and per typology recall when a sidecar is given
Results: research/out/transfer_<name>.json
"""
import json, os, sys
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "out")
CFG = json.load(open(os.path.join(ROOT, "service", "models", "cascade_config.json")))
F = CFG["features"]; FN = [f for f in F if f != "fmt_code"]; L1T = CFG["l1_threshold"]
RNG = np.random.default_rng(5)
PARAMS = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "tree_method": "hist", "nthread": 4}


def m(y, s):
    o = {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": round(float(y.mean()), 5),
         "pr_auc": round(float(AP(y, s)), 4), "roc_auc": round(float(AUC(y, s)), 4)}
    k = max(1, int(len(y) * 0.005))  # an alert budget of 0.5% of payments
    top = np.argsort(-s)[:k]
    o["recall_at_0.5pct_alerts"] = round(float(y[top].sum() / max(1, y.sum())), 4)
    o["precision_at_0.5pct_alerts"] = round(float(y[top].mean()), 4)
    return o


def main():
    name, dump, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
    types = sys.argv[4] if len(sys.argv) > 4 else None
    tr_end, te0 = int(n * 0.6), int(n * 0.8)
    neg_share = 0.1 if n < 15_000_000 else 0.03
    trX, trY, teX, teY, te_rows, fmt_rows = [], [], [], [], [], []
    cols = ["row", "y", "fmt", "l1", "l2x", "l2l"] + F
    for ch in pd.read_csv(dump, usecols=cols, chunksize=1_000_000, dtype={c: np.float32 for c in cols if c != "fmt"} | {"fmt": str}):
        r = ch.row.values.astype(np.int64)
        fmt_rows.append(ch.groupby("fmt").y.agg(["count", "sum"]))
        a = ch[r < tr_end]
        keep = (a.y.values == 1) | (RNG.random(len(a)) < neg_share)
        trX.append(a[F].values[keep]); trY.append(a.y.values[keep])
        b = ch[r >= te0]
        teX.append(b[F].values); teY.append(b.y.values)
        te_rows.append(b[["row", "l1", "l2x", "l2l"]].values)
    trX, trY = np.vstack(trX), np.concatenate(trY).astype(int)
    Xt, yt = np.vstack(teX), np.concatenate(teY).astype(int)
    tr = np.vstack(te_rows)
    served = np.where(tr[:, 1] < L1T, tr[:, 1], (tr[:, 2] + tr[:, 3]) / 2)
    fm = pd.concat(fmt_rows).groupby(level=0).sum()
    R = {"dataset": name, "rows": n, "split": "rows < 60% train, >= 80% test (chronological)",
         "laundering_rate_by_format": {k: {"payments": int(v["count"]), "laundering": int(v["sum"]),
                                           "rate": round(float(v["sum"] / v["count"]), 5)} for k, v in fm.iterrows()},
         "served_zero_shot": m(yt, served)}
    fi = [F.index(f) for f in FN]
    for label, path, feats, ix in (("hi_small_control_zero_shot", "control_31_xgb.json", F, list(range(len(F)))),
                                   ("hi_small_no_format_zero_shot", "candidate_no_format_xgb.json", FN, fi)):
        b = xgb.Booster(); b.load_model(os.path.join(OUT, path))
        R[label] = m(yt, b.predict(xgb.DMatrix(Xt[:, ix], missing=np.nan, feature_names=feats)))
    inds = {}
    for label, feats, ix in (("in_domain_control", F, list(range(len(F)))), ("in_domain_no_format", FN, fi)):
        b = xgb.train(dict(PARAMS, seed=0), xgb.DMatrix(trX[:, ix], trY, missing=np.nan, feature_names=feats), 400)
        s = b.predict(xgb.DMatrix(Xt[:, ix], missing=np.nan, feature_names=feats))
        inds[label] = s
        R[label] = m(yt, s)
        imp = b.get_score(importance_type="gain")
        R[label]["top_features"] = [k for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:6]]
    if types:
        tt = pd.read_csv(types, header=None)[0].values[te0:te0 + len(yt)]
        k = max(1, int(len(yt) * 0.005)); top = np.zeros(len(yt), bool); top[np.argsort(-inds["in_domain_no_format"])[:k]] = True
        topc = np.zeros(len(yt), bool); topc[np.argsort(-inds["in_domain_control"])[:k]] = True
        R["typology_recall_at_0.5pct_alerts"] = {t: {"laundering": int(((tt == t) & (yt == 1)).sum()),
                                                     "in_domain_control": round(float(topc[(tt == t) & (yt == 1)].mean()), 3),
                                                     "in_domain_no_format": round(float(top[(tt == t) & (yt == 1)].mean()), 3)}
                                                 for t in sorted(set(tt[yt == 1]))}
    json.dump(R, open(os.path.join(OUT, f"transfer_{name}.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in R.items() if k != "typology_recall_at_0.5pct_alerts"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
