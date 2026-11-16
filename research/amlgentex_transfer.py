"""Cross generator experiment: IBM AMLworld (HI-Small) models and features on AMLgentex (Östman et al., 2025).

Data: Hugging Face AMLGentex/sweden_100K_difficult and AMLGentex/sweden_100K_easy, file tx_log.csv (MIT licence on
the dataset cards). Download them and pass their folder:

    python3 -I research/amlgentex_transfer.py <folder with sweden_difficult.csv and sweden_easy.csv>

Mapping to the AMLworld schema used by the live feature code (stated, not hidden):
  * INITALBALANCE rows (simulation set-up) are dropped; TRANSFER and CASH rows are kept, in file order (sorted by step)
  * timestamp = 2022/09/01 00:00 + step days. AMLgentex has daily steps, so every gap and 24 h window feature works at
    day resolution, unlike AMLworld minutes; this is part of the generator gap being measured
  * payment format: TRANSFER -> ACH, CASH -> Cash (a sensitivity run re-scores TRANSFER as Wire)
  * currency: US Dollar for all (AMLgentex amounts are in one unnamed currency)
  * label: isSAR

Experiments
  1. zero-shot: the served HI-Small cascade (L1 gate, mean of L2 halves) on AMLgentex, last 20% of steps
  2. in-domain reference: XGBoost retrained on the AMLgentex features with a chronological split (first 60% of steps
     train, last 20% test), 3 seeds
  3. easy -> difficult: trained on easy, tested on difficult
  4. AMLgentex -> AMLworld: the difficult-trained model on the HI-Small evaluation window of the bundled slice
Results: research/out/amlgentex_transfer.json
"""
import gzip, json, os, subprocess, sys
import numpy as np, pandas as pd, xgboost as xgb, lightgbm as lgb
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "target", "release", "examples", "research_dump")
CFG = json.load(open(os.path.join(ROOT, "service", "models", "cascade_config.json")))
L1T = CFG["l1_threshold"]
RNG = np.random.default_rng(7)


def convert(src, dst):
    d = pd.read_csv(src, usecols=["step", "type", "amount", "nameOrig", "bankOrig", "nameDest", "bankDest", "isSAR"])
    d = d[d.type != "INITALBALANCE"].reset_index(drop=True)
    day = pd.Timestamp("2022-09-01") + pd.to_timedelta(d.step, unit="D")
    amt = d.amount.map(lambda x: f"{x:.2f}")
    out = pd.DataFrame({"ts": day.dt.strftime("%Y/%m/%d %H:%M"), "fb": d.bankOrig.astype(str), "fa": d.nameOrig.astype(str),
                        "tb": d.bankDest.astype(str), "ta": d.nameDest.astype(str), "amt_recv": amt, "cur_recv": "US Dollar",
                        "amt_paid": amt, "cur_paid": "US Dollar", "fmt": np.where(d.type == "CASH", "Cash", "ACH"),
                        "y": d.isSAR.astype(int)})
    out.to_csv(dst, index=False, compression="gzip")
    return d.step.values, d.type.values


def dump(name, folder):
    csv = os.path.join(folder, f"sweden_{name}.csv")
    gz = os.path.join(folder, f"sweden_{name}_aml.csv.gz")
    out = os.path.join(folder, f"sweden_{name}_dump.csv")
    step, typ = convert(csv, gz)
    if not os.path.exists(out):
        subprocess.run([BIN, gz, out], check=True)
    d = pd.read_csv(out, dtype=np.float32, usecols=lambda c: c not in ("fmt",))
    d["step"] = step; d["type"] = typ
    return d


def models():
    b1 = xgb.Booster(); b1.load_model(os.path.join(ROOT, "service/models/l1_gatekeeper.json"))
    bx = xgb.Booster(); bx.load_model(os.path.join(ROOT, "service/models/l2_xgb.json"))
    bl = lgb.Booster(model_file=os.path.join(ROOT, "ml/exported/l2_lgbm.txt"))
    return b1, bx, bl


def live(X, F, m):
    b1, bx, bl = m
    dm = xgb.DMatrix(X, missing=np.nan, feature_names=F)
    s1 = b1.predict(dm)
    return np.where(s1 < L1T, s1, (bx.predict(dm) + bl.predict(X.astype(np.float64))) / 2)


def metrics(y, s, nb=200):
    out = {"rows": int(len(y)), "positives": int(y.sum()), "prevalence": round(float(y.mean()), 5),
           "pr_auc": round(float(AP(y, s)), 4), "roc_auc": round(float(AUC(y, s)), 4)}
    n = len(y); m = min(n, 400_000)
    aps, aus = [], []
    for _ in range(nb):
        ix = RNG.integers(0, n, m)
        if 0 < y[ix].sum() < m:
            aps.append(AP(y[ix], s[ix])); aus.append(AUC(y[ix], s[ix]))
    out["pr_auc_ci95"] = [round(float(np.percentile(aps, 2.5)), 4), round(float(np.percentile(aps, 97.5)), 4)]
    out["roc_auc_ci95"] = [round(float(np.percentile(aus, 2.5)), 4), round(float(np.percentile(aus, 97.5)), 4)]
    out["note_ci"] = f"bootstrap of {m:,} row resamples" if m < n else "row bootstrap"
    return out


def split(d):
    s = d.step.values; hi = s.max()
    return s < 0.6 * (hi + 1), s >= 0.8 * (hi + 1)


def main():
    folder = sys.argv[1]
    m = models()
    F = CFG["features"]
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
              "tree_method": "hist", "nthread": 4}
    R = {"data": "AMLGentex/sweden_100K_{difficult,easy} tx_log.csv from Hugging Face; mapping in the script docstring",
         "split": "chronological by step: first 60% of steps train, last 20% test"}
    trained = {}
    D = {}
    for name in ("difficult", "easy"):
        d = dump(name, folder)
        D[name] = d
        X = d[F].values.astype(np.float32); y = d.y.values.astype(int)
        tr, te = split(d)
        r = {"rows_total": int(len(d)), "sar_total": int(y.sum()), "steps": int(d.step.max()) + 1,
             "sar_rate_by_type": {t: round(float(y[d.type.values == t].mean()), 4) for t in np.unique(d.type.values)}}
        s = live(X, F, m)
        r["zero_shot_served_model_test"] = metrics(y[te], s[te])
        tt = te & (d.type.values == "TRANSFER")
        r["zero_shot_served_model_test_transfers_only"] = metrics(y[tt], s[tt])
        Xw = X.copy(); Xw[d.type.values == "TRANSFER", F.index("fmt_code")] = CFG["fmt_codes"]["Wire"]
        r["zero_shot_transfer_as_wire_test"] = metrics(y[te], live(Xw, F, m)[te])
        aps = []
        for seed in range(3):
            pos = y[tr].mean()
            b = xgb.train(dict(params, seed=seed, scale_pos_weight=float(min(10.0, (1 - pos) / pos))),
                          xgb.DMatrix(X[tr], y[tr], missing=np.nan, feature_names=F), 300)
            p = b.predict(xgb.DMatrix(X[te], missing=np.nan, feature_names=F))
            aps.append(metrics(y[te], p, nb=50))
            if seed == 0:
                trained[name] = b
                imp = b.get_score(importance_type="gain")
                r["in_domain_top_features_gain"] = [k for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:8]]
        r["in_domain_xgboost_test"] = {"pr_auc_seeds": [a["pr_auc"] for a in aps], "roc_auc_seeds": [a["roc_auc"] for a in aps],
                                       "pr_auc_mean": round(float(np.mean([a["pr_auc"] for a in aps])), 4),
                                       "roc_auc_mean": round(float(np.mean([a["roc_auc"] for a in aps])), 4),
                                       "prevalence": aps[0]["prevalence"], "positives": aps[0]["positives"]}
        R[name] = r
    # easy -> difficult
    d = D["difficult"]; _, te = split(d)
    p = trained["easy"].predict(xgb.DMatrix(d[F].values.astype(np.float32)[te], missing=np.nan, feature_names=F))
    R["easy_trained_on_difficult_test"] = metrics(d.y.values.astype(int)[te], p)
    # AMLgentex -> AMLworld slice evaluation window
    h = pd.read_csv(os.path.join(ROOT, "research", "out", "streams", "base_dump.csv"))
    h = h[h.row >= 100_000]
    Xh = h[F].values.astype(np.float32)
    p = trained["difficult"].predict(xgb.DMatrix(Xh, missing=np.nan, feature_names=F))
    R["amlgentex_difficult_trained_on_hi_slice_eval"] = metrics(h.y.values.astype(int), p, nb=500)
    R["served_model_on_hi_slice_eval_reference"] = metrics(h.y.values.astype(int), live(Xh, F, m), nb=500)
    json.dump(R, open(os.path.join(ROOT, "research", "out", "amlgentex_transfer.json"), "w"), indent=1)
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
