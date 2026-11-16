"""Full IBM AMLworld HI-Small evaluation and a format-robust retrain, under a rule fixed before test is read.

Data: HI-Small_Trans.csv and HI-Small_Patterns.txt (Hugging Face mirror OsamaMIT/IBM-AML-HI-Small, CDLA-Sharing-1.0;
checked against the repository: 5,078,345 payments, 5,177 laundering, and the bundled demo slice matches rows
3,962,676 to 4,122,675 of the time sorted file exactly). Features come from research_dump (served Rust engine).

    python3 -I research/full_hi_eval.py <hi_small_sorted.csv.gz> <hi_dump.csv> <HI-Small_Patterns.txt>

Splits as in the repository: rows < 3,047,007 build state and train; 3,047,007 to 4,062,675 validation; the rest test.

ADOPTION RULE (fixed here before any test number is computed; the script evaluates validation, records the decision,
and only then scores test):
  candidate = one XGBoost on the 30 live features without fmt_code, isotonic calibrated on validation, no L1 gate
  adopt if, on HI-Small validation,
    R1  candidate PR-AUC >= served live score PR-AUC - 0.03                                (keeps accuracy)
    R2  candidate worst PR-AUC when every laundering payment is re-sent in another format
        >= 0.8 x candidate clean PR-AUC                                                     (robust to the attack)
    R3  served worst PR-AUC under the same attack < 0.5 x served clean PR-AUC              (the attack is real)
Results: research/out/full_hi_eval.json
"""
import json, os, sys, time
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "service", "models", "cascade_config.json")))
L1T, VER, HOLD, BLK = CFG["l1_threshold"], CFG["verify_threshold"], CFG["hold_threshold"], CFG["block_threshold"]
VAL0, TEST0 = 3_047_007, 4_062_676
F = CFG["features"]
FNOFMT = [f for f in F if f != "fmt_code"]
RULE = {"R1_max_drop": 0.03, "R2_min_ratio": 0.8, "R3_served_max_ratio": 0.5}
RNG = np.random.default_rng(11)
PARAMS = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 1, "tree_method": "hist", "eval_metric": "aucpr", "nthread": 4}
ROUNDS = 400


def boot_ap(y, s, n=200):
    idx_p, idx_n = np.where(y == 1)[0], np.where(y == 0)[0]
    out = []
    for _ in range(n):  # stratified: keeps prevalence fixed, resamples both classes
        ix = np.concatenate([RNG.choice(idx_p, len(idx_p)), RNG.choice(idx_n, len(idx_n))])
        out.append(AP(y[ix], s[ix]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def pr(y, s, nb=200):
    return {"pr_auc": round(float(AP(y, s)), 4), "ci95": boot_ap(y, s, nb), "roc_auc": round(float(AUC(y, s)), 4),
            "positives": int(y.sum()), "rows": int(len(y))}


def band_counts(y, s, thr):
    f = s >= thr
    return {"flags": int(f.sum()), "tp": int((f & (y == 1)).sum()), "precision": round(float(y[f].mean()), 4) if f.any() else None,
            "recall": round(float((f & (y == 1)).sum() / max(1, y.sum())), 4)}


def thr_for_rate(s, rate):
    return float(np.quantile(s, 1 - rate))


def main():
    sorted_csv, dump, patterns = sys.argv[1:4]
    t0 = time.time()
    meta = pd.read_csv(sorted_csv, usecols=["ts", "fb", "fa", "tb", "ta", "amt_paid", "fmt"], dtype=str)
    d = pd.read_csv(dump, dtype=np.float32, usecols=lambda c: c not in ("fmt",))
    assert len(d) == len(meta) == 5_078_345
    y = d.y.values.astype(np.int8)
    X = d[F].values.astype(np.float32)
    fi = F.index("fmt_code")
    live = np.where(d.l1 < L1T, d.l1, (d.l2x + d.l2l) / 2).astype(np.float64)
    day = meta.ts.str.slice(0, 10).values
    R = {"data": "HI-Small full (mirror verified against the repository); features from the served Rust engine",
         "rule": RULE, "splits": {"train_rows": VAL0, "validation_rows": TEST0 - VAL0, "test_rows": len(d) - TEST0}}
    va, te = slice(VAL0, TEST0), slice(TEST0, None)
    b1 = xgb.Booster(); b1.load_model(os.path.join(ROOT, "service/models/l1_gatekeeper.json"))
    bx = xgb.Booster(); bx.load_model(os.path.join(ROOT, "service/models/l2_xgb.json"))
    import lightgbm as lgb
    bl = lgb.Booster(model_file=os.path.join(ROOT, "ml/exported/l2_lgbm.txt"))

    def served(Xm):
        dm = xgb.DMatrix(Xm, missing=np.nan, feature_names=F)
        s1 = b1.predict(dm)
        return np.where(s1 < L1T, s1, (bx.predict(dm) + bl.predict(Xm.astype(np.float64))) / 2)

    formats = {k: v for k, v in CFG["fmt_codes"].items() if k not in ("ACH", "Reinvestment")}

    def attack(Xm, yy, score_fn):
        """worst PR-AUC when every laundering payment is re-sent in one other format"""
        res = {}
        for name, code in formats.items():
            Xa = Xm.copy(); Xa[yy == 1, fi] = code
            res[name] = round(float(AP(yy, score_fn(Xa))), 4)
        return res

    # ------------------------------------------------ served model, validation
    yv, Xv = y[va], X[va]
    R["served_validation"] = pr(yv, live[va], 100)
    R["served_validation_attack"] = attack(Xv, yv, served)

    # ------------------------------------------------ candidates trained on train rows (all positives + 10% negatives)
    tr_idx = np.arange(VAL0)
    pos = tr_idx[y[:VAL0] == 1]; neg = tr_idx[y[:VAL0] == 0]
    sub = np.sort(np.concatenate([pos, RNG.choice(neg, len(neg) // 10, replace=False)]))
    R["train_sample"] = {"positives": int(len(pos)), "negatives": int(len(neg) // 10)}
    models = {}
    for name, feats in (("control_31_features", F), ("candidate_no_format", FNOFMT)):
        cols = [F.index(f) for f in feats]
        b = xgb.train(dict(PARAMS, seed=0), xgb.DMatrix(X[sub][:, cols], y[sub], missing=np.nan, feature_names=feats), ROUNDS)
        models[name] = (b, cols, feats)

    def raw(name, Xm):
        b, cols, feats = models[name]
        return b.predict(xgb.DMatrix(Xm[:, cols], missing=np.nan, feature_names=feats))

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(raw("candidate_no_format", Xv), yv)

    def cand(Xm):
        return iso.predict(raw("candidate_no_format", Xm))

    for name in models:
        R[f"{name}_validation"] = pr(yv, raw(name, Xv), 100)
    R["control_31_features_validation_attack"] = attack(Xv, yv, lambda Xm: raw("control_31_features", Xm))
    cv = cand(Xv)
    R["candidate_calibrated_validation"] = pr(yv, cv, 100)
    R["candidate_validation_attack"] = attack(Xv, yv, cand)
    # thresholds: keep the served validation alert rates (verify, hold, block)
    rates = {b: float((live[va] >= t).mean()) for b, t in (("verify", VER), ("hold", HOLD), ("block", BLK))}
    cthr = {b: thr_for_rate(cv, r) for b, r in rates.items()}
    R["candidate_thresholds_from_served_validation_rates"] = {"rates": rates, "thresholds": cthr}

    # ------------------------------------------------ decision (before test)
    s_clean, c_clean = R["served_validation"]["pr_auc"], R["candidate_calibrated_validation"]["pr_auc"]
    s_worst, c_worst = min(R["served_validation_attack"].values()), min(R["candidate_validation_attack"].values())
    checks = {"R1": c_clean >= s_clean - RULE["R1_max_drop"], "R2": c_worst >= RULE["R2_min_ratio"] * c_clean,
              "R3": s_worst < RULE["R3_served_max_ratio"] * s_clean}
    R["decision"] = {"checks": checks, "adopt": all(checks.values()),
                     "values": {"served_clean": s_clean, "served_worst_attack": s_worst, "candidate_clean": c_clean,
                                "candidate_worst_attack": c_worst},
                     "decided_at_seconds": round(time.time() - t0, 1), "note": "decided on validation only; test scored after this line"}
    json.dump(R, open(os.path.join(ROOT, "research", "out", "full_hi_eval.json"), "w"), indent=1)

    # ------------------------------------------------ test (read once, after the decision)
    yt, Xt = y[te], X[te]
    dt = day[te]
    tail = np.isin(dt, ["2022/09/11", "2022/09/12", "2022/09/13", "2022/09/14", "2022/09/15", "2022/09/16", "2022/09/17", "2022/09/18"])
    ct = cand(Xt)
    scores = {"served_live": live[te], "control_31_features": raw("control_31_features", Xt), "candidate_no_format": ct}
    T = {}
    for k, s in scores.items():
        T[k] = {"all_test": pr(yt, s), "without_sep11_18_tail": pr(yt[~tail], s[~tail]),
                "sep09_10_only": pr(yt[np.isin(dt, ["2022/09/09", "2022/09/10"])], s[np.isin(dt, ["2022/09/09", "2022/09/10"])])}
    T["tail"] = {"rows": int(tail.sum()), "laundering": int(yt[tail].sum()), "share_of_test_laundering": round(float(yt[tail].sum() / yt.sum()), 4)}
    T["served_attack"] = attack(Xt, yt, served)
    T["candidate_attack"] = attack(Xt, yt, cand)
    T["served_bands"] = {b: band_counts(yt, live[te], t) for b, t in (("verify", VER), ("hold", HOLD), ("block", BLK))}
    T["candidate_bands"] = {b: band_counts(yt, ct, t) for b, t in cthr.items()}
    # bands under the Cheque attack
    Xa = Xt.copy(); Xa[yt == 1, fi] = CFG["fmt_codes"]["Cheque"]
    sa, ca = served(Xa), cand(Xa)
    T["bands_under_cheque_attack"] = {"served": {b: band_counts(yt, sa, t) for b, t in (("verify", VER), ("hold", HOLD), ("block", BLK))},
                                      "candidate": {b: band_counts(yt, ca, t) for b, t in cthr.items()}}
    # ------------------------------------------------ typology recall (patterns file)
    typ, cur = {}, None
    for line in open(patterns):
        line = line.strip()
        if line.startswith("BEGIN"):
            cur = line.split("-", 1)[1].split(":")[0].strip()
        elif line.startswith("END") or not line:
            continue
        elif cur:
            p = line.split(",")
            typ[(p[0], p[1], p[2], p[3], p[4], p[7])] = cur
    mt = meta.iloc[TEST0:].reset_index(drop=True)
    keys = list(zip(mt.ts, mt.fb, mt.fa, mt.tb, mt.ta, mt.amt_paid))
    tt = np.array([typ.get(k, "unattributed") for k in keys])
    by = {}
    for name in sorted(set(tt[yt == 1])):
        m = (tt == name) & (yt == 1)
        by[name] = {"laundering": int(m.sum()),
                    "served_verify": round(float((live[te][m] >= VER).mean()), 3), "served_block": round(float((live[te][m] >= BLK).mean()), 3),
                    "candidate_verify": round(float((ct[m] >= cthr["verify"]).mean()), 3), "candidate_block": round(float((ct[m] >= cthr["block"]).mean()), 3)}
    T["typology_recall"] = by
    R["test"] = T
    # export the candidate for serving (only meaningful if adopted)
    models["candidate_no_format"][0].save_model(os.path.join(ROOT, "research", "out", "candidate_no_format_xgb.json"))
    models["control_31_features"][0].save_model(os.path.join(ROOT, "research", "out", "control_31_xgb.json"))
    json.dump({"x": [float(v) for v in iso.X_thresholds_], "y": [float(v) for v in iso.y_thresholds_]},
              open(os.path.join(ROOT, "research", "out", "candidate_isotonic.json"), "w"))
    json.dump(R, open(os.path.join(ROOT, "research", "out", "full_hi_eval.json"), "w"), indent=1)
    print(json.dumps(R["decision"], indent=1)); print(json.dumps({k: v["all_test"] for k, v in T.items() if isinstance(v, dict) and "all_test" in v}, indent=1))


if __name__ == "__main__":
    main()
