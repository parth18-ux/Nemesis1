"""Unified fraud training pipeline over every transaction-shaped dataset: bank AML (AMLworld HI-Small and HI-Medium,
SAML-D, AMLgentex), cards (BankSim, Sparkov) and mobile wallets (PaySim). Every dataset is streamed through the served
Rust feature code (research_dump, gzip output with 13 extra causal indicators) and split by time per dataset.

    python3 -I research/unified_train.py <v2 folder> <sources json>

Features
  BASE  the 31 live features minus fmt_code (format is a generator shortcut: 92% of HI-Small laundering is ACH, every
        SAML-D cash payment differs, PaySim fraud lives only in TRANSFER/CASH_OUT) and minus cur_code (a categorical id
        whose meaning differs per dataset mapping). 29 features.
  NEW   the 13 indicators added by this review (bursts, mule collection, structuring pairs, amount anomaly, hour
        anomaly, payee age, new-payee velocity). All causal, none reads a label.
Leakage controls
  * chronological 60/20/20 per dataset; models never see validation or test rows
  * label-adjacent columns never enter (PaySim balances and isFlaggedFraud, SAML-D Laundering_type, AMLgentex alertID)
  * an "unseen accounts" test: test rows touching any account seen in a laundering/fraud row of the train part are removed,
    because SAML-D reuses laundering accounts across time (47% of its test laundering touches train laundering accounts)
  * genuine rows are subsampled for training and evaluation; weights restore the true prevalence in every metric
Pre-registered question (written before any test is read): do the NEW indicators help? Adopt them if the pooled
BASE+NEW model's mean test ROC-AUC across datasets beats pooled BASE with a paired bootstrap interval above zero.
Results: research/out/unified_train.json
"""
import gzip, json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "out")
CFG = json.load(open(os.path.join(ROOT, "service", "models", "cascade_config.json")))
L1T = CFG["l1_threshold"]
BASE = [f for f in CFG["features"] if f not in ("fmt_code", "cur_code")]
NEW = ["s_cnt_1h", "s_usd_1h", "d_cnt_1h", "d_fanin_1h", "d_usd_24h", "pair_cnt_24h", "pair_usd_24h", "amt_z", "amt_to_max",
       "just_below_round", "hour_dev", "payee_age_min", "s_new_payees_24h"]
RNG = np.random.default_rng(29)
NEG_TRAIN, NEG_EVAL = 300_000, 600_000
PARAMS = {"objective": "binary:logistic", "max_depth": 7, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
          "min_child_weight": 2, "tree_method": "hist", "nthread": 4}


def count_rows(path):
    n = 0
    with gzip.open(path, "rt") as f:
        for _ in f:
            n += 1
    return n - 1


def load(name, path, src):
    """train / val / test parts with sampled genuine rows and weights; ids of payer and payee for the unseen-account test"""
    n = count_rows(path)
    tr_end, va_end = int(n * 0.6), int(n * 0.8)
    rates = {"tr": min(1.0, NEG_TRAIN / tr_end), "va": min(1.0, NEG_EVAL / (va_end - tr_end)), "te": min(1.0, NEG_EVAL / (n - va_end))}
    cols = ["row", "y", "l1", "l2x", "l2l"] + BASE + NEW
    ids = pd.read_csv(src, usecols=["fa", "ta"], dtype=str) if src else None
    parts = {k: [] for k in rates}
    for ch in pd.read_csv(path, usecols=cols, chunksize=1_000_000, dtype=np.float32):
        r = ch.row.values.astype(np.int64); y = ch.y.values.astype(np.int8)
        u = RNG.random(len(ch))
        for k, m in (("tr", r < tr_end), ("va", (r >= tr_end) & (r < va_end)), ("te", r >= va_end)):
            keep = m & ((y == 1) | (u < rates[k]))
            if keep.any():
                served = np.where(ch.l1.values[keep] < L1T, ch.l1.values[keep], (ch.l2x.values[keep] + ch.l2l.values[keep]) / 2)
                parts[k].append((ch[BASE + NEW].values[keep], y[keep], np.where(y[keep] == 1, 1.0, 1.0 / rates[k]), served, r[keep]))
    D = {k: tuple(np.concatenate([p[i] for p in v]) for i in range(5)) for k, v in parts.items()}
    D["n"] = n
    if ids is not None:
        rtr = D["tr"][4][D["tr"][1] == 1]
        bad = set(ids.fa.values[rtr]) | set(ids.ta.values[rtr])
        rte = D["te"][4]
        D["te_unseen"] = ~(ids.fa.isin(bad).values[rte] | ids.ta.isin(bad).values[rte])
    return D


def train(D, names, cols):
    X = np.vstack([D[n]["tr"][0][:, cols] for n in names]); y = np.concatenate([D[n]["tr"][1] for n in names])
    w = np.concatenate([D[n]["tr"][2] / D[n]["tr"][2].sum() * 1e6 for n in names])  # every dataset the same total weight
    return xgb.train(dict(PARAMS, seed=0), xgb.DMatrix(X, y, weight=w, missing=np.nan), 500)


def m(y, s, w):
    k = s >= np.quantile(s, 0.995)
    return {"positives": int(y.sum()), "pr_auc": round(float(AP(y, s, sample_weight=w)), 4), "roc_auc": round(float(AUC(y, s, sample_weight=w)), 4),
            "recall_top_0.5pct": round(float(w[k & (y == 1)].sum() / w[y == 1].sum()), 4)}


def structuring(folder, models, ib, inew):
    """HI-Small demo slice (inside the test period, after all training rows): every laundering payment of the evaluation
    window split into 4 parts (same minute, or 30 minutes apart), re-streamed through the engine and the indicators.
    Each model flags the same share of payments as the served block band does on the clean slice (46 of 60,000);
    a laundering payment counts as caught if any part is flagged."""
    out = {}
    origin = lambda n: np.loadtxt(os.path.join(ROOT, "research", "out", "streams", f"{n}.origin"), dtype=np.int64)
    cols = ["y", "l1", "l2x", "l2l"] + BASE + NEW
    base = pd.read_csv(os.path.join(folder, "slice_base.csv.gz"), usecols=cols, dtype=np.float32)
    ev = origin("base") >= 100_000
    rate = 46 / 60_000
    def scores(df, label):
        if label == "served_cascade":
            return np.where(df.l1.values < L1T, df.l1.values, (df.l2x.values + df.l2l.values) / 2)
        cols_ = ib if label == "pooled_base" else inew
        return models[label].predict(xgb.DMatrix(df[BASE + NEW].values[:, cols_], missing=np.nan))
    for label in ("served_cascade", "pooled_base", "pooled_base_new"):
        sb = scores(base, label)[ev]; yb = base.y.values[ev]
        thr = np.quantile(sb, 1 - rate)
        r = {"clean_recall": round(float((sb[yb == 1] >= thr).mean()), 4)}
        for n in ("split4", "split4_spread"):
            df = pd.read_csv(os.path.join(folder, f"slice_{n}.csv.gz"), usecols=cols, dtype=np.float32)
            o = origin(n); keep = o >= 100_000
            s_ = scores(df, label)[keep]; y_ = df.y.values[keep]; o_ = o[keep]
            caught = pd.Series(s_[y_ == 1] >= thr).groupby(o_[y_ == 1]).any()
            r[f"{n}_recall_any_part"] = round(float(caught.mean()), 4)
        out[label] = r
    out["note"] = "97 laundering payments in the window; the served models' block threshold is replaced by an equal alert rate per model"
    return out


def main():
    folder, sources = sys.argv[1], json.loads(sys.argv[2])
    names = [n for n in sources if os.path.exists(os.path.join(folder, f"{n}.csv.gz"))]
    D = {n: load(n, os.path.join(folder, f"{n}.csv.gz"), sources[n]) for n in names}
    ib = list(range(len(BASE))); inew = list(range(len(BASE) + len(NEW)))
    R = {"datasets": {n: {"rows": D[n]["n"], "train_positives": int(D[n]["tr"][1].sum()), "test_positives": int(D[n]["te"][1].sum())} for n in names},
         "features": {"base": BASE, "new": NEW}}
    models = {"pooled_base": train(D, names, ib), "pooled_base_new": train(D, names, inew)}
    loo = {n: train(D, [x for x in names if x != n], inew) for n in names}
    T, diffs = {}, []
    for n in names:
        X, y, w, served, _ = D[n]["te"]
        sb = models["pooled_base"].predict(xgb.DMatrix(X[:, ib], missing=np.nan))
        sn = models["pooled_base_new"].predict(xgb.DMatrix(X[:, inew], missing=np.nan))
        sl = loo[n].predict(xgb.DMatrix(X[:, inew], missing=np.nan))
        T[n] = {"served_cascade": m(y, served, w), "pooled_base": m(y, sb, w), "pooled_base_new": m(y, sn, w), "leave_dataset_out": m(y, sl, w)}
        if "te_unseen" in D[n]:
            u = D[n]["te_unseen"]
            T[n]["unseen_accounts"] = {"test_positives_kept": int(y[u].sum()), "pooled_base_new": m(y[u], sn[u], w[u]), "served_cascade": m(y[u], served[u], w[u])}
        diffs.append((y, w, sb, sn))
    # paired bootstrap of the mean ROC-AUC difference across datasets
    boots = []
    for _ in range(200):
        d = []
        for y, w, sb, sn in diffs:
            ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
            ix = np.r_[RNG.choice(ip, len(ip)), RNG.choice(ineg, len(ineg))]
            d.append(AUC(y[ix], sn[ix], sample_weight=w[ix]) - AUC(y[ix], sb[ix], sample_weight=w[ix]))
        boots.append(np.mean(d))
    point = float(np.mean([T[n]["pooled_base_new"]["roc_auc"] - T[n]["pooled_base"]["roc_auc"] for n in names]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    R["new_indicators_test"] = {"mean_roc_gain": round(point, 4), "ci95": [round(float(lo), 4), round(float(hi), 4)], "adopt": bool(lo > 0)}
    R["test"] = T
    R["structuring_attack"] = structuring(folder, models, ib, inew)
    imp = models["pooled_base_new"].get_score(importance_type="gain")
    names_all = BASE + NEW
    R["top_features_gain"] = [names_all[int(k[1:])] for k, _ in sorted(imp.items(), key=lambda kv: -kv[1])[:12]]
    models["pooled_base_new"].save_model(os.path.join(OUT, "unified_pooled_xgb.json"))
    json.dump(R, open(os.path.join(OUT, "unified_train.json"), "w"), indent=1)
    print(json.dumps(R, indent=1)[:6000])


if __name__ == "__main__":
    main()
