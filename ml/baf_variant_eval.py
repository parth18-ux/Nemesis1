"""L0 onboarding gate on the full Feedzai BAF files Parth uploaded: Base (1,000,000 rows) and Variant III (bias stress set).

The live L0 model (service/models/l0_onboarding_lgbm.json, trained on Base months 0 to 4) is scored with its own age group
thresholds (l0_config.json). Variant III was never used for training; it adds two synthetic features (x1, x2) that leak
the label through a group shortcut. L0 does not read x1 or x2, so this tests whether the gate keeps its accuracy and its
false alarm parity on a population whose bias is shifted. A control model trained on Variant III WITH x1 and x2 shows
what the shortcut does when a model is allowed to use it.
Writes results/baf_variant_eval.json.
"""
import glob, json, os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from lgbm_lite import LiteBooster  # noqa: E402
sys.path.insert(0, os.path.join(ROOT, "ml")); import lite_fast  # noqa: E402

DATA = os.environ.get("RUSETECT_DATA", "/mnt/user-data/uploads/iit delhi")
cfg = json.load(open(os.path.join(ROOT, "service", "models", "l0_config.json")))
L0 = LiteBooster(os.path.join(ROOT, "service", "models", "l0_onboarding_lgbm.json"))
T0 = time.time()

def load(name):
    return pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(DATA, "for_claude", f"baf_{name}_part*.csv.gz")))], ignore_index=True)

def matrix(df):
    cols = []
    for k in cfg["features"]:
        if k in cfg["categorical"]:
            lst = cfg["categories"][k]; m = {v: float(i) for i, v in enumerate(lst)}
            cols.append(df[k].map(m).astype(float).to_numpy())
        else:
            cols.append(df[k].astype(float).to_numpy())
    return np.column_stack(cols)

def evaluate(df, label):
    M = matrix(df); lite_fast.check(L0, M, 500); p = lite_fast.predict(L0, M); y = df["fraud_bool"].to_numpy().astype(bool); old = df["customer_age"].to_numpy() >= 50
    thr = np.where(old, cfg["age_50_plus"], cfg["age_under_50"]); flag = p >= thr
    fpr = lambda m: float((flag & ~y & m).sum() / max((~y & m).sum(), 1))
    tpr = lambda m: float((flag & y & m).sum() / max((y & m).sum(), 1))
    f, t, _ = roc_curve(y, p); r5 = float(np.interp(0.05, f, t))
    return {"set": label, "applications": int(len(df)), "fraud": int(y.sum()), "months": sorted(int(m) for m in df["month"].unique()),
            "roc_auc": round(float(roc_auc_score(y, p)), 4), "recall_at_5pct_fpr": round(r5, 4),
            "with_age_thresholds": {"recall": round(tpr(np.ones_like(y)), 4), "fpr": round(fpr(np.ones_like(y)), 4),
                                    "fpr_50_plus": round(fpr(old), 4), "fpr_under_50": round(fpr(~old), 4),
                                    "fpr_parity_ratio": round(min(fpr(old), fpr(~old)) / max(fpr(old), fpr(~old)), 3),
                                    "recall_50_plus": round(tpr(old), 4), "recall_under_50": round(tpr(~old), 4),
                                    "flag_rate": round(float(flag.mean()), 4)}}

base = load("Base"); v3 = load("Variant_III")
print(f"loaded Base {len(base):,} and Variant III {len(v3):,} in {time.time() - T0:.0f}s; V3 months {sorted(v3.month.unique())}", flush=True)
out = {"model": "live L0 LightGBM, trained on BAF Base months 0 to 4; thresholds per age group from l0_config.json",
       "base_test_months_6_7": evaluate(base[base.month >= 6], "Base, months 6 to 7 (benchmark test period)"),
       "variant_iii_all": evaluate(v3, "Variant III, all uploaded rows (never trained on)")}
if (v3.month >= 6).any():
    out["variant_iii_months_6_7"] = evaluate(v3[v3.month >= 6], "Variant III, months 6 to 7")

# ---- control: what happens if a model may use the leaked bias features
try:
    import lightgbm as lgb
    tr, te = v3[v3.month <= 4], v3[v3.month >= 6] if (v3.month >= 6).any() else v3[v3.month > 4]
    def fit(cols):
        X = tr[cols].copy(); Xt = te[cols].copy()
        for c in cfg["categorical"]:
            if c in cols:
                X[c] = X[c].astype("category"); Xt[c] = pd.Categorical(Xt[c], categories=X[c].cat.categories)
        m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, verbose=-1).fit(X, tr["fraud_bool"])
        p = m.predict_proba(Xt)[:, 1]; y = te["fraud_bool"].to_numpy().astype(bool)
        f, t, th = roc_curve(y, p); thr = th[np.searchsorted(f, 0.05)]
        flag = p >= thr; old = te["customer_age"].to_numpy() >= 50
        fp = lambda msk: float((flag & ~y & msk).sum() / max((~y & msk).sum(), 1))
        return {"roc_auc": round(float(roc_auc_score(y, p)), 4), "recall_at_5pct_fpr": round(float(np.interp(0.05, f, t)), 4),
                "fpr_50_plus": round(fp(old), 4), "fpr_under_50": round(fp(~old), 4),
                "fpr_parity_ratio": round(min(fp(old), fp(~old)) / max(fp(old), fp(~old)), 3)}
    feats = list(cfg["features"])
    out["control_trained_on_variant_iii"] = {
        "note": "Fresh LightGBM trained on Variant III months 0 to 4 and tested on its later months, one global threshold at 5% FPR.",
        "without_bias_features": fit(feats), "with_bias_features_x1_x2": fit(feats + ["x1", "x2"])}
except ImportError:
    out["control_trained_on_variant_iii"] = "lightgbm not installed"
out["runtime_seconds"] = round(time.time() - T0, 1)
out["caveats"] = ["Feedzai BAF is synthetic.", "Only the first 200,000 Variant III rows were uploaded."]
json.dump(out, open(os.path.join(ROOT, "results", "baf_variant_eval.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
