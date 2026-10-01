"""
RuseTect — Layer 0 onboarding gate (mule-account opening), final model + fairness post-processing.
LightGBM on BAF Base, trained months 0-4, thresholds on month 5, tested on months 6-7.
Fairness fix: group-specific thresholds so applicants >= 50 and < 50 each get 5% FPR on validation.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import glob, json
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score

files = sorted(glob.glob(f"{DATA}/baf_Base_part*.csv.gz"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
CAT = list(df.select_dtypes(exclude="number").columns)
for c in CAT: df[c] = df[c].astype("category")
y = df.fraud_bool.values; old = (df.customer_age >= 50).values
feats = [c for c in df.columns if c not in ("fraud_bool", "month")]
tr, va, te = (df.month <= 4).values, (df.month == 5).values, (df.month >= 6).values
spw = ((y[tr] == 0).sum() / (y[tr] == 1).sum()) ** 0.5
m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8,
                       subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw, n_jobs=2, verbose=-1)
m.fit(df.loc[tr, feats], y[tr])
p = m.predict_proba(df[feats])[:, 1]

def report(thr_old, thr_young, name):
    t = np.where(old, thr_old, thr_young); pred = p >= t
    sel = te; yt, pt, ot = y[sel], pred[sel], old[sel]
    fpr = lambda g: (pt & (yt == 0) & g).sum() / ((yt == 0) & g).sum()
    tpr = lambda g: (pt & (yt == 1) & g).sum() / ((yt == 1) & g).sum()
    r = dict(recall=float((pt & (yt == 1)).sum() / (yt == 1).sum()), fpr=float((pt & (yt == 0)).sum() / (yt == 0).sum()),
             fpr_old=float(fpr(ot)), fpr_young=float(fpr(~ot)), tpr_old=float(tpr(ot)), tpr_young=float(tpr(~ot)))
    r["fpr_ratio"] = min(r["fpr_old"], r["fpr_young"]) / max(r["fpr_old"], r["fpr_young"])
    print(name, {k: round(v, 3) for k, v in r.items()}); return r

q = lambda g: np.quantile(p[va & g & (y == 0)], 0.95)
glob_t = np.quantile(p[va & (y == 0)], 0.95)
out = dict(roc_auc=float(roc_auc_score(y[te], p[te])),
           single_threshold=report(glob_t, glob_t, "single threshold"),
           group_thresholds=report(q(old), q(~old), "group thresholds"),
           thresholds=dict(global_=float(glob_t), age_50_plus=float(q(old)), age_under_50=float(q(~old))))
imp = sorted(zip(feats, m.booster_.feature_importance("gain")), key=lambda x: -x[1])[:10]
out["top_features_gain"] = [(f, round(float(v) / sum(v for _, v in imp), 3)) for f, v in imp]
print(out["top_features_gain"])
json.dump(out, open(f"{WORK}/results_l0.json", "w"), indent=1)
m.booster_.save_model(f"{EXPORT}/l0_onboarding_lgbm.txt")
json.dump(m.booster_.dump_model(), open(f"{MODELS}/l0_onboarding_lgbm.json", "w"), separators=(",", ":"))  # served by service/lgbm_lite.py
json.dump(dict(features=feats, categorical=CAT, categories={c: list(df[c].cat.categories) for c in CAT}, **out["thresholds"]),
          open(f"{MODELS}/l0_config.json", "w"), indent=1)
print("saved")
