"""
RuseTect — Layer 0 (account-opening / mule-account onboarding) benchmark on Feedzai BAF Base (1M applications).
Protocol from the BAF paper (Jesus et al., NeurIPS 2022): train months 0-5, test months 6-7,
metric = TPR (recall) at 5% FPR, fairness = FPR ratio (age >= 50 vs < 50) at that threshold.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import glob, json, time, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb, catboost as cb

OUT = f"{WORK}/results_baf.json"
files = sorted(glob.glob(f"{DATA}/baf_Base_part*.csv.gz"))
df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
print("rows", len(df), "fraud rate", round(df.fraud_bool.mean(), 4), "months", sorted(df.month.unique()))
CAT = list(df.select_dtypes(exclude="number").columns)
for c in CAT: df[c] = df[c].astype("category")
y = df.fraud_bool.values; old = (df.customer_age >= 50).values
feats = [c for c in df.columns if c not in ("fraud_bool", "month")]
tr = (df.month <= 4).values; va = (df.month == 5).values; te = (df.month >= 6).values
print("train", tr.sum(), "val", va.sum(), "test", te.sum(), "test fraud", y[te].sum())

X = df[feats]
Xoh = pd.get_dummies(X, columns=CAT, dtype=np.float32).astype(np.float32)
sc = StandardScaler().fit(Xoh[tr]); Xs = sc.transform(Xoh).astype(np.float32)

def thr_at_fpr(yv, p, fpr=0.05):
    return np.quantile(p[yv == 0], 1 - fpr)

def metrics(p, thr):
    yt, pt, ot = y[te], p[te], old[te]
    pred = pt >= thr
    fpr = lambda g: (pred & (yt == 0) & g).sum() / max(((yt == 0) & g).sum(), 1)
    f_old, f_young = fpr(ot), fpr(~ot)
    return dict(roc_auc=float(roc_auc_score(yt, pt)), pr_auc=float(average_precision_score(yt, pt)),
                recall_at_5fpr=float((pred & (yt == 1)).sum() / (yt == 1).sum()),
                fpr_test=float((pred & (yt == 0)).sum() / (yt == 0).sum()),
                fpr_ratio_age=float(min(f_old, f_young) / max(f_old, f_young)))

results = {}
def run(name, fam, make, Xm):
    t = time.time(); m = make()
    if isinstance(Xm, pd.DataFrame): m.fit(Xm[tr], y[tr]); p = m.predict_proba(Xm)[:, 1]
    else: m.fit(Xm[tr], y[tr]); p = m.predict_proba(Xm)[:, 1]
    r = metrics(p, thr_at_fpr(y[va], p[va])); r.update(family=fam, train_secs=round(time.time() - t, 1))
    results[name] = r; json.dump(results, open(OUT, "w"), indent=1)
    print(f"{name:20s} ROC {r['roc_auc']:.3f} PR {r['pr_auc']:.3f} Recall@5%FPR {r['recall_at_5fpr']:.3f} FPR-ratio(age) {r['fpr_ratio_age']:.2f} ({r['train_secs']:.0f}s)", flush=True)
    return m, p

spw = ((y[tr] == 0).sum() / (y[tr] == 1).sum()) ** 0.5
run("LightGBM", "boosting", lambda: lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw, n_jobs=2, verbose=-1), X)
run("XGBoost", "boosting", lambda: xgb.XGBClassifier(n_estimators=600, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.7, tree_method="hist", enable_categorical=True, scale_pos_weight=spw, n_jobs=2), X)
run("CatBoost", "boosting", lambda: cb.CatBoostClassifier(iterations=800, depth=6, learning_rate=0.06, cat_features=CAT, thread_count=2, verbose=0, scale_pos_weight=spw), X.assign(**{c: X[c].astype(str) for c in CAT}))
run("AdaBoost", "boosting", lambda: AdaBoostClassifier(n_estimators=200, learning_rate=0.5), Xoh.values)
run("LogisticRegression", "classical", lambda: LogisticRegression(max_iter=3000, class_weight="balanced"), Xs)
run("DecisionTree", "classical", lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=200, class_weight="balanced"), Xoh.values)
run("RandomForest", "classical", lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=20, max_features="sqrt", class_weight="balanced_subsample", n_jobs=2), Xoh.values)
run("MLP", "deep", lambda: MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3, batch_size=2048, max_iter=30, early_stopping=True, random_state=0), Xs)
print("done")
