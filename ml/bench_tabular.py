"""
RuseTect: benchmark of boosting + classical ML models on IBM AMLworld HI-Small.
Same training rows for every model, threshold chosen on validation, metrics on held-out test.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import json, time, sys, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import load, to_np, train_sample, best_f1_threshold, evaluate, RAW, ALL
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, IsolationForest
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
import xgboost as xgb, lightgbm as lgb, catboost as cb

OUT = f"{WORK}/results_tabular.json"
d, split = load()
y = d["y"].to_numpy().astype(np.int8)
ts = d["ts"].to_numpy()
core = (split == 2) & (ts < np.datetime64("2022-09-11"))   # test window without the tail artifact
tr = train_sample(y, split)
va = np.where(split == 1)[0]
te = np.where(split == 2)[0]
print("train rows", len(tr), "pos", y[tr].sum(), "| val", len(va), "| test", len(te), "pos", y[te].sum())

def slog(X):  # sign-log + impute for distance/linear models
    X = np.nan_to_num(X, nan=-1.0)
    return np.sign(X) * np.log1p(np.abs(X))

results = json.load(open(OUT)) if len(sys.argv) > 1 and sys.argv[1] == "--resume" else {}

def record(name, fam, feats, p_va, p_te, secs, note=""):
    thr = best_f1_threshold(y[va], p_va)
    r = evaluate(y[te], p_te, thr)
    core_te = core[te]
    r["pr_auc_core"] = float(average_precision_score(y[te][core_te], p_te[core_te]))
    r.update(family=fam, features=feats, train_secs=round(secs, 1), note=note)
    results[name] = r
    json.dump(results, open(OUT, "w"), indent=1)
    print(f"{name:28s} PR-AUC {r['pr_auc']:.3f} core {r['pr_auc_core']:.3f} ROC {r['roc_auc']:.3f} "
          f"F1 {r['f1']:.3f} R@0.2% {r['recall_at_budget']:.3f} ({secs:.0f}s)", flush=True)

def run(name, fam, feats, make, scale=False, proba=True, impute=False):
    if name in results: return
    cols = RAW if feats == "raw" else ALL
    Xtr, Xva, Xte = to_np(d[tr], cols), to_np(d[va], cols), to_np(d[te], cols)
    if impute:
        Xtr, Xva, Xte = [np.nan_to_num(X, nan=-1.0) for X in (Xtr, Xva, Xte)]
    if scale:
        Xtr, Xva, Xte = slog(Xtr), slog(Xva), slog(Xte)
        sc = StandardScaler().fit(Xtr); Xtr, Xva, Xte = sc.transform(Xtr), sc.transform(Xva), sc.transform(Xte)
    t = time.time(); m = make(); m.fit(Xtr, y[tr]); secs = time.time() - t
    f = (lambda X: m.predict_proba(X)[:, 1]) if proba else m.decision_function
    record(name, fam, feats, f(Xva), f(Xte), secs)
    return m

spw = float((y[tr] == 0).sum() / max((y[tr] == 1).sum(), 1))

# ---- 1. Boosting frameworks
run("XGBoost", "boosting", "raw", lambda: xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, scale_pos_weight=spw ** 0.5))
run("XGBoost+graph", "boosting", "all", lambda: xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, scale_pos_weight=spw ** 0.5))
run("LightGBM+graph", "boosting", "all", lambda: lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1, scale_pos_weight=spw ** 0.5))
run("CatBoost+graph", "boosting", "all", lambda: cb.CatBoostClassifier(iterations=600, depth=6, learning_rate=0.08, thread_count=2, verbose=0, scale_pos_weight=spw ** 0.5))
run("AdaBoost+graph", "boosting", "all", lambda: AdaBoostClassifier(n_estimators=300, learning_rate=0.5), impute=True)

# ---- 2. Classical ML
run("LogisticRegression", "classical", "raw", lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True)
run("LogisticRegression+graph", "classical", "all", lambda: LogisticRegression(max_iter=2000, class_weight="balanced"), scale=True)
run("DecisionTree+graph", "classical", "all", lambda: DecisionTreeClassifier(max_depth=12, min_samples_leaf=20, class_weight="balanced"), impute=True)
run("RandomForest+graph", "classical", "all", lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt", n_jobs=2, class_weight="balanced_subsample"), scale=True)
run("LinearSVM+graph", "classical", "all", lambda: LinearSVC(C=0.5, class_weight="balanced", max_iter=5000), scale=True, proba=False)

# ---- RBF-SVM and KNN cannot score 1M rows x 300K references on 2 CPUs.
# Train on a 40K subsample, score on all test positives + 10% of test negatives,
# and re-weight negatives x10 so PR-AUC is an unbiased estimate of full-test PR-AUC.
rng = np.random.default_rng(1)
def subsample_eval(name, make):
    if name in results: return
    tr_s = np.concatenate([tr[y[tr] == 1], rng.choice(tr[y[tr] == 0], 38000, replace=False)])
    te_pos, te_neg = te[y[te] == 1], te[y[te] == 0]
    te_s = np.concatenate([te_pos, rng.choice(te_neg, len(te_neg) // 10, replace=False)])
    w = np.where(y[te_s] == 1, 1.0, 10.0)
    Xtr, Xte = slog(to_np(d[tr_s], ALL)), slog(to_np(d[te_s], ALL))
    sc = StandardScaler().fit(Xtr); Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    t = time.time(); m = make(); m.fit(Xtr, y[tr_s]); secs = time.time() - t
    p = m.predict_proba(Xte)[:, 1] if hasattr(m, "predict_proba") else m.decision_function(Xte)
    r = {"pr_auc": float(average_precision_score(y[te_s], p, sample_weight=w)),
         "roc_auc": float(roc_auc_score(y[te_s], p, sample_weight=w)),
         "family": "classical", "features": "all", "train_secs": round(secs, 1),
         "note": "40K-row train subsample; test estimated on positives + 10% negatives (re-weighted)"}
    results[name] = r; json.dump(results, open(OUT, "w"), indent=1)
    print(f"{name:28s} PR-AUC {r['pr_auc']:.3f} ROC {r['roc_auc']:.3f} ({secs:.0f}s)", flush=True)

subsample_eval("KNN+graph (k=50)", lambda: KNeighborsClassifier(n_neighbors=50, weights="distance", n_jobs=2))
subsample_eval("RBF-SVM+graph", lambda: SVC(C=1.0, gamma="scale", class_weight="balanced"))

# ---- Unsupervised anomaly baseline (no labels used)
if "IsolationForest+graph" not in results:
    Xtr = np.nan_to_num(to_np(d[tr], ALL), nan=-1); t = time.time()
    m = IsolationForest(n_estimators=300, max_samples=4096, n_jobs=2, random_state=0).fit(Xtr[y[tr] == 0])
    secs = time.time() - t
    s = lambda X: -m.score_samples(np.nan_to_num(X, nan=-1))
    record("IsolationForest+graph", "unsupervised", "all", s(to_np(d[va], ALL)), s(to_np(d[te], ALL)), secs, "trained on negatives only")

# ---- Ablation: does XGBoost still work without payment format (the ACH shortcut)?
if "XGBoost+graph −format" not in results:
    cols = [c for c in ALL if c != "fmt_code"]
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, scale_pos_weight=spw ** 0.5)
    t = time.time(); m.fit(to_np(d[tr], cols), y[tr]); secs = time.time() - t
    record("XGBoost+graph −format", "boosting", "all−fmt", m.predict_proba(to_np(d[va], cols))[:, 1], m.predict_proba(to_np(d[te], cols))[:, 1], secs, "payment format removed")
print("done")
