"""
RuseTect: reproducible comparison of tabular classifiers for payment-level laundering detection (the live L2 task).

Task      : binary classification of each payment in IBM AMLworld HI-Small (synthetic).
Positive  : `Is Laundering == 1`, the generator's own label.
Features  : the 31 causal features from ml/aml_features.py (each uses only earlier payments).
Split     : time-ordered 60 / 20 / 20 by payment order (train / validation / test). No shuffling.
Training  : every training positive + a fixed 10% random sample of training negatives (seed 0).
            Resampling and all preprocessing are fitted on training rows only.
Tuning    : a fixed grid or seeded random search per model (BUDGET below). Configurations are ranked by average
            precision on a validation subsample (all validation positives + 20% of validation negatives, negatives
            weighted x5, which keeps AP an unbiased estimate of full-validation AP).
Threshold : the chosen configuration is scored on the FULL validation set; the threshold maximising F1 there is fixed.
Test      : scored once, at the end, on all 1,015,669 test payments (except where a model cannot score 1M rows;
            such rows are marked "subsampled" and use a re-weighted stratified test sample).
Metrics   : prevalence, precision, recall, F1, confusion matrix, FP/FN, average precision
            (sklearn.metrics.average_precision_score: step-wise sum (R_n - R_{n-1}) * P_n, no interpolation),
            ROC-AUC (supplementary), recall/precision if analysts can review only the top 0.2% of payments
            (explicit alert-budget assumption), training time, single-payment latency, batch throughput, model size.
Outputs   : results/model_comparison_tabular.json
Usage     : RUSETECT_WORK=... RUSETECT_DATA=... python ml/compare_models.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, REPO
import gc, glob, hashlib, json, pickle, platform, subprocess, time, warnings
import numpy as np
warnings.filterwarnings("ignore")
from common import load, to_np, train_sample, ALL
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier,
                              HistGradientBoostingClassifier, IsolationForest)
from sklearn.neural_network import MLPClassifier
import sklearn, xgboost as xgb, lightgbm as lgb, catboost as cb

SEED = 0
BUDGET_FRAC = 0.002
OUT_DIR = os.path.join(REPO, "results"); os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, "model_comparison_tabular.json")
rng = np.random.default_rng(SEED)

t0 = time.time()
d, split = load()
y = d["y"].to_numpy().astype(np.int8)
tr = train_sample(y, split, neg_frac=0.10, seed=SEED)
va = np.where(split == 1)[0]; te = np.where(split == 2)[0]
va_sub = np.sort(np.concatenate([va[y[va] == 1], rng.choice(va[y[va] == 0], (y[va] == 0).sum() // 5, replace=False)]))
w_vs = np.where(y[va_sub] == 1, 1.0, 5.0)
te_sub = np.sort(np.concatenate([te[y[te] == 1], rng.choice(te[y[te] == 0], (y[te] == 0).sum() // 10, replace=False)]))
w_ts = np.where(y[te_sub] == 1, 1.0, 10.0)
Xtr_raw, Xva_raw, Xte_raw, Xvs_raw, Xts_raw = (to_np(d[i], ALL) for i in (tr, va, te, va_sub, te_sub))
ytr, yva, yte, yvs, yts = y[tr], y[va], y[te], y[va_sub], y[te_sub]
src = d["src_id"].to_numpy(); seen = np.zeros(src.max() + 1, bool); seen[src[split == 0]] = True
unseen_te = ~seen[src[te]]
print(f"data {time.time()-t0:.0f}s | train {len(tr):,} ({ytr.sum()} pos) | val {len(va):,} ({yva.sum()}) | test {len(te):,} ({yte.sum()})", flush=True)

def slog(X): X = np.nan_to_num(X, nan=-1.0); return np.sign(X) * np.log1p(np.abs(X))
sc = StandardScaler().fit(slog(Xtr_raw))
PREP = {"native": lambda X: X, "impute": lambda X: np.nan_to_num(X, nan=-1.0),
        "scaled": lambda X: sc.transform(slog(X)).astype(np.float32)}
spw = float((ytr == 0).sum() / (ytr == 1).sum())

def rs(space, n, seed):
    r = np.random.default_rng(seed)
    return [{k: (v[r.integers(len(v))] if isinstance(v, list) else v) for k, v in space.items()} for _ in range(n)]

XGB0 = {"max_depth": 6, "learning_rate": 0.08, "n_estimators": 400, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_weight": 1, "reg_lambda": 1.0}
LGB0 = {"num_leaves": 63, "learning_rate": 0.05, "n_estimators": 600, "subsample": 0.8, "colsample_bytree": 0.8, "min_child_samples": 20, "reg_lambda": 0.0}
MODELS = {
 "Dummy (class prior)": dict(prep="native", family="baseline", grid=[{}], make=lambda p: DummyClassifier(strategy="prior")),
 "Logistic Regression": dict(prep="scaled", family="linear", grid=[{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
     make=lambda p: LogisticRegression(class_weight="balanced", max_iter=3000, **p)),
 "Gaussian Naive Bayes": dict(prep="scaled", family="probabilistic", grid=[{"var_smoothing": v} for v in (1e-9, 1e-6, 1e-3)],
     make=lambda p: GaussianNB(**p)),
 "Linear SVM": dict(prep="scaled", family="linear", scorer="decision", grid=[{"C": c} for c in (0.01, 0.1, 1.0)],
     make=lambda p: LinearSVC(class_weight="balanced", max_iter=10000, **p)),
 "KNN (k-nearest neighbours)": dict(prep="scaled", family="instance-based", subsample_fit=40000, subsample_eval=True,
     grid=[{"n_neighbors": k} for k in (25, 50)], make=lambda p: KNeighborsClassifier(weights="distance", n_jobs=2, **p)),
 "Decision Tree": dict(prep="impute", family="tree", grid=[{"max_depth": a, "min_samples_leaf": b} for a in (6, 10, 14) for b in (20, 100)],
     make=lambda p: DecisionTreeClassifier(class_weight="balanced", random_state=SEED, **p)),
 "Random Forest": dict(prep="impute", family="bagging", grid=[{"max_features": "sqrt", "min_samples_leaf": 3}, {"max_features": 0.5, "min_samples_leaf": 3}],
     make=lambda p: RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=SEED, **p)),
 "Extra Trees": dict(prep="impute", family="bagging", grid=[{"max_features": "sqrt", "min_samples_leaf": 3}, {"max_features": 0.5, "min_samples_leaf": 2}],
     make=lambda p: ExtraTreesClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=SEED, **p)),
 "AdaBoost": dict(prep="impute", family="boosting", grid=[{"n_estimators": 300, "learning_rate": 0.5}, {"n_estimators": 300, "learning_rate": 1.0}],
     make=lambda p: AdaBoostClassifier(random_state=SEED, **p)),
 "HistGradientBoosting": dict(prep="native", family="boosting",
     grid=[{"max_leaf_nodes": 31, "learning_rate": 0.1, "max_iter": 400}] + rs({"max_leaf_nodes": [15, 31, 63], "learning_rate": [0.05, 0.1], "max_iter": [300, 600], "l2_regularization": [0.0, 1.0]}, 3, 3),
     make=lambda p: HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, early_stopping=False, **p)),
 "XGBoost": dict(prep="native", family="boosting",
     grid=[XGB0] + rs({"max_depth": [4, 5, 6, 8], "learning_rate": [0.03, 0.05, 0.08, 0.12], "n_estimators": [300, 500, 800], "subsample": [0.7, 0.8, 0.9],
                       "colsample_bytree": [0.6, 0.8, 1.0], "min_child_weight": [1, 5, 10], "reg_lambda": [0.5, 1.0, 5.0]}, 5, 1),
     make=lambda p: xgb.XGBClassifier(tree_method="hist", n_jobs=2, random_state=SEED, scale_pos_weight=spw ** 0.5, **p)),
 "LightGBM": dict(prep="native", family="boosting",
     grid=[LGB0] + rs({"num_leaves": [15, 31, 63, 127], "learning_rate": [0.03, 0.05, 0.08], "n_estimators": [300, 600, 900], "subsample": [0.7, 0.8, 0.9],
                       "colsample_bytree": [0.6, 0.8, 1.0], "min_child_samples": [10, 20, 50], "reg_lambda": [0.0, 1.0, 5.0]}, 5, 2),
     make=lambda p: lgb.LGBMClassifier(subsample_freq=1, n_jobs=2, random_state=SEED, verbose=-1, scale_pos_weight=spw ** 0.5, **p)),
 "CatBoost": dict(prep="native", family="boosting",
     grid=[{"depth": 6, "learning_rate": 0.08, "iterations": 600, "l2_leaf_reg": 3}, {"depth": 8, "learning_rate": 0.06, "iterations": 800, "l2_leaf_reg": 3},
           {"depth": 4, "learning_rate": 0.1, "iterations": 800, "l2_leaf_reg": 1}],
     make=lambda p: cb.CatBoostClassifier(thread_count=2, verbose=0, random_seed=SEED, scale_pos_weight=spw ** 0.5, allow_writing_files=False, **p)),
 "MLP (small neural net)": dict(prep="scaled", family="neural", grid=[{"hidden_layer_sizes": (64, 32), "alpha": 1e-4}, {"hidden_layer_sizes": (128, 64), "alpha": 1e-3}],
     make=lambda p: MLPClassifier(batch_size=1024, max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=SEED, **p)),
}

def score(m, X, how): return m.decision_function(X) if how == "decision" else m.predict_proba(X)[:, 1]

def size_bytes(m):
    if isinstance(m, xgb.XGBClassifier): return len(m.get_booster().save_raw("ubj"))
    if isinstance(m, lgb.LGBMClassifier): return len(m.booster_.model_to_string().encode())
    if isinstance(m, cb.CatBoostClassifier):
        f = os.path.join(WORK, "_cb.cbm"); m.save_model(f); return os.path.getsize(f)
    return len(pickle.dumps(m, protocol=pickle.HIGHEST_PROTOCOL))

def latency(m, X, how, reps=200):
    for _ in range(5): score(m, X[:1], how)
    ts = []
    for i in range(reps):
        t = time.perf_counter(); score(m, X[i:i + 1], how); ts.append(time.perf_counter() - t)
    n = min(10000, len(X)); t = time.perf_counter(); score(m, X[:n], how); b = time.perf_counter() - t
    return float(np.median(ts) * 1e3), float(np.percentile(ts, 99) * 1e3), float(n / b)

def metrics(yt, p, thr, w=None):
    pred = p >= thr; ww = np.ones(len(yt)) if w is None else w
    tp = float(ww[pred & (yt == 1)].sum()); fp = float(ww[pred & (yt == 0)].sum())
    fn = float(ww[~pred & (yt == 1)].sum()); tn = float(ww[~pred & (yt == 0)].sum())
    prec = tp / max(tp + fp, 1e-12); rec = tp / max(tp + fn, 1e-12)
    out = dict(threshold=float(thr), precision=prec, recall=rec, f1=2 * prec * rec / max(prec + rec, 1e-12),
               tp=round(tp), fp=round(fp), fn=round(fn), tn=round(tn), alerts=round(tp + fp),
               average_precision=float(average_precision_score(yt, p, sample_weight=w)), roc_auc=float(roc_auc_score(yt, p, sample_weight=w)))
    if w is None:
        k = max(1, int(len(yt) * BUDGET_FRAC)); top = np.argpartition(-p, k)[:k]
        out.update(recall_at_top_0_2pct=float(yt[top].sum() / yt.sum()), precision_at_top_0_2pct=float(yt[top].mean()))
    return out

def best_thr(yt, p, w=None):
    pr, rc, th = precision_recall_curve(yt, p, sample_weight=w)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None); return float(th[int(np.nanargmax(f1[:-1]))])

results = {}
if os.environ.get("RESUME") and os.path.exists(OUT.replace(".json", ".partial.json")):   # skip models already finished with identical data/protocol
    results = json.load(open(OUT.replace(".json", ".partial.json")))["models"]
for name, spec in MODELS.items():
    if name in results: print("   skip (resumed)", name, flush=True); continue
    P = PREP[spec["prep"]]; how = spec.get("scorer", "proba")
    fit_idx = np.arange(len(tr))
    if spec.get("subsample_fit"):
        fit_idx = np.sort(np.concatenate([np.where(ytr == 1)[0], rng.choice(np.where(ytr == 0)[0], spec["subsample_fit"], replace=False)]))
    Xf, yf = P(Xtr_raw[fit_idx]), ytr[fit_idx]; Xvs = P(Xvs_raw)
    trials = []
    for i, params in enumerate(spec["grid"]):
        t = time.time(); m = spec["make"](params); m.fit(Xf, yf); fs = time.time() - t
        ap = float(average_precision_score(yvs, score(m, Xvs, how), sample_weight=w_vs))
        trials.append(dict(params={k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()}, val_ap=round(ap, 5), fit_seconds=round(fs, 1)))
        print(f"   {name:28s} trial {i+1}/{len(spec['grid'])} val-AP {ap:.4f} ({fs:.0f}s)", flush=True)
        del m; gc.collect()
    b = max(range(len(trials)), key=lambda i: trials[i]["val_ap"])
    t = time.time(); m = spec["make"](spec["grid"][b]); m.fit(Xf, yf); train_s = time.time() - t
    if spec.get("subsample_eval"):
        p_vs = score(m, Xvs, how); thr = best_thr(yvs, p_vs, w_vs)
        Xts = P(Xts_raw); p_ts = score(m, Xts, how)
        test = metrics(yts, p_ts, thr, w_ts); evaluation = "subsampled: test positives + 10% of test negatives (x10 weight); counts are weighted estimates"
        lat = latency(m, Xts, how); unseen = None
    else:
        Xva, Xte = P(Xva_raw), P(Xte_raw)
        p_va, p_te = score(m, Xva, how), score(m, Xte, how)
        thr = best_thr(yva, p_va) if name != "Dummy (class prior)" else 0.5
        test = metrics(yte, p_te, thr); evaluation = "full test set"
        lat = latency(m, Xte, how)
        unseen = dict(rows=int(unseen_te.sum()), positives=int(yte[unseen_te].sum()),
                      average_precision=float(average_precision_score(yte[unseen_te], p_te[unseen_te])))
        if name == "XGBoost": xgb_m = m; p_xgb_te = p_te
    results[name] = dict(family=spec["family"], preprocessing=spec["prep"], evaluation=evaluation, tuning_trials=trials,
                         chosen_params=trials[b]["params"], test=test, unseen_sender_test=unseen,
                         train_rows=int(len(fit_idx)), train_seconds=round(train_s, 1), latency_ms_p50=round(lat[0], 3),
                         latency_ms_p99=round(lat[1], 3), throughput_rows_per_s=round(lat[2]), model_size_bytes=size_bytes(m))
    json.dump({"models": results}, open(OUT, "w"), indent=1)
    print(f"== {name:28s} AP {test['average_precision']:.4f} ROC {test['roc_auc']:.4f} P {test['precision']:.3f} R {test['recall']:.3f} F1 {test['f1']:.3f} "
          f"FP {test['fp']} FN {test['fn']} | train {train_s:.0f}s | {lat[0]:.2f} ms/row | {results[name]['model_size_bytes']/1e6:.2f} MB", flush=True)
    if name != "XGBoost": del m
    gc.collect()

# ---- XGBoost explanations: mean |TreeSHAP| over 20,000 random test payments
sel = rng.choice(len(te), 20000, replace=False)
contrib = xgb_m.get_booster().predict(xgb.DMatrix(Xte_raw[sel], feature_names=ALL), pred_contribs=True)[:, :-1]
shap = sorted(zip(ALL, np.abs(contrib).mean(0).round(4).tolist()), key=lambda x: -x[1])

# ---- Isolation Forest: separate, unsupervised setup (labels not used in fitting)
t = time.time(); iso = IsolationForest(n_estimators=300, max_samples=4096, n_jobs=2, random_state=SEED).fit(PREP["impute"](Xtr_raw)); iso_s = time.time() - t
a_tr, a_va, a_te = (-iso.score_samples(PREP["impute"](X)) for X in (Xtr_raw, Xva_raw, Xte_raw))
iso_alone = metrics(yte, a_te, best_thr(yva, a_va)); iso_alone["train_seconds"] = round(iso_s, 1)
xp = xgb.XGBClassifier(tree_method="hist", n_jobs=2, random_state=SEED, scale_pos_weight=spw ** 0.5, **results["XGBoost"]["chosen_params"]).fit(np.c_[Xtr_raw, a_tr], ytr)
pv, pt = xp.predict_proba(np.c_[Xva_raw, a_va])[:, 1], xp.predict_proba(np.c_[Xte_raw, a_te])[:, 1]
iso_plus = metrics(yte, pt, best_thr(yva, pv))

def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while (b := f.read(1 << 20)): h.update(b)
    return h.hexdigest()
try: lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
except Exception: lscpu = ""
cpu = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if l.startswith("Model name")), platform.processor())
out = dict(
  task="payment-level laundering detection (binary); the live L2 task",
  dataset=dict(name="IBM AMLworld HI-Small (synthetic)", files=[dict(file=os.path.basename(f), sha256=sha(f)) for f in sorted(glob.glob(f"{DATA}/aml_HI-Small_part0*.csv.gz"))],
               features_parquet_sha256=sha(os.path.join(WORK, "aml_feat.parquet")), rows=int(len(y)), positives=int(y.sum()),
               positive_class="Is Laundering = 1 (IBM generator label)", split="time-ordered 60/20/20 by (timestamp, file order)",
               train_rows_used=int(len(tr)), train_positives=int(ytr.sum()), train_resampling="all positives + 10% of negatives (seed 0), training period only",
               val_rows=int(len(va)), val_positives=int(yva.sum()), test_rows=int(len(te)), test_positives=int(yte.sum()), test_prevalence=float(yte.mean()),
               unseen_sender_test_rows=int(unseen_te.sum()),
               entity_note="Accounts recur over time, as in a bank; no account id is a feature. 'unseen_sender_test' reports AP on test payments whose sender never paid during training."),
  environment=dict(python=platform.python_version(), numpy=np.__version__, scikit_learn=sklearn.__version__, xgboost=xgb.__version__,
                   lightgbm=lgb.__version__, catboost=cb.__version__, cpu=cpu, logical_cpus=os.cpu_count(),
                   ram_gb=round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1), seed=SEED),
  metric_notes=dict(average_precision="sklearn.metrics.average_precision_score (step-wise, no interpolation)",
                    threshold="maximises F1 on the full validation set (re-weighted subsample for KNN)",
                    alert_budget=f"assumption: analysts review the top {BUDGET_FRAC:.1%} of payments (~{int(len(te)*BUDGET_FRAC):,} test alerts)",
                    latency="median / p99 of 200 single-row predict calls, model only (no feature computation), this hardware"),
  models=results,
  not_evaluated={"Gaussian Process classifier": "O(n^3) training and O(n^2) memory; infeasible for 307,000 training rows on 7 GB RAM",
                 "RBF-kernel SVM": "kernel SVM cost grows ~quadratically with rows; not run at this scale"},
  isolation_forest=dict(setup="unsupervised: fitted on training-period rows without labels; threshold picked on validation labels for comparability",
                        standalone=iso_alone, xgboost_plus_anomaly_score=iso_plus),
  xgboost_mean_abs_tree_shap=shap, runtime_seconds=round(time.time() - t0))
json.dump(out, open(OUT, "w"), indent=1)
print("saved", OUT, f"{time.time()-t0:.0f}s", flush=True)
