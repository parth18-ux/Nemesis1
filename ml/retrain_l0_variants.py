"""Does adding the unused BAF variant files to training make the L0 account opening gate better?

Data: Feedzai BAF Base and Variants I to V (1M applications each). Same protocol as ml/l0_onboarding.py: train on
months 0 to 4, set thresholds on month 5 (5% false alarms separately for applicants 50+ and under 50), test on
months 6 to 7. Variants III and V carry two leaked label features (x1, x2); they are dropped, as in the live model.

Candidates:
  current            the model in service/models today (Base months 0 to 4 only)
  retrain_base       the same recipe retrained (control)
  retrain_variants   Base months 0 to 4 + every variant's months 0 to 4 (all fraud cases, 30% of the others per
                     variant to keep training time sane; class weights handle the imbalance)
Adoption rule, fixed before reading test numbers: on month 5 (validation), mean ROC-AUC over the six populations at
least 0.003 higher than current, Base ROC-AUC no more than 0.003 lower, and the worst age parity (lower group
false alarm rate / higher) at least 0.8. Test months reported either way. --adopt writes the model and thresholds.
Writes results/retrain_l0_variants.json.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml")); sys.path.insert(0, os.path.join(ROOT, "service"))
from paths import WORK, MODELS, EXPORT  # noqa: E402
from lgbm_lite import LiteBooster  # noqa: E402
import lite_fast  # noqa: E402

DATA = os.environ.get("RUSETECT_DATA") or os.path.dirname(next(iter(sorted(glob.glob("/mnt/user-data/uploads/*/for_claude"))), "data/for_claude"))
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
cfg = json.load(open(os.path.join(MODELS, "l0_config.json")))
FEATS, CAT, CATS = cfg["features"], cfg["categorical"], cfg["categories"]
POPS = ["Base", "Variant_I", "Variant_II", "Variant_III", "Variant_IV", "Variant_V"]


def load(name):
    df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(os.path.join(DATA, "for_claude", f"baf_{name}_part*.csv.gz")))], ignore_index=True)
    for c in CAT:
        df[c] = pd.Categorical(df[c], categories=CATS[c])          # the serving model's category order
    return df[FEATS + ["fraud_bool", "month"]]


data = {p: load(p) for p in POPS}
log("loaded", {p: len(d) for p, d in data.items()})


def lite_matrix(df):
    return np.column_stack([df[k].cat.codes.replace(-1, np.nan).astype(float).to_numpy() if k in CAT else df[k].astype(float).to_numpy() for k in FEATS])


cur = LiteBooster(os.path.join(MODELS, "l0_onboarding_lgbm.json"))
predict = {"current": lambda df: lite_fast.predict(cur, lite_matrix(df))}


def fit(frames, tag):
    tr = pd.concat(frames, ignore_index=True); y = tr["fraud_bool"].to_numpy()
    spw = ((y == 0).sum() / (y == 1).sum()) ** 0.5
    log(tag, f"training rows {len(tr):,}, fraud {int(y.sum()):,}")
    m = lgb.LGBMClassifier(n_estimators=800, learning_rate=0.03, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.7, reg_lambda=1.0, scale_pos_weight=spw, n_jobs=2, verbose=-1).fit(tr[FEATS], y)
    log(tag, "trained")
    return m


rng = np.random.default_rng(0)
base_tr = data["Base"][data["Base"].month <= 4]
m_base = fit([base_tr], "retrain_base")
var_frames = [base_tr]
for p in POPS[1:]:
    d = data[p][data[p].month <= 4]
    keep = (d.fraud_bool == 1).to_numpy() | (rng.random(len(d)) < 0.30)
    var_frames.append(d[keep])
m_var = fit(var_frames, "retrain_variants")
del var_frames
predict["retrain_base"] = lambda df: m_base.predict_proba(df[FEATS])[:, 1]
predict["retrain_variants"] = lambda df: m_var.predict_proba(df[FEATS])[:, 1]


def evaluate(name):
    pb = predict[name](data["Base"]); b = data["Base"]; val = (b.month == 5).to_numpy(); yb = b.fraud_bool.to_numpy()
    old_b = (b.customer_age >= 50).to_numpy()
    q = lambda g: float(np.quantile(pb[val & g & (yb == 0)], 0.95))
    thr = {"age_50_plus": q(old_b), "age_under_50": q(~old_b)}
    out = {"thresholds": thr}
    for p in POPS:
        d = data[p]; s = pb if p == "Base" else predict[name](d); y = d.fraud_bool.to_numpy(); old = (d.customer_age >= 50).to_numpy()
        t = np.where(old, thr["age_50_plus"], thr["age_under_50"]); flag = s >= t
        r = {}
        for part, msk in (("validation_month_5", (d.month == 5).to_numpy()), ("test_months_6_7", (d.month >= 6).to_numpy())):
            yy, ff, oo, ss = y[msk].astype(bool), flag[msk], old[msk], s[msk]
            fpr = lambda g: float((ff & ~yy & g).sum() / max((~yy & g).sum(), 1))
            f, tp_, _ = roc_curve(yy, ss)
            r[part] = {"roc_auc": round(float(roc_auc_score(yy, ss)), 4), "recall_at_5pct_fpr": round(float(np.interp(0.05, f, tp_)), 4),
                       "recall": round(float((ff & yy).sum() / yy.sum()), 4), "fpr": round(fpr(np.ones_like(yy)), 4),
                       "fpr_parity": round(min(fpr(oo), fpr(~oo)) / max(fpr(oo), fpr(~oo)), 3)}
        out[p] = r
    log(name, {p: (out[p]["validation_month_5"]["roc_auc"], out[p]["test_months_6_7"]["roc_auc"]) for p in POPS})
    return out


res = {"candidates": {n: evaluate(n) for n in ("current", "retrain_base", "retrain_variants")}}
C, V = res["candidates"]["current"], res["candidates"]["retrain_variants"]
mean_val = lambda R: float(np.mean([R[p]["validation_month_5"]["roc_auc"] for p in POPS]))
adopt = (mean_val(V) >= mean_val(C) + 0.003 and V["Base"]["validation_month_5"]["roc_auc"] >= C["Base"]["validation_month_5"]["roc_auc"] - 0.003
         and min(V[p]["validation_month_5"]["fpr_parity"] for p in POPS) >= 0.8)
res["mean_validation_auc"] = {n: round(mean_val(R), 4) for n, R in res["candidates"].items()}
res["mean_test_auc"] = {n: round(float(np.mean([R[p]["test_months_6_7"]["roc_auc"] for p in POPS])), 4) for n, R in res["candidates"].items()}
res["adoption_rule"] = "month 5: mean AUC over six populations >= current + 0.003, Base AUC >= current - 0.003, worst parity >= 0.8"
res["adopt"] = bool(adopt)
if adopt and "--adopt" in sys.argv:
    import shutil
    for f in ("l0_onboarding_lgbm.json", "l0_config.json"):
        shutil.copy(os.path.join(MODELS, f), os.path.join(WORK, f"previous_{f}"))
    m_var.booster_.save_model(os.path.join(EXPORT, "l0_onboarding_lgbm.txt"))
    json.dump(m_var.booster_.dump_model(), open(os.path.join(MODELS, "l0_onboarding_lgbm.json"), "w"), separators=(",", ":"))
    thr = V["thresholds"]
    json.dump({**cfg, "age_50_plus": thr["age_50_plus"], "age_under_50": thr["age_under_50"],
               "trained_on": "BAF Base + Variants I to V, months 0 to 4 (variants: all fraud, 30% of others); x1, x2 dropped"},
              open(os.path.join(MODELS, "l0_config.json"), "w"), indent=1)
    log("adopted: L0 written to service/models (previous copies in work/)")
res["notes"] = ["All data synthetic (Feedzai BAF).", "Variant IV's months 6 to 7 repeat Variant III's applications, reordered."]
res["runtime_seconds"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(ROOT, "results", "retrain_l0_variants.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
