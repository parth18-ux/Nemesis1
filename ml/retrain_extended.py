"""Does adding the unused IBM LI-Small file to training make the live payment models better?

Inputs: $RUSETECT_WORK/live_feat_HI.npz and live_feat_LI.npz from ml/build_live_features.py (features from the
serving code, so there is no training/serving skew). Each file is split by time: 60% train, 20% validation, 20% test.

Three candidates, all scored exactly like the live path (L1 gatekeeper; survivors get the mean of L2 XGBoost and
L2 LightGBM; everything L1 clears scores 0):
  current       the models in service/models today (trained on HI-Small only)
  retrain_HI    same recipe as ml/cascade.py on HI-Small train weeks only (control: isolates the effect of new data
                from the effect of retraining itself)
  retrain_HI_LI same recipe on HI-Small + LI-Small train weeks
Training rows: every laundering payment in the train weeks + 10% of the others (ml/common.train_sample, seed 0).
L1 threshold keeps 99% of the validation laundering of the training data.
Metrics on each test set: PR-AUC, ROC-AUC, L1 pass rate and laundering kept, and laundering caught at the SAME alert
volume as the current models' verify and hold levels (so a better number means more laundering for the same work).

Adoption rule, fixed before looking at test numbers: retrain_HI_LI replaces the live models only if, on the
VALIDATION weeks, its PR-AUC is at least the current models' on HI-Small minus 0.005 and at least 0.01 higher on
LI-Small. Test weeks are reported either way. With --adopt the winning models are written to service/models with
thresholds that keep the current verify / hold / block alert rates on HI-Small validation weeks.
Writes results/retrain_extended.json.
"""
import json, os, shutil, sys, time
import numpy as np
import xgboost as xgb, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml")); sys.path.insert(0, os.path.join(ROOT, "service"))
from paths import WORK, MODELS, EXPORT  # noqa: E402
from common import train_sample  # noqa: E402
from lgbm_lite import LiteBooster  # noqa: E402
import lite_fast  # noqa: E402

T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
cfg = json.load(open(os.path.join(MODELS, "cascade_config.json")))
D = {}
for s in ("HI", "LI"):
    z = np.load(os.path.join(WORK, f"live_feat_{s}.npz"))
    assert list(z["features"]) == cfg["features"], "feature order differs from cascade_config"
    D[s] = {"X": z["X"], "y": z["y"].astype(np.int8), "split": z["split"]}
    log(s, f"{len(z['y']):,} payments; train/val/test laundering",
        [int(D[s]["y"][D[s]["split"] == k].sum()) for k in (0, 1, 2)])


class Live:
    """A cascade scored like Engine.score_payment, in batch."""
    def __init__(self, l1, l2x, l2l_predict, l1_thr):
        self.l1, self.l2x, self.l2l, self.thr = l1, l2x, l2l_predict, l1_thr

    def score(self, X):
        dm = xgb.DMatrix(X, feature_names=cfg["features"])
        s1 = self.l1.predict(dm); esc = s1 >= self.thr; out = np.zeros(len(X))
        if esc.any():
            Xe = X[esc]; out[esc] = (self.l2x.predict(xgb.DMatrix(Xe, feature_names=cfg["features"])) + self.l2l(Xe)) / 2
        return out, esc


def current():
    l1 = xgb.Booster(); l1.load_model(os.path.join(MODELS, "l1_gatekeeper.json"))
    l2x = xgb.Booster(); l2x.load_model(os.path.join(MODELS, "l2_xgb.json"))
    lb = LiteBooster(os.path.join(MODELS, "l2_lgbm.json"))
    return Live(l1, l2x, lambda X: lite_fast.predict(lb, X), cfg["l1_threshold"])


def train(sets, tag):
    idx = {s: train_sample(D[s]["y"], D[s]["split"]) for s in sets}
    Xtr = np.vstack([D[s]["X"][idx[s]] for s in sets]); ytr = np.concatenate([D[s]["y"][idx[s]] for s in sets])
    spw = float((ytr == 0).sum() / (ytr == 1).sum()) ** 0.5
    log(tag, f"training rows {len(ytr):,}, laundering {int(ytr.sum()):,}, scale_pos_weight {spw:.1f}")
    g = xgb.XGBClassifier(n_estimators=30, max_depth=3, learning_rate=0.3, tree_method="hist", n_jobs=2, scale_pos_weight=spw).fit(Xtr, ytr)
    mx = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist",
                           n_jobs=2, scale_pos_weight=spw).fit(Xtr, ytr)
    ml = lgb.LGBMClassifier(n_estimators=600, num_leaves=63, learning_rate=0.05, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                            n_jobs=2, verbose=-1, scale_pos_weight=spw).fit(Xtr, ytr)
    # L1 threshold: keep 99% of the validation laundering of the training data
    pv = np.concatenate([g.predict_proba(D[s]["X"][(D[s]["split"] == 1) & (D[s]["y"] == 1)])[:, 1] for s in sets])
    thr = float(np.sort(pv)[int(0.01 * len(pv))])
    l1, l2x = g.get_booster(), mx.get_booster()
    l1.feature_names = l2x.feature_names = cfg["features"]
    log(tag, f"trained, L1 threshold {thr:.4f}")
    return Live(l1, l2x, lambda X: ml.predict_proba(X)[:, 1], thr), (g, mx, ml)


def metrics(model, s, part, budgets=None):
    m = D[s]["split"] == part; X, y = D[s]["X"][m], D[s]["y"][m].astype(bool)
    sc, esc = model.score(X)
    out = {"payments": int(m.sum()), "laundering": int(y.sum()), "pr_auc": round(float(average_precision_score(y, sc)), 4),
           "roc_auc": round(float(roc_auc_score(y, sc)), 4), "l1_pass_rate": round(float(esc.mean()), 4),
           "l1_laundering_kept": round(float((esc & y).sum() / y.sum()), 4)}
    for name, rate in (budgets or {}).items():
        k = max(1, int(round(rate * len(y)))); top = np.argsort(-sc, kind="stable")[:k]
        out[f"caught_at_{name}_volume"] = {"alerts": k, "caught": int(y[top].sum()), "recall": round(float(y[top].mean() * k / y.sum()), 4),
                                          "precision": round(float(y[top].mean()), 4)}
    return out, sc


def rates(model, s, part):
    """Share of payments at or above the live verify / hold / block thresholds."""
    m = D[s]["split"] == part; sc, esc = model.score(D[s]["X"][m])
    return {k: float(((sc >= cfg[f"{k}_threshold"]) & esc).mean()) for k in ("verify", "hold", "block")}


cur = current()
budget = {s: {"verify": rates(cur, s, 2)["verify"], "hold": rates(cur, s, 2)["hold"]} for s in ("HI", "LI")}
res = {"candidates": {}}
models = {"current": cur}
models["retrain_HI"], _ = train(["HI"], "retrain_HI")
models["retrain_HI_LI"], arte = train(["HI", "LI"], "retrain_HI_LI")
for name, mdl in models.items():
    res["candidates"][name] = {s: {"validation": metrics(mdl, s, 1)[0], "test": metrics(mdl, s, 2, budget[s])[0]} for s in ("HI", "LI")}
    log(name, {s: (res["candidates"][name][s]["validation"]["pr_auc"], res["candidates"][name][s]["test"]["pr_auc"]) for s in ("HI", "LI")})

C, N = res["candidates"]["current"], res["candidates"]["retrain_HI_LI"]
adopt = (N["HI"]["validation"]["pr_auc"] >= C["HI"]["validation"]["pr_auc"] - 0.005 and
         N["LI"]["validation"]["pr_auc"] >= C["LI"]["validation"]["pr_auc"] + 0.01)
res["adoption_rule"] = "validation PR-AUC: HI >= current - 0.005 and LI >= current + 0.01 (fixed before reading test numbers)"
res["adopt"] = bool(adopt)
res["budgets_from_current_models"] = budget

if adopt and "--adopt" in sys.argv:
    g, mx, ml = arte; new = models["retrain_HI_LI"]
    # thresholds that keep the current verify / hold / block alert rates on HI-Small validation weeks
    target = rates(cur, "HI", 1); m = D["HI"]["split"] == 1; sc, esc = new.score(D["HI"]["X"][m]); live = np.sort(sc[esc])[::-1]
    thr = {k: float(live[max(0, int(round(target[k] * m.sum())) - 1)]) for k in ("verify", "hold", "block")}
    for f in ("l1_gatekeeper.json", "l2_xgb.json", "l2_lgbm.json", "cascade_config.json"):
        shutil.copy(os.path.join(MODELS, f), os.path.join(WORK, f"previous_{f}"))
    g.save_model(os.path.join(MODELS, "l1_gatekeeper.json")); mx.save_model(os.path.join(MODELS, "l2_xgb.json"))
    ml.booster_.save_model(os.path.join(EXPORT, "l2_lgbm.txt"))
    json.dump(ml.booster_.dump_model(), open(os.path.join(MODELS, "l2_lgbm.json"), "w"), separators=(",", ":"))
    new_cfg = {**cfg, "l1_threshold": new.thr, "verify_threshold": thr["verify"], "hold_threshold": thr["hold"], "block_threshold": thr["block"],
               "trained_on": "IBM AMLworld HI-Small + LI-Small train weeks (first 60% of each, by time), live path features",
               "thresholds_note": "verify / hold / block keep the previous models' alert rates on HI-Small validation weeks"}
    json.dump(new_cfg, open(os.path.join(MODELS, "cascade_config.json"), "w"), indent=1)
    res["adopted_thresholds"] = {"l1": new.thr, **thr}
    log("adopted: models written to service/models (previous copies in work/)")
res["notes"] = ["All data synthetic (IBM AMLworld).", "LI-Small test weeks were never used for training or for choosing anything.",
                "The current models were trained on offline features; retrain_HI uses the serving code's features, so the control also checks that the two agree."]
res["runtime_seconds"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(ROOT, "results", "retrain_extended.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "candidates"}, indent=1))
