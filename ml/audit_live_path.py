"""
Audit the LIVE payment path exactly as service/rusetect_engine.py scores it:
    risk = mean(L2 XGBoost, L2 LightGBM) if L1 XGBoost >= l1_threshold else 0
using the committed models and thresholds in service/models/ (nothing is retrained).

Reports, on the validation and test periods:
  * average precision of the live score, and of each L2 model alone
  * block-band precision / recall / F1 / false alarms / misses at the committed block_threshold
  * what a block threshold re-tuned on validation (max F1) for the live score would give on test (not applied)
Output: results/live_path_audit.json
Usage:  RUSETECT_WORK=... python ml/audit_live_path.py      (needs aml_feat.parquet from ml/aml_features.py)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import REPO, MODELS
import json
import numpy as np, xgboost as xgb
from common import load, to_np, ALL
from sklearn.metrics import average_precision_score, precision_recall_curve
sys.path.insert(0, os.path.join(REPO, "service"))
from lgbm_lite import LiteBooster

cfg = json.load(open(os.path.join(MODELS, "cascade_config.json")))
l1 = xgb.Booster(); l1.load_model(os.path.join(MODELS, "l1_gatekeeper.json"))
l2x = xgb.Booster(); l2x.load_model(os.path.join(MODELS, "l2_xgb.json"))
try:                                     # native LightGBM is identical to LiteBooster (tested) and much faster in bulk
    import lightgbm as lgb
    _bl = lgb.Booster(model_file=os.path.join(REPO, "ml", "exported", "l2_lgbm.txt")); l2l = _bl.predict
except Exception:
    l2l = LiteBooster(os.path.join(MODELS, "l2_lgbm.json")).predict

d, split = load(); y = d["y"].to_numpy()


def band(yt, r, thr):
    blk = r >= thr; tp = int((blk & (yt == 1)).sum()); fp = int((blk & (yt == 0)).sum()); fn = int((~blk & (yt == 1)).sum())
    p = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(threshold=float(thr), alerts=tp + fp, tp=tp, fp=fp, fn=fn, precision=p, recall=rc, f1=2 * p * rc / max(p + rc, 1e-12))


out = {}
for name, s in (("validation", 1), ("test", 2)):
    idx = np.where(split == s)[0]; X = to_np(d[idx], ALL); dm = xgb.DMatrix(X, feature_names=ALL)
    g, sx, sl = l1.predict(dm), l2x.predict(dm), l2l(X)
    risk = np.where(g >= cfg["l1_threshold"], (sx + sl) / 2, 0.0); yt = y[idx]
    out[name] = dict(rows=int(len(idx)), positives=int(yt.sum()),
                     average_precision=dict(live_score=float(average_precision_score(yt, risk)), l2_xgboost_alone=float(average_precision_score(yt, sx)),
                                            l2_lightgbm_alone=float(average_precision_score(yt, sl))),
                     l1_pass_rate=float((g >= cfg["l1_threshold"]).mean()), l1_recall=float(yt[g >= cfg["l1_threshold"]].sum() / yt.sum()),
                     block_band_committed_threshold=band(yt, risk, cfg["block_threshold"]),
                     hold_or_block_rate=float((risk >= cfg["hold_threshold"]).mean()), step_up_or_above_rate=float((risk >= cfg["verify_threshold"]).mean()))
    if s == 1:
        pr, rc, th = precision_recall_curve(yt, risk); f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
        retuned = float(th[int(np.nanargmax(f1[:-1]))])
    else:
        out[name]["block_band_if_retuned_on_validation"] = band(yt, risk, retuned)
out["note"] = ("block_threshold and hold_threshold in cascade_config.json keep the alert rates of the earlier thresholds, which "
               "ml/cascade.py fitted on the offline L3 blend score (ml/retrain_extended.py carried those rates over to the retrained models); "
               "the live engine applies them to mean(L2 XGBoost, L2 LightGBM). 'block_band_if_retuned_on_validation' shows the effect of "
               "re-fitting on the live score; it is NOT applied to the served configuration.")
os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
json.dump(out, open(os.path.join(REPO, "results", "live_path_audit.json"), "w"), indent=1)
t = out["test"]
print(json.dumps({k: t[k] for k in ("average_precision", "block_band_committed_threshold", "block_band_if_retuned_on_validation")}, indent=1))
