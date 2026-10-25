"""How much does each half of the live L2 average contribute? Scores the served models on the live path features of
HI-Small and LI-Small (from ml/build_live_features.py) exactly as the engine does: payments the L1 gatekeeper clears get
risk 0; the rest get XGBoost alone, LightGBM alone, or their average. Nothing is retrained.

Usage: RUSETECT_WORK=... python ml/l2_halves_check.py        writes results/l2_halves_check.json
"""
import json, os, sys
import numpy as np, xgboost as xgb, lightgbm as lgb
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
from paths import WORK, MODELS, EXPORT  # noqa: E402

cfg = json.load(open(os.path.join(MODELS, "cascade_config.json")))
l1 = xgb.Booster(); l1.load_model(os.path.join(MODELS, "l1_gatekeeper.json"))
l2x = xgb.Booster(); l2x.load_model(os.path.join(MODELS, "l2_xgb.json"))
l2l = lgb.Booster(model_file=os.path.join(EXPORT, "l2_lgbm.txt"))     # identical to the served LiteBooster (tests check this)

out = {"note": "average precision of the live score with each L2 half alone and with the served average; same L1 gate"}
for s in ("HI", "LI"):
    z = np.load(os.path.join(WORK, f"live_feat_{s}.npz"))
    assert list(z["features"]) == cfg["features"], "feature order differs from cascade_config"
    X, y, split = z["X"], z["y"].astype(np.int8), z["split"]
    out[s] = {}
    for part, k in (("validation", 1), ("test", 2)):
        m = split == k; Xp, yp = X[m], y[m]
        esc = l1.predict(xgb.DMatrix(Xp, feature_names=cfg["features"])) >= cfg["l1_threshold"]
        px, pl_ = np.zeros(len(yp)), np.zeros(len(yp))
        px[esc] = l2x.predict(xgb.DMatrix(Xp[esc], feature_names=cfg["features"])); pl_[esc] = l2l.predict(Xp[esc])
        out[s][part] = {"payments": int(m.sum()), "laundering": int(yp.sum()),
                        "xgboost_alone": round(float(average_precision_score(yp, px)), 4),
                        "lightgbm_alone": round(float(average_precision_score(yp, pl_)), 4),
                        "average_served": round(float(average_precision_score(yp, (px + pl_) / 2)), 4)}
        print(s, part, out[s][part], flush=True)
    del X, z
json.dump(out, open(os.path.join(ROOT, "results", "l2_halves_check.json"), "w"), indent=1)
