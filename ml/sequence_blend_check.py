"""Would a sequence model improve the live L2 score? Blends the served L2 score with each sequence model's predictions
(from ml/bench_deep.py, saved in $RUSETECT_WORK/pred_*_va.npy and pred_*_te.npy) and compares average precision.

The rule, from docs/model-selection.md, is decided on validation only: a sequence model is worth adding only if the
blend beats the served score on validation by at least 0.01. The live engine does not compute 12 event sequences, so a
pass would mean engineering work, not a model swap. Blend = mean of percentile ranks (the scores are on different
scales). Test numbers are reported either way.

Usage: RUSETECT_WORK=... python ml/sequence_blend_check.py        writes results/sequence_blend_check.json
"""
import json, os, sys
import numpy as np, xgboost as xgb, lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "ml"))
from paths import WORK, MODELS, EXPORT  # noqa: E402
from common import load, to_np, ALL  # noqa: E402

cfg = json.load(open(os.path.join(MODELS, "cascade_config.json")))
assert cfg["features"] == ALL
l1 = xgb.Booster(); l1.load_model(os.path.join(MODELS, "l1_gatekeeper.json"))
l2x = xgb.Booster(); l2x.load_model(os.path.join(MODELS, "l2_xgb.json"))
l2l = lgb.Booster(model_file=os.path.join(EXPORT, "l2_lgbm.txt"))
d, split = load(); y = d["y"].to_numpy().astype(np.int8)
va, te = np.where(split == 1)[0], np.where(split == 2)[0]


def live(idx):
    X = to_np(d, ALL)[idx]; dm = xgb.DMatrix(X, feature_names=ALL)
    esc = l1.predict(dm) >= cfg["l1_threshold"]; r = np.zeros(len(idx))
    r[esc] = (l2x.predict(xgb.DMatrix(X[esc], feature_names=ALL)) + l2l.predict(X[esc])) / 2
    return r


rank = lambda s: rankdata(s) / len(s)
base = {"validation": live(va), "test": live(te)}
out = {"rule": "add a sequence model only if the rank blend beats the served L2 score on validation by >= 0.01 average precision",
       "served_l2": {p: round(float(average_precision_score(y[i], base[p])), 4) for p, i in (("validation", va), ("test", te))}, "blends": {}}
for name in ("TCN", "LSTM", "Transformer", "1D-CNN"):
    pv, pt = np.load(os.path.join(WORK, f"pred_{name}_va.npy")), np.load(os.path.join(WORK, f"pred_{name}_te.npy"))
    bv, bt = (rank(base["validation"]) + rank(pv)) / 2, (rank(base["test"]) + rank(pt)) / 2
    r = {"validation": round(float(average_precision_score(y[va], bv)), 4), "test": round(float(average_precision_score(y[te], bt)), 4)}
    r["passes_rule"] = bool(r["validation"] >= out["served_l2"]["validation"] + 0.01)
    out["blends"][f"served L2 + {name}"] = r
    print(name, r, flush=True)
print("served", out["served_l2"])
json.dump(out, open(os.path.join(ROOT, "results", "sequence_blend_check.json"), "w"), indent=1)
