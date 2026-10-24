"""Vectorised batch inference for service/lgbm_lite.LiteBooster (same trees, same split rules), for offline runs over
millions of rows. The live API keeps the per-row walker. check() asserts both give identical scores."""
import numpy as np


def _arrays(t, max_cat):
    feat, thr, cats, kind, dleft, miss, left, right, leaf = t
    n = len(feat)
    A = {"feat": np.array(feat, dtype=np.int64), "thr": np.array([x if x is not None else 0.0 for x in thr]),
         "kind": np.array([k if k is not None else 0 for k in kind], dtype=np.int8),
         "dleft": np.array([bool(d) for d in dleft]), "miss": np.array([m if m is not None else 0 for m in miss], dtype=np.int8),
         "left": np.array([l if l is not None else 0 for l in left], dtype=np.int64),
         "right": np.array([r if r is not None else 0 for r in right], dtype=np.int64),
         "leaf": np.array([v if v is not None else 0.0 for v in leaf])}
    lut = np.zeros((n, max_cat + 1), dtype=bool)
    for i, c in enumerate(cats):
        if c:
            for v in c:
                if v <= max_cat:
                    lut[i, v] = True
    A["lut"] = lut
    return A


def raw(booster, X, max_cat=256):
    X = np.asarray(X, dtype=np.float64)
    if not hasattr(booster, "_fast"):
        booster._fast = [_arrays(t, max_cat) for t in booster.trees]
    out = np.zeros(len(X)); rows = np.arange(len(X))
    for A in booster._fast:
        node = np.zeros(len(X), dtype=np.int64)
        active = A["feat"][node] != -1
        while active.any():
            r, nd = rows[active], node[active]
            v = X[r, A["feat"][nd]]
            isnan = np.isnan(v)
            mt, dl = A["miss"][nd], A["dleft"][nd]
            num = np.where((mt == 2) & isnan, dl, np.where((mt == 1) & (isnan | (v == 0.0)), dl, np.where(isnan, 0.0, v) <= A["thr"][nd]))
            vi = np.clip(np.where(isnan, 0, v), 0, max_cat).astype(np.int64)
            catg = (~isnan) & (v >= 0) & A["lut"][nd, vi]
            go_left = np.where(A["kind"][nd] == 1, catg, num)
            node[active] = np.where(go_left, A["left"][nd], A["right"][nd])
            active = A["feat"][node] != -1
        out += A["leaf"][node]
    return out


def predict(booster, X):
    z = raw(booster, X)
    return 1.0 / (1.0 + np.exp(-booster.sigmoid * z)) if booster.binary else z


def check(booster, X, n=2000):
    idx = np.linspace(0, len(X) - 1, min(n, len(X))).astype(int)
    a, b = booster.predict(X[idx]), predict(booster, X[idx])
    assert np.allclose(a, b, atol=1e-9), f"fast LiteBooster mismatch: max diff {np.abs(a - b).max()}"
    return float(np.abs(a - b).max())
