"""
Dependency-free LightGBM inference from a `Booster.dump_model()` JSON file.

Why: the lightgbm wheel needs the system libgomp, which serverless runtimes (Vercel, AWS Lambda)
do not ship. Trees are flattened into NumPy arrays at load; prediction walks them in plain Python.
Supports what our models use: numerical '<=' and categorical '==' splits, missing_type None/NaN/Zero,
binary sigmoid objective.
"""
import json, math
import numpy as np


class LiteBooster:
    def __init__(self, path):
        m = path if isinstance(path, dict) else json.loads(open(path).read())
        self.sigmoid = 1.0
        obj = m.get("objective", "")
        if "sigmoid:" in obj:
            self.sigmoid = float(obj.split("sigmoid:")[1].split()[0])
        self.binary = obj.startswith("binary")
        self.feature_names = m["feature_names"]
        self.trees = [self._flatten(t["tree_structure"]) for t in m["tree_info"]]

    @staticmethod
    def _flatten(root):
        feat, thr, cats, kind, dleft, miss, left, right, leaf = [], [], [], [], [], [], [], [], []
        def add(n):
            i = len(feat)
            for lst in (feat, thr, cats, kind, dleft, miss, left, right, leaf): lst.append(None)
            if "split_index" not in n:
                feat[i] = -1; leaf[i] = float(n["leaf_value"]); return i
            feat[i] = n["split_feature"]; dleft[i] = bool(n["default_left"])
            miss[i] = {"None": 0, "Zero": 1, "NaN": 2}[n["missing_type"]]
            if n["decision_type"] == "==":
                kind[i] = 1; cats[i] = frozenset(int(c) for c in str(n["threshold"]).split("||"))
            else:
                kind[i] = 0; thr[i] = float(n["threshold"])
            left[i] = add(n["left_child"]); right[i] = add(n["right_child"])
            return i
        add(root)
        return feat, thr, cats, kind, dleft, miss, left, right, leaf

    @staticmethod
    def _walk(t, x):
        feat, thr, cats, kind, dleft, miss, left, right, leaf = t
        i = 0
        while feat[i] != -1:
            v = x[feat[i]]
            isnan = v != v
            if kind[i] == 0:
                mt = miss[i]
                if mt == 2 and isnan:
                    go_left = dleft[i]
                elif mt == 1 and (isnan or v == 0.0):
                    go_left = dleft[i]
                else:
                    if isnan: v = 0.0
                    go_left = v <= thr[i]
            else:
                go_left = (not isnan) and v >= 0 and int(v) in cats[i]
            i = left[i] if go_left else right[i]
        return leaf[i]

    def raw(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X[None, :]
        out = np.empty(len(X))
        for r, x in enumerate(X.tolist()):
            out[r] = sum(self._walk(t, x) for t in self.trees)
        return out

    def predict(self, X):
        z = self.raw(X)
        return 1.0 / (1.0 + np.exp(-self.sigmoid * z)) if self.binary else z
