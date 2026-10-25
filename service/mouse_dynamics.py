"""
Mouse dynamics (the "person" level, the mouse half): is the person at the mouse the account owner?

The SDK sends raw pointer events (time in seconds, x, y, button, state). They are cut into ACTIONS, the unit used in
the mouse biometrics literature (Antal and Egyed 2019, on the same Balabit data): a mouse move, a point and click
(a move that ends in a click), or a drag and drop. Each action becomes 24 numbers about its shape and speed.
A per customer model (trained on that customer's own actions against everyone else's) scores each action; a
session score is the mean over its actions, so a longer session gives a steadier answer.

Validated on the real Balabit Mouse Dynamics Challenge data (10 users, labelled impostor sessions):
ml/balabit_eval.py, results/balabit_eval.json. It is a SUPPORTING signal: a mismatch adds a small number of session
points (as typing_anomaly does) and can never block on its own.
"""
from __future__ import annotations
import math
import numpy as np

MAX_GAP_S = 3.0          # a pause longer than this ends an action
MIN_POINTS = 4
FEATURES = ["kind_move", "kind_click", "kind_drag", "duration", "travel", "straight", "straightness", "n_points",
            "v_mean", "v_std", "v_max", "vx_mean", "vy_mean", "a_mean", "a_std", "a_max", "jerk_mean",
            "curv_mean", "curv_std", "angle_change_mean", "largest_deviation", "dir_sin", "dir_cos", "pause_frac"]


def actions(events: list[tuple]) -> list[dict]:
    """events: (t_seconds, x, y, button, state) in time order. Returns actions with their points and kind."""
    out, cur, dragging = [], [], False
    for t, x, y, button, state in events:
        if button == "Scroll" or x > 10000 or y > 10000:            # wheel events and sentinel coordinates carry no shape
            continue
        if cur and t - cur[-1][0] > MAX_GAP_S:
            if len(cur) >= MIN_POINTS:
                out.append({"pts": cur, "kind": "move"})
            cur, dragging = [], False
        cur.append((t, x, y))
        if state == "Drag":
            dragging = True
        elif state == "Released":
            if len(cur) >= MIN_POINTS:
                out.append({"pts": cur, "kind": "drag" if dragging else "click"})
            cur, dragging = [], False
    if len(cur) >= MIN_POINTS:
        out.append({"pts": cur, "kind": "move"})
    return out


def features(action: dict) -> list[float]:
    p = np.asarray(action["pts"], dtype=float)
    t, x, y = p[:, 0], p[:, 1], p[:, 2]
    dt = np.diff(t); dt[dt <= 0] = 1e-3
    dx, dy = np.diff(x), np.diff(y); ds = np.hypot(dx, dy)
    v = ds / dt; vx = dx / dt; vy = dy / dt
    a = np.diff(v) / dt[1:] if len(v) > 1 else np.zeros(1)
    j = np.diff(a) / dt[2:] if len(a) > 1 else np.zeros(1)
    ang = np.arctan2(dy, dx); dang = np.abs(np.angle(np.exp(1j * np.diff(ang)))) if len(ang) > 1 else np.zeros(1)
    curv = dang / np.maximum(ds[1:], 1e-3) if len(ds) > 1 else np.zeros(1)
    travel = float(ds.sum()); straight = float(math.hypot(x[-1] - x[0], y[-1] - y[0]))
    # largest perpendicular distance from the straight start-end line
    if straight > 0:
        nx, ny = (y[-1] - y[0]) / straight, -(x[-1] - x[0]) / straight
        dev = float(np.max(np.abs((x - x[0]) * nx + (y - y[0]) * ny)))
    else:
        dev = 0.0
    theta = math.atan2(y[-1] - y[0], x[-1] - x[0])
    k = action["kind"]
    return [k == "move", k == "click", k == "drag", float(t[-1] - t[0]), travel, straight, straight / travel if travel else 0.0, len(p),
            float(v.mean()), float(v.std()), float(v.max()), float(np.abs(vx).mean()), float(np.abs(vy).mean()),
            float(np.abs(a).mean()), float(a.std()), float(np.abs(a).max()), float(np.abs(j).mean()),
            float(curv.mean()), float(curv.std()), float(dang.mean()), dev, math.sin(theta), math.cos(theta),
            float((dt > 0.1).mean())]


def session_matrix(events: list[tuple]) -> np.ndarray:
    acts = actions(events)
    return np.array([features(a) for a in acts], dtype=np.float32) if acts else np.zeros((0, len(FEATURES)), dtype=np.float32)


class MouseProfile:
    """Per customer model. fit(own, others) then score(session events) -> mean probability that it is the owner."""
    def __init__(self, n_estimators: int = 200, seed: int = 0):
        from sklearn.ensemble import ExtraTreesClassifier
        self.model = ExtraTreesClassifier(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=2, random_state=seed, class_weight="balanced")

    def fit(self, own: np.ndarray, others: np.ndarray):
        X = np.vstack([own, others]); y = np.r_[np.ones(len(own)), np.zeros(len(others))]
        self.model.fit(np.nan_to_num(X), y)
        return self

    def action_scores(self, M: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.nan_to_num(M))[:, 1] if len(M) else np.zeros(0)

    def score(self, events: list[tuple], max_actions: int | None = None) -> dict:
        s = self.action_scores(session_matrix(events)[:max_actions])
        return {"actions": int(len(s)), "owner_probability": round(float(s.mean()), 3) if len(s) else None}
