"""
Typing rhythm check: is the person typing the account owner?

Enrol from at least three sessions of key to key timings for pairs of consecutive letters (digraphs, mean ms per
pair); at login or payment, compare the new session with the profile. A flag marks typing_anomaly in the session
signals, which adds 5 session points (and, on an unenrolled device, asks the customer to verify).

Two scorers, same input:
  * the person-independent verifier in service/models/typing_verifier.json when present: a small model over the
    features below, trained on real keystrokes of thousands of people (Aalto 136M Keystrokes, CHI 2018) and tested
    on people it never saw (ml/aalto_eval.py, results/aalto_eval.json);
  * otherwise the original statistic: median z score over shared key pairs, threshold 2.5.
Both stay a supporting signal: the typing check alone never holds a payment.
"""
from __future__ import annotations
import json, math, os, statistics

MIN_SESSIONS = 3
THRESHOLD = 2.5           # the median z statistic's threshold (used when no verifier model is installed)
MIN_PAIRS = 5

VERIFIER_FEATURES = ["median_z", "mean_z_clipped", "frac_z_over_2", "frac_z_over_3", "log_shared_pairs", "frac_shared",
                     "median_log_ratio", "abs_median_log_ratio", "mean_abs_log_ratio", "spread_log_ratio",
                     "owner_variability", "attempt_speed", "owner_speed"]


def verifier_features(stats: dict, attempt: dict) -> list[float]:
    """Attempt-vs-profile features. stats: {pair: (mean, population std, count)} from enrolment; attempt: {pair: ms}.
    Returns NaNs when fewer than MIN_PAIRS pairs were seen in at least MIN_SESSIONS enrolment sessions."""
    shared = [(float(v), stats[k]) for k, v in attempt.items() if k in stats and stats[k][2] >= MIN_SESSIONS and v > 0]
    if len(shared) < MIN_PAIRS or not attempt:
        return [math.nan] * len(VERIFIER_FEATURES)
    z = sorted(abs(v - mu) / max(sd, 0.15 * mu, 5.0) for v, (mu, sd, _) in shared)
    r = sorted(math.log(v / mu) for v, (mu, sd, _) in shared if mu > 0)
    med_r = statistics.median(r)
    return [statistics.median(z), statistics.fmean(min(x, 10.0) for x in z), sum(x > 2 for x in z) / len(z),
            sum(x > 3 for x in z) / len(z), math.log1p(len(shared)), len(shared) / len(attempt), med_r, abs(med_r),
            statistics.fmean(abs(x) for x in r), statistics.median(abs(x - med_r) for x in r),
            statistics.median(sd / mu for _, (mu, sd, _) in shared if mu > 0),
            math.log(statistics.median(float(v) for v in attempt.values() if v > 0)),
            math.log(statistics.median(mu for _, (mu, _, _) in shared))]


def _load_verifier():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "typing_verifier.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = json.load(f)
    if m.get("features") != VERIFIER_FEATURES:
        return None
    if m["kind"] == "lightgbm":
        from lgbm_lite import LiteBooster
        booster = LiteBooster(m["booster"])
        m["predict"] = lambda f: float(booster.predict([f])[0])
    else:
        mean, scale, coef, b = m["mean"], m["scale"], m["coef"], m["intercept"]
        m["predict"] = lambda f: 1.0 / (1.0 + math.exp(-(b + sum(c * (x - mu) / s for c, x, mu, s in zip(coef, f, mean, scale)))))
    return m


VERIFIER = _load_verifier()


class TypingProfile:
    def __init__(self):
        self.sessions = []

    def enrol(self, digraphs: dict):
        self.sessions.append({k: float(v) for k, v in digraphs.items()})

    @property
    def ready(self) -> bool:
        return len(self.sessions) >= MIN_SESSIONS

    def stats(self) -> dict:
        vals = {}
        for s in self.sessions:
            for k, v in s.items():
                vals.setdefault(k, []).append(v)
        return {k: (statistics.fmean(v), statistics.pstdev(v), len(v)) for k, v in vals.items()}

    def score(self, digraphs: dict) -> dict:
        """Original statistic: median z over shared pairs (kept for the explanation and as the fallback)."""
        if not self.ready:
            return {"ready": False, "score": None, "anomaly": False}
        zs = []
        for k, v in digraphs.items():
            vals = [s[k] for s in self.sessions if k in s]
            if len(vals) >= MIN_SESSIONS:
                mu = statistics.fmean(vals); sd = max(statistics.pstdev(vals), 0.15 * mu, 5.0)
                zs.append(abs(float(v) - mu) / sd)
        if len(zs) < MIN_PAIRS:
            return {"ready": True, "score": None, "anomaly": False, "note": "too few shared key pairs"}
        z = statistics.median(zs)
        return {"ready": True, "score": round(z, 2), "anomaly": z > THRESHOLD, "pairs": len(zs)}

    def check(self, digraphs: dict) -> dict:
        """What the API uses: the trained verifier when installed, else the median z statistic."""
        base = self.score(digraphs)
        if VERIFIER is None or not base["ready"]:
            return base | {"method": "median_z"}
        f = verifier_features(self.stats(), digraphs)
        if any(x != x for x in f):
            return base | {"method": "verifier", "score": None, "anomaly": False, "note": "too few shared key pairs"}
        p = VERIFIER["predict"](f)
        return {"ready": True, "method": "verifier", "score": round(p, 3), "anomaly": p > VERIFIER["threshold"],
                "threshold": round(VERIFIER["threshold"], 3), "median_z": base["score"], "pairs": base.get("pairs")}
