"""
Typing rhythm check (experimental; the "person" level).

Enrol from at least three sessions of key to key timings for common key pairs (digraphs, in milliseconds); at login
or payment, compare the new session with the profile: the median over shared digraphs of |x - mean| / std. A score
above the threshold marks typing_anomaly in the session signals, which adds 5 session points.

Honest scope: session layer planned a BehaveFormer + Siamese model on the Aalto keystroke data (1,500 unseen users). That
data is not in this repository, so this is a simple statistical profile, tested only on synthetic timings
(tests/test_session_network.py). Treat it as a weak, supporting signal until it is validated on real keystroke data.
"""
from __future__ import annotations
import statistics

MIN_SESSIONS = 3
THRESHOLD = 2.5


class TypingProfile:
    def __init__(self):
        self.sessions = []

    def enrol(self, digraphs: dict):
        self.sessions.append({k: float(v) for k, v in digraphs.items()})

    @property
    def ready(self) -> bool:
        return len(self.sessions) >= MIN_SESSIONS

    def score(self, digraphs: dict) -> dict:
        if not self.ready:
            return {"ready": False, "score": None, "anomaly": False}
        zs = []
        for k, v in digraphs.items():
            vals = [s[k] for s in self.sessions if k in s]
            if len(vals) >= MIN_SESSIONS:
                mu = statistics.fmean(vals); sd = max(statistics.pstdev(vals), 0.15 * mu, 5.0)
                zs.append(abs(float(v) - mu) / sd)
        if len(zs) < 5:
            return {"ready": True, "score": None, "anomaly": False, "note": "too few shared key pairs"}
        z = statistics.median(zs)
        return {"ready": True, "score": round(z, 2), "anomaly": z > THRESHOLD, "pairs": len(zs)}
