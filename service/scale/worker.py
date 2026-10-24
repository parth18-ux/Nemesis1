"""
Stateless scoring worker: reads payments from one Redis Stream partition, keeps account state in Redis, scores the
payment cascade in batches and the context layer per payment, and writes decisions to the "decisions" stream.

  python -m service.scale.worker --redis redis://localhost:6379/0 --partition 0
Partition = hash(payer) mod partitions (see producer.py), so every payer's payments are handled in order by one worker.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE)); sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "ml"))
from rusetect_engine import Engine                                       # noqa: E402
from context_rules import assess, combine, from_engine_features          # noqa: E402
from scale.redis_state import RedisState                                 # noqa: E402

DECISION = {0: "Approve", 1: "Step-up verification", 2: "Hold for analyst", 3: "Block + STR draft"}


class Scorer:
    def __init__(self, r, prefix=""):
        import xgboost as xgb
        try:
            import lite_fast
            self._lgbm = lambda X: lite_fast.predict(self.eng.l2l, X)
        except ImportError:
            self._lgbm = lambda X: self.eng.l2l.predict(X)
        self.xgb = xgb; self.eng = Engine(load_models=True); self.state = RedisState(r, self.eng.cfg, prefix)

    def score(self, payments: list[dict]) -> list[dict]:
        feats, sfs = [], []
        for p in payments:
            feats.append(self.state.features(p)); sfs.append(self.state.last_sender_fan_in)
        X = np.array([[f[k] for k in self.eng.feats] for f in feats], dtype=np.float32)
        dm = self.xgb.DMatrix(X, feature_names=self.eng.feats); cfg = self.eng.cfg
        s1 = self.eng.l1.predict(dm); esc = s1 >= cfg["l1_threshold"]; risk = np.zeros(len(X))
        if esc.any():
            Xe = X[esc]; dme = self.xgb.DMatrix(Xe, feature_names=self.eng.feats)
            risk[esc] = (self.eng.l2x.predict(dme) + self._lgbm(Xe)) / 2
        out = []
        for i, p in enumerate(payments):
            rank = 0
            if esc[i]:
                rank = 3 if risk[i] >= cfg["block_threshold"] else 2 if risk[i] >= cfg["hold_threshold"] else 1 if risk[i] >= cfg["verify_threshold"] else 0
            ctx = assess(None, from_engine_features(feats[i], sfs[i]), True)
            c = combine(DECISION[rank], ctx)
            out.append({"eid": p.get("eid"), "payment_decision": DECISION[rank], "risk": round(float(risk[i]), 5) if esc[i] else None,
                        "context_band": ctx.band, "final_decision": c["final_decision"], "raised_by_context": c["raised_by_context"]})
        return out


def run(redis_url, partition, batch=256, group="rusetect", prefix=""):
    import redis
    r = redis.Redis.from_url(redis_url, decode_responses=True)
    stream = f"{prefix}payments:{partition}"
    try:
        r.xgroup_create(stream, group, id="0", mkstream=True)
    except redis.ResponseError:
        pass
    sc = Scorer(r, prefix); n = 0
    while True:
        resp = r.xreadgroup(group, f"worker-{partition}", {stream: ">"}, count=batch, block=2000)
        if not resp:
            continue
        msgs = resp[0][1]; stop = False; pays, ids, enq = [], [], []
        for mid, fields in msgs:
            if fields.get("stop"):
                stop = True; ids.append(mid); continue
            p = json.loads(fields["p"]); pays.append(p); ids.append(mid); enq.append(int(fields["enq_ns"]))
        if pays:
            res = sc.score(pays); now = time.time_ns(); pl = r.pipeline(transaction=False)
            for d, e in zip(res, enq):
                d["latency_ms"] = round((now - e) / 1e6, 2); d["partition"] = partition
                pl.xadd(f"{prefix}decisions", {"d": json.dumps(d)})
            pl.xack(stream, group, *ids); pl.execute(); n += len(pays)
        elif ids:
            r.xack(stream, group, *ids)
        if stop:
            r.xadd(f"{prefix}decisions", {"done": partition, "n": n})
            return n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    ap.add_argument("--partition", type=int, default=int(os.environ.get("PARTITION", 0)))
    ap.add_argument("--batch", type=int, default=256)
    a = ap.parse_args()
    print("scored", run(a.redis, a.partition, a.batch))
