"""Mouse dynamics on the real Balabit Mouse Dynamics Challenge data (https://github.com/balabit/Mouse-Dynamics-Challenge).

Protocol (the challenge's own split):
  * train: each user's training sessions are that user's genuine behaviour; for user u, positives are u's actions,
    negatives are actions from the other nine users' training sessions (sampled to 3x positives)
  * test: the test sessions in u's folder that have a public label (public_labels.csv: is_illegal 1 = someone else)
  * session score = mean owner probability over the first N actions (N = 1, 5, 10, 20, 50, all), to show how much
    behaviour a bank app needs to see before the signal is useful
Metrics: ROC-AUC and equal error rate per user, averaged over users, plus pooled over all sessions.
Writes results/balabit_eval.json.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from mouse_dynamics import session_matrix, MouseProfile  # noqa: E402
import statistics

SRC = os.environ.get("BALABIT_DIR") or sys.argv[1]
T0 = time.time()


def load(path):
    d = pd.read_csv(path)
    d = d.sort_values("client timestamp", kind="mergesort")
    return list(zip(d["client timestamp"].to_numpy(float), d["x"].to_numpy(float), d["y"].to_numpy(float),
                    d["button"].astype(str).to_numpy(), d["state"].astype(str).to_numpy()))


labels = pd.read_csv(os.path.join(SRC, "public_labels.csv")).set_index("filename")["is_illegal"].to_dict()
users = sorted(os.path.basename(p) for p in glob.glob(os.path.join(SRC, "training_files", "user*")))
train = {u: np.vstack([session_matrix(load(f)) for f in sorted(glob.glob(os.path.join(SRC, "training_files", u, "session_*")))]) for u in users}
print(f"[{time.time() - T0:4.0f}s] training actions per user:", {u: len(m) for u, m in train.items()}, flush=True)
tests = {}
for u in users:
    for f in sorted(glob.glob(os.path.join(SRC, "test_files", u, "session_*"))):
        if os.path.basename(f) in labels:
            tests.setdefault(u, []).append((int(labels[os.path.basename(f)]), session_matrix(load(f))))
print(f"[{time.time() - T0:4.0f}s] labelled test sessions:", {u: len(v) for u, v in tests.items()}, flush=True)

NS = [1, 5, 10, 20, 50, None]
rng = np.random.default_rng(0)
per_user, pooled = {}, {str(n): ([], []) for n in NS}


def eer(y, s):
    f, t, _ = roc_curve(y, s); fnr = 1 - t; i = int(np.nanargmin(np.abs(fnr - f)))
    return float((f[i] + fnr[i]) / 2)


for u in users:
    own = train[u]; others = np.vstack([train[v] for v in users if v != u])
    others = others[rng.choice(len(others), size=min(len(others), 3 * len(own)), replace=False)]
    prof = MouseProfile().fit(own, others)
    ts = [(lab, M) for lab, M in tests.get(u, []) if len(M) > 0]
    if not ts or len({lab for lab, _ in ts}) < 2:
        continue
    r = {"test_sessions": len(ts), "impostor_sessions": sum(lab for lab, _ in ts)}
    for n in NS:
        ys = np.array([lab for lab, _ in ts]); sc = np.array([1 - prof.action_scores(M[:n]).mean() for _, M in ts])   # impostor score
        r[f"auc_first_{n or 'all'}"] = round(float(roc_auc_score(ys, sc)), 3); r[f"eer_first_{n or 'all'}"] = round(eer(ys, sc), 3)
        pooled[str(n)][0].extend(ys.tolist()); pooled[str(n)][1].extend(sc.tolist())
    per_user[u] = r
    print(f"[{time.time() - T0:4.0f}s] {u}: {r}", flush=True)

summary = {}
for n in NS:
    k = n or "all"
    summary[f"first_{k}_actions"] = {"mean_user_auc": round(float(np.mean([r[f"auc_first_{k}"] for r in per_user.values()])), 3),
                                     "mean_user_eer": round(float(np.mean([r[f"eer_first_{k}"] for r in per_user.values()])), 3),
                                     "pooled_auc": round(float(roc_auc_score(*pooled[str(n)])), 3),
                                     "median_user_auc": round(statistics.median([r[f"auc_first_{k}"] for r in per_user.values()]), 3),
                                     "median_user_eer": round(statistics.median([r[f"eer_first_{k}"] for r in per_user.values()]), 3)}
out = {"dataset": "Balabit Mouse Dynamics Challenge (real, 10 users), public labelled test sessions",
       "source": "https://github.com/balabit/Mouse-Dynamics-Challenge", "model": "per user ExtraTrees on 24 action features",
       "summary": summary, "per_user": per_user,
       "caveats": ["Desktop mouse data from remote sessions, not phone touch; a banking app SDK would need its own enrolment data.",
                   "Pooled AUC mixes users with different score scales; the per user mean is the fairer number.",
                   "One user (user7) is perfectly separable even from a single action, most likely because of a different screen setup; the median user is the more honest summary."],
       "runtime_seconds": round(time.time() - T0, 1)}
json.dump(out, open(os.path.join(ROOT, "results", "balabit_eval.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
