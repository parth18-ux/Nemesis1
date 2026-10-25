"""Typing rhythm check on the real Aalto 136M Keystrokes data: is the person typing the account owner?

Data: https://userinterfaces.aalto.fi/136Mkeystrokes/ (Keystrokes.zip, 168,593 participants, 15 sentences each; free for
non-commercial use with attribution: Dhakal, Feit, Kristensson, Oulasvirta, "Observations on Typing from 136 Million
Keystrokes", CHI 2018). The full zip is 1.6 GB, so the run uses two samples cut from it with Python's zipfile:
  sample   8,000 participants drawn uniformly at random (seed 0)             -> train / validation / test people
  boost    up to 3,000 participants aged 50+ and 3,000 on phone keyboards   -> extra test groups only (fairness)
No participant is in more than one role, and the people in the test group are never seen in training.

Per attempt the input is exactly what the live API receives (POST /api/v2/behaviour/check): a dict of mean key to key
times per pair of consecutive letters, in ms. Enrolment = a participant's first 8 sentences. Genuine attempts = their
sentences 9 to 14, scored one at a time and as two 3-sentence sessions. Impostor attempts = the same number of sentences
and 3-sentence sessions from other people in the same group.

Candidates
  current    the live TypingProfile: median z score over shared key pairs (threshold 2.5)
  log_dist   mean |log(attempt / owner's mean)| over shared key pairs
  verifier   a person-independent verifier (logistic regression or LightGBM) on 13 attempt-vs-profile features (service/behaviour.py, shared with the live API),
             trained on the train people: a Siamese style verifier in a form small enough to serve
Rule fixed before any test number is read: pick the candidate with the best validation ROC-AUC (mean of 1-sentence
and 3-sentence attempts); it replaces the live check only if it beats `current` by at least 0.02. The live threshold is
set on validation people to 5% false alarms on genuine 1-sentence attempts. Test people are scored once, at the end.

Usage: python ml/aalto_eval.py SAMPLE.zip [BOOST.zip] [--adopt]       writes results/aalto_eval.json
"""
import csv, io, json, math, os, pickle, random, sys, time, zipfile
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "ml"))
import behaviour  # noqa: E402
from paths import WORK, MODELS  # noqa: E402

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
SAMPLE, BOOST = ARGS[0], (ARGS[1] if len(ARGS) > 1 else None)
ENROL, NEED = 8, 14
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)


# ------------------------------------------------------------------ parsing
def sentences(text: str) -> list[dict]:
    """One {pair: mean ms} dict per typed sentence, in the order typed (letters and space only, gaps under 2 s)."""
    # plain tab split, indexed from the end (typed text can hold stray tabs or line breaks):
    # ... TEST_SECTION_ID(1) ... PRESS_TIME(-4) RELEASE_TIME(-3) LETTER(-2) KEYCODE(-1)
    by_section = {}
    for line in text.split("\n")[1:]:
        f = line.rstrip("\r").split("\t")
        if len(f) < 9:
            continue
        try:
            ev = (float(f[-4]), f[-2].lower())
        except ValueError:
            continue
        by_section.setdefault(f[1], []).append(ev)
    out = []
    for _, ev in sorted(by_section.items(), key=lambda kv: min(t for t, _ in kv[1])):
        ev.sort(); acc = {}
        for (t0, a), (t1, b) in zip(ev, ev[1:]):
            if len(a) == 1 and len(b) == 1 and (a.isalpha() or a == " ") and (b.isalpha() or b == " ") and 0 < t1 - t0 < 2000:
                acc.setdefault(a + b, []).append(t1 - t0)
        if len(acc) >= 5:
            out.append({k: float(np.mean(v)) for k, v in acc.items()})
    return out


def load(path, tag):
    cache = os.path.join(WORK, f"aalto_{tag}.pkl")
    if os.path.exists(cache) and os.path.getmtime(cache) > os.path.getmtime(path):
        return pickle.load(open(cache, "rb"))
    users, meta = {}, {}
    with zipfile.ZipFile(path) as zf:
        for n in zf.namelist():
            if n.endswith("metadata_participants.txt"):
                for m in csv.DictReader(io.StringIO(zf.read(n).decode("utf-8", "ignore")), delimiter="\t"):
                    meta[m["PARTICIPANT_ID"]] = {"age": int(m["AGE"]) if m["AGE"].isdigit() else None, "gender": m["GENDER"],
                                                 "keyboard": m["KEYBOARD_TYPE"], "wpm": float(m["AVG_WPM_15"] or 0)}
            elif n.endswith("_keystrokes.txt"):
                s = sentences(zf.read(n).decode("utf-8", errors="ignore"))
                if len(s) >= NEED:
                    users[os.path.basename(n).split("_")[0]] = s[:NEED]
    pickle.dump((users, meta), open(cache, "wb"))
    return users, meta


users, meta = load(SAMPLE, "sample")
log(f"sample: {len(users):,} participants with {NEED}+ usable sentences")
boost = {}
if BOOST:
    boost, _ = load(BOOST, "boost")
    boost = {u: s for u, s in boost.items() if u not in users}
    log(f"boost: {len(boost):,} participants")


# ------------------------------------------------------------------ attempts and features
class Profile:
    def __init__(self, sents):
        vals = {}
        for s in sents:
            for k, v in s.items():
                vals.setdefault(k, []).append(v)
        self.stats = {k: (float(np.mean(v)), float(np.std(v)), len(v)) for k, v in vals.items()}
        self.tp = behaviour.TypingProfile()
        for s in sents:
            self.tp.enrol(s)


def merge(sents):
    acc = {}
    for s in sents:
        for k, v in s.items():
            acc.setdefault(k, []).append(v)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def attempts(group, rnd):
    """(profile, attempt dict, label 1 = impostor, length, owner) for every person in the group."""
    ids = list(group); out = []
    for u in ids:
        p = Profile(group[u][:ENROL]); rest = group[u][ENROL:]
        gen = [(s, 1) for s in rest] + [(merge(rest[0:3]), 3), (merge(rest[3:6]), 3)]
        others = rnd.sample([x for x in ids if x != u], 6)
        imp = [(group[v][rnd.randrange(NEED)], 1) for v in others]
        imp += [(merge(group[v][(i := rnd.randrange(NEED - 2)):i + 3]), 3) for v in others[:2]]
        out += [(p, a, 0, n, u) for a, n in gen] + [(p, a, 1, n, u) for a, n in imp]
    return out


FEATURES = behaviour.VERIFIER_FEATURES


def features(p, a):
    return behaviour.verifier_features(p.stats, a)


def scores(rows, model=None):
    cur = np.array([np.nan if (r := p.tp.score(a))["score"] is None else r["score"] for p, a, *_ in rows])
    X = np.array([features(p, a) for p, a, *_ in rows])
    return cur, X


def evaluate(y, s, n, thr=None):
    out = {}
    for length in (1, 3):
        m = (n == length) & ~np.isnan(s)
        r = {"attempts": int(m.sum()), "unscored": int(((n == length) & np.isnan(s)).sum()),
             "roc_auc": round(float(roc_auc_score(y[m], s[m])), 4)}
        f, t, _ = roc_curve(y[m], s[m]); i = int(np.argmin(np.abs((1 - t) - f)))
        r["equal_error_rate"] = round(float((f[i] + 1 - t[i]) / 2), 4)
        if thr is not None:
            fl = s > thr
            r["false_alarm_rate"] = round(float(fl[m & (y == 0)].mean()), 4)
            r["impostors_caught"] = round(float(fl[m & (y == 1)].mean()), 4)
        out[f"{length}_sentence"] = r
    return out


rnd = random.Random(0)
ids = sorted(users); rnd.shuffle(ids)
cut1, cut2 = int(0.4 * len(ids)), int(0.6 * len(ids))
groups = {"train": {u: users[u] for u in ids[:cut1]}, "validation": {u: users[u] for u in ids[cut1:cut2]},
          "test": {u: users[u] for u in ids[cut2:]}}
rows = {g: attempts(d, random.Random(i + 1)) for i, (g, d) in enumerate(groups.items())}
log({g: len(r) for g, r in rows.items()}, "attempts")
data = {}
for g, r in rows.items():
    cur, X = scores(r)
    data[g] = {"y": np.array([x[2] for x in r]), "n": np.array([x[3] for x in r]), "owner": [x[4] for x in r], "current": cur, "X": X}
log("features done")

# ------------------------------------------------------------------ fit on train people
tr = data["train"]; ok = ~np.isnan(tr["X"]).any(1)
mu, sd = tr["X"][ok].mean(0), tr["X"][ok].std(0) + 1e-9
lr = LogisticRegression(C=1.0, max_iter=2000).fit((tr["X"][ok] - mu) / sd, tr["y"][ok])
import lightgbm as lgb  # noqa: E402
gbm = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=50, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=2).fit(tr["X"][ok], tr["y"][ok])


def cand_scores(d):
    X = d["X"]; bad = np.isnan(X).any(1); Xf = np.nan_to_num(X)
    lrs = lr.predict_proba((Xf - mu) / sd)[:, 1]; gb = gbm.predict_proba(Xf)[:, 1]
    lrs[bad] = np.nan; gb[bad] = np.nan
    logd = X[:, FEATURES.index("mean_abs_log_ratio")].copy()
    return {"current": d["current"], "log_dist": logd, "verifier_lr": lrs, "verifier_gbm": gb}


val = data["validation"]; vs = cand_scores(val)
res = {"dataset": "Aalto 136M Keystrokes (real; desktop, laptop and phone keyboards)", "source": "https://userinterfaces.aalto.fi/136Mkeystrokes/",
       "citation": "Dhakal, Feit, Kristensson, Oulasvirta. Observations on Typing from 136 Million Keystrokes. CHI 2018.",
       "people": {g: len(d) for g, d in groups.items()} | {"boost": len(boost)}, "enrol_sentences": ENROL,
       "validation": {k: evaluate(val["y"], s, val["n"]) for k, s in vs.items()}}
mean_auc = {k: float(np.mean([v["1_sentence"]["roc_auc"], v["3_sentence"]["roc_auc"]])) for k, v in res["validation"].items()}
best = max((k for k in mean_auc if k != "current"), key=mean_auc.get)
adopt = mean_auc[best] >= mean_auc["current"] + 0.02
chosen = best if adopt else "current"


def threshold(s, y, n):
    m = (n == 1) & (y == 0) & ~np.isnan(s)
    return float(np.quantile(s[m], 0.95))


thr = {k: threshold(s, val["y"], val["n"]) for k, s in vs.items()}
thr["current_live"] = behaviour.THRESHOLD
res["validation_mean_auc"] = {k: round(v, 4) for k, v in mean_auc.items()}
res["thresholds_5pct_false_alarms_on_validation"] = {k: round(v, 4) for k, v in thr.items()}
res["rule"] = "best validation ROC-AUC (mean of 1 and 3 sentence attempts); replaces the live check only if >= current + 0.02"
res["chosen"] = chosen; res["adopt"] = bool(adopt)
log("validation mean AUC", res["validation_mean_auc"], "chosen", chosen)

# ------------------------------------------------------------------ test people, once
te = data["test"]; ts = cand_scores(te)
res["test"] = {k: evaluate(te["y"], s, te["n"], thr[k]) for k, s in ts.items()}
res["test"]["current_at_live_threshold_2.5"] = evaluate(te["y"], ts["current"], te["n"], behaviour.THRESHOLD)


def fairness(d, s, t):
    """False alarm rate on genuine 1-sentence attempts per group (age, keyboard, gender)."""
    out = {}
    m = (d["n"] == 1) & (d["y"] == 0) & ~np.isnan(s); fl = s > t
    owners = np.array(d["owner"])
    def rate(sel):
        k = m & sel
        return {"genuine_attempts": int(k.sum()), "false_alarm_rate": round(float(fl[k].mean()), 4) if k.sum() else None}
    age = np.array([meta.get(o, {}).get("age") or -1 for o in owners])
    kb = np.array([meta.get(o, {}).get("keyboard", "?") for o in owners])
    gd = np.array([meta.get(o, {}).get("gender", "?") for o in owners])
    out["age_under_50"] = rate((age >= 0) & (age < 50)); out["age_50_plus"] = rate(age >= 50)
    for k in ("full", "laptop", "small", "on-screen"):
        out[f"keyboard_{k}"] = rate(kb == k)
    for g in ("female", "male"):
        out[f"gender_{g}"] = rate(gd == g)
    return out


res["test_fairness"] = {k: fairness(te, ts[k], thr[k]) for k in ("current", chosen)}

if boost:
    # older and phone participants: enrol, genuine attempts, impostors drawn from the random test people
    rb = random.Random(9); test_ids = list(groups["test"]); brow = []
    for u, s in boost.items():
        p = Profile(s[:ENROL]); rest = s[ENROL:]
        brow += [(p, a, 0, 1, u) for a in rest]
        brow += [(p, groups["test"][v][rb.randrange(NEED)], 1, 1, u) for v in rb.sample(test_ids, 6)]
    cur, X = scores(brow)
    bd = {"y": np.array([x[2] for x in brow]), "n": np.array([x[3] for x in brow]), "owner": [x[4] for x in brow], "current": cur, "X": X}
    bs = cand_scores(bd)
    res["boost_fairness"] = {k: fairness(bd, bs[k], thr[k]) for k in ("current", chosen)}
    res["boost_catch_rate"] = {k: round(float((bs[k] > thr[k])[(bd["y"] == 1) & ~np.isnan(bs[k])].mean()), 4) for k in ("current", chosen)}

if adopt and "--adopt" in sys.argv:
    import shutil
    if chosen == "verifier_lr":
        model = {"kind": "logistic", "features": FEATURES, "mean": mu.tolist(), "scale": sd.tolist(),
                 "coef": lr.coef_[0].tolist(), "intercept": float(lr.intercept_[0])}
    else:
        model = {"kind": "lightgbm", "features": FEATURES, "booster": gbm.booster_.dump_model()}
    model |= {"threshold": thr[chosen], "trained_on": f"Aalto 136M Keystrokes, {len(groups['train']):,} random participants",
              "threshold_rule": "5% false alarms on genuine 1-sentence attempts of validation participants"}
    json.dump(model, open(os.path.join(MODELS, "typing_verifier.json"), "w"), separators=(",", ":"))
    log("adopted:", chosen, "-> service/models/typing_verifier.json")

res["notes"] = ["One sentence is a short sample (about 40 characters); a banking session usually gives more typing.",
                "Aalto was collected on a typing test website; phone banking typing needs its own pilot data.",
                "Sentences differ between attempts, so the check compares shared key pairs, as the live API does."]
res["runtime_seconds"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(ROOT, "results", "aalto_eval.json"), "w"), indent=1)
print(json.dumps({k: res[k] for k in ("people", "validation_mean_auc", "chosen", "adopt")}, indent=1))
print(json.dumps(res["test"], indent=1))
print(json.dumps(res["test_fairness"], indent=1))
print(json.dumps(res.get("boost_fairness"), indent=1), res.get("boost_catch_rate"))
