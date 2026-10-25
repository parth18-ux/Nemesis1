"""Does the trained message model make a better context signal than phrase rules, on messages it
did NOT train on? Template-grouped 5-fold CV on the curated Indian corpus (seed 7 fills, the Part 4 holdout):
each fold trains a fresh hashed TF-IDF + logistic regression (same featurizer as the served L-1 model) on 4/5 of
the TEMPLATES and scores the held-out templates. The out-of-fold band then feeds the context layer (variant D).
Thresholds = the served model's (caution 0.6155, high 0.7546). Writes results/fusion_oof.json."""
import json, os, sys
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "ml"))
from scam_text import counts, N_BUCKETS, hard_signals
from scam_corpus import generate
from context_rules import assess, PaymentContext

msgs = generate(n_per_template=3, seed=7)
texts = [m[0] for m in msgs]; y = np.array([m[1] for m in msgs]); groups = np.array([m[3] for m in msgs])
C = [counts(t) for t in texts]
def mat(idx, idf=None):
    rows, cols, vals = [], [], []
    for r, i in enumerate(idx):
        for h, c in C[i].items(): rows.append(r); cols.append(h); vals.append(1 + np.log(c))
    X = csr_matrix((vals, (rows, cols)), shape=(len(idx), N_BUCKETS), dtype=np.float32)
    if idf is None:
        df = np.bincount(X.indices, minlength=N_BUCKETS); idf = np.log((1 + len(idx)) / (1 + df)) + 1
    X = X.multiply(idf).tocsr(); n = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()) + 1e-12
    return csr_matrix(X.multiply(1 / n[:, None])), idf
ug = np.unique(groups); rng = np.random.default_rng(0); rng.shuffle(ug); fold_of = {g: i % 5 for i, g in enumerate(ug)}
fold = np.array([fold_of[g] for g in groups]); oof = np.zeros(len(y))
for k in range(5):
    tr, te = np.where(fold != k)[0], np.where(fold == k)[0]
    Xtr, idf = mat(tr); Xte, _ = mat(te, idf)
    m = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced").fit(Xtr, y[tr]); oof[te] = m.predict_proba(Xte)[:, 1]
CAU, HIGH = 0.6155, 0.7546
def band(p, t):
    hard = hard_signals(t)
    return "high" if (p >= HIGH or (hard and p >= CAU) or len(hard) >= 2) else "caution" if (p >= CAU or hard) else "safe"
bands = [band(p, t) for p, t in zip(oof, texts)]
pay = PaymentContext(new_beneficiary=True, amount_deviation=True)
res = {}
for name, use_model in (("B rules + negation (default ctx-1.1)", False), ("D + out-of-fold intent model", True)):
    rs = [assess(t, pay, True, model_band=(b if use_model else None)) for t, b in zip(texts, bands)]
    esc = np.array([r.band in ("HIGH", "CRITICAL") for r in rs]); ver = np.array([r.band != "LOW" for r in rs])
    res[name] = {"scam_escalated": round(esc[y == 1].mean(), 3), "scam_verify_or_more": round(ver[y == 1].mean(), 3),
                 "legit_escalated": round(esc[y == 0].mean(), 3), "legit_verify_or_more": round(ver[y == 0].mean(), 3)}
out = {"messages": len(y), "scam": int(y.sum()), "templates": int(len(ug)), "payment_setting": "new beneficiary + amount deviation (worst case)",
       "oof_model_bands": {"scam": {b: int(sum(1 for bb, yy in zip(bands, y) if bb == b and yy == 1)) for b in ("high", "caution", "safe")},
                           "legit": {b: int(sum(1 for bb, yy in zip(bands, y) if bb == b and yy == 0)) for b in ("high", "caution", "safe")}},
       "variants": res, "caveat": "Hard-signal list and corpus share authors (optimistic); synthetic template text, not real reported messages."}
json.dump(out, open(os.path.join(ROOT, "results", "fusion_oof.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
