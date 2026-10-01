"""
Train and evaluate the L-1 lure detector (scam message / call-transcript classifier).

Data
  * UCI SMS Spam Collection (5,574 real English SMS, 747 spam), file data/sms.tsv
    (download: https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv).
    Spam there is mostly UK prize / premium-rate scams and marketing; it is labelled 1.
  * ml/scam_corpus.py: Indian scam workflows (label 1) and Indian legitimate messages (label 0), synthetic, template-based.

Protocol
  * 5-fold grouped cross-validation. A group is one curated template (all its re-filled copies) or one distinct UCI text,
    so no test message is a near-copy of a training message.
  * Models compared on the same folds: rules only (scam_text.tag), Multinomial Naive Bayes, logistic regression on TF-IDF
    (plain and with Indian rows up-weighted), and an LR + NB blend.
  * Thresholds come from the out-of-fold scores: `high` = highest-recall threshold with precision >= 0.95 on Indian messages,
    `caution` = threshold that flags <= 10% of legitimate Indian messages. The final model is refit on all rows and exported.

Outputs: service/models/message_model.json, results/message_model_eval.json
"""
import json, os, sys, time
import numpy as np
from scipy import sparse
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(REPO, "service"))
import scam_corpus
from scam_text import N_BUCKETS, counts, hard_signals, normalize, tag

SEED = 0
DOMAIN_WEIGHT = 4.0      # Indian rows count 4x: the target is Indian UPI-era scams, UCI is older UK SMS
UCI = os.path.join(os.environ.get("RUSETECT_DATA") or os.environ.get("RUSETECT_DATA") or os.path.join(REPO, "data"), "sms.tsv")


def load():
    rows = []
    with open(UCI, encoding="utf-8") as fh:
        for line in fh:
            lab, txt = line.rstrip("\n").split("\t", 1)
            rows.append(dict(text=txt, y=int(lab == "spam"), source="uci", workflow="uci_spam" if lab == "spam" else "uci_ham", group="uci:" + normalize(txt)))
    for txt, y, wf, tid in scam_corpus.generate(n_per_template=14, seed=SEED):
        rows.append(dict(text=txt, y=y, source="india", workflow=wf, group="in:" + tid))
    return rows


def folds(groups, y, k=5, seed=SEED):
    """Grouped folds, balanced by sorting groups by label then dealing them round-robin."""
    g = sorted(set(groups)); r = np.random.default_rng(seed); r.shuffle(g)
    gy = {}
    for gi, yi in zip(groups, y): gy[gi] = max(gy.get(gi, 0), yi)
    g.sort(key=lambda x: gy[x])
    fold_of = {gi: i % k for i, gi in enumerate(g)}
    return np.array([fold_of[gi] for gi in groups])


def matrix(cnts):
    ind, ptr, dat = [], [0], []
    for c in cnts:
        ind.extend(c.keys()); dat.extend(c.values()); ptr.append(len(ind))
    return sparse.csr_matrix((np.array(dat, np.float64), np.array(ind), np.array(ptr)), shape=(len(cnts), N_BUCKETS))


def idf_of(C):
    n = C.shape[0]; df = np.bincount(C.indices, minlength=N_BUCKETS)
    return np.log((1 + n) / (1 + df)) + 1.0, df


def tfidf(C, idf):
    X = C.copy(); X.data = 1.0 + np.log(X.data); X = X.multiply(idf).tocsr()
    nrm = np.sqrt(np.asarray(X.multiply(X).sum(1)).ravel()); nrm[nrm == 0] = 1
    return sparse.diags(1 / nrm) @ X


def fit_lr(X, y, C=4.0, dom=None):
    """L2 logistic regression, classes balanced; dom multiplies each row's weight (domain weighting)."""
    w_pos = 0.5 * len(y) / y.sum(); w_neg = 0.5 * len(y) / (len(y) - y.sum())
    sw = np.where(y == 1, w_pos, w_neg) * (1.0 if dom is None else dom)
    def f(wb):
        w, b = wb[:-1], wb[-1]; z = X @ w + b
        p = 1 / (1 + np.exp(-z)); loss = np.sum(sw * (np.logaddexp(0, z) - y * z)) + 0.5 / C * w @ w
        g = sw * (p - y); return loss, np.r_[X.T @ g + w / C, g.sum()]
    res = minimize(f, np.zeros(X.shape[1] + 1), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return res.x[:-1], res.x[-1]


def fit_nb(C, y, alpha=0.1):
    lp = []
    for k in (0, 1):
        s = np.asarray(C[y == k].sum(0)).ravel() + alpha; lp.append(np.log(s / s.sum()))
    prior = np.log(np.array([(y == 0).mean(), (y == 1).mean()]))
    return lp, prior


def nb_prob(model, C):
    lp, prior = model; z = np.c_[C @ lp[0] + prior[0], C @ lp[1] + prior[1]]
    z -= z.max(1, keepdims=True); e = np.exp(z); return e[:, 1] / e.sum(1)


def rules_score(texts):
    out = []
    for t in texts:
        g = tag(t); w = g["workflows"][0]["matches"] if g["workflows"] else 0
        out.append(min(1.0, 0.25 * w + 0.12 * len([x for x in g["tactics"] if x["id"] != "link"])))
    return np.array(out)


def ap(y, p):
    o = np.argsort(-p, kind="stable"); y = y[o]; tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / max(y.sum(), 1))


def at(y, p, thr):
    pred = p >= thr; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(threshold=round(float(thr), 4), precision=round(pr, 4), recall=round(rc, 4), f1=round(2 * pr * rc / max(pr + rc, 1e-9), 4),
                tp=tp, fp=fp, fn=fn, flagged_legit_per_1000=round(1000 * fp / max((y == 0).sum(), 1), 2))


def main():
    t0 = time.time(); rows = load()
    y = np.array([r["y"] for r in rows]); texts = [r["text"] for r in rows]
    src = np.array([r["source"] for r in rows]); wf = np.array([r["workflow"] for r in rows])
    fold = folds([r["group"] for r in rows], y)
    C = matrix([counts(t) for t in texts])
    print(f"{len(rows):,} messages ({y.sum():,} scam) | india {np.sum(src=='india'):,} | uci {np.sum(src=='uci'):,} | {time.time()-t0:.0f}s", flush=True)

    ind = src == "india"; dom = np.where(ind, DOMAIN_WEIGHT, 1.0)
    oof = {m: np.zeros(len(y)) for m in ("Rules only", "Naive Bayes", "Logistic regression", "Logistic regression, India-weighted")}
    grid = {c: np.zeros(len(y)) for c in (1.0, 4.0, 16.0)}
    for k in range(5):
        tr, te = fold != k, fold == k
        idf, _ = idf_of(C[tr]); Xtr, Xte = tfidf(C[tr], idf), tfidf(C[te], idf)
        w, b = fit_lr(Xtr, y[tr], 16.0); oof["Logistic regression"][te] = 1 / (1 + np.exp(-(Xte @ w + b)))
        for c in grid:
            w, b = fit_lr(Xtr, y[tr], c, dom[tr]); grid[c][te] = 1 / (1 + np.exp(-(Xte @ w + b)))
        oof["Naive Bayes"][te] = nb_prob(fit_nb(C[tr], y[tr]), C[te])
        print(f"fold {k} done {time.time()-t0:.0f}s", flush=True)
    oof["Rules only"] = rules_score(texts)
    grid_ap = {c: ap(y[ind], p[ind]) for c, p in grid.items()}; best_c = max(grid_ap, key=grid_ap.get)
    oof["Logistic regression, India-weighted"] = grid[best_c]
    oof["Blend: LR India-weighted + NB"] = 0.5 * (grid[best_c] + oof["Naive Bayes"])

    # Thresholds for the served model (India-weighted LR), chosen on out-of-fold Indian scores:
    #   high    = lowest threshold whose precision is >= 0.95 (an interrupting warning must almost always be right)
    #   caution = lowest threshold that flags <= 10% of legitimate Indian messages (shown inline only, never pushed)
    pl = oof["Logistic regression, India-weighted"]; yi, pi = y[ind], pl[ind]
    cand = np.unique(np.round(pi, 4))
    hi = min([t for t in cand if at(yi, pi, t)["precision"] >= 0.95] or [0.9])
    ca = min([t for t in cand if at(yi, pi, t)["flagged_legit_per_1000"] <= 100] or [hi])
    ca = min(ca, hi)
    thr = {"caution": round(float(ca), 4), "high": round(float(hi), 4)}

    def report(p):
        out = {}
        for name, m in (("all", np.ones(len(y), bool)), ("india", ind), ("uci", src == "uci")):
            out[name] = dict(average_precision=round(ap(y[m], p[m]), 4), at_high=at(y[m], p[m], thr["high"]), at_caution=at(y[m], p[m], thr["caution"]))
        return out
    models = {m: report(p) for m, p in oof.items()}
    # rules have no learned threshold: report them at their own best-F1 cut on Indian messages
    pr = oof["Rules only"][ind]; rc = max(np.unique(pr), key=lambda t: at(yi, pr, t)["f1"])
    models["Rules only"]["india"]["at_best_f1"] = at(yi, pr, rc)

    # Served decision = model bands + hard-signal floor (scam_text.MessageModel.score). The hard-signal list was written
    # by the same people as the Indian templates, so its effect on this corpus is optimistic; its false-alarm count on
    # legitimate messages is the number that matters.
    n_hard = np.array([len(hard_signals(t)) for t in texts]); hard = n_hard > 0
    def served(p, nh):
        return np.where((p >= thr["high"]) | ((nh > 0) & (p >= thr["caution"])) | (nh >= 2), 2, np.where((p >= thr["caution"]) | (nh > 0), 1, 0))
    band = served(pl, n_hard)
    def band_stats(m):
        yy, bb = y[m], band[m]
        return {name: at(yy, (bb >= lvl).astype(float), 0.5) for name, lvl in (("caution_or_high", 1), ("high", 2))}
    served_eval = dict(india=band_stats(ind), uci=band_stats(src == "uci"),
                       hard_signal_hits=dict(india_scam=int((hard & ind & (y == 1)).sum()), india_legit=int((hard & ind & (y == 0)).sum()),
                                             uci_spam=int((hard & (src == "uci") & (y == 1)).sum()), uci_ham=int((hard & (src == "uci") & (y == 0)).sum())),
                       legit_examples_with_hard_signal=[texts[i] for i in np.where(hard & (y == 0))[0][:6]],
                       note="optimistic on the Indian corpus (same authors as the hard-signal list); see legit hits for false alarms")

    per_wf = {}
    for w in sorted(set(wf[ind])):
        m = ind & (wf == w)
        per_wf[w] = dict(n=int(m.sum()), scam=bool(y[m].max()), flagged_high=round(float((pl[m] >= thr["high"]).mean()), 3),
                         flagged_caution=round(float((pl[m] >= thr["caution"]).mean()), 3),
                         served_high=round(float((band[m] == 2).mean()), 3), served_any=round(float((band[m] >= 1).mean()), 3))
    misses = sorted([(float(pl[i]), texts[i], wf[i]) for i in np.where(ind & (y == 1) & (pl < thr["caution"]))[0]])[:8]
    false_alarms = sorted([(float(pl[i]), texts[i], wf[i]) for i in np.where(ind & (y == 0) & (pl >= thr["caution"]))[0]], reverse=True)[:8]

    # final model on all rows
    idf, df = idf_of(C); X = tfidf(C, idf); w, b = fit_lr(X, y, best_c, dom)
    keep = np.where(df > 0)[0]
    n = C.shape[0]
    os.makedirs(os.path.join(REPO, "service", "models"), exist_ok=True)
    json.dump(dict(buckets=keep.tolist(), idf=np.round(idf[keep], 5).tolist(), coef=np.round(w[keep], 5).tolist(),
                   idf_unseen=float(np.log(1 + n) + 1.0), intercept=float(b), thresholds=thr,
                   meta=dict(model="logistic regression on hashed TF-IDF (word 1-2 grams + char 3-5 grams), Indian rows weighted x4", C=best_c, n_buckets=N_BUCKETS,
                             trained_on=f"{n} messages: UCI SMS Spam Collection + curated Indian scam/legit corpus (synthetic)")),
              open(os.path.join(REPO, "service", "models", "message_model.json"), "w"), separators=(",", ":"))
    os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
    out = dict(task="Is this message / call transcript part of a scam workflow? (binary)",
               data=dict(total=len(y), scam=int(y.sum()), india=int(ind.sum()), india_scam=int(y[ind].sum()), uci=int((src == "uci").sum()), uci_spam=int(y[src == "uci"].sum()),
                         note="Indian rows are synthetic, template-based (ml/scam_corpus.py). UCI rows are real but UK-centric and older."),
               protocol="5-fold grouped CV; a group is one template (Indian) or one distinct text (UCI). Thresholds picked on out-of-fold Indian scores.",
               served_decision=served_eval, lr_C_grid_ap_india=grid_ap, domain_weight_india=DOMAIN_WEIGHT, chosen_C=best_c, thresholds=thr, models=models, india_by_workflow=per_wf,
               examples=dict(missed_scams=[dict(score=round(s, 3), workflow=w, text=t) for s, t, w in misses],
                             false_alarms=[dict(score=round(s, 3), workflow=w, text=t) for s, t, w in false_alarms]),
               runtime_seconds=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(REPO, "results", "message_model_eval.json"), "w"), indent=1, ensure_ascii=False)
    for m, r in models.items():
        print(f"{m:20s} AP all {r['all']['average_precision']:.3f} india {r['india']['average_precision']:.3f} uci {r['uci']['average_precision']:.3f} | "
              f"india@high P {r['india']['at_high']['precision']:.3f} R {r['india']['at_high']['recall']:.3f}")
    print("served india", {k: (v["precision"], v["recall"], v["flagged_legit_per_1000"]) for k, v in served_eval["india"].items()})
    print("hard hits", served_eval["hard_signal_hits"], served_eval["legit_examples_with_hard_signal"][:3])
    print("thresholds", thr, "C", best_c, f"{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
