"""Experiments on the labelled IBM AMLworld HI-Small slice bundled with the service (service/models/demo_stream.csv.gz).

All scores come from the served models via research_dump (Rust engine; Python XGBoost and LightGBM reproduce them to
3e-8, checked below). The slice is 160,000 consecutive payments of 2022/09/08; rows 0 to 99,999 build account state and
rows 100,000 to 159,999 (16:12 to 19:16, 97 laundering payments) are the evaluation window, as in the console demo.

IMPORTANT: this is NOT the full HI-Small protocol (4.06M rows of history, 1.02M test rows). Account state here holds about
five hours of history, so absolute numbers are lower than results/full_eval_HI.json. The experiments measure effects
(differences, robustness, calibration, attribution) on the served models; every number is written to
research/out/slice_results.json with its sample size and a bootstrap interval where one applies.

Usage: python3 -I research/analyze_slice.py   (after research/make_streams.py)
"""
import json, os
import numpy as np, pandas as pd
import xgboost as xgb, lightgbm as lgb
from scipy import stats
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S = os.path.join(ROOT, "research", "out", "streams")
OUT = os.path.join(ROOT, "research", "out")
CFG = json.load(open(os.path.join(ROOT, "service", "models", "cascade_config.json")))
L1T, VER, HOLD, BLK = CFG["l1_threshold"], CFG["verify_threshold"], CFG["hold_threshold"], CFG["block_threshold"]
BLK_RETUNED = 0.9345272756338533  # results/full_eval_HI.json, fitted on full validation, not served
EVAL_FROM = 100_000
RNG = np.random.default_rng(20261111)
NB = 2000
R = {"meta": {
    "data": "service/models/demo_stream.csv.gz: 160,000 consecutive IBM AMLworld HI-Small payments (2022/09/08 11:06 to 19:16)",
    "evaluation_window": "rows 100,000 to 159,999; account state built from rows 0 to 99,999 only",
    "thresholds": {"l1": L1T, "verify": VER, "hold": HOLD, "block": BLK, "block_retuned_full_validation": BLK_RETUNED},
    "bootstrap": f"{NB} resamples, percentile 95% intervals; 'block' resamples 10 minute blocks to respect time dependence",
    "caveat": "five hours of account history instead of about thirteen days; absolute metrics are not comparable with the full protocol"}}


def load(name):
    d = pd.read_csv(os.path.join(S, name + "_dump.csv"))
    d["orig"] = np.loadtxt(os.path.join(S, name + ".origin"), dtype=np.int64)
    return d


def live(d):
    return np.where(d.l1 < L1T, d.l1, (d.l2x + d.l2l) / 2)


def band(r):
    return np.select([r >= BLK, r >= HOLD, r >= VER], [3, 2, 1], 0)


def ci(vals):
    v = np.asarray(vals)
    return [round(float(np.nanpercentile(v, 2.5)), 4), round(float(np.nanpercentile(v, 97.5)), 4)]


def boot(y, scores: dict, fn, blocks=None):
    """Paired bootstrap of fn(y, s) for several score vectors; returns point, CI and CIs of differences to the first."""
    n = len(y)
    keys = list(scores)
    pts = {k: fn(y, scores[k]) for k in keys}
    samples = {k: [] for k in keys}
    if blocks is not None:
        ub = np.unique(blocks)
        members = {b: np.where(blocks == b)[0] for b in ub}
    for _ in range(NB):
        if blocks is None:
            ix = RNG.integers(0, n, n)
        else:
            ix = np.concatenate([members[b] for b in RNG.choice(ub, len(ub))])
        if y[ix].sum() == 0:
            continue
        for k in keys:
            samples[k].append(fn(y[ix], scores[k][ix]))
    out = {k: {"point": round(pts[k], 4), "ci95": ci(samples[k])} for k in keys}
    base = np.array(samples[keys[0]])
    for k in keys[1:]:
        dlt = np.array(samples[k]) - base
        out[k]["minus_" + keys[0]] = {"point": round(pts[k] - pts[keys[0]], 4), "ci95": ci(dlt),
                                      "share_of_resamples_above_0": round(float((dlt > 0).mean()), 3)}
    return out


def pr_at(y, flag):
    tp = int((flag & (y == 1)).sum()); n = int(flag.sum())
    return {"flags": n, "tp": tp, "fp": n - tp, "precision": round(tp / n, 4) if n else None,
            "recall": round(tp / max(1, int(y.sum())), 4)}


def ece(y, p, bins=15):
    """Expected calibration error with equal mass bins (scores are very skewed)."""
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); q[-1] += 1e-12
    e = 0.0
    for a, b in zip(q[:-1], q[1:]):
        m = (p >= a) & (p < b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return e


def main():
    raw = pd.read_csv(os.path.join(ROOT, "service", "models", "demo_stream.csv.gz"), dtype=str)
    d = load("base")
    F = list(d.columns[7:-1])
    X = d[F].values.astype(np.float32)

    # ---------------------------------------------------------------- parity of the offline libraries
    bx = xgb.Booster(); bx.load_model(os.path.join(ROOT, "service/models/l2_xgb.json"))
    b1 = xgb.Booster(); b1.load_model(os.path.join(ROOT, "service/models/l1_gatekeeper.json"))
    bl = lgb.Booster(model_file=os.path.join(ROOT, "ml/exported/l2_lgbm.txt"))
    def score(Xm):
        dm = xgb.DMatrix(Xm, missing=np.nan, feature_names=F)
        return b1.predict(dm), bx.predict(dm), bl.predict(Xm.astype(np.float64))
    s1, sx, sl = score(X)
    R["parity"] = {"max_abs_diff_l1": float(np.abs(s1 - d.l1).max()), "max_abs_diff_l2_xgb": float(np.abs(sx - d.l2x).max()),
                   "max_abs_diff_l2_lgbm": float(np.abs(sl - d.l2l).max()),
                   "note": "Python xgboost/lightgbm on the Rust feature vectors vs the Rust served scores"}

    # ---------------------------------------------------------------- A. dataset audit
    y_all = raw.y.astype(int).values
    usd = d.usd.values
    acc_s = raw.fb + ":" + raw.fa; acc_d = raw.tb + ":" + raw.ta
    by_fmt = raw.assign(y=y_all).groupby("fmt").y.agg(["count", "sum"])
    by_fmt["rate"] = by_fmt["sum"] / by_fmt["count"]
    by_cur = raw.assign(y=y_all).groupby("cur_paid").y.agg(["count", "sum"])
    launder_accts = set(acc_s[y_all == 1]) | set(acc_d[y_all == 1])
    R["A_dataset_audit"] = {
        "rows": len(raw), "laundering": int(y_all.sum()), "prevalence": round(float(y_all.mean()), 6),
        "eval_rows": int((d.row >= EVAL_FROM).sum()), "eval_laundering": int(d.y[d.row >= EVAL_FROM].sum()),
        "time_span": [raw.ts.iloc[0], raw.ts.iloc[-1]],
        "exact_duplicate_rows": int(raw.duplicated().sum()), "empty_fields": int((raw == "").sum().sum()),
        "accounts": int(len(set(acc_s) | set(acc_d))), "accounts_touching_laundering": len(launder_accts),
        "self_loops": int((acc_s == acc_d).sum()),
        "cross_currency_share": round(float((raw.cur_paid != raw.cur_recv).mean()), 4),
        "laundering_by_format": {k: {"payments": int(v["count"]), "laundering": int(v["sum"]), "rate": round(float(v["rate"]), 5)}
                                 for k, v in by_fmt.iterrows()},
        "laundering_by_currency": {k: int(v["sum"]) for k, v in by_cur.iterrows() if v["sum"] > 0},
        "amount_usd_median": {"legit": round(float(np.median(usd[y_all == 0])), 2), "laundering": round(float(np.median(usd[y_all == 1])), 2)},
        "amount_mannwhitney_p": float(stats.mannwhitneyu(usd[y_all == 1], usd[y_all == 0]).pvalue),
        "ach_share_of_laundering": round(float(by_fmt.loc["ACH", "sum"] / y_all.sum()), 4),
        "labels_per_payment_only": "no typology, ring id or account label in the slice; ring and typology analyses need the full HI-Small patterns file",
    }

    # ---------------------------------------------------------------- B. feature intelligence (evaluation window)
    ev = d.row.values >= EVAL_FROM
    ye = d.y.values[ev]
    Xe = pd.DataFrame(X[ev], columns=F)
    uni = {}
    for f in F:
        v = Xe[f].values.astype(float)
        miss = np.isnan(v)
        fill = np.where(miss, np.nanmin(v) - 1 if (~miss).any() else 0, v)
        a = AUC(ye, fill) if len(np.unique(fill)) > 1 else 0.5
        uni[f] = {"missing_share": round(float(miss.mean()), 4), "univariate_auc": round(float(max(a, 1 - a)), 4),
                  "direction": "higher is riskier" if a >= 0.5 else "lower is riskier",
                  "mean_legit": round(float(np.nanmean(v[ye == 0])), 3), "mean_laundering": round(float(np.nanmean(v[ye == 1])), 3)}
    rho = Xe.rank().corr(method="pearson").values  # Spearman with pairwise NaN handling via ranks
    pairs = [(F[i], F[j], round(float(rho[i, j]), 3)) for i in range(len(F)) for j in range(i + 1, len(F))
             if abs(rho[i, j]) >= 0.85]
    # history truncation: the same features averaged by how much history the window had
    R["B_feature_intelligence"] = {
        "univariate_ranked": dict(sorted(uni.items(), key=lambda kv: -kv[1]["univariate_auc"])),
        "redundant_pairs_spearman_ge_0.85": sorted(pairs, key=lambda z: -abs(z[2])),
        "constant_features": [f for f in F if Xe[f].nunique(dropna=True) <= 1],
    }

    # ---------------------------------------------------------------- C. headline metrics with intervals
    de = d[ev].reset_index(drop=True)
    r_live = live(de)
    r_x = np.where(de.l1 < L1T, de.l1, de.l2x)
    r_l = np.where(de.l1 < L1T, de.l1, de.l2l)
    tmin = pd.to_datetime(raw.ts.iloc[EVAL_FROM:].values, format="%Y/%m/%d %H:%M")
    blocks10 = ((tmin - tmin[0]).total_seconds() // 600).astype(int).values
    sc = {"live_mean": r_live, "xgb_half_only": r_x, "lgbm_half_only": r_l}
    R["C_headline_with_intervals"] = {
        "pr_auc_iid": boot(ye, sc, lambda y, s: AP(y, s)),
        "pr_auc_time_block": boot(ye, sc, lambda y, s: AP(y, s), blocks10),
        "roc_auc_iid": boot(ye, sc, lambda y, s: AUC(y, s)),
        "l1_pass_rate": round(float((de.l1 >= L1T).mean()), 4),
        "l1_laundering_kept": round(float((de.l1[ye == 1] >= L1T).mean()), 4),
    }
    bands = {}
    for name, t in [("verify_or_more", VER), ("hold_or_more", HOLD), ("block_served", BLK), ("block_retuned", BLK_RETUNED)]:
        m = pr_at(ye, r_live >= t)
        pb, rb = [], []
        for _ in range(NB):
            ix = RNG.integers(0, len(ye), len(ye)); f = r_live[ix] >= t; yy = ye[ix]
            if f.sum():
                pb.append((yy[f] == 1).mean())
            rb.append((f & (yy == 1)).sum() / max(1, yy.sum()))
        m["precision_ci95"] = ci(pb); m["recall_ci95"] = ci(rb)
        bands[name] = m
    R["C_headline_with_intervals"]["bands"] = bands

    # ---------------------------------------------------------------- D. calibration and threshold transfer
    # chronological halves of the evaluation window: fit calibrators on the first, judge on the second
    half = len(de) // 2
    cal, tst = slice(0, half), slice(half, None)
    gate = (de.l1 >= L1T).values
    D = {"split": {"calibration_rows": half, "calibration_laundering": int(ye[cal].sum()),
                   "test_rows": len(de) - half, "test_laundering": int(ye[tst].sum())}}
    raw_scores = {"live_mean_uncalibrated": r_live, "xgb": de.l2x.values, "lgbm": de.l2l.values}
    for k, s in raw_scores.items():
        D[k] = {"brier": float(np.mean((s[tst] - ye[tst]) ** 2)), "ece": float(ece(ye[tst], s[tst])),
                "mean_score": float(s[tst].mean()), "observed_rate": float(ye[tst].mean())}
    def iso(s):
        m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(s[cal][gate[cal]], ye[cal][gate[cal]])
        return m.predict(s)
    def platt(s):
        z = np.log(np.clip(s, 1e-7, 1 - 1e-7) / (1 - np.clip(s, 1e-7, 1 - 1e-7))).reshape(-1, 1)
        m = LogisticRegression(C=1e6, max_iter=1000).fit(z[cal][gate[cal]], ye[cal][gate[cal]])
        return m.predict_proba(z)[:, 1]
    for nm, f in [("isotonic", iso), ("platt", platt)]:
        c = (f(de.l2x.values) + f(de.l2l.values)) / 2
        c = np.where(gate, c, 0.0)
        D[f"mean_of_{nm}_calibrated_halves"] = {
            "brier": float(np.mean((c[tst] - ye[tst]) ** 2)), "ece": float(ece(ye[tst], c[tst])),
            "pr_auc_test_half": round(float(AP(ye[tst], c[tst])), 4)}
    D["pr_auc_test_half_uncalibrated"] = round(float(AP(ye[tst], r_live[tst])), 4)
    # threshold transfer: pick the block threshold for precision >= 0.9 on the calibration half, apply to the test half
    def thr_for_precision(y, s, target=0.9, min_tp=5):
        o = np.argsort(-s); ys = y[o]; tp = np.cumsum(ys); n = np.arange(1, len(ys) + 1)
        ok = np.where((tp / n >= target) & (tp >= min_tp))[0]
        return float(s[o][ok[-1]]) if len(ok) else None
    tt = {}
    for nm, s in [("uncalibrated_mean", r_live), ("xgb_half_only", r_x)]:
        t = thr_for_precision(ye[cal], s[cal])
        tt[nm] = {"threshold_fitted_on_first_half": t, "second_half": pr_at(ye[tst], s[tst] >= t) if t else None}
    D["threshold_transfer_precision_0.9_target"] = tt
    D["served_block_on_second_half"] = pr_at(ye[tst], r_live[tst] >= BLK)
    R["D_calibration"] = D

    # ---------------------------------------------------------------- E. operational metrics a bank would read
    hours = (len(de) and (tmin[-1] - tmin[0]).total_seconds() / 3600)
    o = np.argsort(-r_live)
    E = {"window_hours": round(hours, 2), "payments_per_hour": round(len(de) / hours, 1)}
    E["precision_at_alert_budget"] = {}
    for per10k in (1, 5, 10, 25, 50, 100):
        k = int(round(len(de) * per10k / 10_000))
        tp = int(ye[o[:k]].sum())
        E["precision_at_alert_budget"][f"{per10k}_per_10k_payments"] = {"alerts": k, "tp": tp, "precision": round(tp / k, 4),
                                                                        "recall": round(tp / ye.sum(), 4)}
    neg = np.sort(r_live[ye == 0])[::-1]
    E["recall_at_fpr"] = {}
    for fpr in (0.0005, 0.001, 0.005, 0.01):
        t = neg[int(fpr * len(neg))]
        E["recall_at_fpr"][str(fpr)] = {"threshold": float(t), "recall": round(float((r_live[ye == 1] > t).mean()), 4)}
    ue = de.usd.values
    E["amount_weighted_recall"] = {nm: {"count_recall": round(float((r_live[ye == 1] >= t).mean()), 4),
                                        "usd_recall": round(float(ue[(ye == 1) & (r_live >= t)].sum() / ue[ye == 1].sum()), 4)}
                                   for nm, t in [("verify", VER), ("hold", HOLD), ("block", BLK)]}
    # cost: missed laundering costs its USD amount (capped) times c, a false hold costs a fixed review + friction cost
    cost = {}
    for ratio in (10, 50, 100, 500):
        grid = np.unique(np.quantile(r_live[gate], np.linspace(0, 1, 400)))
        best = min(((float(((r_live >= t) & (ye == 0)).sum() + ratio * ((r_live < t) & (ye == 1)).sum()), t) for t in grid))
        cost[f"fn_cost_{ratio}x_fp"] = {"cost_minimising_threshold": round(best[1], 4), "cost": best[0],
                                         "cost_at_served_verify": float(((r_live >= VER) & (ye == 0)).sum() + ratio * ((r_live < VER) & (ye == 1)).sum()),
                                         "cost_at_served_hold": float(((r_live >= HOLD) & (ye == 0)).sum() + ratio * ((r_live < HOLD) & (ye == 1)).sum()),
                                         "cost_at_served_block": float(((r_live >= BLK) & (ye == 0)).sum() + ratio * ((r_live < BLK) & (ye == 1)).sum())}
    E["cost_curve_count_based"] = cost
    R["E_operational"] = E

    # ---------------------------------------------------------------- F. missing field robustness
    fi = F.index("fmt_code"); ci_ = F.index("cur_code")
    Xev = X[ev].copy()
    true_fmt = de.fmt.values
    G = {}
    variants = {"true_format": Xev}
    Xn = Xev.copy(); Xn[:, fi] = np.nan; variants["format_missing_as_nan"] = Xn
    Xa = Xev.copy(); Xa[:, fi] = CFG["fmt_codes"]["ACH"]; variants["format_missing_defaulted_to_ACH (current API)"] = Xa
    for nm, Xm in variants.items():
        a1, ax, al = score(Xm)
        r = np.where(a1 < L1T, a1, (ax + al) / 2)
        G[nm] = {"pr_auc": round(float(AP(ye, r)), 4), "bands": {b: int((band(r) == i).sum()) for i, b in enumerate(["approve", "verify", "hold", "block"])},
                 "legit_flagged_verify_plus": int(((r >= VER) & (ye == 0)).sum())}
        if nm != "true_format":
            G[nm]["decisions_changed_vs_true"] = int((band(r) != band(r_live)).sum())
            G[nm]["escalated_vs_true"] = int((band(r) > band(r_live)).sum())
            G[nm]["escalated_non_ach_legit"] = int(((band(r) > band(r_live)) & (ye == 0) & (true_fmt != "ACH")).sum())
    R["F_missing_field"] = G

    # ---------------------------------------------------------------- G. payment side evasion
    lau = ye == 1
    H = {"laundering_payments_in_window": int(lau.sum()),
         "baseline": {b: round(float((r_live[lau] >= t).mean()), 4) for b, t in [("verify", VER), ("hold", HOLD), ("block", BLK)]}}
    # (1) format switching: the payer chooses the rail; feature only change
    sw = {}
    for fname, code in CFG["fmt_codes"].items():
        if fname in ("Reinvestment",):
            continue
        Xm = Xev.copy(); Xm[lau, fi] = code
        a1, ax, al = score(Xm)
        r = np.where(a1 < L1T, a1, (ax + al) / 2)
        sw[fname] = {b: round(float((r[lau] >= t).mean()), 4) for b, t in [("verify", VER), ("hold", HOLD), ("block", BLK)]}
        sw[fname]["pr_auc"] = round(float(AP(ye, r)), 4)
    H["format_switch_all_laundering_to"] = sw
    # (2) amount nudging: avoid round hundreds and shave 1%; feature only (log_usd, round_100)
    Xm = Xev.copy(); lu = F.index("log_usd"); rh = F.index("round_100")
    Xm[lau, lu] = np.log1p(np.expm1(Xm[lau, lu]) * 0.99); Xm[lau, rh] = 0
    a1, ax, al = score(Xm); r = np.where(a1 < L1T, a1, (ax + al) / 2)
    H["amount_shave_1pct_no_round"] = {b: round(float((r[lau] >= t).mean()), 4) for b, t in [("verify", VER), ("hold", HOLD), ("block", BLK)]}
    # (3) structuring, re-simulated through the engine (state changes): caught if ANY part reaches the band
    for name in ("split2", "split4", "split4_spread"):
        s = load(name)
        s = s[s.orig >= EVAL_FROM]
        s["risk"] = live(s)
        g = s[s.y == 1].groupby("orig").risk
        anyp = g.max(); allp = g.min()
        legit = s[s.y == 0]
        H[f"structuring_{name}"] = {
            "original_payments": int(len(anyp)),
            **{f"{b}_any_part": round(float((anyp >= t).mean()), 4) for b, t in [("verify", VER), ("hold", HOLD), ("block", BLK)]},
            **{f"{b}_all_parts": round(float((allp >= t).mean()), 4) for b, t in [("verify", VER), ("block", BLK)]},
            "legit_flagged_verify_plus": int((legit.risk >= VER).sum()),
        }
    H["legit_flagged_verify_plus_baseline"] = int(((r_live >= VER) & (ye == 0)).sum())
    R["G_evasion"] = H

    # ---------------------------------------------------------------- H. explanations: XGBoost-only reasons vs both halves
    gi = np.where(gate)[0]
    dmx = xgb.DMatrix(Xev[gi], missing=np.nan, feature_names=F)
    phx = bx.predict(dmx, pred_contribs=True)[:, :-1]
    phl = bl.predict(Xev[gi].astype(np.float64), pred_contrib=True)[:, :-1]
    px_, pl_ = de.l2x.values[gi], de.l2l.values[gi]
    # first order map of each half's log odds attributions into probability space, then average like the served score
    full = 0.5 * (phx * (px_ * (1 - px_))[:, None] + phl * (pl_ * (1 - pl_))[:, None])
    def topk(m, k=4):
        return [set(np.argsort(-row)[:k][row[np.argsort(-row)[:k]] > 0]) for row in m]
    tx, tl, tf = topk(phx), topk(phl), topk(full)
    def jac(a, b):
        return np.mean([len(p & q) / max(1, len(p | q)) for p, q in zip(a, b)])
    top1 = lambda m: np.argmax(m, axis=1)
    hi = r_live[gi] >= HOLD
    def summary(mask):
        idx = np.where(mask)[0]
        return {"payments": int(len(idx)),
                "top4_jaccard_xgb_vs_lgbm": round(float(jac([tx[i] for i in idx], [tl[i] for i in idx])), 3),
                "top4_jaccard_xgb_vs_full": round(float(jac([tx[i] for i in idx], [tf[i] for i in idx])), 3),
                "top1_agreement_xgb_vs_lgbm": round(float((top1(phx[idx]) == top1(phl[idx])).mean()), 3),
                "top1_agreement_xgb_vs_full": round(float((top1(phx[idx]) == top1(full[idx])).mean()), 3)}
    imp_x = np.abs(phx).mean(0); imp_l = np.abs(phl).mean(0)
    R["H_explanations"] = {
        "method": "TreeSHAP per half (log odds), then each half mapped to probability space by p(1-p) and averaged like the served score",
        "all_l2_payments": summary(np.ones(len(gi), bool)), "hold_or_more": summary(hi),
        "global_rank_spearman_xgb_vs_lgbm": round(float(stats.spearmanr(imp_x, imp_l).statistic), 3),
        "top10_global_xgb": [F[i] for i in np.argsort(-imp_x)[:10]],
        "top10_global_lgbm": [F[i] for i in np.argsort(-imp_l)[:10]],
    }

    # ---------------------------------------------------------------- I. history depth
    I = {}
    for name, hist in [("base", 100_000), ("warm_90000", 10_000), ("warm_75000", 25_000), ("warm_50000", 50_000)]:
        s = load(name); s = s[s.orig >= EVAL_FROM]
        r = live(s)
        b = boot(s.y.values, {"live": r}, lambda y, q: AP(y, q))["live"]
        I[f"history_{hist}_payments"] = {"pr_auc": b["point"], "ci95": b["ci95"], "roc_auc": round(float(AUC(s.y, r)), 4),
                                         "block_flags": int((r >= BLK).sum()), "block_precision": round(float(s.y.values[r >= BLK].mean()), 4) if (r >= BLK).any() else None}
    I["full_protocol_reference"] = {"history_payments": 4_062_676, "pr_auc": 0.7107, "source": "results/full_eval_HI.json"}
    R["I_history_depth"] = I

    # ---------------------------------------------------------------- J. random vs chronological split (retrained)
    # retrain an XGBoost with the served hyperparameter style on the slice's features; same rows, two split policies
    yall = d.y.values; n = len(d)
    tr_end, te_from = int(n * 0.6), int(n * 0.8)
    params = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1, "subsample": 0.8, "colsample_bytree": 0.8,
              "eval_metric": "aucpr", "tree_method": "hist", "scale_pos_weight": 10.0}
    def fit_eval(tr, te, seed):
        p = dict(params, seed=seed)
        b = xgb.train(p, xgb.DMatrix(X[tr], yall[tr], missing=np.nan), 300)
        return AP(yall[te], b.predict(xgb.DMatrix(X[te], missing=np.nan)))
    chrono = [fit_eval(np.arange(tr_end), np.arange(te_from, n), s) for s in range(5)]
    rnd, grp = [], []
    sender = (raw.fb + ":" + raw.fa).values
    for s in range(5):
        rs = np.random.default_rng(s)
        perm = rs.permutation(n)
        rnd.append(fit_eval(perm[:tr_end], perm[te_from:], s))
        us = np.unique(sender); rs.shuffle(us)
        cut = set(us[: int(len(us) * 0.6)])
        intr = np.array([x in cut for x in sender])
        tr = np.where(intr)[0]; te = np.where(~intr)[0]
        te = te[rs.permutation(len(te))[: n - te_from]]
        grp.append(fit_eval(tr, te, s))
    R["J_random_vs_chronological"] = {
        "setup": "XGBoost retrained on the slice's 31 live features; 60% train, last 20% test; 5 seeds",
        "chronological": {"pr_auc_mean": round(float(np.mean(chrono)), 4), "sd": round(float(np.std(chrono)), 4),
                          "test_laundering": int(yall[te_from:].sum())},
        "random_row_split": {"pr_auc_mean": round(float(np.mean(rnd)), 4), "sd": round(float(np.std(rnd)), 4)},
        "random_split_grouped_by_sender": {"pr_auc_mean": round(float(np.mean(grp)), 4), "sd": round(float(np.std(grp)), 4)},
        "caveat": "features carry the causal state of a five hour stream, so the gap measures split policy only, on 236 positives",
    }

    with open(os.path.join(OUT, "slice_results.json"), "w") as f:
        json.dump(R, f, indent=1, default=float)
    print(json.dumps(R, indent=1, default=float)[:200000])


if __name__ == "__main__":
    main()
