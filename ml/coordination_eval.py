"""Tune and test the flagged sender rule on the full IBM files, using the dumps written by ml/full_dataset_eval.py.

Rows from 60% of the time ordered stream onward: 60 to 80% validation, 80 to 100% test. A sender counts as flagged
once one of its payments got a live payment decision at or above the chosen level. For each payment we count the
distinct flagged senders that paid the same payee within the window BEFORE this payment, so nothing looks ahead.
Selection on HI validation only: best F1 for catching laundering the payment model MISSED (pay_rank == 0), among
rules that flag at most 0.2% of payments. Then the same rule is reported on HI test and on LI (never tuned there).
Usage: RUSETECT_DUMP=... python ml/coordination_eval.py      writes results/coordination_eval.json
"""
import glob, itertools, json, os, sys
from collections import defaultdict, deque
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DUMP = os.environ.get("RUSETECT_DUMP", "/tmp")
DATA = os.environ.get("RUSETECT_DATA") or "data"   # folder with the dataset files (see README)


def minutes(SET, collect_from):
    parts = sorted(glob.glob(os.path.join(DATA, f"aml_{SET}-Small_part*.csv.gz")))
    ts = pd.concat([pd.read_csv(p, header=0, usecols=[0], names=["ts"], dtype=str) for p in parts], ignore_index=True)["ts"]
    t = pd.to_datetime(ts, format="%Y/%m/%d %H:%M").astype("datetime64[s]").astype("int64").to_numpy() // 60
    return np.sort(t, kind="stable")[collect_from:]          # same stable order as full_dataset_eval


def counts(D, t, window, flag_rank):
    sd = D["src_dst"]; pr = D["pay_rank"]; n = len(pr)
    snd = sd[:, 0].astype(np.int64) * 10_000_000 + sd[:, 1]; pay = sd[:, 2].astype(np.int64) * 10_000_000 + sd[:, 3]
    flagged_at, inc = {}, defaultdict(deque); out = np.zeros(n, dtype=np.int32)
    for j in range(n):
        tj, s, d = int(t[j]), int(snd[j]), int(pay[j])
        q = inc[d]
        while q and q[0][0] <= tj - window:
            q.popleft()
        if q:
            lo = tj - window
            out[j] = len({x for (_, x) in q if flagged_at.get(x, -10 ** 12) > lo})
        if pr[j] >= flag_rank:
            flagged_at[s] = tj
        q.append((tj, s))
    return out


def stats(mask, y, rows):
    m, yy = mask[rows], y[rows]; n, tp, P = int(m.sum()), int((m & yy).sum()), int(yy.sum())
    p, r = (tp / n if n else 0.0), (tp / P if P else 0.0)
    return {"flagged": n, "flag_rate": round(n / rows.sum(), 5), "caught": tp, "laundering": P,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}


def evaluate(SET, rule=None):
    D = np.load(os.path.join(DUMP, f"ctx_dump_{SET}.npz")); y = D["y"].astype(bool); split = int(D["split"]); pr = D["pay_rank"]
    n = len(y); collect_from = None
    res = json.load(open(os.path.join(ROOT, "results", f"full_eval_{SET}.json")))
    t = minutes(SET, res["validation_from_row"]); assert len(t) == n, (len(t), n)
    val, test = np.arange(n) < split, np.arange(n) >= split
    missed = pr == 0
    grid, cache = [], {}
    combos = [(rule["window_min"], rule["flag_rank"])] if rule else list(itertools.product([24 * 60, 3 * 24 * 60, 7 * 24 * 60], [1, 2]))
    for W, R in combos:
        cache[(W, R)] = counts(D, t, W, R)
        for K in ([rule["k"]] if rule else [2, 3, 5, 8]):
            m = cache[(W, R)] >= K
            grid.append({"window_min": W, "flag_rank": R, "k": K, "val_missed": stats(m & missed, y, val), "val_all": stats(m, y, val)})
    if rule is None:
        ok = [g for g in grid if g["val_all"]["flag_rate"] <= 0.002 and g["val_missed"]["caught"] > 0] or grid
        best = max(ok, key=lambda g: (g["val_missed"]["f1"], g["val_missed"]["precision"]))
    else:
        best = grid[0]
    m = cache[(best["window_min"], best["flag_rank"])] >= best["k"]
    final_ext = np.maximum(pr, np.where(m, 1, 0))      # the rule adds graph points; alone it reaches verification at most
    return {"rule": {k: best[k] for k in ("window_min", "flag_rank", "k")},
            "validation": {"rule_on_missed": best["val_missed"], "rule_all": best["val_all"]},
            "test": {"rule_all": stats(m, y, test), "rule_on_missed": stats(m & missed, y, test),
                     "payment_model_verify_or_more": stats(pr >= 1, y, test), "payment_model_plus_rule_verify_or_more": stats(final_ext >= 1, y, test)},
            "grid_top": sorted(grid, key=lambda g: -g["val_missed"]["f1"])[:6] if rule is None else None}


if __name__ == "__main__" and "--fusion" not in sys.argv:
    hi = evaluate("HI")
    out = {"HI": hi, "selection": "HI validation weeks only: best F1 on laundering the payment model missed, flag rate <= 0.2%",
           "note": "The rule turns on graph points (coordinated_accounts); alone it can reach verification, never a hold."}
    if os.path.exists(os.path.join(DUMP, "ctx_dump_LI.npz")):
        out["LI_unchanged_rule"] = evaluate("LI", hi["rule"])
    json.dump(out, open(os.path.join(ROOT, "results", "coordination_eval.json"), "w"), indent=1)
    print(json.dumps({k: (v if k != "HI" else {kk: vv for kk, vv in v.items() if kk != "grid_top"}) for k, v in out.items()}, indent=1))


# ---------------------------------------------------------------- inside the real fusion (ctx-1.4 bands), not as a standalone alert
def fused(SET, rules):
    D = np.load(os.path.join(DUMP, f"ctx_dump_{SET}.npz")); y = D["y"].astype(bool); split = int(D["split"]); pr = D["pay_rank"]
    RAW = list(D["raw"]); R = D["R"]; col = lambda k: R[:, RAW.index(k)]
    newp, amt, rapid, fan = col("new_pair") > 0, col("amount_deviation") > 0, col("rapid") > 0, col("fan_in_24h")
    sf, ratio, msi, ud = col("sender_fan_in_24h"), col("s_out_over_last_in"), col("s_mins_since_in"), col("s_uniq_dst")
    sig = (sf >= 4) & rapid & ~np.isnan(ratio) & (ratio >= 0.95) & ~np.isnan(msi) & (msi <= 2) & (ud <= 10)    # ctx-1.3
    res = json.load(open(os.path.join(ROOT, "results", f"full_eval_{SET}.json")))
    t = minutes(SET, res["validation_from_row"]); n = len(y); val, test = np.arange(n) < split, np.arange(n) >= split

    def bands(coord):
        txn = np.minimum(30, 15 * newp + 15 * amt); graph = np.minimum(30, 15 * (fan >= 4) + 15 * rapid + 15 * coord + 15 * sig)
        tot = txn + graph; doms = (txn > 0).astype(int) + (graph > 0)
        b = np.select([tot >= 85, tot >= 60, tot >= 40], [3, 2, 1], 0); b = np.where((doms < 2) & (b >= 2), 1, b)
        return np.minimum(np.where(sig & (b < 1), 1, b), 2)
    base = np.maximum(pr, bands(np.zeros(n, bool)))
    out = {"without_rule": {"val": stats(base >= 1, y, val), "test": stats(base >= 1, y, test)}}
    for W, Rk, K in rules:
        c = counts(D, t, W, Rk) >= K; f = np.maximum(pr, bands(c))
        out[f"window {W // 1440} d, flag level {Rk}, k {K}"] = {"val": stats(f >= 1, y, val), "test": stats(f >= 1, y, test),
                                                             "added_flags_test": int(((f >= 1) & ~(base >= 1))[test].sum()),
                                                             "added_laundering_test": int(((f >= 1) & ~(base >= 1) & y)[test].sum())}
    return out


if __name__ == "__main__" and "--fusion" in sys.argv:
    rules = [(3 * 1440, 1, 2), (3 * 1440, 1, 3), (7 * 1440, 1, 5), (7 * 1440, 2, 3)]
    out = json.load(open(os.path.join(ROOT, "results", "coordination_eval.json")))
    out["fusion"] = {s: fused(s, rules) for s in ("HI", "LI") if os.path.exists(os.path.join(DUMP, f"ctx_dump_{s}.npz"))}
    json.dump(out, open(os.path.join(ROOT, "results", "coordination_eval.json"), "w"), indent=1)
    print(json.dumps(out["fusion"], indent=1))
