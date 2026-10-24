"""Tune the sender mule signature on the VALIDATION weeks of the full IBM file and report it once on the TEST weeks.

Input: the dump written by ml/full_dataset_eval.py (rows from 60% of the time ordered stream; split marks 80%).
Step 1 re-implements the ctx-1.2 band logic in NumPy and asserts it reproduces assess() on every row.
Step 2 searches a small grid of stricter signatures (all rules a human can read) on validation only:
    sender paid by >= F distinct accounts in 24 h, forwards >= R of the last deposit within M minutes,
    and has sent to at most D distinct payees so far.
    Pick: highest validation F1 among rules that flag at most 0.2% of payments.
Step 3 reports the chosen rule, and the combined decision, on the test weeks. Writes results/context_tuning_<set>.json.
Usage: python ml/tune_context.py HI   (needs $RUSETECT_DUMP/ctx_dump_HI.npz)
"""
import itertools, json, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SET = (sys.argv[1] if len(sys.argv) > 1 else "HI").upper()
FIXED = sys.argv[sys.argv.index("--fixed") + 1] if "--fixed" in sys.argv else None   # apply a rule chosen elsewhere (no tuning here)
D = np.load(os.path.join(os.environ.get("RUSETECT_DUMP", "/tmp"), f"ctx_dump_{SET}.npz"), allow_pickle=False)
RAW = list(D["raw"]); R = D["R"]; y = D["y"].astype(bool); split = int(D["split"])
col = lambda k: R[:, RAW.index(k)]
new_pair, amt_dev, rapid, fan_in = col("new_pair") > 0, col("amount_deviation") > 0, col("rapid") > 0, col("fan_in_24h")
sf, uniq_dst, ratio, msi = col("sender_fan_in_24h"), col("s_uniq_dst"), col("s_out_over_last_in"), col("s_mins_since_in")
pay_rank = D["pay_rank"]
val, test = np.arange(len(y)) < split, np.arange(len(y)) >= split

def bands(sig):
    txn = np.minimum(30, 15 * new_pair + 15 * amt_dev)
    graph = np.minimum(30, 15 * (fan_in >= 4) + 15 * rapid + 15 * sig)
    total = txn + graph; doms = (txn > 0).astype(int) + (graph > 0)
    b = np.select([total >= 85, total >= 60, total >= 40], [3, 2, 1], 0)
    b = np.where((doms < 2) & (b >= 2), 1, b)                       # HIGH or CRITICAL needs two domains
    b = np.where(sig & (b < 1), 1, b)                                # signature forces at least MEDIUM
    return np.minimum(b, 2)                                          # CONTEXT_TO_RANK: HIGH and CRITICAL both map to hold

shipped_sig = (sf >= 4) & rapid
assert np.array_equal(bands(shipped_sig), D["ctx_rank"]), "NumPy replica does not match assess(); stop"

def stats(mask, rows):
    m, t = mask[rows], y[rows]; n, tp, P = int(m.sum()), int((m & t).sum()), int(t.sum())
    p, r = (tp / n if n else 0.0), tp / P
    return {"flagged": n, "flag_rate": round(n / rows.sum(), 5), "caught": tp, "laundering": P,
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0}

grid = []
okr = ~np.isnan(ratio); okm = ~np.isnan(msi)
for F, Rmin, M, Dmax in itertools.product([4, 6, 8, 12, 20], [0.8, 0.9, 0.95], [10, 5, 2], [1, 2, 3, 5, 10, 10 ** 9]):
    sig = (sf >= F) & okr & (ratio >= Rmin) & okm & (msi <= M) & (uniq_dst <= Dmax)
    v = stats(sig, val)
    grid.append({"F": F, "R": Rmin, "M": M, "D": Dmax, "val": v})
ok = [g for g in grid if g["val"]["flag_rate"] <= 0.002 and g["val"]["caught"] > 0]
best = max(ok or grid, key=lambda g: (g["val"]["f1"], g["val"]["precision"]))
F, Rmin, M, Dmax = best["F"], best["R"], best["M"], best["D"]
if FIXED:
    cr = json.load(open(FIXED))["chosen_rule"]
    F, Rmin, M = cr["sender_fan_in_24h_min"], cr["forward_ratio_min"], cr["minutes_since_deposit_max"]
    Dmax = cr["sender_distinct_payees_max"] if cr["sender_distinct_payees_max"] is not None else 10 ** 9
    best = {"val": stats((sf >= F) & okr & (ratio >= Rmin) & okm & (msi <= M) & (uniq_dst <= Dmax), val)}
new_sig = (sf >= F) & okr & (ratio >= Rmin) & okm & (msi <= M) & (uniq_dst <= Dmax)
old_ctx, new_ctx = bands(shipped_sig), bands(new_sig)

out = {"set": SET, "selection": (f"fixed rule from {os.path.basename(FIXED)} (chosen on another dataset; nothing tuned here)" if FIXED
                                  else "highest validation F1 among signatures flagging at most 0.2% of validation payments"),
       "chosen_rule": {"sender_fan_in_24h_min": F, "forward_ratio_min": Rmin, "minutes_since_deposit_max": M, "sender_distinct_payees_max": None if Dmax > 10 ** 8 else Dmax},
       "validation": {"shipped_ctx_1_2": stats(shipped_sig, val), "chosen": best["val"]},
       "test": {
           "signature": {"shipped_ctx_1_2": stats(shipped_sig, test), "chosen": stats(new_sig, test)},
           "context_layer_alone_verify_or_more": {"shipped_ctx_1_2": stats(old_ctx >= 1, test), "chosen": stats(new_ctx >= 1, test)},
           "combined_verify_or_more": {"shipped_ctx_1_2": stats(np.maximum(pay_rank, old_ctx) >= 1, test), "chosen": stats(np.maximum(pay_rank, new_ctx) >= 1, test)},
           "combined_hold_or_more": {"shipped_ctx_1_2": stats(np.maximum(pay_rank, old_ctx) >= 2, test), "chosen": stats(np.maximum(pay_rank, new_ctx) >= 2, test)},
           "payment_model_verify_or_more": stats(pay_rank >= 1, test), "payment_model_hold_or_more": stats(pay_rank >= 2, test),
           "raised_by_context_chosen": stats((new_ctx > pay_rank) & (new_ctx >= 1), test)},
       "top_validation_candidates": sorted(ok, key=lambda g: -g["val"]["f1"])[:8]}

# ring coverage on test with the chosen rule
ring_j, ring_id, typ = D["ring_j"], D["ring_id"], D["ring_typ"]
def rings(mask):
    seen, caught = {}, {}
    for j, r in zip(ring_j, ring_id):
        if j < split:
            continue
        seen.setdefault(r, False); caught[r] = caught.get(r, False) or bool(mask[j])
    return {"rings": len(seen), "caught": int(sum(caught.values())), "rate": round(sum(caught.values()) / max(len(seen), 1), 3)}
out["test"]["rings_combined_verify_or_more"] = {"shipped_ctx_1_2": rings(np.maximum(pay_rank, old_ctx) >= 1), "chosen": rings(np.maximum(pay_rank, new_ctx) >= 1)}
out["test"]["rings_combined_hold_or_more"] = {"shipped_ctx_1_2": rings(np.maximum(pay_rank, old_ctx) >= 2), "chosen": rings(np.maximum(pay_rank, new_ctx) >= 2)}
out["test"]["rings_payment_model_verify_or_more"] = rings(pay_rank >= 1)
json.dump(out, open(os.path.join(ROOT, "results", f"context_tuning_{SET}" + ("_fixed" if FIXED else "") + ".json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "top_validation_candidates"}, indent=1))
