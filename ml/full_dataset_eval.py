"""Full dataset evaluation of the live path on the IBM AMLworld files Parth uploaded (not the bundled 160k sample).

  python ml/full_dataset_eval.py HI     all 5,078,345 HI-Small payments; metrics on the time-ordered last 20% (the benchmark test period)
  python ml/full_dataset_eval.py LI     LI-Small sample (first 500,000 payments of LI-Small, never used for training): 40% warm-up, 60% scored

Every payment is streamed in time order through the live engine's past-only feature state (service/rusetect_engine.py),
then scored by the live L1 gatekeeper and the L2 XGBoost + LightGBM average, and by the context layer's payment and
graph checks (service/context_rules.py, ctx-1.2; IBM data has no messages, so the message domain is empty).
Labelled laundering payments are matched to the ground-truth rings in <set>-Small_Patterns.txt to measure ring coverage.
Data folder: env RUSETECT_DATA (default: the uploads folder). Writes results/full_eval_<set>.json.
"""
import calendar, gzip, glob, json, math, os, sys, time
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from rusetect_engine import Engine                                   # noqa: E402
from context_rules import assess, combine, from_engine_features, sender_fan_in_24h, PAYMENT_RANK, CONTEXT_TO_RANK, RULESET_VERSION  # noqa: E402

DATA = os.environ.get("RUSETECT_DATA") or os.path.dirname(next(iter(sorted(glob.glob("/mnt/user-data/uploads/*/for_claude"))), "data/for_claude"))   # folder holding for_claude/ and the archive
SET = (sys.argv[1] if len(sys.argv) > 1 else "HI").upper()
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:7.1f}s]", *a, flush=True)

# ------------------------------------------------------------------ load and sort by time (the files are not time ordered)
parts = sorted(glob.glob(os.path.join(DATA, "for_claude", f"aml_{SET}-Small_part*.csv.gz")))
cols = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
dt = {"ts": "string", "fb": "category", "fa": "category", "tb": "category", "ta": "category", "amt_recv": "string", "cur_recv": "category",
      "amt_paid": "string", "cur_paid": "category", "fmt": "category", "y": "int8"}
df = pd.concat([pd.read_csv(p, header=0, names=cols, dtype=dt) for p in parts], ignore_index=True)
for c in ("fb", "fa", "tb", "ta", "cur_recv", "cur_paid", "fmt"):
    df[c] = df[c].astype("string").astype("category")
df["tmin"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M").astype("datetime64[s]").astype("int64") // 60   # minutes; explicit unit (pandas 3 parses to microseconds)
df = df.sort_values("tmin", kind="mergesort").reset_index(drop=True)
N = len(df)
collect_from, test_from = int(N * 0.6), int(N * 0.8)   # 60 to 80% = validation (rule tuning only), 80 to 100% = test (all metrics)
span = (str(df["ts"].iloc[0]), str(df["ts"].iloc[-1]))
log(f"{SET}: {N:,} payments from {len(parts)} files, {int(df.y.sum()):,} laundering; validation rows {collect_from:,}+, test rows {test_from:,}+")

# ------------------------------------------------------------------ ground truth rings
rings, key2ring = [], {}
with open(os.path.join(DATA, "archive (6) 8 gb", f"{SET}-Small_Patterns.txt")) as fh:
    cur = None
    for line in fh:
        line = line.strip()
        if line.startswith("BEGIN"):
            typ = line.split(" - ", 1)[1].split(":")[0].strip()
            cur = {"typology": typ, "keys": []}; rings.append(cur)
        elif line.startswith("END"):
            cur = None
        elif line and cur is not None:
            f = line.split(",")
            k = (f[0], f[1], f[2], f[3], f[4], f[7]); cur["keys"].append(k); key2ring.setdefault(k, []).append(len(rings) - 1)
log(f"{len(rings)} labelled rings, {len(key2ring):,} ring payments")

# ------------------------------------------------------------------ stream through the live engine
eng = Engine(load_models=True); eng.reset()
feats = eng.feats
cat = {c: df[c].cat.categories.to_numpy() for c in ("fb", "fa", "tb", "ta", "cur_recv", "cur_paid", "fmt")}
codes = {c: df[c].cat.codes.to_numpy() for c in cat}
tmin, ys = df["tmin"].to_numpy(), df["y"].to_numpy()
amt = df["amt_paid"].astype(float).to_numpy()
pos_idx = np.flatnonzero(ys)
pos_key = {int(i): (df["ts"].iat[i], df["amt_paid"].iat[i]) for i in pos_idx}   # strings only for labelled rows (ring matching)
del df
import gc; gc.collect()
nt = N - collect_from
RAW = ["new_pair", "amount_deviation", "rapid", "fan_in_24h", "sender_fan_in_24h", "s_uniq_dst", "s_in_n", "s_out_n",
       "s_out_over_last_in", "s_mins_since_in", "d_uniq_src", "log_usd", "s_passthrough"]
R = np.zeros((nt, len(RAW)), dtype=np.float32)
X = np.empty((nt, len(feats)), dtype=np.float32)
BANDS = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
ctx_rank = np.zeros(nt, dtype=np.int8); ctx_band = np.zeros(nt, dtype=np.int8); mule_sig = np.zeros(nt, dtype=bool)
ring_rows = {}                                     # test row -> ring ids (labelled ring payments only)
src_dst = np.zeros((nt, 4), dtype=np.int32)         # category codes (fb, fa, tb, ta) for L4 case linking
for i in range(N):
    p = {"from_bank": cat["fb"][codes["fb"][i]], "from_account": cat["fa"][codes["fa"][i]], "to_bank": cat["tb"][codes["tb"][i]],
         "to_account": cat["ta"][codes["ta"][i]], "amount_paid": float(amt[i]), "payment_currency": cat["cur_paid"][codes["cur_paid"][i]],
         "receiving_currency": cat["cur_recv"][codes["cur_recv"][i]], "payment_format": cat["fmt"][codes["fmt"][i]], "tmin": int(tmin[i])}
    if i < collect_from:
        eng.features(p)
    else:
        j = i - collect_from
        sf = sender_fan_in_24h(eng, p)
        f = eng.features(p)
        X[j] = [f[k] for k in feats]
        pc = from_engine_features(f, sf)
        R[j] = (pc.new_beneficiary, pc.amount_deviation, pc.rapid_pass_through, pc.fan_in_24h, sf, f["s_uniq_dst"], f["s_in_n"], f["s_out_n"],
                f["s_out_over_last_in"], f["s_mins_since_in"], f["d_uniq_src"], f["log_usd"], f["s_passthrough"])
        r = assess(None, pc, True)
        ctx_band[j] = BANDS.index(r.band); ctx_rank[j] = CONTEXT_TO_RANK[r.band]
        mule_sig[j] = "account graph: sender_mule_signature" in r.reasons
        src_dst[j] = (codes["fb"][i], codes["fa"][i], codes["tb"][i], codes["ta"][i])
        if ys[i]:
            k = (pos_key[i][0], p["from_bank"], p["from_account"], p["to_bank"], p["to_account"], pos_key[i][1])
            if k in key2ring:
                ring_rows[j] = key2ring[k]
    if i % 500_000 == 0:
        log(f"  streamed {i:,} / {N:,}")
log("stream done; batch scoring the live models")
eng.accts = None; eng.pairs = None; eng.bank_pairs = None
import gc; gc.collect()

# ------------------------------------------------------------------ live models, batch (same thresholds and maths as Engine.score_payment)
import xgboost as xgb  # noqa: E402
cfg = eng.cfg
dm = xgb.DMatrix(X, feature_names=feats)
s1 = eng.l1.predict(dm)
sys.path.insert(0, os.path.join(ROOT, "ml")); import lite_fast  # noqa: E402
lite_fast.check(eng.l2l, X)
risk = (eng.l2x.predict(dm) + lite_fast.predict(eng.l2l, X)) / 2           # L2 average, computed for every row so PR AUC is defined
pay_rank = np.zeros(nt, dtype=np.int8)
esc = s1 >= cfg["l1_threshold"]
pay_rank[esc & (risk >= cfg["verify_threshold"])] = 1
pay_rank[esc & (risk >= cfg["hold_threshold"])] = 2
pay_rank[esc & (risk >= cfg["block_threshold"])] = 3
# validation + test dump for rule tuning (ml/tune_context.py); metrics below use the test rows only
np.savez_compressed(os.path.join(os.environ.get("RUSETECT_DUMP", "/tmp"), f"ctx_dump_{SET}.npz"), R=R, raw=np.array(RAW), y=ys[collect_from:],
                    pay_rank=pay_rank, risk=risk, esc=esc, ctx_rank=ctx_rank, split=test_from - collect_from,
                    src_dst=src_dst, ring_j=np.array(sorted(ring_rows), dtype=np.int64),
                    ring_id=np.array([ring_rows[j][0] for j in sorted(ring_rows)], dtype=np.int64),
                    ring_typ=np.array([r["typology"] for r in rings]))
np.save(os.path.join(os.environ.get("RUSETECT_DUMP", "/tmp"), f"live_X_{SET}.npy"), X)   # live features, rows collect_from.., for offline parity checks
off = test_from - collect_from
X, R, ctx_rank, ctx_band, mule_sig, src_dst = X[off:], R[off:], ctx_rank[off:], ctx_band[off:], mule_sig[off:], src_dst[off:]
s1, risk, esc, pay_rank = s1[off:], risk[off:], esc[off:], pay_rank[off:]
ring_rows = {j - off: v for j, v in ring_rows.items() if j >= off}
nt = N - test_from
final_rank = np.maximum(pay_rank, ctx_rank)
y = ys[test_from:].astype(bool)
live_score = np.where(esc, risk, 0.0)

from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

def level(mask):
    n, tp, P = int(mask.sum()), int((mask & y).sum()), int(y.sum())
    return {"flagged": n, "flag_rate": round(n / nt, 5), "laundering_caught": tp, "laundering": P,
            "recall": round(tp / P, 4) if P else None, "precision": round(tp / n, 4) if n else None}

def ring_cov(mask):
    seen, caught = Counter(), Counter()
    rows_by_ring = defaultdict(list)
    for j, rs in ring_rows.items():
        for r in rs:
            rows_by_ring[r].append(j)
    for r, js in rows_by_ring.items():
        t = rings[r]["typology"]; seen[t] += 1
        if any(mask[j] for j in js):
            caught[t] += 1
    tot_s, tot_c = sum(seen.values()), sum(caught.values())
    return {"rings_in_scored_period": tot_s, "rings_caught": tot_c, "rate": round(tot_c / tot_s, 3) if tot_s else None,
            "by_typology": {t: {"rings": seen[t], "caught": caught[t], "rate": round(caught[t] / seen[t], 3)} for t in sorted(seen)}}

# account -> owning entity (accounts file), used to merge cases that share an owner
ent = {}
for acc_path in [os.path.join(DATA, "archive (6) 8 gb", f"{SET}-Small_accounts.csv")] + sorted(glob.glob(os.path.join(DATA, "for_claude", f"aml_{SET}-Small_accounts_part*.csv.gz"))):
    if os.path.exists(acc_path):
        a = pd.read_csv(acc_path, dtype=str)
        ent.update({(str(int(b)) if b.isdigit() else b, n): e for b, n, e in zip(a["Bank ID"], a["Account Number"], a["Entity ID"])})
        if "archive" in acc_path:
            break                                    # the archive file is complete; otherwise read every part
norm = lambda acct: (str(int(acct[0])) if acct[0].isdigit() else acct[0], acct[1])

def cases(mask, by_entity=False):
    parent = {}
    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for j in np.flatnonzero(mask):
        s = (cat["fb"][src_dst[j, 0]], cat["fa"][src_dst[j, 1]]); d = (cat["tb"][src_dst[j, 2]], cat["ta"][src_dst[j, 3]]); parent[find(s)] = find(d)
        if by_entity:
            for a in (s, d):
                e = ent.get(norm(a))
                if e: parent[find(a)] = find(("entity", e))
    return len({find((cat["fb"][src_dst[j, 0]], cat["fa"][src_dst[j, 1]])) for j in np.flatnonzero(mask)})

systems = {
    "payment_model": {"verify_or_more": pay_rank >= 1, "hold_or_more": pay_rank >= 2, "block": pay_rank >= 3},
    "context_layer_alone": {"verify_or_more": ctx_rank >= 1, "hold_or_more": ctx_rank >= 2},
    "combined_final": {"verify_or_more": final_rank >= 1, "hold_or_more": final_rank >= 2},
}
out = {
    "dataset": f"IBM AMLworld {SET}-Small, all payments" + ("" if SET == "HI" else " (never used for training)"),
    "payments_total": N, "laundering_total": int(ys.sum()), "scored_from_row": test_from, "validation_from_row": collect_from, "scored_payments": nt, "scored_laundering": int(y.sum()),
    "protocol": "time ordered; 0 to 60% builds account state; 60 to 80% is validation (used only to tune context rules); metrics on the last 20%"
                + ("" if SET == "HI" else "; payment models were trained on HI-Small only, so every LI payment is unseen"),
    "ruleset": RULESET_VERSION,
    "live_model_pr_auc": round(float(average_precision_score(y, live_score)), 4),
    "live_model_roc_auc": round(float(roc_auc_score(y, live_score)), 4),
    "l2_average_pr_auc_all_rows": round(float(average_precision_score(y, risk)), 4),
    "l1_pass_rate": round(float(esc.mean()), 4), "l1_laundering_kept": round(float((esc & y).sum() / max(y.sum(), 1)), 4),
    "levels": {s: {k: level(m) for k, m in d.items()} for s, d in systems.items()},
    "raised_by_context": level((ctx_rank > pay_rank)),
    "mule_signature": level(mule_sig),
    "context_band_counts": {BANDS[k]: int(v) for k, v in sorted(Counter(ctx_band.tolist()).items())},
    "rings": {s: {k: ring_cov(m) for k, m in d.items()} for s, d in systems.items()},
    "ring_payments_matched": len(ring_rows),
    "l4_cases_from_blocks": cases(pay_rank >= 3), "l4_cases_from_holds": cases(final_rank >= 2),
    "l4_cases_from_holds_entity_merged": cases(final_rank >= 2, by_entity=True), "accounts_with_entity": len(ent),
    "time_span": span,
    "runtime_seconds": round(time.time() - T0, 1),
    "caveats": ["IBM AMLworld is synthetic.", "No messages exist in IBM data, so the context layer runs on payment and graph checks only and "
                "single domain results cap at MEDIUM by design.",
                "Context and graph checks were designed after looking at the bundled sample, not at this full file."]

}
json.dump(out, open(os.path.join(ROOT, "results", f"full_eval_{SET}.json"), "w"), indent=1)
log(json.dumps({k: v for k, v in out.items() if k not in ("rings",)}, indent=1))
log(json.dumps({s: {k: {kk: vv for kk, vv in v.items() if kk != "by_typology"} for k, v in d.items()} for s, d in out["rings"].items()}, indent=1))
