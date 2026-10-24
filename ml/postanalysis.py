"""
RuseTect: analyses that reuse the saved test-period predictions of ml/layered_engine.py (nothing is refit).
  1. anomaly layer as a cross-verifier of payments the supervised fusion cleared
  2. real example alerts with human-readable reasons (XGBoost TreeSHAP + triggered rules + anomaly percentile)
  3. one real laundering ring from the test period: how each payment looks alone vs. in the network
Output: results/postanalysis.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK, REPO
import json, pickle, numpy as np, polars as pl
import xgboost as xgb
from common import load, to_np, ALL

d, split = load(); P = np.load(f"{WORK}/layered_preds.npz", allow_pickle=True); M = pickle.load(open(f"{WORK}/layered_models.pkl", "rb"))
R = json.load(open(os.path.join(REPO, "results", "layered_engine.json")))
te = P["te"]; yte = P["y"]; fused = P["fused"]; iso = P["iso"]; ae = P["ae"]; rules = P["rules"]; n = len(te)
thr = [v for k, v in R["challengers"].items() if k.startswith("RuseTect fused")][0]["threshold"]
out = {}

# ---- 1. cross-verifier
flag = fused >= thr; resid = np.where(~flag)[0]
res = dict(operating_threshold_on_fused_score=thr, supervised_alerts=int(flag.sum()), supervised_caught=int(yte[flag].sum()), laundering_total=int(yte.sum()),
           missed_by_supervised=int(yte[resid].sum()))
for frac in (0.0005, 0.001):
    k = int(n * frac)
    for nm, s in (("autoencoder", ae), ("isolation_forest", iso)):
        top = resid[np.argsort(-s[resid])[:k]]
        res[f"{nm}_top_{frac*100:.2f}pct"] = dict(extra_alerts=int(k), extra_laundering_caught=int(yte[top].sum()), precision=float(yte[top].mean()),
                                                  recall_gain=float(yte[top].sum() / yte.sum()))
# a rule-based second opinion for comparison: payments that trip >= 3 rules among the cleared ones
for kk in (3, 4):
    m_ = rules.sum(1) >= kk; sel = np.where(m_ & ~flag)[0]
    res[f"rules_ge_{kk}_on_cleared"] = dict(extra_alerts=int(len(sel)), extra_laundering_caught=int(yte[sel].sum()), precision=float(yte[sel].mean()) if len(sel) else None)
out["cross_verifier"] = res

# ---- 2. example alerts
xg = M["flat"]; names = ALL
Xte = to_np(d[te], ALL)
typ = pl.read_parquet(f"{WORK}/typology.parquet"); tm = dict(zip(typ["rid"].to_list(), zip(typ["typ"].to_list(), typ["ring"].to_list())))
rid = d["rid"].to_numpy()[te]; src = d["src_id"].to_numpy()[te]; dst = d["dst_id"].to_numpy()[te]; usd = d["usd"].to_numpy()[te]; tmin = d["tmin"].to_numpy()[te]
fmt = d["fmt"].to_numpy()[te]
PH = {"s_passthrough": "forwards {v:.0%} of what it received", "s_mins_since_in": "sent money {v:.0f} min after money arrived",
      "s_fanout_24h": "paid {v:.0f} different accounts in 24 h", "d_fanin_24h": "receiver took money from {v:.0f} accounts in 24 h",
      "s_cnt_24h": "{v:.0f} payments from this account in 24 h", "new_pair": "first payment between these two accounts",
      "s_out_over_last_in": "outgoing amount is {v:.1f}x the last incoming", "reciprocal": "receiver pays back to the sender",
      "log_usd": "amount ${e:,.0f}", "s_age_out": "account active for {v:.0f} min only", "d_in_n": "receiver has {v:.0f} earlier incoming payments",
      "s_uniq_dst": "sender has paid {v:.0f} distinct accounts so far", "s_usd_24h": "${e:,.0f} sent in 24 h", "fmt_code": "payment channel unusual",
      "d_gap_in": "receiver's previous incoming payment {v:.0f} min ago", "s_gap_out": "{v:.0f} min since sender's previous payment", "d_in_usd_log": "receiver's incoming volume ${e:,.0f}",
      "s_in_n": "sender received {v:.0f} payments before", "d_out_n": "receiver has made {v:.0f} payments", "d_uniq_src": "receiver has {v:.0f} distinct payers",
      "new_bank_pair": "first payment between these banks", "cross_bank": "cross-bank payment", "cross_cur": "currency conversion", "round_100": "round-number amount"}
def reason(i, cvals):
    k = int(np.argmax(-np.abs(cvals))) if False else None
    top = np.argsort(-np.abs(cvals))[:4]; txt = []
    for j in top:
        if cvals[j] <= 0: continue
        nm = ALL[j]; v = Xte[i, j]
        if nm not in PH or not np.isfinite(v): continue
        e = float(np.expm1(v)) if nm in ("log_usd", "d_in_usd_log", ) else float(v)
        txt.append(dict(feature=nm, phrase=PH[nm].format(v=v, e=e), shap=float(cvals[j])))
    return txt
order = np.argsort(-fused)
cands = {"true_positive": [i for i in order[:3000] if yte[i] == 1][:3], "false_positive": [i for i in order[:3000] if yte[i] == 0][:2]}
ex = {}
pc = np.argsort(np.argsort(ae)) / n * 100
for kind, idxs in cands.items():
    ex[kind] = []
    for i in idxs:
        c = xg.get_booster().predict(xgb.DMatrix(Xte[i:i + 1][:, :], feature_names=ALL), pred_contribs=True)[0][:-1]
        tr_ = [nm for nm, f in zip(R["rules"]["definitions"], rules[i]) if f]
        ex[kind].append(dict(row=int(i), typology=(tm.get(int(rid[i])) or ("", ""))[0] or ("not a labelled typology" if yte[i] else ""), is_laundering=bool(yte[i]),
                             fused_score=float(fused[i]), calibrated_probability=float(P["fused_cal"][i]), amount_usd=float(usd[i]), channel=str(fmt[i]),
                             anomaly_percentile_autoencoder=float(pc[i]), rules_triggered=[R["rules"]["definitions"][r] for r in tr_], reasons=reason(i, c)))
out["example_alerts"] = ex

# ---- 3. one real ring in the test period
ring_of = np.array([(tm.get(int(r)) or (None, -1))[1] if (tm.get(int(r)) or (None, -1))[1] is not None else -1 for r in rid])
typ_of = np.array([(tm.get(int(r)) or ("", -1))[0] or "" for r in rid])
fp = np.argsort(np.argsort(fused)) / n * 100
tx = P["L_transaction"]; txp = np.argsort(np.argsort(tx)) / n * 100
amt_pct = np.argsort(np.argsort(usd)) / n * 100
best = None
for r in np.unique(ring_of[ring_of >= 0]):
    ii = np.where(ring_of == r)[0]
    if len(ii) < 8 or typ_of[ii[0]] not in ("SCATTER-GATHER", "GATHER-SCATTER"): continue
    caught = int((fused[ii] >= thr).sum())
    score = caught - 0.02 * np.abs(len(ii) - 14)
    if best is None or score > best[0]: best = (score, r, ii)
_, r, ii = best; ii = ii[np.argsort(tmin[ii])]
ids = {}
lab = lambda a: ids.setdefault(int(a), f"A{len(ids)+1:02d}")
edges = [dict(src=lab(src[i]), dst=lab(dst[i]), usd=float(usd[i]), amount_percentile=float(amt_pct[i]), transaction_layer_percentile=float(txp[i]),
              fused_percentile=float(fp[i]), flagged=bool(fused[i] >= thr), t_min=int(tmin[i] - tmin[ii[0]]), rules_triggered=int(rules[i].sum()), autoencoder_percentile=float(pc[i]), fused_score=float(fused[i]), calibrated=float(P["fused_cal"][i])) for i in ii]
out["ring"] = dict(ring=int(r), typology=str(typ_of[ii[0]]), payments=len(ii), accounts=len(ids), flagged=int(sum(e["flagged"] for e in edges)),
                   median_amount_percentile=float(np.median([e["amount_percentile"] for e in edges])), median_transaction_layer_percentile=float(np.median([e["transaction_layer_percentile"] for e in edges])),
                   median_fused_percentile=float(np.median([e["fused_percentile"] for e in edges])), edges=edges)
json.dump(out, open(os.path.join(REPO, "results", "postanalysis.json"), "w"), indent=1, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "example_alerts"}, indent=1, default=float)[:3500])
