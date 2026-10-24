"""
RuseTect: fairness-of-comparison checks on the saved test predictions of ml/layered_engine.py.
  1. alert queue needed to catch 30/50/70% of laundering, TIE-AWARE (expected value when tied scores are reviewed in
     random order; matters for coarse scores such as a count of triggered rules)
  2. a stronger rules baseline: logistic-regression weights on the 9 rule flags, fitted on validation period A
  3. ring linker on the CURRENT ensemble's alerts (p >= 0.40): union-find over shared accounts -> number of cases
  4. shortcut check: tuned XGBoost re-fitted without the payment-format field
Output: results/checks.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK, REPO
import json, numpy as np, xgboost as xgb
from common import load, to_np, train_sample, ALL
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

P = np.load(f"{WORK}/layered_preds.npz"); te = P["te"]; yte = P["y"]

def queue(yt, s, target):
    """Expected number of reviewed payments (highest score first, ties in random order) to catch `target` of positives."""
    need = int(np.ceil(target * yt.sum()))
    vals, inv = np.unique(-s, return_inverse=True)            # groups in descending score order
    g_n = np.bincount(inv); g_k = np.bincount(inv, weights=yt).astype(int)
    before_n = np.concatenate([[0], np.cumsum(g_n)[:-1]]); before_k = np.concatenate([[0], np.cumsum(g_k)[:-1]])
    j = int(np.searchsorted(np.cumsum(g_k), need))            # first group where cumulative positives reach `need`
    m = need - before_k[j]; g, k = g_n[j], g_k[j]
    return float(before_n[j] + m * (g + 1) / (k + 1))         # expected draws to get m positives from g items holding k

d, split = load(); y = d["y"].to_numpy().astype(np.int8); X = to_np(d, ALL); col = {c: i for i, c in enumerate(ALL)}
tr = train_sample(y, split, 0.10, 0); va = np.where(split == 1)[0]; vaA = va[: len(va) // 2]
neg_tr = tr[y[tr] == 0]
q = lambda c, qq: float(np.nanquantile(X[neg_tr, col[c]], qq))
RULES = [lambda Z: Z[:, col["s_cnt_24h"]] >= q("s_cnt_24h", .99), lambda Z: Z[:, col["s_fanout_24h"]] >= q("s_fanout_24h", .99),
         lambda Z: Z[:, col["d_fanin_24h"]] >= q("d_fanin_24h", .99), lambda Z: Z[:, col["s_passthrough"]] >= q("s_passthrough", .99),
         lambda Z: Z[:, col["s_mins_since_in"]] <= q("s_mins_since_in", .01), lambda Z: (Z[:, col["new_pair"]] == 1) & (Z[:, col["cross_bank"]] == 1),
         lambda Z: (Z[:, col["round_100"]] == 1) & (Z[:, col["cross_cur"]] == 1), lambda Z: Z[:, col["log_usd"]] >= q("log_usd", .99),
         lambda Z: Z[:, col["reciprocal"]] == 1]
rm = lambda idx: np.stack([np.nan_to_num(f(X[idx]).astype(np.float32)) for f in RULES], 1)
R_A, R_T = rm(vaA), rm(te)
assert np.array_equal(R_T, P["rules"]), "rule flags differ from layered_engine.py"
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(R_A, y[vaA]); s_lr = lr.decision_function(R_T)

scores = {"rules: count of triggered rules": P["rules"].sum(1).astype(float), "rules: learned weights (logistic regression on 9 rule flags)": s_lr,
          "Isolation Forest": P["iso"], "Autoencoder": P["ae"], "XGBoost, payment fields only": P["L_transaction"], "XGBoost, + sender behaviour": P["L_behaviour"],
          "XGBoost, + network features": P["L_graph"], "XGBoost, all features": P["L_flat_all_xgb"], "LightGBM, all features": P["L_flat_all_lgbm"],
          "Random Forest, all features": P["L_flat_all_rf"], "RuseTect stacked ensemble (raw stacker score)": P["fused"]}
out = {"queue_note": "expected alerts when tied scores are reviewed in random order", "queue": {}, "pr_auc": {}}
for k, s in scores.items():
    out["queue"][k] = {f"catch_{int(t*100)}pct": round(queue(yte, s, t)) for t in (0.3, 0.5, 0.7)}
    out["pr_auc"][k] = float(average_precision_score(yte, s))
    print(f"{k:60s} AP {out['pr_auc'][k]:.4f} {out['queue'][k]}", flush=True)
out["pr_auc"]["RuseTect calibrated probability (isotonic, has ties)"] = float(average_precision_score(yte, P["fused_cal"]))

# ---- ring linker on current alerts
src = d["src_id"].to_numpy()[te]; dst = d["dst_id"].to_numpy()[te]
al = np.where(P["fused_cal"] >= 0.40)[0]
par = {}
def f(a):
    par.setdefault(a, a)
    while par[a] != a: par[a] = par[par[a]]; a = par[a]
    return a
for i in al: par[f(int(src[i]))] = f(int(dst[i]))
cases = {}
for i in al: cases.setdefault(f(int(src[i])), []).append(i)
sizes = np.array([len(v) for v in cases.values()])
out["ring_linker_current"] = dict(alerts=int(len(al)), cases=int(len(cases)), largest_case_payments=int(sizes.max()),
                                  cases_with_2plus_payments=int((sizes >= 2).sum()), laundering_in_alerts=int(yte[al].sum()))
print(out["ring_linker_current"], flush=True)

# ---- shortcut check with the tuned XGBoost
cmp_ = json.load(open(os.path.join(REPO, "results", "model_comparison_tabular.json")))["models"]["XGBoost"]["chosen_params"]
spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum()) ** 0.5
for name, cols in (("all features", ALL), ("without payment format", [c for c in ALL if c != "fmt_code"])):
    ci = [col[c] for c in cols]
    m = xgb.XGBClassifier(**cmp_, tree_method="hist", n_jobs=2, random_state=0, scale_pos_weight=spw).fit(X[tr][:, ci], y[tr])
    out.setdefault("shortcut_check", {})[name] = float(average_precision_score(yte, m.predict_proba(X[te][:, ci])[:, 1]))
print(out["shortcut_check"])
json.dump(out, open(os.path.join(REPO, "results", "checks.json"), "w"), indent=1)
