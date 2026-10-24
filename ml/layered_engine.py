"""
RuseTect: layered engine experiment. Every layer is trained and scored separately, then fused, then compared with
single-model challengers under ONE protocol (same time-ordered 60/20/20 split as ml/compare_models.py).

Layers (each answers a different question):
  L1 rules      : 9 hand-written velocity / pass-through / counterparty rules, thresholds = 99th percentile of TRAINING negatives
  L2 transaction: XGBoost on payment fields only            (is this payment odd in itself?)
  L2 behaviour  : XGBoost on payment fields + sender-history features   (is it odd for THIS sender?)
  L3 anomaly    : Isolation Forest, autoencoder reconstruction error (no labels used)
  L4 graph      : XGBoost on payment fields + counterparty/network features (fan-in/out, new pairs, pass-through, reciprocity)
  L5 fusion     : logistic-regression or LightGBM stacker fitted on the first half of the validation period,
                  isotonic calibration + threshold on the second half. Test period is scored once.
Also run: unseen-typology hold-out (train without one IBM laundering typology, test on it), alert-queue analysis,
          calibration, and component latency / size.
Outputs: results/layered_engine.json, $RUSETECT_WORK/layered_preds.npz
Usage  : RUSETECT_DATA=... RUSETECT_WORK=... python ml/layered_engine.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK, REPO
import gc, json, time, warnings, pickle
import numpy as np, polars as pl
warnings.filterwarnings("ignore")
from common import load, to_np, train_sample, ALL, RAW, GRAPH
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, brier_score_loss
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
import xgboost as xgb, lightgbm as lgb

SEED = 0
BUDGET = 0.002
t0 = time.time()
BEH = ["s_out_n", "s_out_mean_usd", "s_gap_out", "s_age_out", "s_cnt_24h", "s_usd_24h", "s_in_n", "s_mins_since_in", "s_out_over_last_in", "s_passthrough"]
NET = ["new_pair", "s_uniq_dst", "s_fanout_24h", "d_in_n", "d_gap_in", "d_uniq_src", "d_out_n", "d_in_usd_log", "d_fanin_24h", "d_cnt_24h", "reciprocal", "new_bank_pair"]
assert sorted(BEH + NET) == sorted(GRAPH)
GROUPS = {"transaction": RAW, "behaviour": RAW + BEH, "graph": RAW + NET}

d, split = load()
y = d["y"].to_numpy().astype(np.int8); n = len(y)
typ_df = pl.read_parquet(f"{WORK}/typology.parquet")
typ = np.full(n, "", dtype=object)
rid = d["rid"].to_numpy(); m = dict(zip(typ_df["rid"].to_list(), typ_df["typ"].to_list()))
for i in np.where(y == 1)[0]: typ[i] = m.get(int(rid[i])) or "UNLABELLED"
X = to_np(d, ALL); col = {c: i for i, c in enumerate(ALL)}
tr = train_sample(y, split, 0.10, SEED); va = np.where(split == 1)[0]; te = np.where(split == 2)[0]
vaA, vaB = va[: len(va) // 2], va[len(va) // 2:]
yte = y[te]
print(f"data {time.time()-t0:.0f}s | train {len(tr):,} ({y[tr].sum()} pos) | valA {len(vaA):,} ({y[vaA].sum()}) valB {len(vaB):,} ({y[vaB].sum()}) | test {len(te):,} ({yte.sum()})", flush=True)
del d; gc.collect()
spw = float((y[tr] == 0).sum() / (y[tr] == 1).sum()) ** 0.5
# hyper-parameters: the configurations ml/compare_models.py selected on VALIDATION average precision (test never used)
_cmp = json.load(open(os.path.join(REPO, "results", "model_comparison_tabular.json")))["models"]
XGBP = dict(**_cmp["XGBoost"]["chosen_params"], tree_method="hist", n_jobs=2, random_state=SEED, scale_pos_weight=spw)
LGBP = dict(**_cmp["LightGBM"]["chosen_params"], subsample_freq=1, n_jobs=2, random_state=SEED, verbose=-1, scale_pos_weight=spw)
RFP = dict(**_cmp["Random Forest"]["chosen_params"])

# ------------------------------------------------------------------ metrics helpers
def best_thr(yt, p):
    pr, rc, th = precision_recall_curve(yt, p); f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None); return float(th[int(np.nanargmax(f1[:-1]))])

def summarize(yt, p, thr):
    pred = p >= thr; tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum()); fn = int((~pred & (yt == 1)).sum()); tn = int(len(yt) - tp - fp - fn)
    pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    k = max(1, int(len(yt) * BUDGET)); top = np.argpartition(-p, k)[:k]
    return dict(pr_auc=float(average_precision_score(yt, p)), roc_auc=float(roc_auc_score(yt, p)), threshold=float(thr), precision=pr, recall=rc,
                f1=2 * pr * rc / max(pr + rc, 1e-12), tp=tp, fp=fp, fn=fn, tn=tn, alerts=tp + fp, fpr=fp / max(fp + tn, 1),
                recall_at_top_0_2pct=float(yt[top].sum() / yt.sum()), precision_at_top_0_2pct=float(yt[top].mean()))

def alerts_for_recall(yt, p, target):
    """Smallest number of flagged payments (taking the highest scores first) that catches `target` of all laundering payments."""
    # tie-aware: tied scores are reviewed in random order, so return the expected count (see ml/queue_and_checks.py)
    need = int(np.ceil(target * yt.sum())); vals, inv = np.unique(-p, return_inverse=True)
    g_n = np.bincount(inv); g_k = np.bincount(inv, weights=yt).astype(int); ck = np.cumsum(g_k)
    j = int(np.searchsorted(ck, need)); bn = int(g_n[:j].sum()); bk = int(g_k[:j].sum())
    return int(round(bn + (need - bk) * (g_n[j] + 1) / (g_k[j] + 1)))

def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

# ------------------------------------------------------------------ L1 rules
neg_tr = tr[y[tr] == 0]
q = lambda c, qq: float(np.nanquantile(X[neg_tr, col[c]], qq))
RULES = {  # name -> (human-readable text, function)
 "velocity":    (f"sender made >= {q('s_cnt_24h', .99):.0f} payments in 24 h", lambda Z: Z[:, col["s_cnt_24h"]] >= q("s_cnt_24h", .99)),
 "fan_out":     (f"sender paid >= {q('s_fanout_24h', .99):.0f} different accounts in 24 h", lambda Z: Z[:, col["s_fanout_24h"]] >= q("s_fanout_24h", .99)),
 "fan_in":      (f"receiver got money from >= {q('d_fanin_24h', .99):.0f} different accounts in 24 h", lambda Z: Z[:, col["d_fanin_24h"]] >= q("d_fanin_24h", .99)),
 "pass_through":(f"sender forwards >= {q('s_passthrough', .99):.2f} of what it received", lambda Z: Z[:, col["s_passthrough"]] >= q("s_passthrough", .99)),
 "rapid_forward":(f"money leaves <= {np.nanquantile(X[neg_tr, col['s_mins_since_in']], .01):.0f} min after it arrived", lambda Z: Z[:, col["s_mins_since_in"]] <= np.nanquantile(X[neg_tr, col["s_mins_since_in"]], .01)),
 "new_counterparty": ("first payment between these two accounts to a different bank", lambda Z: (Z[:, col["new_pair"]] == 1) & (Z[:, col["cross_bank"]] == 1)),
 "round_cross_currency": ("round amount paid in a different currency", lambda Z: (Z[:, col["round_100"]] == 1) & (Z[:, col["cross_cur"]] == 1)),
 "large_amount": (f"amount >= ${np.expm1(q('log_usd', .99)):,.0f}", lambda Z: Z[:, col["log_usd"]] >= q("log_usd", .99)),
 "money_returns": ("receiver pays money straight back (round trip)", lambda Z: Z[:, col["reciprocal"]] == 1),
}
def rule_matrix(idx):
    Z = X[idx]; return np.stack([np.nan_to_num(f(Z).astype(np.float32)) for _, f in RULES.values()], 1)
rules_va, rules_te = rule_matrix(va), rule_matrix(te)
rules_vaA, rules_vaB = rules_va[: len(vaA)], rules_va[len(vaA):]
print(f"L1 rules {time.time()-t0:.0f}s | rule-only AP test {average_precision_score(yte, rules_te.sum(1)):.4f}", flush=True)

# ------------------------------------------------------------------ L3 anomaly (no labels)
def slog(Z): Z = np.nan_to_num(Z, nan=-1.0); return np.sign(Z) * np.log1p(np.abs(Z))
sc = StandardScaler().fit(slog(X[tr]))
Zs = lambda idx: sc.transform(slog(X[idx])).astype(np.float32)
t = time.time(); iso = IsolationForest(n_estimators=300, max_samples=4096, n_jobs=2, random_state=SEED).fit(np.nan_to_num(X[tr], nan=-1.0)); iso_fit = time.time() - t
iso_s = lambda idx: -iso.score_samples(np.nan_to_num(X[idx], nan=-1.0))
import torch, torch.nn as nn
torch.set_num_threads(2); torch.manual_seed(SEED)
ae = nn.Sequential(nn.Linear(len(ALL), 24), nn.ReLU(), nn.Linear(24, 8), nn.ReLU(), nn.Linear(8, 24), nn.ReLU(), nn.Linear(24, len(ALL)))
Ztr = torch.tensor(Zs(neg_tr)); opt = torch.optim.Adam(ae.parameters(), 2e-3); t = time.time()
for ep in range(6):
    perm = torch.randperm(len(Ztr))
    for i in range(0, len(Ztr), 2048):
        b = Ztr[perm[i:i + 2048]]; loss = ((ae(b) - b) ** 2).mean(); opt.zero_grad(); loss.backward(); opt.step()
ae_fit = time.time() - t
def ae_s(idx):
    out = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), 200000):
            z = torch.tensor(Zs(idx[i:i + 200000])); out[i:i + 200000] = ((ae(z) - z) ** 2).mean(1).numpy()
    return out
iso_va, iso_te, ae_va, ae_te = iso_s(va), iso_s(te), ae_s(va), ae_s(te)
iso_vaA, iso_vaB, ae_vaA, ae_vaB = iso_va[: len(vaA)], iso_va[len(vaA):], ae_va[: len(vaA)], ae_va[len(vaA):]
print(f"L3 anomaly {time.time()-t0:.0f}s | IF AP {average_precision_score(yte, iso_te):.4f} | AE AP {average_precision_score(yte, ae_te):.4f}", flush=True)
# LOF (novelty) on a stratified evaluation subsample only: scoring 1M rows against a reference set is too slow here
rng = np.random.default_rng(SEED)
sub = np.sort(np.concatenate([np.where(yte == 1)[0], rng.choice(np.where(yte == 0)[0], int((yte == 0).sum() * 0.05), replace=False)]))
wsub = np.where(yte[sub] == 1, 1.0, 20.0)
t = time.time(); lof = LocalOutlierFactor(n_neighbors=30, novelty=True, n_jobs=2).fit(Zs(rng.choice(neg_tr, 20000, replace=False)))
lof_sub = -lof.score_samples(Zs(te[sub])); lof_s = time.time() - t
lof_ap = float(average_precision_score(yte[sub], lof_sub, sample_weight=wsub))
print(f"LOF (subsampled) AP {lof_ap:.4f} ({lof_s:.0f}s)", flush=True)

# ------------------------------------------------------------------ supervised layers
def fit_xgb(cols, idx, **kw):
    ci = [col[c] for c in cols]; return xgb.XGBClassifier(**{**XGBP, **kw}).fit(X[idx][:, ci], y[idx]), ci
def layer_scores(models, idx): return {k: m.predict_proba(X[idx][:, ci])[:, 1] for k, (m, ci) in models.items()}

def train_layers(train_idx, with_flat=True):
    mod = {k: fit_xgb(c, train_idx) for k, c in GROUPS.items()}
    if with_flat: mod["flat_all_xgb"] = fit_xgb(ALL, train_idx)
    return mod
t = time.time(); L = train_layers(tr); print(f"supervised layers fit {time.time()-t:.0f}s", flush=True)
S_vaA, S_vaB, S_te = layer_scores(L, vaA), layer_scores(L, vaB), layer_scores(L, te)
ci_all = [col[c] for c in ALL]
t = time.time(); lg = lgb.LGBMClassifier(**LGBP).fit(X[tr], y[tr]); lg_fit = time.time() - t
t = time.time(); rf = RandomForestClassifier(n_estimators=300, class_weight="balanced_subsample", n_jobs=2, random_state=SEED, **RFP).fit(np.nan_to_num(X[tr], nan=-1.0), y[tr]); rf_fit = time.time() - t
for S, idx in ((S_vaA, vaA), (S_vaB, vaB), (S_te, te)):
    S["flat_all_lgbm"] = lg.predict_proba(X[idx])[:, 1]; S["flat_all_rf"] = rf.predict_proba(np.nan_to_num(X[idx], nan=-1.0))[:, 1]
print(f"flat LGBM/RF fit {lg_fit:.0f}s/{rf_fit:.0f}s | total {time.time()-t0:.0f}s", flush=True)
for k, v in S_te.items(): print(f"   {k:16s} test AP {average_precision_score(yte, v):.4f}", flush=True)

# ------------------------------------------------------------------ L5 fusion
def design(S, iso_, ae_, rules, names):
    cols = []
    for nme in names:
        if nme == "iso": cols.append(iso_)
        elif nme == "ae": cols.append(np.log(ae_ + 1e-6))
        elif nme == "rules": cols += [rules.sum(1)]
        else: cols.append(logit(S[nme]))
    return np.stack(cols, 1).astype(np.float32)
FUSIONS = {
  "layers only (txn+behaviour+graph)":             ["transaction", "behaviour", "graph"],
  "layers + rules":                                 ["transaction", "behaviour", "graph", "rules"],
  "layers + rules + anomaly (IF+AE)":               ["transaction", "behaviour", "graph", "rules", "iso", "ae"],
  "full ensemble (layers+rules+anomaly+XGB+LGBM+RF)": ["transaction", "behaviour", "graph", "rules", "iso", "ae", "flat_all_xgb", "flat_all_lgbm", "flat_all_rf"],
}
def fuse(names, learner, trainsets):
    (SA, iA, aA, rA, yA), (SB, iB, aB, rB, yB), (ST, iT, aT, rT) = trainsets
    A, B, T = design(SA, iA, aA, rA, names), design(SB, iB, aB, rB, names), design(ST, iT, aT, rT, names)
    if learner == "logreg":
        s = StandardScaler().fit(A); mdl = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced").fit(s.transform(A), yA)
        f = lambda Z: mdl.decision_function(s.transform(Z))
    else:
        mdl = lgb.LGBMClassifier(num_leaves=8, n_estimators=200, learning_rate=0.05, min_child_samples=50, subsample=0.8, subsample_freq=1, n_jobs=2, random_state=SEED, verbose=-1, scale_pos_weight=20).fit(A, yA)
        f = lambda Z: logit(mdl.predict_proba(Z)[:, 1])
    return f(B), f(T), mdl, (s if learner == "logreg" else None)
packA = (S_vaA, iso_vaA, ae_vaA, rules_vaA, y[vaA]); packB = (S_vaB, iso_vaB, ae_vaB, rules_vaB, y[vaB]); packT = (S_te, iso_te, ae_te, rules_te)
fusion_res, best = {}, None
for fname, names in FUSIONS.items():
    for learner in ("logreg", "lgbm_stacker"):
        sB, sT, mdl, scl = fuse(names, learner, (packA, packB, packT))
        apB = float(average_precision_score(y[vaB], sB)); thr = best_thr(y[vaB], sB)
        fusion_res[f"{fname} | {learner}"] = dict(inputs=names, learner=learner, val_ap=apB, test=summarize(yte, sT, thr))
        if best is None or apB > best[0]: best = (apB, f"{fname} | {learner}", sB, sT, mdl, scl, names, learner)
        print(f"   fusion {fname:52s} {learner:13s} val-AP {apB:.4f} test-AP {fusion_res[f'{fname} | {learner}']['test']['pr_auc']:.4f}", flush=True)
_, best_name, sB, sT, best_mdl, best_scl, best_names, best_learner = best
print("selected on validation:", best_name, flush=True)
cal = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(sB, y[vaB]); pT = cal.predict(sT)
pT_adj = np.clip(pT, 0, 1)
def reliability(yt, p, edges=(0, .001, .01, .05, .1, .25, .5, .75, .9, 1.0001)):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mk = (p >= lo) & (p < hi)
        if mk.sum(): rows.append(dict(lo=lo, hi=min(hi, 1.0), n=int(mk.sum()), mean_pred=float(p[mk].mean()), observed=float(yt[mk].mean())))
    return rows
rel = reliability(yte, pT_adj)
ece = float(sum(r["n"] * abs(r["mean_pred"] - r["observed"]) for r in rel) / len(yte))
cal_info = dict(method="isotonic regression fitted on validation period B", brier_calibrated=float(brier_score_loss(yte, pT_adj)),
                brier_uncalibrated_sigmoid=float(brier_score_loss(yte, 1 / (1 + np.exp(-sT)))), ece=ece, reliability=rel)
# calibrated threshold = value that maximised F1 on validation B, mapped through the calibrator
thr_cal = float(cal.predict([best_thr(y[vaB], sB)])[0])
cal_info["f1_optimal_probability_on_validation"] = thr_cal

# ------------------------------------------------------------------ challengers + queue analysis (test)
single = {k: v for k, v in S_te.items()}
chall = {"rules only (count of triggered rules)": rules_te.sum(1).astype(float), "Isolation Forest (unsupervised)": iso_te, "Autoencoder (unsupervised)": ae_te,
         "XGBoost, transaction fields": S_te["transaction"], "XGBoost, + sender behaviour": S_te["behaviour"], "XGBoost, + network features": S_te["graph"],
         "XGBoost, all features (flat)": S_te["flat_all_xgb"], "LightGBM, all features (flat)": S_te["flat_all_lgbm"], "Random Forest, all features (flat)": S_te["flat_all_rf"],
         f"RuseTect fused ({best_name})": sT}
queue, challenger_tbl = {}, {}
for nme, sc_ in chall.items():
    VB = {"rules only (count of triggered rules)": rules_vaB.sum(1), "Isolation Forest (unsupervised)": iso_vaB, "Autoencoder (unsupervised)": ae_vaB,
                           "XGBoost, transaction fields": S_vaB["transaction"], "XGBoost, + sender behaviour": S_vaB["behaviour"], "XGBoost, + network features": S_vaB["graph"],
                           "XGBoost, all features (flat)": S_vaB["flat_all_xgb"], "LightGBM, all features (flat)": S_vaB["flat_all_lgbm"],
                           "Random Forest, all features (flat)": S_vaB["flat_all_rf"], f"RuseTect fused ({best_name})": sB}
    thr = best_thr(y[vaB], VB[nme])
    challenger_tbl[nme] = summarize(yte, sc_, thr)
    challenger_tbl[nme]["val_B_pr_auc"] = float(average_precision_score(y[vaB], VB[nme]))
    queue[nme] = {f"alerts_to_catch_{int(r*100)}pct": alerts_for_recall(yte, sc_, r) for r in (0.3, 0.5, 0.7)}
challenger_tbl["LOF (novelty, subsample of 5% negatives, weighted)"] = dict(pr_auc=lof_ap, note="scored on a stratified test subsample; not run on all rows")
rule_hits = {k: dict(text=RULES[k][0], flagged=int(rules_te[:, i].sum()), laundering_flagged=int(rules_te[yte == 1][:, i].sum()),
                     precision=float(rules_te[yte == 1][:, i].sum() / max(rules_te[:, i].sum(), 1))) for i, k in enumerate(RULES)}
print("challenger table done", f"{time.time()-t0:.0f}s", flush=True)

# ------------------------------------------------------------------ latency / size of components (single-row, model only)
def lat(fn, reps=200):
    for _ in range(5): fn(0)
    ts = []
    for i in range(reps): a = time.perf_counter(); fn(i); ts.append(time.perf_counter() - a)
    return round(float(np.median(ts) * 1e3), 3), round(float(np.percentile(ts, 99) * 1e3), 3)
r1 = te[:300]; Xr = X[r1]
lat_tbl = {"XGBoost layer (300 trees, depth 8)": lat(lambda i: L["behaviour"][0].predict_proba(Xr[i:i + 1, L["behaviour"][1]])),
           "LightGBM (300 trees)": lat(lambda i: lg.predict_proba(Xr[i:i + 1])), "Random Forest (300 trees)": lat(lambda i: rf.predict_proba(np.nan_to_num(Xr[i:i + 1], nan=-1.0))),
           "Isolation Forest (300 trees)": lat(lambda i: iso.score_samples(np.nan_to_num(Xr[i:i + 1], nan=-1.0))),
           "Autoencoder (31-24-8-24-31)": lat(lambda i: ae(torch.tensor(Zs(r1[i:i + 1]))))}
size_tbl = {"XGBoost layer": len(L["behaviour"][0].get_booster().save_raw("ubj")), "LightGBM": len(lg.booster_.model_to_string().encode()),
            "Random Forest": len(pickle.dumps(rf)), "Isolation Forest": len(pickle.dumps(iso)), "Autoencoder": sum(p.numel() for p in ae.parameters()) * 4}

# ------------------------------------------------------------------ unseen-typology hold-out
ty_te = typ[te]
holdout = {}
def hold_eval(T):
    keep = lambda idx: idx[typ[idx] != T]          # drop the held-out typology's positives from every labelled set
    trH, vaAH, vaBH = keep(tr), keep(vaA), keep(vaB)
    LH = train_layers(trH)
    SA, SB, ST = layer_scores(LH, vaAH), layer_scores(LH, vaBH), layer_scores(LH, te)
    mA, mB = typ[vaA] != T, typ[vaB] != T
    def run(names):
        sB_, sT_, *_ = fuse(names, "logreg", ((SA, iso_vaA[mA], ae_vaA[mA], rules_vaA[mA], y[vaAH]), (SB, iso_vaB[mB], ae_vaB[mB], rules_vaB[mB], y[vaBH]), (ST, iso_te, ae_te, rules_te)))
        return sT_
    out = {}
    Tm = ty_te == T; k = int(len(yte) * BUDGET)
    def rec(sc_):
        top = np.zeros(len(yte), bool); top[np.argpartition(-sc_, k)[:k]] = True
        return dict(recall_at_top_0_2pct=float(top[Tm].mean()),
                    median_percentile_rank=float(np.median((np.searchsorted(np.sort(sc_), sc_[Tm]) / len(sc_)) * 100)))
    out["XGBoost, all features (flat, supervised only)"] = rec(ST["flat_all_xgb"])
    out["fused: layers + rules (no anomaly layer)"] = rec(run(["transaction", "behaviour", "graph", "rules"]))
    out["fused: layers + rules + anomaly (IF+AE)"] = rec(run(["transaction", "behaviour", "graph", "rules", "iso", "ae"]))
    out["Isolation Forest alone (never saw labels)"] = rec(iso_te)
    out["_test_payments_of_held_out_typology"] = int(Tm.sum())
    return out
for T in ("GATHER-SCATTER", "SCATTER-GATHER", "STACK", "CYCLE"):
    t = time.time(); holdout[T] = hold_eval(T); print(f"hold-out {T} {time.time()-t:.0f}s: ", {k: (round(v['recall_at_top_0_2pct'], 3) if isinstance(v, dict) else v) for k, v in holdout[T].items()}, flush=True)
# same typologies when they ARE in training (reference)
ref = {}
for T in holdout:
    Tm = ty_te == T; k = int(len(yte) * BUDGET)
    for nme, sc_ in (("XGBoost, all features (flat, supervised only)", S_te["flat_all_xgb"]), (f"RuseTect fused ({best_name})", sT)):
        top = np.zeros(len(yte), bool); top[np.argpartition(-sc_, k)[:k]] = True
        ref.setdefault(T, {})[nme] = float(top[Tm].mean())

# ------------------------------------------------------------------ save
def curve(yt, p, nn=200):
    pr, rc, th = precision_recall_curve(yt, p); i = np.unique(np.linspace(0, len(pr) - 1, nn).astype(int)); return dict(precision=pr[i].tolist(), recall=rc[i].tolist())
out = dict(
  protocol=dict(split="time-ordered 60/20/20 (same as compare_models.py)", train_rows=int(len(tr)), val_A_rows=int(len(vaA)), val_B_rows=int(len(vaB)), test_rows=int(len(te)),
                test_positives=int(yte.sum()), test_prevalence=float(yte.mean()), alert_budget=f"top {BUDGET:.1%} of payments = {int(len(te)*BUDGET):,} alerts",
                fusion_fit="stacker on validation A; isotonic calibration + threshold on validation B; test scored once",
                selection="fusion variant selected by validation-B average precision (test not used)"),
  rules=dict(definitions={k: v[0] for k, v in RULES.items()}, test_hits=rule_hits),
  challengers=challenger_tbl, queue_to_catch=queue, fusion_variants=fusion_res, selected_fusion=best_name,
  calibration=cal_info, latency_ms_single_row=lat_tbl, model_size_bytes=size_tbl,
  fit_seconds=dict(isolation_forest=round(iso_fit, 1), autoencoder=round(ae_fit, 1), lightgbm=round(lg_fit, 1), random_forest=round(rf_fit, 1)),
  unseen_typology_holdout=holdout, same_typology_when_seen=ref,
  typology_counts_in_test={t_: int((ty_te == t_).sum()) for t_ in sorted(set(ty_te.tolist())) if t_},
  pr_curves={k: curve(yte, v) for k, v in {"rules": chall["rules only (count of triggered rules)"], "XGBoost, transaction fields": S_te["transaction"], "XGBoost, all features (flat)": S_te["flat_all_xgb"], "fused": sT, "Isolation Forest": iso_te}.items()},
  runtime_seconds=round(time.time() - t0))
os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
json.dump(out, open(os.path.join(REPO, "results", "layered_engine.json"), "w"), indent=1)
np.savez_compressed(f"{WORK}/layered_preds.npz", te=te, y=yte, fused=sT, fused_cal=pT_adj, iso=iso_te, ae=ae_te, rules=rules_te,
                    **{f"L_{k}": v for k, v in S_te.items()})
pickle.dump(dict(best_mdl=best_mdl, scaler=best_scl, names=best_names, learner=best_learner, cal=cal, flat=L["flat_all_xgb"][0]), open(f"{WORK}/layered_models.pkl", "wb"))
print("saved results/layered_engine.json", f"{time.time()-t0:.0f}s", flush=True)
