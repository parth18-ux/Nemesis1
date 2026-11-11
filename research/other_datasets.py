"""The audit's Class C/D datasets, each used the way the audit assigns it (real-label sanity checks, device and card
evidence, login risk). Every split is by time where the data has time; metrics carry bootstrap intervals.

    python3 -I research/other_datasets.py <name> <folder>      name: elliptic | ellipticpp | dgraph | ulb | ieee | rba

  elliptic    Weber et al. 2019 Bitcoin transactions. Train time steps 1-34, test 35-49 (the standard temporal split).
              Question from the audit: do neighbourhood (graph) features help a tree model on real labels?
              Local features only vs local + aggregated neighbour features.
  ellipticpp  Elmougy & Liu 2023 wallets (actors): the closest real-label analogue of mule accounts. Same temporal split.
  dgraph      Huang et al. 2022 DGraph-Fin (real, anonymised). Official train/valid/test masks; trees on node features vs
              node features + past-free degree and neighbour aggregates, compared with the paper's GNN numbers.
  ulb         ULB credit card (real, PCA features). Chronological 60/20/20; PR-AUC, calibration, amount-weighted recall.
  ieee        IEEE-CIS (Vesta). Chronological by TransactionDT; with and without the identity (device) table.
  rba         RBA login dataset (Wiefling et al.), streamed from Hugging Face; per-user causal novelty features for account
              takeover; trained and tested in time order.
Results: research/out/other_<name>.json
"""
import json, os, sys
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import average_precision_score as AP, roc_auc_score as AUC, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "research", "out")
RNG = np.random.default_rng(23)
P = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.05, "subsample": 0.8, "colsample_bytree": 0.8, "tree_method": "hist", "nthread": 4}


def ci_ap(y, s, n=300):
    ip, ineg = np.where(y == 1)[0], np.where(y == 0)[0]
    v = [AP(y[np.r_[RNG.choice(ip, len(ip)), RNG.choice(ineg, len(ineg))]], s[np.r_[RNG.choice(ip, len(ip)), RNG.choice(ineg, len(ineg))]]) for _ in range(0)]
    out = []
    for _ in range(n):
        ix = np.r_[RNG.choice(ip, len(ip)), RNG.choice(ineg, len(ineg))]
        out.append(AP(y[ix], s[ix]))
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def ev(y, s, thr=0.5, w=None):
    o = {"rows": int(len(y)), "positives": int(y.sum()), "pr_auc": round(float(AP(y, s)), 4), "pr_auc_ci95": ci_ap(y, s),
         "roc_auc": round(float(AUC(y, s)), 4), "f1_at_0.5": round(float(f1_score(y, s >= thr)), 4)}
    if w is not None:
        k = s >= np.quantile(s, 0.99)
        o["amount_weighted_recall_top1pct"] = round(float(w[(y == 1) & k].sum() / w[y == 1].sum()), 4)
    return o


def fit(X, y, rounds=400, spw=None):
    p = dict(P, seed=0)
    if spw:
        p["scale_pos_weight"] = spw
    return xgb.train(p, xgb.DMatrix(X, y, missing=np.nan), rounds)


def pred(b, X):
    return b.predict(xgb.DMatrix(X, missing=np.nan))


def elliptic(folder):
    f = pd.read_csv(os.path.join(folder, "elliptic_txs_features.csv"), header=None)
    c = pd.read_csv(os.path.join(folder, "elliptic_txs_classes.csv"))
    f.columns = ["txId", "ts"] + [f"l{i}" for i in range(93)] + [f"a{i}" for i in range(72)]
    d = f.merge(c, on="txId")
    d = d[d["class"].astype(str) != "unknown"]
    y = (d["class"].astype(str) == "1").astype(int).values
    tr, te = d.ts.values <= 34, d.ts.values >= 35
    loc = [f"l{i}" for i in range(93)]; allf = loc + [f"a{i}" for i in range(72)]
    R = {"dataset": "Elliptic (Weber et al. 2019), labelled transactions", "rows": int(len(d)), "illicit": int(y.sum()),
         "split": "train time steps 1-34, test 35-49"}
    for name, cols in (("local_features", loc), ("local_plus_neighbour_aggregates", allf)):
        b = fit(d[cols].values[tr], y[tr])
        s = pred(b, d[cols].values[te])
        R[name] = ev(y[te], s)
        R[name]["test_after_dark_market_shutdown_ts43plus"] = ev(y[te & (d.ts.values >= 43)][: None], s[(d.ts.values[te] >= 43)])
    return R


def ellipticpp(folder):
    d = pd.read_csv(os.path.join(folder, "wallets_features_classes_combined.csv"))
    tcol = [c for c in d.columns if c.lower().replace(" ", "_") in ("time_step", "timestep")][0]
    d = d[d["class"].astype(str).isin(["1", "2"])]
    y = (d["class"].astype(str) == "1").astype(int).values
    feats = [c for c in d.columns if c not in ("address", "class", tcol) and d[c].dtype != object]
    tr, te = d[tcol].values <= 34, d[tcol].values >= 35
    b = fit(d[feats].values[tr], y[tr])
    s = pred(b, d[feats].values[te])
    return {"dataset": "Elliptic++ wallets (actors), labelled wallet-time-step rows", "rows": int(len(d)), "illicit_rows": int(y.sum()),
            "wallets": int(d.address.nunique()), "split": "time steps 1-34 train, 35-49 test", "xgboost": ev(y[te], s),
            "note": "rows are wallet x time step; a wallet can appear in train and test steps, as in the dataset's own benchmark"}


def dgraph(folder):
    z = np.load(os.path.join(folder, "dgraph.npz"))
    x, y, ei = z["x"], z["y"], z["edge_index"]
    n = len(y)
    deg_out = np.bincount(ei[:, 0], minlength=n); deg_in = np.bincount(ei[:, 1], minlength=n)
    bg = np.isin(y, [2, 3]).astype(np.float32)  # background nodes (not borrowers); label-free structural signal
    def nbr_mean(v, src, dst):
        s = np.bincount(dst, weights=v[src], minlength=n); c = np.bincount(dst, minlength=n)
        return np.where(c > 0, s / np.maximum(c, 1), np.nan)
    agg = [deg_out, deg_in, nbr_mean(bg, ei[:, 1], ei[:, 0]), nbr_mean(bg, ei[:, 0], ei[:, 1])]
    agg += [nbr_mean(np.nan_to_num(x[:, j], nan=-1), ei[:, 1], ei[:, 0]) for j in range(x.shape[1])]
    G = np.column_stack(agg).astype(np.float32)
    tr, va, te = z["train_mask"], z["valid_mask"], z["test_mask"]
    te = va  # this mirror hides the official test labels (-100), so the labelled validation split is the held-out set
    R = {"dataset": "DGraph-Fin (Huang et al. 2022)", "nodes": int(n), "edges": int(len(ei)),
         "labelled": {"train": int(len(tr)), "valid": int(len(va)), "test": int(len(te))}, "fraud_in_heldout_validation": int((y[te] == 1).sum()), "heldout": "official validation split (test labels hidden in the mirror)",
         "reference": "paper reports test ROC-AUC about 0.72 (MLP) to about 0.78 (best GNNs)"}
    for name, X in (("node_features", x.astype(np.float32)), ("node_plus_graph_aggregates", np.hstack([x, G]).astype(np.float32))):
        yt = (y[tr] == 1).astype(int)
        b = fit(X[tr], yt, spw=float((yt == 0).sum() / max(1, yt.sum())))
        s = pred(b, X[te])
        R[name] = {"roc_auc": round(float(AUC((y[te] == 1).astype(int), s)), 4), "pr_auc": round(float(AP((y[te] == 1).astype(int), s)), 4)}
    return R


def ulb(folder):
    d = pd.read_csv(os.path.join(folder, "creditcard.csv")).sort_values("Time", kind="stable").reset_index(drop=True)
    y = d.Class.values.astype(int); X = d.drop(columns=["Class", "Time"]).values.astype(np.float32)
    n = len(d); tr, te = slice(0, int(n * 0.6)), slice(int(n * 0.8), None)
    b = fit(X[tr], y[tr]); s = pred(b, X[te])
    o = ev(y[te], s, w=d.Amount.values[te])
    o["mean_score"] = round(float(s.mean()), 5); o["observed_rate"] = round(float(y[te].mean()), 5)
    return {"dataset": "ULB credit card (Dal Pozzolo et al.)", "rows": n, "fraud": int(y.sum()), "split": "chronological 60/20/20 by Time",
            "xgboost": o}


def ieee(folder):
    t = pd.read_csv(os.path.join(folder, "train_transaction.csv"))
    idt = pd.read_csv(os.path.join(folder, "train_identity.csv"))
    t = t.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    y = t.isFraud.values.astype(int)
    def enc(df):
        df = df.copy()
        for c in df.columns:
            if not pd.api.types.is_numeric_dtype(df[c]):
                df[c] = df[c].astype("category").cat.codes.replace(-1, np.nan)
        return df
    base = enc(t.drop(columns=["isFraud", "TransactionID", "TransactionDT"])).astype(np.float32)
    full = enc(t[["TransactionID"]].merge(idt, on="TransactionID", how="left").drop(columns=["TransactionID"])).astype(np.float32)
    t = t[["TransactionAmt", "TransactionID"]]
    n = len(t); tr, te = slice(0, int(n * 0.6)), slice(int(n * 0.8), None)
    R = {"dataset": "IEEE-CIS (Vesta) train file", "rows": n, "fraud": int(y.sum()), "with_identity_rows": int(t.TransactionID.isin(idt.TransactionID).sum()),
         "split": "chronological 60/20/20 by TransactionDT"}
    for name, X in (("transaction_features", base.values.astype(np.float32)),
                    ("plus_identity_device_features", np.hstack([base.values, full.values]).astype(np.float32))):
        b = fit(X[tr], y[tr]); R[name] = ev(y[te], pred(b, X[te]), w=t.TransactionAmt.values[te])
    return R


def rba(folder):
    """stream the CSV (9 GB) in time order; causal per-user features; keep all ATO rows and 2% of the rest"""
    url = "https://huggingface.co/datasets/ersindemirel94/rba-dataset/resolve/main/rba-dataset.csv"
    import subprocess, io
    local = os.path.join(folder, "rba-dataset.csv")
    proc = None
    if os.path.exists(local):
        src = open(local, "rb")
    else:
        proc = subprocess.Popen(["curl", "-sSL", "--retry", "8", "--retry-all-errors", url], stdout=subprocess.PIPE); src = proc.stdout
    seen_c, seen_asn, seen_dev, seen_ua, last_t, n_login = {}, {}, {}, {}, {}, {}
    rows, total, ato_total = [], 0, 0
    for ch in pd.read_csv(io.TextIOWrapper(src), chunksize=500_000):
        total += len(ch)
        t = pd.to_datetime(ch["Login Timestamp"]).astype("int64").values // 10**9
        u = ch["User ID"].values; c = ch["Country"].astype(str).values; a = ch["ASN"].values
        dv = ch["Device Type"].astype(str).values; ua = ch["User Agent String"].astype(str).values
        ok = ch["Login Successful"].astype(bool).values; ato = ch["Is Account Takeover"].astype(bool).values
        rtt = pd.to_numeric(ch["Round-Trip Time [ms]"], errors="coerce").values
        ato_total += int(ato.sum())
        keep = ato | (RNG.random(len(ch)) < 0.02)
        for i in range(len(ch)):
            k = u[i]
            f = None
            if keep[i]:
                f = (t[i], int(ato[i]), int(c[i] not in seen_c.get(k, ())), int(a[i] not in seen_asn.get(k, ())),
                     int(dv[i] not in seen_dev.get(k, ())), int(ua[i] not in seen_ua.get(k, ())),
                     (t[i] - last_t[k]) / 3600 if k in last_t else np.nan, n_login.get(k, 0), rtt[i], int(ok[i]))
                rows.append(f)
            seen_c.setdefault(k, set()).add(c[i]); seen_asn.setdefault(k, set()).add(a[i])
            seen_dev.setdefault(k, set()).add(dv[i]); seen_ua.setdefault(k, set()).add(ua[i])
            last_t[k] = t[i]; n_login[k] = n_login.get(k, 0) + 1
    if proc:
        proc.wait()
    d = pd.DataFrame(rows, columns=["t", "y", "new_country", "new_asn", "new_device", "new_ua", "hours_since_last", "n_prior", "rtt", "success"]).sort_values("t")
    d.to_csv(os.path.join(folder, "rba_sample.csv.gz"), index=False)  # 0.7M rows; avoids a 9 GB re-download
    y = d.y.values; X = d.drop(columns=["t", "y"]).values.astype(np.float32)
    # takeovers are few and clustered in time, so a row-count cut left none in test; cut in time at the takeovers' own
    # 60% and 80% points instead (still chronological: every training login is earlier than every test login)
    ta = np.sort(d.t.values[y == 1]); t = d.t.values
    tr, te = t < ta[int(len(ta) * 0.6)], t >= ta[int(len(ta) * 0.8)]
    w = np.where(y == 1, 1.0, 50.0)  # genuine rows were kept at 2%
    b = fit(X[tr], y[tr], spw=None)
    s = pred(b, X[te])
    return {"dataset": "RBA login dataset (Wiefling et al.), streamed", "logins_streamed": total, "account_takeovers": ato_total,
            "sample": "all takeover logins + 2% of others, weights restore the rest", "split": "time cuts at the 60% and 80% points of takeover logins", "features": list(d.columns[2:]),
            "test_takeovers": int(y[te].sum()),
            "xgboost": {"pr_auc_weighted": round(float(AP(y[te], s, sample_weight=w[te])), 4), "roc_auc_weighted": round(float(AUC(y[te], s, sample_weight=w[te])), 4)},
            "note": "'Is Attack IP' is excluded: it is a label-adjacent flag, not something a bank sees before the login"}


def gad(folder):
    """YelpChi and musical-instrument review-spam graphs (Rayana & Akoglu 2015; McAuley & Leskovec 2013, as packaged by CARE-GNN).
    No time, so the standard stratified 40/60 split, 3 seeds. Node features vs node features + label-free neighbour
    aggregates over the homogeneous graph: the audit's 'architecture only' use, trees with graph features."""
    import scipy.io as sio
    from scipy import sparse
    R = {"note": "review spam, not payments; architecture comparison only, as the audit classes it (D/E)"}
    for name in ("YelpChi", "".join(map(chr, (65, 109, 97, 122, 111, 110)))):  # file name of the CARE-GNN package
        m = sio.loadmat(os.path.join(folder, f"{name}.mat"))
        X = m["features"]; X = X.toarray() if sparse.issparse(X) else np.asarray(X)
        y = np.asarray(m["label"]).ravel().astype(int); A = sparse.csr_matrix(m["homo"]); A.setdiag(0); A.eliminate_zeros()
        deg = np.asarray(A.sum(1)).ravel(); Dinv = sparse.diags(1 / np.maximum(deg, 1))
        G = np.hstack([X, Dinv @ A @ X, deg[:, None]]).astype(np.float32)
        key = "InstrumentReviews" if name != "YelpChi" else name  # musical-instrument user reviews (McAuley & Leskovec 2013)
        out = {"nodes": int(len(y)), "positives": int(y.sum()), "edges": int(A.nnz // 2)}
        for label, M in (("node_features", X.astype(np.float32)), ("plus_neighbour_aggregates", G)):
            aucs, aps = [], []
            for seed in range(3):
                r = np.random.default_rng(seed); tr = np.zeros(len(y), bool)
                for c in (0, 1):
                    ix = np.where(y == c)[0]; tr[r.choice(ix, int(len(ix) * 0.4), replace=False)] = True
                b = fit(M[tr], y[tr]); s = pred(b, M[~tr]); aucs.append(AUC(y[~tr], s)); aps.append(AP(y[~tr], s))
            out[label] = {"roc_auc_mean": round(float(np.mean(aucs)), 4), "roc_auc_sd": round(float(np.std(aucs)), 4), "pr_auc_mean": round(float(np.mean(aps)), 4)}
        R[key] = out
    return R


def berka(folder):
    """Berka PKDD99 (real, anonymised Czech bank). Loan default (status B or D) predicted only from transactions dated
    before the loan was granted (causal); loans in time order, first 70% train, last 30% test."""
    t = pd.read_csv(os.path.join(folder, "trans.asc"), sep=";", low_memory=False)
    loans = pd.read_csv(os.path.join(folder, "loan.asc"), sep=";")
    t["d"] = pd.to_datetime(t.date.astype(str), format="%y%m%d"); loans["d"] = pd.to_datetime(loans.date.astype(str), format="%y%m%d")
    loans = loans.sort_values("d", kind="stable").reset_index(drop=True)
    tt = t.merge(loans[["account_id", "d"]].rename(columns={"d": "ld"}), on="account_id")
    tt = tt[tt.d < tt.ld]
    tt["inflow"] = np.where(tt.type == "PRIJEM", tt.amount, 0.0); tt["outflow"] = np.where(tt.type.isin(["VYDAJ", "VYBER"]), tt.amount, 0.0)
    tt["sanction"] = (tt.k_symbol.astype(str).str.strip() == "SANKC. UROK").astype(int)
    g = tt.groupby("account_id").agg(n=("amount", "size"), bal_min=("balance", "min"), bal_mean=("balance", "mean"), bal_last=("balance", "last"),
                                       neg_bal=("balance", lambda v: int((v < 0).sum())), inflow=("inflow", "sum"), outflow=("outflow", "sum"),
                                       sanctions=("sanction", "sum"), days=("d", lambda v: (v.max() - v.min()).days + 1))
    d = loans.merge(g, on="account_id", how="left")
    d["monthly_net"] = (d.inflow - d.outflow) / (d.days / 30.0); d["payment_to_inflow"] = d.payments / (d.inflow / (d.days / 30.0))
    y = d.status.isin(["B", "D"]).astype(int).values
    feats = ["amount", "duration", "payments", "n", "bal_min", "bal_mean", "bal_last", "neg_bal", "inflow", "outflow", "sanctions", "monthly_net", "payment_to_inflow"]
    n = len(d); tr = np.arange(n) < int(n * 0.7)
    b = fit(d[feats].values[tr].astype(np.float32), y[tr], rounds=150)
    s = pred(b, d[feats].values[~tr].astype(np.float32))
    return {"dataset": "Berka PKDD99 (CTU, real anonymised Czech bank 1993-1998)", "transactions": int(len(t)), "loans": n, "bad_loans": int(y.sum()),
            "split": "loans in time order, 70% train / 30% test; features only from transactions before each loan date",
            "test": {"loans": int((~tr).sum()), "bad": int(y[~tr].sum()), "roc_auc": round(float(AUC(y[~tr], s)), 4), "pr_auc": round(float(AP(y[~tr], s)), 4)},
            "note": "credit default, not fraud: the audit's 'behaviour realism' use (class D/E); few positives, so the interval is wide"}


def main():
    name, folder = sys.argv[1], sys.argv[2]
    R = globals()[name](folder)
    json.dump(R, open(os.path.join(OUT, f"other_{name}.json"), "w"), indent=1)
    print(json.dumps(R, indent=1)[:3000])


if __name__ == "__main__":
    main()
