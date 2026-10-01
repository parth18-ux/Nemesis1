"""
RuseTect — deep learning benchmark on IBM AMLworld HI-Small.

Sequence models (1D-CNN, LSTM, Transformer) read the last K events (money in / money out)
of BOTH the sender and the receiver strictly before the current payment, plus the current
payment's own features. MLP sees only the tabular features (same as the tree models).
GAN: tabular GAN trained on training positives; synthetic positives added to XGBoost training.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import json, sys, time, math
import numpy as np, polars as pl, torch, torch.nn as nn
from common import load, to_np, train_sample, best_f1_threshold, evaluate, ALL
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
torch.set_num_threads(2); torch.manual_seed(0); np.random.seed(0)

OUT = f"{WORK}/results_deep.json"
K = 12
results = {}
try: results = json.load(open(OUT))
except Exception: pass

d, split = load()
y = d["y"].to_numpy().astype(np.int8)
ts = d["ts"].to_numpy()
core = (split == 2) & (ts < np.datetime64("2022-09-11"))
tr = train_sample(y, split); va = np.where(split == 1)[0]; te = np.where(split == 2)[0]

# ---------------- tabular matrix (sign-log + standardise)
def slog(X): X = np.nan_to_num(X, nan=-1.0); return np.sign(X) * np.log1p(np.abs(X))
Xall = slog(to_np(d, ALL)); sc = StandardScaler().fit(Xall[tr]); Xall = sc.transform(Xall).astype(np.float32)

# ---------------- event table for sequences
n = d.height
src = d["src_id"].to_numpy(); dst = d["dst_id"].to_numpy(); order = d["order"].to_numpy()
tmin = d["tmin"].to_numpy().astype(np.int64); lusd = d["log_usd"].to_numpy().astype(np.float32)
fmt = d["fmt_code"].to_numpy().astype(np.int64); xb = d["cross_bank"].to_numpy().astype(np.float32)
nf = int(fmt.max()) + 1
aid = np.concatenate([src, dst]); eo = np.concatenate([order, order]); role = np.concatenate([np.zeros(n, np.int8), np.ones(n, np.int8)])
cp = np.concatenate([dst, src])                                    # counterparty of the event
perm = np.lexsort((role, eo, aid)); aid, eo, role, cp = aid[perm], eo[perm], role[perm], cp[perm]
pos_of = np.empty(2 * n, np.int64); pos_of[perm] = np.arange(2 * n)
p_send = pos_of[:n]          # position of txn i as an OUT event of its sender
p_recv = pos_of[n:]          # position of txn i as an IN event of its receiver
E_t = tmin[eo]; E_l = lusd[eo]; E_f = fmt[eo]; E_x = xb[eo]
del perm, pos_of

EV_DIM = 5 + nf
def seq_batch(idx):
    """Return (B, 2, K, EV_DIM) tensor of sender/receiver history + mask."""
    B = len(idx); out = np.zeros((B, 2, K, EV_DIM), np.float32); mask = np.zeros((B, 2, K), bool)
    for side, (p, owner, other) in enumerate([(p_send, src, dst), (p_recv, dst, src)]):
        base = p[idx]
        for k in range(K):
            j = base - 1 - k
            ok = (j >= 0)
            jj = np.where(ok, j, 0)
            ok &= (aid[jj] == owner[idx]) & (eo[jj] < order[idx])
            gap = np.log1p(np.clip(tmin[idx] - E_t[jj], 0, None)).astype(np.float32)
            v = np.zeros((B, EV_DIM), np.float32)
            v[:, 0] = E_l[jj] / 10.0; v[:, 1] = role[jj]; v[:, 2] = gap / 10.0
            v[:, 3] = E_x[jj]; v[:, 4] = (cp[jj] == other[idx])        # same counterparty as now?
            v[np.arange(B), 5 + E_f[jj]] = 1.0
            v[~ok] = 0.0
            out[:, side, K - 1 - k] = v; mask[:, side, K - 1 - k] = ok
    return torch.from_numpy(out), torch.from_numpy(mask)

# ---------------- encoders
class CNNEnc(nn.Module):
    def __init__(s, h=32):
        super().__init__(); s.net = nn.Sequential(nn.Conv1d(EV_DIM, h, 3, padding=1), nn.ReLU(), nn.Conv1d(h, h, 3, padding=1), nn.ReLU())
    def forward(s, x, m): z = s.net(x.transpose(1, 2)).transpose(1, 2); z = z.masked_fill(~m[..., None], -1e4); return z.max(1).values.clamp(min=0)
class LSTMEnc(nn.Module):
    def __init__(s, h=32): super().__init__(); s.rnn = nn.LSTM(EV_DIM, h, batch_first=True)
    def forward(s, x, m): o, _ = s.rnn(x); return o[:, -1]           # left-padded, last step = most recent
class TransEnc(nn.Module):
    def __init__(s, h=32):
        super().__init__(); s.inp = nn.Linear(EV_DIM, h); s.pos = nn.Parameter(torch.randn(1, K, h) * 0.02)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(h, 4, 64, 0.1, batch_first=True), 2)
    def forward(s, x, m):
        m2 = m.clone(); m2[:, -1] |= ~m.any(1)                        # avoid all-masked rows
        z = s.enc(s.inp(x) + s.pos, src_key_padding_mask=~m2); z = z * m2[..., None]
        return z.sum(1) / m2.sum(1, keepdim=True)

class Model(nn.Module):
    def __init__(s, enc=None, tab=len(ALL), h=32):
        super().__init__(); s.enc = enc
        s.encs = nn.ModuleList([enc(h), enc(h)]) if enc else None
        s.head = nn.Sequential(nn.Linear(tab + (2 * h if enc else 0), 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    def forward(s, xt, xs=None, ms=None):
        parts = [xt]
        if s.encs is not None:
            parts += [s.encs[i](xs[:, i], ms[:, i]) for i in range(2)]
        return s.head(torch.cat(parts, 1)).squeeze(1)

def predict(m, idx, seq, bs=8192):
    m.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]; xt = torch.from_numpy(Xall[b])
            if seq: xs, ms = seq_batch(b); out.append(torch.sigmoid(m(xt, xs, ms)).numpy())
            else: out.append(torch.sigmoid(m(xt)).numpy())
    return np.concatenate(out)

def train_nn(name, enc, epochs=6, bs=1024):
    if name in results: print("skip", name); return
    seq = enc is not None; m = Model(enc); opt = torch.optim.AdamW(m.parameters(), 2e-3, weight_decay=1e-4)
    pw = torch.tensor(math.sqrt((y[tr] == 0).sum() / (y[tr] == 1).sum()))
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw); t = time.time(); best = (-1, None)
    va_s = np.concatenate([va[y[va] == 1], np.random.default_rng(3).choice(va[y[va] == 0], 100000, replace=False)])
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]; xt = torch.from_numpy(Xall[b]); yb = torch.from_numpy(y[b].astype(np.float32))
            logit = m(xt, *seq_batch(b)) if seq else m(xt)
            loss = lossf(logit, yb); opt.zero_grad(); loss.backward(); opt.step()
        ap = average_precision_score(y[va_s], predict(m, va_s, seq))
        print(f"  {name} ep{ep} val-sample AP {ap:.3f} ({time.time()-t:.0f}s)", flush=True)
        if ap > best[0]: best = (ap, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1]); secs = time.time() - t
    p_va = predict(m, va, seq); p_te = predict(m, te, seq)
    thr = best_f1_threshold(y[va], p_va); r = evaluate(y[te], p_te, thr)
    r["pr_auc_core"] = float(average_precision_score(y[te][core[te]], p_te[core[te]]))
    r.update(family="deep", features="all" + (" + sender/receiver sequences" if seq else ""), train_secs=round(secs, 1))
    results[name] = r; json.dump(results, open(OUT, "w"), indent=1)
    np.save(f"{WORK}/pred_{name.split()[0]}_te.npy", p_te); np.save(f"{WORK}/pred_{name.split()[0]}_va.npy", p_va)
    print(f"{name:28s} PR-AUC {r['pr_auc']:.3f} core {r['pr_auc_core']:.3f} F1 {r['f1']:.3f} R@0.2% {r['recall_at_budget']:.3f}", flush=True)

which = sys.argv[1:] or ["mlp", "cnn", "lstm", "transformer", "gan"]
if "mlp" in which: train_nn("MLP", None, epochs=8)
if "cnn" in which: train_nn("1D-CNN (sequence)", CNNEnc)
if "lstm" in which: train_nn("LSTM (sequence)", LSTMEnc)
if "transformer" in which: train_nn("Transformer (sequence)", TransEnc)

# ---------------- GAN augmentation for XGBoost
if "gan" in which and "XGBoost+graph + GAN aug" not in results:
    import xgboost as xgb
    Xp = torch.from_numpy(Xall[tr][y[tr] == 1]); D_ = Xp.shape[1]; Z = 16
    G = nn.Sequential(nn.Linear(Z, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, D_))
    Dn = nn.Sequential(nn.Linear(D_, 64), nn.LeakyReLU(0.2), nn.Linear(64, 32), nn.LeakyReLU(0.2), nn.Linear(32, 1))
    og, od = torch.optim.Adam(G.parameters(), 1e-3, betas=(0.5, 0.9)), torch.optim.Adam(Dn.parameters(), 1e-3, betas=(0.5, 0.9))
    bce = nn.BCEWithLogitsLoss(); t = time.time()
    for step in range(3000):
        real = Xp[torch.randint(0, len(Xp), (256,))]; fake = G(torch.randn(256, Z))
        ld = bce(Dn(real), torch.ones(256, 1)) + bce(Dn(fake.detach()), torch.zeros(256, 1))
        od.zero_grad(); ld.backward(); od.step()
        lg = bce(Dn(fake), torch.ones(256, 1)); og.zero_grad(); lg.backward(); og.step()
    with torch.no_grad(): synth = G(torch.randn(len(Xp) * 3, Z)).numpy()
    Xtr = np.vstack([Xall[tr], synth]); ytr = np.concatenate([y[tr], np.ones(len(synth), np.int8)])
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    m = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=2, scale_pos_weight=spw ** 0.5).fit(Xtr, ytr)
    p_va, p_te = m.predict_proba(Xall[va])[:, 1], m.predict_proba(Xall[te])[:, 1]
    r = evaluate(y[te], p_te, best_f1_threshold(y[va], p_va))
    r["pr_auc_core"] = float(average_precision_score(y[te][core[te]], p_te[core[te]]))
    r.update(family="deep (generative)", features="all + 3x GAN-synthetic positives", train_secs=round(time.time() - t, 1))
    results["XGBoost+graph + GAN aug"] = r; json.dump(results, open(OUT, "w"), indent=1)
    print(f"GAN-aug XGB PR-AUC {r['pr_auc']:.3f} core {r['pr_auc_core']:.3f} F1 {r['f1']:.3f}", flush=True)
print("done")
