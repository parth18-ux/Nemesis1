"""
RuseTect: directed edge-level GNN (GraphSAGE-style, separate in/out aggregation) for mule-ring payments.

Strictly causal protocol: node embeddings come only from the graph of PAST payments.
  train: graph = first 40% of payments  -> classify payments in 40-60%
  val:   graph = first 60%              -> classify payments in 60-80%
  test:  graph = first 80%              -> classify payments in 80-100%
Edge classifier input = [h_sender, h_receiver, payment features].
Two variants: GNN with raw payment fields only, and GNN + the causal graph features.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import json, sys, time, math
import numpy as np, torch, torch.nn as nn
from common import load, to_np, best_f1_threshold, evaluate, ALL, RAW
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
torch.set_num_threads(2); torch.manual_seed(0)
OUT = f"{WORK}/results_gnn.json"
try: results = json.load(open(OUT))
except Exception: results = {}

d, split = load()
y = d["y"].to_numpy().astype(np.int8); n = d.height
ts = d["ts"].to_numpy(); core = (split == 2) & (ts < np.datetime64("2022-09-11"))
src = d["src_id"].to_numpy().astype(np.int64); dst = d["dst_id"].to_numpy().astype(np.int64)
N = int(max(src.max(), dst.max())) + 1
lusd = d["log_usd"].to_numpy().astype(np.float32); fmt = d["fmt_code"].to_numpy().astype(np.int64); nf = int(fmt.max()) + 1
xb = d["cross_bank"].to_numpy().astype(np.float32)
c40, c60, c80 = int(.4 * n), int(.6 * n), int(.8 * n)

def slog(X): X = np.nan_to_num(X, nan=-1.0); return np.sign(X) * np.log1p(np.abs(X))

def node_feats(cut):
    """Per-account statistics from payments [0, cut)."""
    s, t = src[:cut], dst[:cut]
    f = np.zeros((N, 6 + 2 * nf), np.float32)
    outd = np.bincount(s, minlength=N); ind = np.bincount(t, minlength=N)
    f[:, 0] = np.log1p(outd); f[:, 1] = np.log1p(ind)
    f[:, 2] = np.bincount(s, lusd[:cut], N) / np.maximum(outd, 1)
    f[:, 3] = np.bincount(t, lusd[:cut], N) / np.maximum(ind, 1)
    f[:, 4] = (outd > 0) & (ind > 0)                            # both receives and sends
    f[:, 5] = np.bincount(s, xb[:cut], N) / np.maximum(outd, 1)
    for k in range(nf):
        m = fmt[:cut] == k
        f[:, 6 + k] = np.bincount(s[m], minlength=N) / np.maximum(outd, 1)
        f[:, 6 + nf + k] = np.bincount(t[m], minlength=N) / np.maximum(ind, 1)
    return torch.from_numpy(f)

class SAGE(nn.Module):
    def __init__(s, fin, h):
        super().__init__(); s.self_ = nn.Linear(fin, h); s.inn = nn.Linear(fin, h); s.out = nn.Linear(fin, h)
    def forward(s, x, es, ed, deg_in, deg_out):
        agg_in = torch.zeros_like(x).index_add_(0, ed, x[es]) / deg_in      # mean over senders into v
        agg_out = torch.zeros_like(x).index_add_(0, es, x[ed]) / deg_out    # mean over receivers from v
        return torch.relu(s.self_(x) + s.inn(agg_in) + s.out(agg_out))

class GNN(nn.Module):
    def __init__(s, fin, fe, h=32):
        super().__init__(); s.l1 = SAGE(fin, h); s.l2 = SAGE(h, h)
        s.head = nn.Sequential(nn.Linear(2 * h + fe, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))
    def embed(s, g):
        x, es, ed, di, do = g; h = s.l1(x, es, ed, di, do); return s.l2(h, es, ed, di, do)
    def forward(s, H, idx, E):
        return s.head(torch.cat([H[src_t[idx]], H[dst_t[idx]], E[idx]], 1)).squeeze(1)

src_t, dst_t = torch.from_numpy(src), torch.from_numpy(dst)
def graph(cut):
    # de-duplicate parallel edges to save memory; keep one edge per (src,dst) pair
    key = np.unique(src[:cut] * N + dst[:cut]); es = torch.from_numpy(key // N); ed = torch.from_numpy(key % N)
    di = torch.bincount(ed, minlength=N).clamp(min=1).float()[:, None]; do = torch.bincount(es, minlength=N).clamp(min=1).float()[:, None]
    x = node_feats(cut); x = (x - x.mean(0)) / (x.std(0) + 1e-6)
    return (x, es, ed, di, do)

def run(name, cols):
    if name in results: print("skip", name); return
    E = slog(to_np(d, cols)); E = StandardScaler().fit(E[c40:c60]).transform(E).astype(np.float32); E = torch.from_numpy(E)
    g_tr, g_va, g_te = graph(c40), graph(c60), graph(c80)
    tr = np.arange(c40, c60); va = np.arange(c60, c80); te = np.arange(c80, n)
    rng = np.random.default_rng(0)
    m = GNN(g_tr[0].shape[1], E.shape[1]); opt = torch.optim.AdamW(m.parameters(), 3e-3, weight_decay=1e-4)
    pos = tr[y[tr] == 1]; negs = tr[y[tr] == 0]
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0)); t = time.time(); best = (-1, None)
    va_s = np.concatenate([va[y[va] == 1], rng.choice(va[y[va] == 0], 100000, replace=False)])
    for step in range(301):
        m.train(); b = np.concatenate([pos, rng.choice(negs, 30000, replace=False)])
        H = m.embed(g_tr); loss = lossf(m(H, torch.from_numpy(b), E), torch.from_numpy(y[b].astype(np.float32)))
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 25 == 0:
            m.eval()
            with torch.no_grad(): p = torch.sigmoid(m(m.embed(g_va), torch.from_numpy(va_s), E)).numpy()
            ap = average_precision_score(y[va_s], p); print(f"  {name} step {step} loss {loss.item():.3f} val-AP {ap:.3f} ({time.time()-t:.0f}s)", flush=True)
            if ap > best[0]: best = (ap, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[1]); m.eval()
    with torch.no_grad():
        p_va = torch.sigmoid(m(m.embed(g_va), torch.from_numpy(va), E)).numpy()
        p_te = torch.sigmoid(m(m.embed(g_te), torch.from_numpy(te), E)).numpy()
    r = evaluate(y[te], p_te, best_f1_threshold(y[va], p_va))
    r["pr_auc_core"] = float(average_precision_score(y[te][core[te]], p_te[core[te]]))
    r.update(family="graph neural network", features=name, train_secs=round(time.time() - t, 1),
             note="causal: embeddings from past graph only; trained on 40-60% slice")
    results[name] = r; json.dump(results, open(OUT, "w"), indent=1)
    np.save(f"{WORK}/pred_GNN{'_all' if cols is ALL else '_raw'}_te.npy", p_te); np.save(f"{WORK}/pred_GNN{'_all' if cols is ALL else '_raw'}_va.npy", p_va)
    print(f"{name:28s} PR-AUC {r['pr_auc']:.3f} core {r['pr_auc_core']:.3f} F1 {r['f1']:.3f} R@0.2% {r['recall_at_budget']:.3f}", flush=True)
    if cols is ALL: torch.save(m.state_dict(), f"{WORK}/gnn_all.pt")

run("GNN (raw payment fields)", RAW)
run("GNN + graph features", ALL)
print("done")
