"""Check that the streaming engine reproduces the offline training features exactly."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import glob, sys, time, calendar
import numpy as np, polars as pl
sys.path.insert(0, os.path.join(REPO, "service"))
from rusetect_engine import Engine
from common import load, ALL
N, CHECK = 1_200_000, 50_000
cols = ["ts","fb","fa","tb","ta","amt_recv","cur_recv","amt_paid","cur_paid","fmt","y"]
raw = pl.concat([pl.read_csv(f, has_header=True, infer_schema=False, new_columns=cols) for f in
                 sorted(glob.glob(f"{DATA}/aml_HI-Small_part0*.csv.gz"))]).with_row_index("rid")
d, _ = load()
d = d.head(N)
raw = d.select("rid", "order").join(raw, on="rid").sort("order")
eng = Engine(load_models=False); t = time.time(); got = []
for i, r in enumerate(raw.iter_rows(named=True)):
    tm = calendar.timegm(time.strptime(r["ts"], "%Y/%m/%d %H:%M")) // 60
    f = eng.features(dict(from_bank=r["fb"], from_account=r["fa"], to_bank=r["tb"], to_account=r["ta"],
                          amount_paid=float(r["amt_paid"]), payment_currency=r["cur_paid"], receiving_currency=r["cur_recv"],
                          payment_format=r["fmt"], tmin=tm))
    if i >= N - CHECK: got.append([f[k] for k in ALL])
print(f"replayed {N} payments in {time.time()-t:.0f}s ({(time.time()-t)/N*1e6:.0f} us/payment)")
G = np.array(got, dtype=np.float64)
E = d.tail(CHECK).select(ALL).to_numpy().astype(np.float64)
bad = 0
for j, k in enumerate(ALL):
    a, b = G[:, j], E[:, j]
    ok = np.isclose(a, b, rtol=1e-4, atol=1e-4, equal_nan=True)
    if ok.mean() < 1: bad += 1; print(f"MISMATCH {k}: {1-ok.mean():.4%} rows, e.g. engine {a[~ok][:3]} offline {b[~ok][:3]}")
print("features matching:", len(ALL) - bad, "/", len(ALL))
