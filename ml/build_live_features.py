"""Features for training, computed by the SAME code that serves the API, for every payment of an IBM AMLworld file.

  python ml/build_live_features.py HI      ->  $RUSETECT_WORK/live_feat_HI.npz
  python ml/build_live_features.py LI      ->  $RUSETECT_WORK/live_feat_LI.npz

Each payment is streamed in time order (stable within a minute, i.e. file order) through service/compact_state.py,
which is tested to give exactly the live engine's features (tests/test_compact_state.py). Using the serving code for
training removes any training/serving skew: currency rates, format and currency codes come from the live
cascade_config.json, so a payment gets the same numbers offline and online.
Saved: X (float32, n x 31 in cascade_config feature order), y (int8), tmin (int64 minutes), split (0 train 60%,
1 validation 20%, 2 test 20%, by time).
"""
import glob, os, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "ml"))
from paths import WORK  # noqa: E402
from rusetect_engine import Engine  # noqa: E402
from compact_state import CompactState  # noqa: E402

DATA = os.environ.get("RUSETECT_DATA") or os.path.dirname(next(iter(sorted(glob.glob("/mnt/user-data/uploads/*/for_claude"))), "data/for_claude"))
SET = (sys.argv[1] if len(sys.argv) > 1 else "HI").upper()
T0 = time.time()
parts = sorted(glob.glob(os.path.join(DATA, "for_claude", f"aml_{SET}-Small_part*.csv.gz")))
cols = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
df = pd.concat([pd.read_csv(p, header=0, names=cols, dtype=str, usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9, 10]) for p in parts], ignore_index=True)
df["tmin"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M").astype("datetime64[s]").astype("int64") // 60   # explicit unit
df = df.sort_values("tmin", kind="mergesort").reset_index(drop=True)
N = len(df)
C = {c: (df[c].astype("category").cat.categories.to_numpy(), df[c].astype("category").cat.codes.to_numpy())
     for c in ("fb", "fa", "tb", "ta", "cur_recv", "cur_paid", "fmt")}
amt = df["amt_paid"].astype(float).to_numpy(); tmin = df["tmin"].to_numpy(); y = df["y"].astype(np.int8).to_numpy()
del df
print(f"[{time.time() - T0:5.0f}s] {SET}: {N:,} payments, {int(y.sum()):,} laundering", flush=True)
eng = Engine(load_models=False); st = CompactState(eng.cfg, pairs="exact"); feats = eng.feats
X = np.empty((N, len(feats)), dtype=np.float32)
for i in range(N):
    f = st.features({"from_bank": C["fb"][0][C["fb"][1][i]], "from_account": C["fa"][0][C["fa"][1][i]], "to_bank": C["tb"][0][C["tb"][1][i]],
                     "to_account": C["ta"][0][C["ta"][1][i]], "amount_paid": amt[i], "payment_currency": C["cur_paid"][0][C["cur_paid"][1][i]],
                     "receiving_currency": C["cur_recv"][0][C["cur_recv"][1][i]], "payment_format": C["fmt"][0][C["fmt"][1][i]], "tmin": int(tmin[i])})
    X[i] = [f[k] for k in feats]
    if i % 1_000_000 == 0:
        print(f"[{time.time() - T0:5.0f}s]   {i:,}", flush=True)
X[~np.isfinite(X)] = np.nan
split = np.zeros(N, dtype=np.int8); split[int(0.6 * N):int(0.8 * N)] = 1; split[int(0.8 * N):] = 2
np.savez(os.path.join(WORK, f"live_feat_{SET}.npz"), X=X, y=y, tmin=tmin, split=split, features=np.array(feats))
print(f"[{time.time() - T0:5.0f}s] saved {os.path.join(WORK, f'live_feat_{SET}.npz')}", flush=True)
