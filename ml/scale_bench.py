"""Memory and speed of the account state at full HI-Small scale (5,078,345 payments, 515k accounts).

Runs each state layout in its own process over the whole time ordered file and reports peak memory, state memory
(peak minus memory right after loading the data), throughput, and, for the Bloom index, how often features and live
decisions differ from the exact layout on the test 20%.
  python ml/scale_bench.py            all three layouts
  python ml/scale_bench.py one <kind> internal: run one layout (engine | exact | bloom)
Writes results/scale_bench.json.
"""
import glob, json, os, subprocess, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("RUSETECT_DATA", "/mnt/user-data/uploads/iit delhi")
TMP = os.environ.get("RUSETECT_DUMP", "/tmp")


def rss_mb(field="VmRSS"):
    for line in open("/proc/self/status"):
        if line.startswith(field):
            return int(line.split()[1]) / 1024


def one(kind):
    import pandas as pd
    sys.path.insert(0, os.path.join(ROOT, "service"))
    from rusetect_engine import Engine
    from compact_state import CompactState
    parts = sorted(glob.glob(os.path.join(DATA, "for_claude", "aml_HI-Small_part*.csv.gz")))
    cols = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
    df = pd.concat([pd.read_csv(p, header=0, names=cols, dtype=str, usecols=[0, 1, 2, 3, 4, 6, 7, 8, 9]) for p in parts], ignore_index=True)
    df["tmin"] = (pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M").astype("int64") // 60_000_000_000)
    df = df.sort_values("tmin", kind="mergesort").reset_index(drop=True)
    cat = {c: df[c].astype("category") for c in ("fb", "fa", "tb", "ta", "cur_recv", "cur_paid", "fmt")}
    C = {c: (v.cat.categories.to_numpy(), v.cat.codes.to_numpy()) for c, v in cat.items()}
    amt = df["amt_paid"].astype(float).to_numpy(); tmin = df["tmin"].to_numpy(); N = len(df)
    del df, cat
    import gc; gc.collect()
    base = rss_mb()
    eng = Engine(load_models=False)
    st = eng if kind == "engine" else CompactState(eng.cfg, pairs=kind, capacity=20_000_000, fp_rate=0.001)
    st.reset()
    feats = eng.feats; test_from = int(N * 0.8); X = np.empty((N - test_from, len(feats)), dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(N):
        p = {"from_bank": C["fb"][0][C["fb"][1][i]], "from_account": C["fa"][0][C["fa"][1][i]], "to_bank": C["tb"][0][C["tb"][1][i]],
             "to_account": C["ta"][0][C["ta"][1][i]], "amount_paid": amt[i], "payment_currency": C["cur_paid"][0][C["cur_paid"][1][i]],
             "receiving_currency": C["cur_recv"][0][C["cur_recv"][1][i]], "payment_format": C["fmt"][0][C["fmt"][1][i]], "tmin": int(tmin[i])}
        f = st.features(p)
        if i >= test_from:
            X[i - test_from] = [f[k] for k in feats]
    secs = time.perf_counter() - t0
    np.save(os.path.join(TMP, f"scale_X_{kind}.npy"), X)
    out = {"kind": kind, "payments": N, "seconds": round(secs, 1), "payments_per_second": round(N / secs),
           "peak_mb": round(rss_mb("VmHWM")), "rss_after_load_mb": round(base), "rss_end_mb": round(rss_mb()),
           "state_mb": round(rss_mb() - base)}
    if kind != "engine":
        out["state"] = st.stats()
    print(json.dumps(out), flush=True)


def main():
    res = {}
    for kind in ("engine", "exact", "bloom"):
        r = subprocess.run([sys.executable, __file__, "one", kind], capture_output=True, text=True)
        line = [l for l in r.stdout.splitlines() if l.startswith("{")]
        res[kind] = json.loads(line[-1]) if line else {"error": r.stderr[-800:]}
        print(kind, res[kind], flush=True)
    # feature and decision agreement on the test 20%
    sys.path.insert(0, os.path.join(ROOT, "service")); sys.path.insert(0, os.path.join(ROOT, "ml"))
    import xgboost as xgb, lite_fast
    from rusetect_engine import Engine
    eng = Engine(load_models=True); cfg = eng.cfg
    def decide(X):
        dm = xgb.DMatrix(X, feature_names=eng.feats); s1 = eng.l1.predict(dm)
        risk = (eng.l2x.predict(dm) + lite_fast.predict(eng.l2l, X)) / 2; esc = s1 >= cfg["l1_threshold"]
        r = np.zeros(len(X), dtype=np.int8)
        for k, th in enumerate(("verify_threshold", "hold_threshold", "block_threshold"), 1):
            r[esc & (risk >= cfg[th])] = k
        return r
    try:
        Xs = {k: np.load(os.path.join(TMP, f"scale_X_{k}.npy")) for k in res}
        same = lambda a, b: np.isclose(a, b, equal_nan=True)
        res["agreement_exact_vs_engine"] = {"features_identical": bool(same(Xs["exact"], Xs["engine"]).all())}
        de, db = decide(Xs["exact"]), decide(Xs["bloom"])
        diff_rows = ~same(Xs["exact"], Xs["bloom"]).all(axis=1)
        res["agreement_bloom_vs_exact"] = {"rows_with_any_feature_difference": round(float(diff_rows.mean()), 6),
                                           "decision_agreement": round(float((de == db).mean()), 6),
                                           "flagged_exact": int((de >= 1).sum()), "flagged_bloom": int((db >= 1).sum())}
    except Exception as e:  # noqa: BLE001
        res["agreement_error"] = repr(e)
    res["note"] = ("Single Python process on a 2 core, 8 GB container. state_mb = resident memory after streaming minus resident "
                   "memory right after loading the payments, so it isolates the account state.")
    json.dump(res, open(os.path.join(ROOT, "results", "scale_bench.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    one(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[1] == "one" else main()
