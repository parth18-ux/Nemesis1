"""Scale out benchmark: Redis state + partitioned stateless workers on real HI-Small payments.

  1. throughput: preload P partitions with the first N time ordered payments, start P worker processes, time to drain
  2. latency: a producer publishes at a fixed rate (below capacity) and every decision records publish-to-decision time
  3. memory: Redis bytes per account after the run
Usage: python ml/scale_out_bench.py [N=200000]. Needs redis-server. Writes results/scale_out_bench.json.
On this 2 core container Redis, the producer and the workers share two cores, so the worker scaling here is a floor.
"""
import glob, json, os, socket, subprocess, sys, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
import redis  # noqa: E402
from scale.producer import publish, partition_of  # noqa: E402
DATA = os.environ.get("RUSETECT_DATA", "/mnt/user-data/uploads/iit delhi")
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000

parts = sorted(glob.glob(os.path.join(DATA, "for_claude", "aml_HI-Small_part*.csv.gz")))
cols = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
df = pd.concat([pd.read_csv(p, header=0, names=cols, dtype=str) for p in parts], ignore_index=True)
df["tmin"] = pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M").astype("datetime64[s]").astype("int64") // 60   # minutes; explicit unit (pandas 3 parses to microseconds)
df = df.sort_values("tmin", kind="mergesort").head(N + 20_000).reset_index(drop=True)
P_ALL = [{"eid": i, "from_bank": r.fb, "from_account": r.fa, "to_bank": r.tb, "to_account": r.ta, "amount_paid": float(r.amt_paid),
          "payment_currency": r.cur_paid, "receiving_currency": r.cur_recv, "payment_format": r.fmt, "tmin": int(r.tmin)}
         for i, r in enumerate(df.itertuples())]
del df

s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
srv = subprocess.Popen(["redis-server", "--port", str(port), "--save", "", "--appendonly", "no"], stdout=subprocess.DEVNULL)
url = f"redis://127.0.0.1:{port}/0"
r = redis.Redis.from_url(url, decode_responses=True)
for _ in range(100):
    try:
        r.ping(); break
    except Exception:
        time.sleep(0.05)
WORKER = os.path.join(ROOT, "service", "scale", "worker.py")


def drain(P, pays):
    r.flushdb()
    publish(r, pays, P)
    t0 = time.perf_counter()
    procs = [subprocess.Popen([sys.executable, WORKER, "--redis", url, "--partition", str(k), "--batch", "256"], stdout=subprocess.DEVNULL) for k in range(P)]
    done = set()
    while len(done) < P:
        for _, f in r.xrange("decisions", "-", "+"):
            if "done" in f:
                done.add(f["done"])
        time.sleep(0.5)
    secs = time.perf_counter() - t0
    [p.wait() for p in procs]
    n = sum(1 for _, f in r.xrange("decisions") if "d" in f)
    info = r.info("memory"); accts = sum(1 for _ in r.scan_iter("a:*", count=10_000))
    return {"workers": P, "payments": n, "seconds": round(secs, 1), "payments_per_second": round(n / secs),
            "redis_used_mb": round(info["used_memory"] / 2 ** 20, 1), "accounts": accts,
            "redis_bytes_per_account": round(info["used_memory"] / max(accts, 1)), "note": "includes model loading in each worker"}


def latency(rate, pays):
    r.flushdb()
    P = 2
    procs = [subprocess.Popen([sys.executable, WORKER, "--redis", url, "--partition", str(k), "--batch", "64"], stdout=subprocess.DEVNULL) for k in range(P)]
    time.sleep(8)                                    # let workers load models before timing
    gap = 1.0 / rate; t_next = time.perf_counter()
    for i in range(0, len(pays), 20):                # publish in small groups to hold the rate
        publish(r, pays[i:i + 20], P, stop=False)
        t_next += gap * 20
        time.sleep(max(0.0, t_next - time.perf_counter()))
    for k in range(P):
        r.xadd(f"payments:{k}", {"stop": 1})
    [p.wait(timeout=600) for p in procs]
    lat = np.array([json.loads(f["d"])["latency_ms"] for _, f in r.xrange("decisions") if "d" in f])
    return {"target_rate_per_second": rate, "payments": int(len(lat)), "p50_ms": round(float(np.percentile(lat, 50)), 1),
            "p95_ms": round(float(np.percentile(lat, 95)), 1), "p99_ms": round(float(np.percentile(lat, 99)), 1)}


out = {"payments": N, "cores": os.cpu_count(), "throughput": [drain(1, P_ALL[:N]), drain(2, P_ALL[:N])]}
best = max(t["payments_per_second"] for t in out["throughput"])
out["latency"] = latency(int(best * 0.6), P_ALL[N:N + 20_000] if len(P_ALL) > N else P_ALL[:20_000])
out["note"] = ("Redis, producer and workers share a 2 core container; production would run Redis Cluster (sharded by the {account} "
               "key tag) and many worker processes on separate machines. Latency is publish to decision, including queueing.")
json.dump(out, open(os.path.join(ROOT, "results", "scale_out_bench.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
srv.terminate()
