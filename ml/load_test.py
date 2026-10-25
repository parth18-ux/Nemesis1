"""Load test of POST /api/v2/assess on a running server (default http://localhost:8003).
Sends N payments (half with a consented message) from W threads; reports throughput and latency percentiles
measured at the client. Single uvicorn worker on this machine's CPU; not a bank benchmark.
Usage: python ml/load_test.py [base_url] [n] [workers]. Writes results/load_test.json."""
import json, os, statistics, sys, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8003"; N = int(sys.argv[2]) if len(sys.argv) > 2 else 2000; W = int(sys.argv[3]) if len(sys.argv) > 3 else 8
MSG = "The finance office has revised the vendor details. Please remit the balance to the replacement beneficiary ahead of the afternoon cutoff."
def post(path, body):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())
urllib.request.urlopen(urllib.request.Request(base + "/api/v2/demo/reset?warm=false", method="POST")).read()
lat, lock, errors = [], threading.Lock(), [0]
def one(i):
    body = {"timestamp": f"2022-09-10 {10 + (i // 3600) % 10:02d}:{(i // 60) % 60:02d}", "from_bank": f"{i % 50:03d}", "from_account": f"P{i % 400}",
            "to_bank": f"{(i * 7) % 50:03d}", "to_account": f"R{(i * 13) % 300}", "amount_paid": 500 + (i * 37) % 20000, "payment_currency": "Rupee", "payment_format": "ACH"}
    if i % 2 == 0: body.update(message=MSG, consent=True)
    t = time.perf_counter()
    try: post("/api/v2/assess", body)
    except Exception: errors[0] += 1; return
    with lock: lat.append((time.perf_counter() - t) * 1000)
t0 = time.perf_counter()
with ThreadPoolExecutor(W) as ex: list(ex.map(one, range(N)))
wall = time.perf_counter() - t0; lat.sort(); q = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 2)
out = {"requests": N, "workers": W, "errors": errors[0], "wall_s": round(wall, 2), "throughput_rps": round(len(lat) / wall, 1),
       "latency_ms": {"p50": q(.5), "p95": q(.95), "p99": q(.99), "max": round(lat[-1], 2), "mean": round(statistics.mean(lat), 2)},
       "target": "p95 below 100 ms", "met": q(.95) < 100, "machine": f"{os.cpu_count()} CPU, one uvicorn worker, client on the same machine"}
json.dump(out, open(os.path.join(ROOT, "results", "load_test.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
