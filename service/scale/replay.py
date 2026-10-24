"""Stream an IBM AMLworld style CSV (Timestamp, From Bank, Account, To Bank, Account, Amount Received, Receiving
Currency, Amount Paid, Payment Currency, Payment Format[, Is Laundering]) through the partitioned workers, in time order.
  python service/scale/replay.py --csv payments.csv.gz --partitions 2 [--limit 100000] [--rate 2000]
"""
import argparse, csv, gzip, os, sys, time, calendar
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.dirname(HERE))
from scale.producer import publish  # noqa: E402


def rows(path, limit):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:
        rd = csv.reader(fh); next(rd); out = []
        for i, r in enumerate(rd):
            if limit and i >= limit:
                break
            out.append({"eid": i, "from_bank": r[1], "from_account": r[2], "to_bank": r[3], "to_account": r[4], "amount_paid": float(r[7]),
                        "payment_currency": r[8], "receiving_currency": r[6], "payment_format": r[9],
                        "tmin": calendar.timegm(time.strptime(r[0], "%Y/%m/%d %H:%M")) // 60})
    return sorted(out, key=lambda p: p["tmin"])        # stable: same minute keeps file order


if __name__ == "__main__":
    import redis
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--partitions", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--rate", type=float, default=0, help="payments per second, 0 = as fast as possible")
    a = ap.parse_args(); r = redis.Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"), decode_responses=True)
    pays = rows(a.csv, a.limit); t0 = time.perf_counter()
    if a.rate:
        for i in range(0, len(pays), 50):
            publish(r, pays[i:i + 50], a.partitions, stop=False)
            time.sleep(max(0.0, t0 + (i + 50) / a.rate - time.perf_counter()))
        for k in range(a.partitions):
            r.xadd(f"payments:{k}", {"stop": 1})
    else:
        publish(r, pays, a.partitions)
    print(f"published {len(pays):,} payments to {a.partitions} partitions in {time.perf_counter() - t0:.1f}s")
