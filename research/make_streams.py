"""Build counterfactual payment streams from the labelled demo slice and score each through the served Rust engine.

Every stream keeps the AMLworld column layout, so research_dump (crates/rusetect/examples/research_dump.rs) scores it
with the exact live feature code and models. Outputs go to research/out/streams/.

Streams
  base                 the slice as bundled (160,000 payments, rows 100,000+ are the evaluation window)
  warm_{50000,75000,90000}  the first N rows dropped: the same evaluation payments seen with less account history
  split{2,4}           structuring attack: every laundering payment in the evaluation window replaced by k payments of
                       amount/k at the same minute (attacker controls only their own payments)
  split4_spread        as split4 but the parts are spaced 30 minutes apart, then the stream is re-sorted by time

Usage: python3 -I research/make_streams.py   (from the repository root, after cargo build --release --example research_dump)
"""
import csv, gzip, os, subprocess, sys
from datetime import datetime, timedelta

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "service", "models", "demo_stream.csv.gz")
OUT = os.path.join(ROOT, "research", "out", "streams")
BIN = os.path.join(ROOT, "target", "release", "examples", "research_dump")
EVAL_FROM = 100_000


def read():
    with gzip.open(SRC, "rt", newline="") as f:
        r = csv.DictReader(f)
        return r.fieldnames, list(r)


def write(name, header, rows, origin):
    """origin[i] = index of the bundled row a written row came from (for mapping parts back to the original payment)."""
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name + ".csv.gz")
    with gzip.open(p, "wt", newline="") as f:
        w = csv.DictWriter(f, header)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, name + ".origin"), "w") as f:
        f.write("\n".join(map(str, origin)))
    subprocess.run([BIN, p, os.path.join(OUT, name + "_dump.csv")], check=True)


def fmt_amt(x):
    return f"{x:.2f}"


def main():
    header, rows = read()
    idx = list(range(len(rows)))
    write("base", header, rows, idx)
    for w in (50_000, 75_000, 90_000):
        write(f"warm_{w}", header, rows[w:], idx[w:])
    for k in (2, 4):
        out, org = [], []
        for i, r in enumerate(rows):
            if i >= EVAL_FROM and r["y"] == "1":
                for _ in range(k):
                    s = dict(r)
                    s["amt_paid"] = fmt_amt(float(r["amt_paid"]) / k)
                    s["amt_recv"] = fmt_amt(float(r["amt_recv"]) / k)
                    out.append(s)
                    org.append(i)
            else:
                out.append(r)
                org.append(i)
        write(f"split{k}", header, out, org)
    # spread parts over two hours: needs a re-sort, stable so legitimate order is kept
    out = []
    for i, r in enumerate(rows):
        t = datetime.strptime(r["ts"], "%Y/%m/%d %H:%M")
        if i >= EVAL_FROM and r["y"] == "1":
            for j in range(4):
                s = dict(r)
                s["amt_paid"] = fmt_amt(float(r["amt_paid"]) / 4)
                s["amt_recv"] = fmt_amt(float(r["amt_recv"]) / 4)
                s["ts"] = (t + timedelta(minutes=30 * j)).strftime("%Y/%m/%d %H:%M")
                out.append((s["ts"], i, j, s))
        else:
            out.append((r["ts"], i, 0, r))
    out.sort(key=lambda z: (z[0], z[1], z[2]))
    write("split4_spread", header, [z[3] for z in out], [z[1] for z in out])


if __name__ == "__main__":
    sys.exit(main())
