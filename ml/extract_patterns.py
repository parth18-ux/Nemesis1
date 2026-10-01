"""Parse IBM AMLworld HI-Small_Patterns.txt into one row per laundering payment with its ring id and typology."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK
import polars as pl
src = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/HI-Small_Patterns.txt"
rows, ring, typ = [], -1, None
for line in open(src):
    line = line.strip()
    if line.startswith("BEGIN"):
        ring += 1; typ = line.split("-", 1)[1].split(":")[0].strip()
    elif line and not line.startswith("END"):
        p = line.split(",")
        rows.append((ring, typ, p[0], p[1], p[2], p[3], p[4], float(p[5])))
pat = pl.DataFrame(rows, schema=["ring", "typ", "ts", "fb", "fa", "tb", "ta", "amt"], orient="row")
pat.write_parquet(f"{WORK}/patterns.parquet")
print(ring + 1, "rings,", pat.height, "payments ->", f"{WORK}/patterns.parquet")
