"""Attach the IBM generator's laundering typology (fan-in, cycle, ...) to each laundering payment.
Matches HI-Small_Patterns.txt rows to raw transaction rows on (minute, banks, accounts, amount paid) and
writes $RUSETECT_WORK/typology.parquet with columns rid (raw row index == `rid` in aml_feat.parquet), typ, ring."""
import os, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK
import polars as pl
cols = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
raw = pl.concat([pl.read_csv(f, has_header=True, infer_schema=False, new_columns=cols) for f in sorted(glob.glob(f"{DATA}/aml_HI-Small_part0*.csv.gz"))])
raw = raw.with_row_index("rid").filter(pl.col("y") == "1")
pat = pl.read_parquet(f"{WORK}/patterns.parquet")
key = lambda t, fb, fa, tb, ta, amt: pl.concat_str([t, fb, fa, tb, ta, amt], separator="|")
raw = raw.with_columns(key(pl.col("ts"), pl.col("from_bank"), pl.col("from_acct"), pl.col("to_bank"), pl.col("to_acct"), pl.col("amt_paid").cast(pl.Float64).round(2).cast(pl.Utf8)).alias("k"))
pat = pat.with_columns(key(pl.col("ts"), pl.col("fb"), pl.col("fa"), pl.col("tb"), pl.col("ta"), pl.col("amt").round(2).cast(pl.Utf8)).alias("k")).select("k", "typ", "ring").unique("k")
m = raw.join(pat, on="k", how="left").select(pl.col("rid").cast(pl.UInt32), "typ", "ring")
print("laundering rows", m.height, "with typology", m["typ"].is_not_null().sum())
m.write_parquet(f"{WORK}/typology.parquet")
print(m.group_by("typ").agg(pl.len()).sort("len"))
