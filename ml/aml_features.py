"""
RuseTect: causal graph feature engineering for IBM AMLworld HI-Small.

Every feature for transaction t uses only transactions strictly before t
(or at the same minute but earlier in file order), so a model trained on
early months and tested on later ones never sees the future.

Output: $RUSETECT_WORK/aml_feat.parquet
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import glob, sys, time
import polars as pl
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else f"{DATA}/aml_HI-Small_part0*.csv.gz"
OUT = sys.argv[2] if len(sys.argv) > 2 else f"{WORK}/aml_feat.parquet"
t0 = time.time()

cols = ["ts", "from_bank", "from_acct", "to_bank", "to_acct", "amt_recv", "cur_recv",
        "amt_paid", "cur_paid", "fmt", "y"]
frames = []
for f in sorted(glob.glob(SRC)):
    d = pl.read_csv(f, has_header=True, infer_schema=False, new_columns=cols)
    d = d.with_columns(pl.col("amt_recv", "amt_paid").cast(pl.Float64), pl.col("y").cast(pl.Int8))
    frames.append(d)
df = pl.concat(frames)
n_rows = df.height
print("rows", df.height, "laundering", df["y"].sum(), f"{time.time()-t0:.0f}s")

df = (df.with_row_index("rid")
        .with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M"))
        .with_columns((pl.col("from_bank") + "_" + pl.col("from_acct")).alias("src"),
                      (pl.col("to_bank") + "_" + pl.col("to_acct")).alias("dst"))
        .sort(["ts", "rid"]))

# ---- currency -> USD, rate estimated from cross-currency txns with a USD leg
usd = df.filter((pl.col("cur_recv") == "US Dollar") & (pl.col("cur_paid") != "US Dollar"))
rates = (usd.group_by("cur_paid")
            .agg((pl.col("amt_recv") / pl.col("amt_paid")).median().alias("rate"))
            .rename({"cur_paid": "cur"}))
rates = pl.concat([rates, pl.DataFrame({"cur": ["US Dollar"], "rate": [1.0]})])
df = (df.join(rates.rename({"cur": "cur_paid", "rate": "r"}), on="cur_paid", how="left")
        .with_columns((pl.col("amt_paid") * pl.col("r").fill_null(1.0)).alias("usd"))
        .drop("r"))

# ---- integer ids for accounts (used later by the GNN)
accts = pl.concat([df["src"], df["dst"]]).unique().sort()
amap = pl.DataFrame({"acct": accts, "aid": pl.arange(0, accts.len(), eager=True)})
df = (df.join(amap.rename({"acct": "src", "aid": "src_id"}), on="src", how="left")
        .join(amap.rename({"acct": "dst", "aid": "dst_id"}), on="dst", how="left")
        .sort(["ts", "rid"]))
df = df.with_columns(pl.int_range(0, pl.len()).alias("order"),
                     (pl.col("ts").dt.epoch("s") // 60).cast(pl.Int32).alias("tmin"))
print("accounts", accts.len(), f"{time.time()-t0:.0f}s")

# ---- basic txn features, then drop all strings to fit in memory
df = df.with_columns(
    pl.col("usd").log1p().cast(pl.Float32).alias("log_usd"),
    (pl.col("src") == pl.col("dst")).cast(pl.Int8).alias("self_loop"),
    (pl.col("from_bank") != pl.col("to_bank")).cast(pl.Int8).alias("cross_bank"),
    (pl.col("cur_recv") != pl.col("cur_paid")).cast(pl.Int8).alias("cross_cur"),
    pl.col("ts").dt.hour().cast(pl.Int8).alias("hour"),
    pl.col("ts").dt.weekday().cast(pl.Int8).alias("dow"),
    ((pl.col("usd") % 100) == 0).cast(pl.Int8).alias("round_100"),
    pl.col("from_bank").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("from_bank"),
    pl.col("to_bank").cast(pl.Categorical).to_physical().cast(pl.Int32).alias("to_bank"),
    pl.col("fmt").cast(pl.Categorical).alias("fmt"),
    pl.col("cur_paid").cast(pl.Categorical).alias("cur_paid"),
    pl.col("src_id").cast(pl.Int32), pl.col("dst_id").cast(pl.Int32),
    pl.col("usd").cast(pl.Float32),
).select("order", "rid", "ts", "tmin", "src_id", "dst_id", "from_bank", "to_bank", "usd", "log_usd",
         "fmt", "cur_paid", "self_loop", "cross_bank", "cross_cur", "hour", "dow", "round_100", "y")
del frames, usd, accts, amap
import gc; gc.collect()
print("slim", df.estimated_size("mb"), "MB")

# ---- sender history (strictly past, via cumulative ops over the sender's own sequence)
s = df.sort(["src_id", "order"])
s = s.with_columns(pl.int_range(0, pl.len()).over("src_id").cast(pl.Int32).alias("s_out_n"))
s = s.with_columns(pl.col("usd").cum_sum().over("src_id").alias("_cs"))
s = s.with_columns(pl.col("tmin").diff().over("src_id").alias("s_gap_out"))
s = s.with_columns(pl.col("tmin").first().over("src_id").alias("_s_first"))
s = s.with_columns(((pl.col("_cs") - pl.col("usd")) / pl.col("s_out_n").clip(1)).alias("s_out_mean_usd"),
                   (pl.col("tmin") - pl.col("_s_first")).alias("s_age_out")).drop("_cs", "_s_first")
# first time this src->dst pair appears = new counterparty; cumulative distinct recipients
s = s.with_columns((pl.int_range(0, pl.len()).over(["src_id", "dst_id"]) == 0).cast(pl.Int32).alias("new_pair"))
s = s.with_columns((pl.col("new_pair").cum_sum().over("src_id") - pl.col("new_pair")).alias("s_uniq_dst"))
df = s.sort("order"); del s; gc.collect(); print("sender done", f"{time.time()-t0:.0f}s")

# ---- receiver history
r = df.sort(["dst_id", "order"])
r = r.with_columns(pl.int_range(0, pl.len()).over("dst_id").cast(pl.Int32).alias("d_in_n"))
r = r.with_columns(pl.col("tmin").diff().over("dst_id").alias("d_gap_in"))
r = r.with_columns((pl.int_range(0, pl.len()).over(["dst_id", "src_id"]) == 0).cast(pl.Int32).alias("_np"))
r = r.with_columns((pl.col("_np").cum_sum().over("dst_id") - pl.col("_np")).alias("d_uniq_src")).drop("_np")
df = r.sort("order"); del r; gc.collect(); print("receiver done", f"{time.time()-t0:.0f}s")

# ---- cross-role: how many times has the SENDER received, and the RECEIVER sent, before now
# (pass-through behaviour: money in -> money out = classic mule signature)
ev = pl.concat([
    df.select(pl.col("src_id").alias("aid"), "order", pl.lit(0).alias("role"), "tmin", "usd"),   # out
    df.select(pl.col("dst_id").alias("aid"), "order", pl.lit(1).alias("role"), "tmin", "usd"),   # in
]).sort(["aid", "order", "role"])
ev = ev.with_columns((pl.col("role") == 1).cast(pl.Int32).cum_sum().over("aid").alias("in_cum"))
ev = ev.with_columns((pl.col("role") == 0).cast(pl.Int32).cum_sum().over("aid").alias("out_cum"))
ev = ev.with_columns(pl.when(pl.col("role") == 1).then(pl.col("tmin")).otherwise(None).forward_fill().over("aid").alias("last_in_t"))
ev = ev.with_columns(pl.when(pl.col("role") == 1).then(pl.col("usd")).otherwise(None).forward_fill().over("aid").alias("last_in_usd"))
ev = ev.with_columns(pl.when(pl.col("role") == 1).then(pl.col("usd")).otherwise(0).cum_sum().over("aid").alias("in_usd_cum"))
ev = ev.with_columns(pl.when(pl.col("role") == 0).then(pl.col("usd")).otherwise(0).cum_sum().over("aid").alias("out_usd_cum"))
# sender perspective (role 0 rows): what it had received before this send
so = ev.filter(pl.col("role") == 0).select(
    "order",
    (pl.col("in_cum")).alias("s_in_n"),
    (pl.col("tmin") - pl.col("last_in_t")).alias("s_mins_since_in"),
    (pl.col("usd") / pl.col("last_in_usd")).alias("s_out_over_last_in"),
    ((pl.col("out_usd_cum") - pl.col("usd")) / pl.col("in_usd_cum").clip(1)).alias("s_passthrough"),
)
# receiver perspective (role 1 rows): how much it had sent before this receipt
ro = ev.filter(pl.col("role") == 1).select(
    "order",
    pl.col("out_cum").alias("d_out_n"),
    ((pl.col("in_usd_cum") - pl.col("usd")).log1p()).alias("d_in_usd_log"),
)
df = df.join(so, on="order", how="left").join(ro, on="order", how="left")
del ev, so, ro; gc.collect(); print("passthrough done", f"{time.time()-t0:.0f}s")

# ---- 24h window fan-out / fan-in (distinct counterparties in last day, incl. current)
# unique, order-preserving timestamp: ts + order microseconds (< 6 s, well under the 1-minute resolution)
w = (df.select("order", "ts", "src_id", "dst_id", "usd")
       .with_columns((pl.col("ts") + pl.duration(microseconds=pl.col("order"))).alias("ts"))
       .sort("ts"))
fo = (w.rolling(index_column="ts", period="1d", group_by="src_id")
        .agg(pl.col("order").last(), pl.col("dst_id").n_unique().alias("s_fanout_24h"),
             pl.len().alias("s_cnt_24h"), pl.col("usd").sum().log1p().alias("s_usd_24h")))
df = df.join(fo.select("order", "s_fanout_24h", "s_cnt_24h", "s_usd_24h"), on="order", how="left")
fi = (w.rolling(index_column="ts", period="1d", group_by="dst_id")
        .agg(pl.col("order").last(), pl.col("src_id").n_unique().alias("d_fanin_24h"),
             pl.len().alias("d_cnt_24h")))
df = df.join(fi.select("order", "d_fanin_24h", "d_cnt_24h"), on="order", how="left")
del w, fo, fi
print("windows done", f"{time.time()-t0:.0f}s")

# ---- reciprocity / 2-cycle: has dst ever paid src before this moment?
first_pair = df.group_by(["src_id", "dst_id"]).agg(pl.col("order").min().alias("fp"))
rev = first_pair.rename({"src_id": "dst_id", "dst_id": "src_id", "fp": "rev_first"})
df = df.join(rev, on=["src_id", "dst_id"], how="left").with_columns(
    (pl.col("rev_first").is_not_null() & (pl.col("rev_first") < pl.col("order"))).cast(pl.Int8).alias("reciprocal")
).drop("rev_first")

# ---- account degree totals "so far" at the bank level (bank-pair novelty)
df = df.sort("order").with_columns(
    (pl.int_range(0, pl.len()).over(["from_bank", "to_bank"]) == 0).cast(pl.Int8).alias("new_bank_pair"))

df = df.sort("order")
assert df.height == n_rows, (df.height, n_rows)
df.write_parquet(OUT)
print("saved", OUT, df.shape, f"{time.time()-t0:.0f}s")
print(df.select(pl.col("y").mean()).item())
