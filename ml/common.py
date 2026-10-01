"""Shared data loading, splits and metrics for the RuseTect AML benchmark."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import DATA, WORK, MODELS, EXPORT, REPO
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve

FEAT_PATH = f"{WORK}/aml_feat.parquet"

# Features available at the moment a payment is made (no absolute time, no ids)
RAW = ["log_usd", "fmt_code", "cur_code", "hour", "dow", "cross_bank", "cross_cur", "self_loop", "round_100"]
GRAPH = ["s_out_n", "s_out_mean_usd", "s_gap_out", "s_age_out", "new_pair", "s_uniq_dst",
         "d_in_n", "d_gap_in", "d_uniq_src", "s_in_n", "s_mins_since_in", "s_out_over_last_in",
         "s_passthrough", "d_out_n", "d_in_usd_log", "s_fanout_24h", "s_cnt_24h", "s_usd_24h",
         "d_fanin_24h", "d_cnt_24h", "reciprocal", "new_bank_pair"]
ALL = RAW + GRAPH
FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
CURRENCIES = ["Australian Dollar", "Bitcoin", "Brazil Real", "Canadian Dollar", "Euro", "Mexican Peso", "Ruble", "Rupee",
              "Saudi Riyal", "Shekel", "Swiss Franc", "UK Pound", "US Dollar", "Yen", "Yuan"]
FMT_CODES = {k: i for i, k in enumerate(FORMATS)}
CUR_CODES = {k: i for i, k in enumerate(CURRENCIES)}


def load(cols_extra=()):
    d = pl.read_parquet(FEAT_PATH)
    # fixed, sorted encodings (Polars categorical physical codes are not stable across processes)
    d = d.with_columns(pl.col("fmt").cast(pl.Utf8).replace_strict(FMT_CODES, return_dtype=pl.Int16).alias("fmt_code"),
                       pl.col("cur_paid").cast(pl.Utf8).replace_strict(CUR_CODES, default=-1, return_dtype=pl.Int16).alias("cur_code"),
                       pl.col("s_out_mean_usd").log1p())
    n = d.height
    a, b = int(0.6 * n), int(0.8 * n)
    split = np.zeros(n, dtype=np.int8)
    split[a:b] = 1
    split[b:] = 2
    return d, split


def to_np(d, cols):
    X = d.select(cols).to_numpy().astype(np.float32)
    X[~np.isfinite(X)] = np.nan
    return X


def train_sample(y, split, neg_frac=0.10, seed=0):
    """All training positives + fixed fraction of training negatives (same for every model)."""
    rng = np.random.default_rng(seed)
    tr = np.where(split == 0)[0]
    pos = tr[y[tr] == 1]
    neg = tr[y[tr] == 0]
    neg = rng.choice(neg, int(len(neg) * neg_frac), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    return idx


def best_f1_threshold(y, p):
    pr, rc, th = precision_recall_curve(y, p)
    f1 = 2 * pr * rc / np.clip(pr + rc, 1e-12, None)
    i = np.nanargmax(f1[:-1])
    return th[i]


def evaluate(y, p, thr, budget_frac=0.002):
    """PR-AUC, ROC-AUC, F1 at val-chosen threshold, and recall/precision when analysts
    can only review the top `budget_frac` of traffic (alert budget)."""
    out = {}
    out["pr_auc"] = float(average_precision_score(y, p))
    out["roc_auc"] = float(roc_auc_score(y, p))
    pred = p >= thr
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum()); fn = int((~pred & (y == 1)).sum())
    out["precision"] = tp / max(tp + fp, 1)
    out["recall"] = tp / max(tp + fn, 1)
    out["f1"] = 2 * out["precision"] * out["recall"] / max(out["precision"] + out["recall"], 1e-12)
    out["alerts"] = tp + fp
    out["alerts_per_tp"] = (tp + fp) / max(tp, 1)
    k = max(1, int(len(y) * budget_frac))
    top = np.argsort(-p)[:k]
    out["recall_at_budget"] = float(y[top].sum() / max(y.sum(), 1))
    out["precision_at_budget"] = float(y[top].mean())
    return out
