"""Map the transaction-shaped card datasets to the AMLworld layout read by research_dump, so the payment features and
models can be run on them (the audit's "card benchmarks" used as extra generators).

    python3 -I research/card_convert.py banksim <bs140513_032310.csv> <out.csv.gz>
    python3 -I research/card_convert.py sparkov <train.csv> <test.csv> <out.csv.gz>

BankSim (Lopez-Rojas & Axelsson 2014): customer -> payer, merchant -> payee, step -> day, currency Euro, format Credit Card.
Sparkov (simulated US cards): cc_num -> payer, merchant -> payee, minute timestamps, US Dollar, Credit Card; train and
test files concatenated in time order. Mapping choices are part of the measured generator gap.
"""
import sys
import pandas as pd


def banksim(src, dst):
    d = pd.read_csv(src, quotechar="'")
    ts = (pd.Timestamp("2022-09-01") + pd.to_timedelta(d.step, unit="D")).dt.strftime("%Y/%m/%d %H:%M")
    amt = d.amount.map(lambda x: f"{max(x, 0.01):.2f}")
    out = pd.DataFrame({"ts": ts, "fb": "card", "fa": d.customer.str.strip("'"), "tb": "merchant", "ta": d.merchant.str.strip("'"),
                        "amt_recv": amt, "cur_recv": "Euro", "amt_paid": amt, "cur_paid": "Euro", "fmt": "Credit Card", "y": d.fraud})
    out.to_csv(dst, index=False, compression="gzip"); print("banksim", len(out), int(out.y.sum()))


def sparkov(tr, te, dst):
    d = pd.concat([pd.read_csv(tr), pd.read_csv(te)], ignore_index=True)
    d["_t"] = pd.to_datetime(d.trans_date_trans_time)
    d = d.sort_values("_t", kind="stable").reset_index(drop=True)
    amt = d.amt.map(lambda x: f"{max(x, 0.01):.2f}")
    out = pd.DataFrame({"ts": d._t.dt.strftime("%Y/%m/%d %H:%M"), "fb": "card", "fa": d.cc_num.astype(str), "tb": "merchant",
                        "ta": d.merchant, "amt_recv": amt, "cur_recv": "US Dollar", "amt_paid": amt, "cur_paid": "US Dollar",
                        "fmt": "Credit Card", "y": d.is_fraud})
    out.to_csv(dst, index=False, compression="gzip"); print("sparkov", len(out), int(out.y.sum()))




def paysim(src, dst):
    """PaySim (Lopez-Rojas et al. 2016). Balances and isFlaggedFraud are dropped: every simulated fraud empties the
    account, so balances are a label shortcut (docs/datasets.md), and isFlaggedFraud is the simulator's own rule output."""
    d = pd.read_csv(src, usecols=["step", "type", "amount", "nameOrig", "nameDest", "isFraud"])
    ts = (pd.Timestamp("2022-09-01") + pd.to_timedelta(d.step, unit="h")).dt.strftime("%Y/%m/%d %H:%M")
    fmt = d.type.map({"TRANSFER": "Wire", "CASH_OUT": "Cash", "CASH_IN": "Cash", "PAYMENT": "Credit Card", "DEBIT": "ACH"})
    amt = d.amount.map(lambda x: f"{max(x, 0.01):.2f}")
    out = pd.DataFrame({"ts": ts, "fb": "wallet", "fa": d.nameOrig, "tb": "wallet", "ta": d.nameDest, "amt_recv": amt, "cur_recv": "US Dollar",
                        "amt_paid": amt, "cur_paid": "US Dollar", "fmt": fmt, "y": d.isFraud})
    out.to_csv(dst, index=False, compression="gzip"); print("paysim", len(out), int(out.y.sum()))


def amlgentex(src, dst):
    sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
    import amlgentex_transfer
    amlgentex_transfer.convert(src, dst)


def tabformer(src, dst):
    """IBM TabFormer (Padhi et al. 2021, synthetic US cards, 24M transactions). Chunked; output unsorted, the caller sorts
    by time on disk. Payer = user and card, payee = merchant name, minute timestamps, Credit Card format."""
    first = True
    for ch in pd.read_csv(src, chunksize=1_000_000, dtype=str):
        ts = ch.Year + "/" + ch.Month.str.zfill(2) + "/" + ch.Day.str.zfill(2) + " " + ch.Time
        amt = ch.Amount.str.replace("$", "", regex=False).astype(float).abs().clip(lower=0.01).map(lambda x: f"{x:.2f}")
        out = pd.DataFrame({"ts": ts, "fb": "card", "fa": ch.User + "-" + ch.Card, "tb": "merchant", "ta": ch["Merchant Name"],
                            "amt_recv": amt, "cur_recv": "US Dollar", "amt_paid": amt, "cur_paid": "US Dollar", "fmt": "Credit Card",
                            "y": (ch["Is Fraud?"] == "Yes").astype(int)})
        out.to_csv(dst, index=False, header=False, mode="w" if first else "a"); first = False


if __name__ == "__main__":
    {"banksim": banksim, "sparkov": sparkov, "paysim": paysim, "amlgentex": amlgentex, "tabformer": tabformer}[sys.argv[1]](*sys.argv[2:])
