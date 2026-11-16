"""Convert SAML-D (Oztas et al., IEEE ICEBE 2023; CC BY-NC-SA 4.0) to the AMLworld column layout read by research_dump.

    python3 -I research/samld_convert.py <SAML-D.csv> <out.csv.gz>

Mapping (stated, part of the measured generator gap): bank = bank location (SAML-D has no bank ids); payment type
Credit/Debit card -> Credit Card, Cross-border -> Wire, Cash Deposit/Withdrawal -> Cash, ACH and Cheque as is;
currencies the served config knows are renamed (UK pounds -> UK Pound, Indian rupee -> Rupee, ...); the six it does not
know (Turkish lira, Dirham, Pakistani rupee, Naira, Moroccan dirham, Albanian lek, 1.8% of payments) stay unknown, which
the engine converts at rate 1.0 with a missing currency code. Seconds are dropped (the engine works in minutes).
A sidecar <out>.types lists Laundering_type per row for typology recall.
"""
import sys
import pandas as pd

CUR = {"UK pounds": "UK Pound", "Euro": "Euro", "Swiss franc": "Swiss Franc", "US dollar": "US Dollar", "Yen": "Yen",
       "Mexican Peso": "Mexican Peso", "Indian rupee": "Rupee"}
FMT = {"Credit card": "Credit Card", "Debit card": "Credit Card", "Cheque": "Cheque", "ACH": "ACH", "Cross-border": "Wire",
       "Cash Withdrawal": "Cash", "Cash Deposit": "Cash"}


def main(src, dst):
    d = pd.read_csv(src, dtype=str)
    d["_t"] = pd.to_datetime(d.Date + " " + d.Time)
    d = d.sort_values("_t", kind="stable").reset_index(drop=True)
    amt = d.Amount.astype(float).map(lambda x: f"{x:.2f}")
    out = pd.DataFrame({"ts": d._t.dt.strftime("%Y/%m/%d %H:%M"), "fb": d.Sender_bank_location, "fa": d.Sender_account,
                        "tb": d.Receiver_bank_location, "ta": d.Receiver_account, "amt_recv": amt,
                        "cur_recv": d.Received_currency.map(lambda c: CUR.get(c, c)), "amt_paid": amt,
                        "cur_paid": d.Payment_currency.map(lambda c: CUR.get(c, c)), "fmt": d.Payment_type.map(FMT),
                        "y": d.Is_laundering})
    out.to_csv(dst, index=False, compression="gzip")
    d.Laundering_type.to_csv(dst + ".types", index=False, header=False)
    print("rows", len(out), "laundering", int(out.y.astype(int).sum()))


if __name__ == "__main__":
    main(*sys.argv[1:3])
