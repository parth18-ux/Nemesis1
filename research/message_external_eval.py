"""External test of the L-1 message model on texts written by another author (bolewara Hinglish & English scam text,
Hugging Face, CC BY 4.0). Rows that also appear in the UCI SMS Spam Collection (already in training) are removed, so
what remains is new to the model: hand-written Hinglish scams and benign messages, and curated English phishing.

    python3 -I research/message_external_eval.py <bolewara data.json> <uci sms_spam parquet>
Results: research/out/message_external_eval.json
"""
import json, os, sys
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from scam_text import MessageModel, normalize  # noqa: E402


def main(bole, uci):
    rows = json.load(open(bole))["rows"]
    seen = {normalize(t) for t in pd.read_parquet(uci).sms}
    ext = [r for r in rows if normalize(r["text"]) not in seen]
    hing = [r for r in ext if any(w in r["text"].lower().split() for w in ("aapka", "hai", "karein", "kar", "nahi", "aap", "mein", "me", "ka", "ki", "ho", "jaldi", "turant", "warna"))]
    m = MessageModel(os.path.join(ROOT, "service", "models", "message_model.json"))
    def ev(rs):
        out = {"messages": len(rs), "scam": sum(r["label"] for r in rs)}
        sc = [(r["label"], m.score(r["text"])["band"]) for r in rs]
        for b in ("high", "caution"):
            flag = [(y, band in ("high",) if b == "high" else band in ("high", "caution")) for y, band in sc]
            tp = sum(1 for y, f in flag if f and y == 1); fp = sum(1 for y, f in flag if f and y == 0)
            out[f"{b}_or_above"] = {"recall": round(tp / max(1, out["scam"]), 3), "false_alarms": fp,
                                    "false_alarm_rate": round(fp / max(1, out["messages"] - out["scam"]), 3)}
        return out
    R = {"source": "bolewara/hinglish-scam-text-dataset (CC BY 4.0), rows not in UCI SMS Spam",
         "all_external": ev(ext), "hinglish_subset": ev(hing),
         "missed_hinglish_scams": [r["text"] for r in hing if r["label"] == 1 and m.score(r["text"])["band"] == "safe"][:15],
         "flagged_hinglish_benign": [r["text"] for r in hing if r["label"] == 0 and m.score(r["text"])["band"] != "safe"][:15]}
    os.makedirs(os.path.join(ROOT, "research", "out"), exist_ok=True)
    json.dump(R, open(os.path.join(ROOT, "research", "out", "message_external_eval.json"), "w"), indent=1, ensure_ascii=False)
    print(json.dumps(R, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main(*sys.argv[1:3])
