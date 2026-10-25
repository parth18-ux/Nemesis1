"""Typing rhythm check (service/behaviour.py) on the real Aalto 136M Keystrokes data.

Data: https://userinterfaces.aalto.fi/136Mkeystrokes/  (data/Keystrokes.zip, 1.4 GB; free for research and
non-commercial use with attribution: Dhakal, Feit, Kristensson, Oulasvirta, CHI 2018). The zip is read directly, no
need to extract its 16 GB. Each participant typed 15 sentences; one file per participant (*_keystrokes.txt, tab
separated, columns include PARTICIPANT_ID, TEST_SECTION_ID, PRESS_TIME, LETTER).

Protocol, per sampled participant:
  * enrol on their first 8 sentences (key to key times for each pair of consecutive letters, in ms)
  * genuine attempts: their remaining sentences
  * impostor attempts: one sentence each from 7 other sampled participants
  * score = TypingProfile's median z score over shared key pairs (higher = less like the owner)
Metrics: ROC-AUC and equal error rate pooled over attempts, the share of attempts with too few shared key pairs to
score, and the false alarm / catch rates at the live threshold (2.5).
Usage: python ml/aalto_eval.py path/to/Keystrokes.zip [users=1500]     writes results/aalto_eval.json
"""
import csv, io, json, os, random, sys, time, zipfile
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from behaviour import TypingProfile, THRESHOLD  # noqa: E402

ZIP = sys.argv[1]; USERS = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
ENROL, IMPOSTORS = 8, 7
rnd = random.Random(0); T0 = time.time()


def sentences(text: str) -> list[dict]:
    """One digraph timing dict per typed sentence: {"th": 112.0, ...} (mean when a pair repeats)."""
    rows = csv.DictReader(io.StringIO(text), delimiter="\t")
    by_section = {}
    for r in rows:
        try:
            by_section.setdefault(r["TEST_SECTION_ID"], []).append((float(r["PRESS_TIME"]), (r.get("LETTER") or "").lower()))
        except (KeyError, ValueError):
            continue
    out = []
    for _, ev in sorted(by_section.items(), key=lambda kv: kv[1][0][0] if kv[1] else 0):
        ev.sort(); acc = {}
        for (t0, a), (t1, b) in zip(ev, ev[1:]):
            if len(a) == 1 and len(b) == 1 and (a.isalpha() or a == " ") and (b.isalpha() or b == " ") and 0 < t1 - t0 < 2000:
                acc.setdefault(a + b, []).append(t1 - t0)
        if acc:
            out.append({k: float(np.mean(v)) for k, v in acc.items()})
    return out


with zipfile.ZipFile(ZIP) as zf:
    names = [n for n in zf.namelist() if n.endswith("_keystrokes.txt")]
    rnd.shuffle(names)
    data = {}
    for n in names:
        if len(data) >= USERS:
            break
        s = sentences(zf.read(n).decode("utf-8", errors="ignore"))
        if len(s) >= ENROL + 3:
            data[n] = s
print(f"[{time.time() - T0:4.0f}s] {len(data)} participants with at least {ENROL + 3} sentences (of {len(names):,} files)", flush=True)

ids = list(data); y, sc, unscored = [], [], 0
for u in ids:
    prof = TypingProfile()
    for s in data[u][:ENROL]:
        prof.enrol(s)
    attempts = [(0, s) for s in data[u][ENROL:]] + [(1, data[v][rnd.randrange(len(data[v]))]) for v in rnd.sample([x for x in ids if x != u], IMPOSTORS)]
    for lab, s in attempts:
        r = prof.score(s)
        if r["score"] is None:
            unscored += 1; continue
        y.append(lab); sc.append(r["score"])
y, sc = np.array(y), np.array(sc)
f, t, th = roc_curve(y, sc); i = int(np.nanargmin(np.abs((1 - t) - f)))
flag = sc > THRESHOLD
out = {"dataset": "Aalto 136M Keystrokes (real, desktop keyboards)", "source": "https://userinterfaces.aalto.fi/136Mkeystrokes/",
       "participants": len(ids), "enrol_sentences": ENROL, "attempts_scored": int(len(y)), "attempts_unscored_too_few_pairs": unscored,
       "roc_auc": round(float(roc_auc_score(y, sc)), 3), "equal_error_rate": round(float((f[i] + 1 - t[i]) / 2), 3),
       "at_live_threshold": {"threshold": THRESHOLD, "false_alarm_rate_genuine": round(float(flag[y == 0].mean()), 3),
                             "impostors_caught": round(float(flag[y == 1].mean()), 3)},
       "note": "One sentence is a very short sample; a banking session gives more typing, so these are conservative.",
       "runtime_seconds": round(time.time() - T0, 1)}
json.dump(out, open(os.path.join(ROOT, "results", "aalto_eval.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
