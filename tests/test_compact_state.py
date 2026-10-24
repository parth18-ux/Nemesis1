"""The compact state must produce exactly the live engine's features (exact pair index)."""
import calendar, csv, gzip, math, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "service"))
from rusetect_engine import Engine
from compact_state import CompactState
from context_rules import sender_fan_in_24h


def _rows(n):
    rd = csv.DictReader(gzip.open(os.path.join(ROOT, "service", "models", "demo_stream.csv.gz"), "rt"))
    for i, r in enumerate(rd):
        if i >= n:
            break
        yield dict(from_bank=r["fb"], from_account=r["fa"], to_bank=r["tb"], to_account=r["ta"], amount_paid=float(r["amt_paid"]),
                   payment_currency=r["cur_paid"], receiving_currency=r["cur_recv"], payment_format=r["fmt"],
                   tmin=calendar.timegm(time.strptime(r["ts"], "%Y/%m/%d %H:%M")) // 60)


def same(a, b):
    return (isinstance(a, float) and math.isnan(a) and isinstance(b, float) and math.isnan(b)) or a == b


def test_compact_exact_matches_engine():
    eng = Engine(load_models=False); eng.reset(); cs = CompactState(eng.cfg, pairs="exact")
    for p in _rows(40_000):
        assert sender_fan_in_24h(eng, p) == cs.sender_fan_in_24h(p)
        fa, fb = eng.features(p), cs.features(p)
        bad = [k for k in fa if not same(fa[k], fb[k])]
        assert not bad, (bad, {k: (fa[k], fb[k]) for k in bad})


def test_bloom_only_errs_towards_seen():
    eng = Engine(load_models=False); eng.reset(); cs = CompactState(eng.cfg, pairs="bloom", capacity=200_000, fp_rate=0.01)
    wrong = 0
    for p in _rows(40_000):
        fa, fb = eng.features(p), cs.features(p)
        if fa["new_pair"] != fb["new_pair"]:
            assert fa["new_pair"] == 1 and fb["new_pair"] == 0     # a Bloom miss can only say "seen before"
            wrong += 1
    assert wrong < 400
