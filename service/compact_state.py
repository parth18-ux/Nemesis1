"""
Compact account state for bank scale streaming.

Same features as Engine.features (service/rusetect_engine.py), computed from much smaller state:
  * accounts get integer ids; per account totals live in typed arrays (array module), not Python objects
  * "distinct payees / payers so far" are plain counters: a pair is new exactly when the payee is new to the payer,
    so the counter moves only when the pair index says "first time" (no per account sets)
  * 24 h windows exist only for accounts active in the last 24 h and are dropped when they empty
  * the pair index ("has A paid B before") is pluggable:
        exact  a set of 64 bit pair ids   (identical features; memory grows with distinct pairs)
        bloom  a fixed size Bloom filter  (fixed memory; a false positive can only turn a truly new pair into "seen")
In a bank the same layout maps onto a keyed state store (Flink keyed state, Kafka Streams, Redis): one row of totals
per account, a short window per active account, and one shared pair index per partition.
"""
from __future__ import annotations
import math, time
from array import array
from collections import deque
import numpy as np

MIN24 = 1440


class ExactPairs:
    kind = "exact"
    def __init__(self, **_):
        self.s = set()
    def seen_add(self, key: int) -> bool:
        if key in self.s:
            return True
        self.s.add(key); return False
    def seen(self, key: int) -> bool:
        return key in self.s
    def nbytes(self):
        return len(self.s) * 60            # CPython set entry plus int object, approximate


class BloomPairs:
    """k hash Bloom filter over 64 bit pair ids. m bits sized for capacity n at false positive rate p."""
    kind = "bloom"
    def __init__(self, capacity: int = 20_000_000, fp_rate: float = 0.001, **_):
        m = int(-capacity * math.log(fp_rate) / (math.log(2) ** 2)); self.k = max(1, round(m / capacity * math.log(2)))
        self.m = m; self.bits = bytearray((m + 7) // 8); self.capacity, self.fp_rate = capacity, fp_rate
    def _idx(self, key: int):
        h1 = (key * 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF; h2 = ((key ^ (key >> 29)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF | 1
        return [((h1 + i * h2) & 0xFFFFFFFFFFFFFFFF) % self.m for i in range(self.k)]
    def seen(self, key: int) -> bool:
        b = self.bits
        return all(b[i >> 3] >> (i & 7) & 1 for i in self._idx(key))
    def seen_add(self, key: int) -> bool:
        b = self.bits; hit = True
        for i in self._idx(key):
            j, bit = i >> 3, 1 << (i & 7)
            if not b[j] & bit:
                hit = False; b[j] |= bit
        return hit
    def nbytes(self):
        return len(self.bits)


class Win:
    __slots__ = ("q", "cnt", "usd")
    def __init__(self):
        self.q = deque(); self.cnt = {}; self.usd = 0.0
    def add(self, t, key, usd):
        self.q.append((t, key, usd)); self.cnt[key] = self.cnt.get(key, 0) + 1; self.usd += usd
        while self.q[0][0] <= t - MIN24:
            _, k, u = self.q.popleft(); self.usd -= u
            c = self.cnt[k] - 1
            if c: self.cnt[k] = c
            else: del self.cnt[k]
        return len(self.cnt), len(self.q), max(self.usd, 0.0)
    def distinct_since(self, t0):
        return len({k for (tt, k, _) in self.q if tt > t0})


class CompactState:
    """Drop in replacement for the feature half of Engine. cfg is Engine.cfg (usd rates, format and currency codes)."""
    NONE = -(1 << 62)

    def __init__(self, cfg: dict, pairs: str = "exact", **pair_kw):
        self.cfg = cfg; self.pair_kind = pairs; self.pair_kw = pair_kw
        self.reset()

    def reset(self):
        self.ids = {}
        self.out_n, self.in_n, self.uniq_dst, self.uniq_src = array("q"), array("q"), array("q"), array("q")
        self.out_usd, self.in_usd, self.last_in_usd = array("d"), array("d"), array("d")
        self.last_out_t, self.first_out_t, self.last_in_t = array("q"), array("q"), array("q")
        self.win_out, self.win_in = {}, {}
        self.pairs = (BloomPairs if self.pair_kind == "bloom" else ExactPairs)(**self.pair_kw)
        self.bank_pairs = set(); self.n_seen = 0

    def _id(self, bank, acct):
        k = f"{bank}|{acct}"
        i = self.ids.get(k)
        if i is None:
            i = self.ids[k] = len(self.ids)
            for a in (self.out_n, self.in_n, self.uniq_dst, self.uniq_src): a.append(0)
            for a in (self.out_usd, self.in_usd): a.append(0.0)
            self.last_in_usd.append(math.nan)
            for a in (self.last_out_t, self.first_out_t, self.last_in_t): a.append(self.NONE)
        return i

    def sender_fan_in_24h(self, p: dict) -> int:
        i = self.ids.get(f'{p["from_bank"]}|{p["from_account"]}')
        w = self.win_in.get(i) if i is not None else None
        return w.distinct_since(int(p["tmin"]) - MIN24) if w else 0

    def features(self, p: dict) -> dict:
        s = self._id(str(p["from_bank"]), str(p["from_account"])); d = self._id(str(p["to_bank"]), str(p["to_account"]))
        t = int(p["tmin"]); usd = float(p["amount_paid"]) * self.cfg["usd_rates"].get(p["payment_currency"], 1.0)
        N, nan = self.NONE, math.nan
        f = {}
        f["log_usd"] = math.log1p(usd)
        f["fmt_code"] = self.cfg["fmt_codes"].get(p["payment_format"], nan)
        f["cur_code"] = self.cfg["cur_codes"].get(p["payment_currency"], nan)
        dt = time.gmtime(t * 60); f["hour"] = dt.tm_hour; f["dow"] = dt.tm_wday + 1
        f["cross_bank"] = int(str(p["from_bank"]) != str(p["to_bank"]))
        f["cross_cur"] = int(p.get("receiving_currency", p["payment_currency"]) != p["payment_currency"])
        f["self_loop"] = int(s == d); f["round_100"] = int(usd % 100 == 0)
        on = self.out_n[s]
        f["s_out_n"] = on
        f["s_out_mean_usd"] = math.log1p(self.out_usd[s] / max(on, 1))
        lo = self.last_out_t[s]; fo = self.first_out_t[s]
        f["s_gap_out"] = t - lo if lo != N else nan
        f["s_age_out"] = t - fo if fo != N else 0
        recip = int(self.pairs.seen((d << 32) | s))       # before adding this pair (a self payment is not its own reverse)
        key = (s << 32) | d
        new = not self.pairs.seen_add(key)
        f["new_pair"] = int(new)
        f["s_uniq_dst"] = self.uniq_dst[s]
        f["d_in_n"] = self.in_n[d]
        li = self.last_in_t[d]
        f["d_gap_in"] = t - li if li != N else nan
        f["d_uniq_src"] = self.uniq_src[d]
        f["s_in_n"] = self.in_n[s]
        sli = self.last_in_t[s]
        f["s_mins_since_in"] = t - sli if sli != N else nan
        liu = self.last_in_usd[s]
        f["s_out_over_last_in"] = usd / liu if (liu == liu and liu) else nan
        f["s_passthrough"] = self.out_usd[s] / max(self.in_usd[s], 1.0)
        f["d_out_n"] = self.out_n[d] + (1 if s == d else 0)
        f["d_in_usd_log"] = math.log1p(self.in_usd[d])
        wo = self.win_out.get(s)
        if wo is None: wo = self.win_out[s] = Win()
        a, b, c = wo.add(t, d, usd)
        f["s_fanout_24h"], f["s_cnt_24h"], f["s_usd_24h"] = a, b, math.log1p(c)
        wi = self.win_in.get(d)
        if wi is None: wi = self.win_in[d] = Win()
        a, b, _ = wi.add(t, s, usd)
        f["d_fanin_24h"], f["d_cnt_24h"] = a, b
        f["reciprocal"] = recip
        bp = (str(p["from_bank"]), str(p["to_bank"])); f["new_bank_pair"] = int(bp not in self.bank_pairs)
        # ---- update state (same order as Engine.features)
        if fo == N: self.first_out_t[s] = t
        self.out_n[s] = on + 1; self.out_usd[s] += usd; self.last_out_t[s] = t
        if new: self.uniq_dst[s] += 1; self.uniq_src[d] += 1
        self.in_n[d] += 1; self.in_usd[d] += usd; self.last_in_t[d] = t; self.last_in_usd[d] = usd
        self.bank_pairs.add(bp); self.n_seen += 1
        if self.n_seen % 200_000 == 0:
            self.evict(t)
        return f

    def evict(self, now):
        """Drop windows with nothing in the last 24 h (they would return zeros anyway)."""
        for W in (self.win_out, self.win_in):
            dead = [k for k, w in W.items() if not w.q or w.q[-1][0] <= now - MIN24]
            for k in dead:
                del W[k]

    def stats(self):
        arrays = sum(a.itemsize * len(a) for a in (self.out_n, self.in_n, self.uniq_dst, self.uniq_src, self.out_usd, self.in_usd,
                                                   self.last_in_usd, self.last_out_t, self.first_out_t, self.last_in_t))
        return {"accounts": len(self.ids), "active_windows": len(self.win_out) + len(self.win_in), "pair_index": self.pairs.kind,
                "pair_index_bytes": self.pairs.nbytes(), "account_arrays_bytes": arrays}
