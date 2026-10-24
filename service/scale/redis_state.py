"""
Account state in Redis, so scoring workers hold no state and can be added or removed freely.

Keys (one Redis Cluster hash slot per account via the {account} tag, so a cluster shards by account):
  a:{acct}    hash   out_n out_usd last_out_t first_out_t in_n in_usd last_in_t last_in_usd uniq_dst uniq_src
  p:{acct}    set    payees this account has paid (pair index; "seen before" and the reverse "money flowing back")
  wo:{acct}   zset   outgoing payments in the last 24 h (score = minute), member "eid|counterparty|usd"
  wi:{acct}   zset   incoming payments in the last 24 h
  bp          set    bank corridors seen
Per payment: one pipelined read round trip, features computed in Python exactly as Engine.features, one pipelined
write round trip. With a single writer per account the features equal the in memory engine (tests/test_scale.py);
with several workers, payments are partitioned by payer so each payer's own history stays ordered, and a payee's
receipts from other partitions may lag by milliseconds (the usual trade off of streaming state).
"""
from __future__ import annotations
import math, time

MIN24 = 1440
A_S = ("out_n", "out_usd", "last_out_t", "first_out_t", "in_n", "in_usd", "last_in_t", "last_in_usd", "uniq_dst")
A_D = ("in_n", "last_in_t", "uniq_src", "out_n", "in_usd")


def _num(v, cast=float):
    return None if v is None else cast(v)


def _cp(m):
    """Member "eid|bank|account|usd" -> ("bank|account", usd)."""
    cp, u = m.split("|", 1)[1].rsplit("|", 1)
    return cp, float(u)


def _window(members, t, new_key, usd):
    """Members strictly newer than t - 24 h (already filtered by score) plus the current payment."""
    keys, total = {}, usd
    for m in members:
        k, u = _cp(m); keys[k] = 1; total += u
    keys[new_key] = 1
    return len(keys), len(members) + 1, max(total, 0.0)


class RedisState:
    def __init__(self, r, cfg: dict, prefix: str = ""):
        self.r, self.cfg, self.px = r, cfg, prefix

    def k(self, kind, acct):
        return f"{self.px}{kind}:{{{acct}}}"

    def features_many(self, payments: list[dict]) -> list[dict]:
        """Sequential semantics inside the batch: each payment sees the state left by the previous one."""
        return [self.features(p) for p in payments]

    def features(self, p: dict) -> dict:
        r, cfg = self.r, self.cfg
        s = f'{p["from_bank"]}|{p["from_account"]}'; d = f'{p["to_bank"]}|{p["to_account"]}'
        t = int(p["tmin"]); usd = float(p["amount_paid"]) * cfg["usd_rates"].get(p["payment_currency"], 1.0)
        bp = f'{p["from_bank"]}>{p["to_bank"]}'; eid = str(p.get("eid", f"{t}.{time.perf_counter_ns()}"))
        lo = f"({t - MIN24}"
        pl = r.pipeline(transaction=False)
        pl.hmget(self.k("a", s), A_S); pl.hmget(self.k("a", d), A_D)
        pl.sismember(self.k("p", s), d); pl.sismember(self.k("p", d), s); pl.sismember(f"{self.px}bp", bp)
        pl.zrangebyscore(self.k("wo", s), lo, "+inf"); pl.zrangebyscore(self.k("wi", d), lo, "+inf")
        pl.zrangebyscore(self.k("wi", s), lo, "+inf")
        S, D, seen, recip, bp_seen, wo, wi, wi_s = pl.execute()
        self.last_sender_fan_in = len({_cp(m)[0] for m in wi_s})   # payers of the sender in the last 24 h, before this payment
        S = dict(zip(A_S, S)); D = dict(zip(A_D, D))
        nan = math.nan
        on = _num(S["out_n"], int) or 0; ousd = _num(S["out_usd"]) or 0.0
        f = {"log_usd": math.log1p(usd), "fmt_code": cfg["fmt_codes"].get(p["payment_format"], nan),
             "cur_code": cfg["cur_codes"].get(p["payment_currency"], nan)}
        dt = time.gmtime(t * 60); f["hour"] = dt.tm_hour; f["dow"] = dt.tm_wday + 1
        f["cross_bank"] = int(str(p["from_bank"]) != str(p["to_bank"]))
        f["cross_cur"] = int(p.get("receiving_currency", p["payment_currency"]) != p["payment_currency"])
        f["self_loop"] = int(s == d); f["round_100"] = int(usd % 100 == 0)
        f["s_out_n"] = on; f["s_out_mean_usd"] = math.log1p(ousd / max(on, 1))
        lot, fot = _num(S["last_out_t"], int), _num(S["first_out_t"], int)
        f["s_gap_out"] = t - lot if lot is not None else nan
        f["s_age_out"] = t - fot if fot is not None else 0
        f["new_pair"] = int(not seen); f["s_uniq_dst"] = _num(S["uniq_dst"], int) or 0
        f["d_in_n"] = _num(D["in_n"], int) or 0
        dli = _num(D["last_in_t"], int); f["d_gap_in"] = t - dli if dli is not None else nan
        f["d_uniq_src"] = _num(D["uniq_src"], int) or 0
        f["s_in_n"] = _num(S["in_n"], int) or 0
        sli = _num(S["last_in_t"], int); f["s_mins_since_in"] = t - sli if sli is not None else nan
        liu = _num(S["last_in_usd"]); f["s_out_over_last_in"] = usd / liu if liu else nan
        f["s_passthrough"] = ousd / max(_num(S["in_usd"]) or 0.0, 1.0)
        f["d_out_n"] = (_num(D["out_n"], int) or 0) + (1 if s == d else 0)
        f["d_in_usd_log"] = math.log1p(_num(D["in_usd"]) or 0.0)
        a, b, c = _window(wo, t, d, usd); f["s_fanout_24h"], f["s_cnt_24h"], f["s_usd_24h"] = a, b, math.log1p(c)
        a, b, _ = _window(wi, t, s, usd); f["d_fanin_24h"], f["d_cnt_24h"] = a, b
        f["reciprocal"] = int(bool(recip)); f["new_bank_pair"] = int(not bp_seen)
        # ---- writes
        w = r.pipeline(transaction=False); ka_s, ka_d = self.k("a", s), self.k("a", d)
        w.hsetnx(ka_s, "first_out_t", t); w.hincrby(ka_s, "out_n", 1); w.hincrbyfloat(ka_s, "out_usd", usd); w.hset(ka_s, "last_out_t", t)
        if not seen:
            w.sadd(self.k("p", s), d); w.hincrby(ka_s, "uniq_dst", 1); w.hincrby(ka_d, "uniq_src", 1)
        w.hincrby(ka_d, "in_n", 1); w.hincrbyfloat(ka_d, "in_usd", usd); w.hset(ka_d, mapping={"last_in_t": t, "last_in_usd": repr(usd)})
        mem = f"{eid}|{{}}|{usd!r}"
        w.zadd(self.k("wo", s), {mem.format(d): t}); w.zadd(self.k("wi", d), {mem.format(s): t})
        w.zremrangebyscore(self.k("wo", s), "-inf", t - MIN24); w.zremrangebyscore(self.k("wi", d), "-inf", t - MIN24)
        w.expire(self.k("wo", s), 2 * 86400); w.expire(self.k("wi", d), 2 * 86400)
        if not bp_seen:
            w.sadd(f"{self.px}bp", bp)
        w.execute()
        return f

    def sender_fan_in_24h(self, p: dict) -> int:
        s = f'{p["from_bank"]}|{p["from_account"]}'; t = int(p["tmin"])
        return len({_cp(m)[0] for m in self.r.zrangebyscore(self.k("wi", s), f"({t - MIN24}", "+inf")})
