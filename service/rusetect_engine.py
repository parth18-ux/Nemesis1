"""
RuseTect streaming engine: the online version of ml/aml_features.py + ml/cascade.py.

Keeps per-account state in memory, computes the same causal features for each incoming
payment (only past payments are used), then runs the cascade:
  L1 gatekeeper (30-tree XGBoost)  -> clean traffic approved in ~0.2 ms
  L2 graph-boost (XGBoost + LightGBM) on survivors
  Decision band + top reasons (XGBoost SHAP contributions)
  L4 ring linker: alerted payments are unioned into cases by shared accounts
L0 onboarding (BAF LightGBM with age-group thresholds) scores new account applications.
"""
from __future__ import annotations
import json, math, time
from collections import defaultdict, deque
from pathlib import Path
import numpy as np

MODELS = Path(__file__).parent / "models"
MIN24 = 1440

READABLE = {
    "log_usd": "Payment amount", "fmt_code": "Payment format", "cur_code": "Currency", "hour": "Hour of day",
    "dow": "Day of week", "cross_bank": "Cross bank transfer", "cross_cur": "Currency conversion",
    "self_loop": "Self transfer", "round_100": "Round amount",
    "s_out_n": "Sender: past payments", "s_out_mean_usd": "Sender: usual amount", "s_gap_out": "Sender: time since last payment",
    "s_age_out": "Sender: time active", "new_pair": "First payment to this receiver", "s_uniq_dst": "Sender: distinct receivers",
    "d_in_n": "Receiver: past receipts", "d_gap_in": "Receiver: time since last receipt", "d_uniq_src": "Receiver: distinct senders",
    "s_in_n": "Sender: money received before", "s_mins_since_in": "Sender: minutes since money came in",
    "s_out_over_last_in": "Amount vs. last money received", "s_passthrough": "Sender: pass through ratio",
    "d_out_n": "Receiver: onward payments", "d_in_usd_log": "Receiver: money received so far",
    "s_fanout_24h": "Sender fan out (24h)", "s_cnt_24h": "Sender payments (24h)", "s_usd_24h": "Sender volume (24h)",
    "d_fanin_24h": "Receiver fan in (24h)", "d_cnt_24h": "Receiver receipts (24h)", "reciprocal": "Money flowing back (cycle)",
    "new_bank_pair": "New bank corridor",
}


class Window:
    """24h sliding window with O(1) amortised distinct-counterparty count and running sum."""
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


class Acct:
    __slots__ = ("out_n", "out_usd", "last_out_t", "first_out_t", "dsts", "in_n", "in_usd", "last_in_t", "last_in_usd",
                 "srcs", "win_out", "win_in")
    def __init__(self):
        self.out_n = 0; self.out_usd = 0.0; self.last_out_t = None; self.first_out_t = None; self.dsts = set()
        self.in_n = 0; self.in_usd = 0.0; self.last_in_t = None; self.last_in_usd = None; self.srcs = set()
        self.win_out = Window(); self.win_in = Window()


class Engine:
    def __init__(self, models_dir: Path = MODELS, load_models: bool = True):
        self.cfg = json.loads((models_dir / "cascade_config.json").read_text())
        self.feats = self.cfg["features"]
        missing = {"l1_threshold", "block_threshold", "hold_threshold", "verify_threshold"} - set(self.cfg)
        if missing:
            raise ValueError(f"cascade_config.json is missing thresholds: {sorted(missing)}")
        self.accts: dict = defaultdict(Acct)
        self.pairs: set = set(); self.bank_pairs: set = set()
        self.parent: dict = {}; self.case_alerts: dict = defaultdict(list)
        self.n_seen = 0; self.decisions = defaultdict(int)
        self.models_loaded = False
        if load_models:
            import xgboost as xgb
            from lgbm_lite import LiteBooster      # pure-NumPy LightGBM inference (no libgomp needed)
            self.l1 = xgb.Booster(); self.l1.load_model(str(models_dir / "l1_gatekeeper.json"))
            self.l2x = xgb.Booster(); self.l2x.load_model(str(models_dir / "l2_xgb.json"))
            self.l2l = LiteBooster(models_dir / "l2_lgbm.json")
            self.l0 = LiteBooster(models_dir / "l0_onboarding_lgbm.json")
            self.l0cfg = json.loads((models_dir / "l0_config.json").read_text())
            self.models_loaded = True

    def reset(self):
        """Forget all account state (models stay loaded)."""
        self.accts = defaultdict(Acct); self.pairs = set(); self.bank_pairs = set()
        self.parent = {}; self.case_alerts = defaultdict(list); self.n_seen = 0; self.decisions = defaultdict(int)

    # ------------------------------------------------------------ features
    def features(self, p: dict) -> dict:
        """p: from_bank, from_account, to_bank, to_account, amount_paid, payment_currency,
        receiving_currency, payment_format, tmin (minutes since epoch). Updates state."""
        src = (str(p["from_bank"]), str(p["from_account"])); dst = (str(p["to_bank"]), str(p["to_account"]))
        t = int(p["tmin"]); usd = float(p["amount_paid"]) * self.cfg["usd_rates"].get(p["payment_currency"], 1.0)
        S, D = self.accts[src], self.accts[dst]
        f = {}
        f["log_usd"] = math.log1p(usd)
        f["fmt_code"] = self.cfg["fmt_codes"].get(p["payment_format"], np.nan)
        f["cur_code"] = self.cfg["cur_codes"].get(p["payment_currency"], np.nan)
        dt = time.gmtime(t * 60); f["hour"] = dt.tm_hour; f["dow"] = dt.tm_wday + 1
        f["cross_bank"] = int(src[0] != dst[0]); f["cross_cur"] = int(p.get("receiving_currency", p["payment_currency"]) != p["payment_currency"])
        f["self_loop"] = int(src == dst); f["round_100"] = int(usd % 100 == 0)
        # sender history (before this payment)
        f["s_out_n"] = S.out_n
        f["s_out_mean_usd"] = math.log1p(S.out_usd / max(S.out_n, 1))
        f["s_gap_out"] = t - S.last_out_t if S.last_out_t is not None else np.nan
        f["s_age_out"] = t - S.first_out_t if S.first_out_t is not None else 0
        f["new_pair"] = int((src, dst) not in self.pairs)
        f["s_uniq_dst"] = len(S.dsts)
        # receiver history
        f["d_in_n"] = D.in_n
        f["d_gap_in"] = t - D.last_in_t if D.last_in_t is not None else np.nan
        f["d_uniq_src"] = len(D.srcs)
        # pass-through (mule) behaviour
        f["s_in_n"] = S.in_n
        f["s_mins_since_in"] = t - S.last_in_t if S.last_in_t is not None else np.nan
        f["s_out_over_last_in"] = usd / S.last_in_usd if S.last_in_usd else np.nan
        f["s_passthrough"] = S.out_usd / max(S.in_usd, 1.0)
        f["d_out_n"] = D.out_n + (1 if src == dst else 0)
        f["d_in_usd_log"] = math.log1p(D.in_usd)
        # 24h windows (include current)
        fo, co, uo = S.win_out.add(t, dst, usd)
        f["s_fanout_24h"], f["s_cnt_24h"], f["s_usd_24h"] = fo, co, math.log1p(uo)
        fi, ci, _ = D.win_in.add(t, src, usd)
        f["d_fanin_24h"], f["d_cnt_24h"] = fi, ci
        f["reciprocal"] = int((dst, src) in self.pairs)
        bp = (src[0], dst[0]); f["new_bank_pair"] = int(bp not in self.bank_pairs)
        # ---- update state
        if S.first_out_t is None: S.first_out_t = t
        S.out_n += 1; S.out_usd += usd; S.last_out_t = t; S.dsts.add(dst)
        D.in_n += 1; D.in_usd += usd; D.last_in_t = t; D.last_in_usd = usd; D.srcs.add(src)
        self.pairs.add((src, dst)); self.bank_pairs.add(bp); self.n_seen += 1
        return f

    # ------------------------------------------------------------ cascade
    def _vec(self, f):
        return np.array([[f[k] for k in self.feats]], dtype=np.float32)

    def score_payment(self, p: dict) -> dict:
        import xgboost as xgb
        t0 = time.perf_counter()
        f = self.features(p); x = self._vec(f)
        dm = xgb.DMatrix(x, feature_names=self.feats)
        s1 = float(self.l1.predict(dm)[0])
        out = {"l1_score": round(s1, 5), "features": {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in f.items()}}
        if s1 < self.cfg["l1_threshold"]:
            out.update(layer_exit="L1", risk=round(s1, 5), decision="Approve", band="safe", reasons=[], case_id=None)
        else:
            sx = float(self.l2x.predict(dm)[0]); sl = float(self.l2l.predict(x)[0])
            risk = (sx + sl) / 2
            contrib = self.l2x.predict(dm, pred_contribs=True)[0][:-1]
            top = np.argsort(-contrib)[:4]
            reasons = [{"feature": READABLE.get(self.feats[i], self.feats[i]), "value": out["features"][self.feats[i]],
                        "display": self._display(self.feats[i], out["features"][self.feats[i]]),
                        "impact": round(float(contrib[i]), 3)} for i in top if contrib[i] > 0]
            if risk >= self.cfg["block_threshold"]:
                decision, band = "Block + STR draft", "high"
            elif risk >= self.cfg["hold_threshold"]:
                decision, band = "Hold for analyst", "med"
            elif risk >= self.cfg["verify_threshold"]:
                decision, band = "Step-up verification", "med"
            else:
                decision, band = "Approve", "safe"
            case_id = self._link(p) if band == "high" else None
            out.update(layer_exit="L2", risk=round(risk, 5), l2_xgb=round(sx, 5), l2_lgbm=round(sl, 5),
                       decision=decision, band=band, reasons=reasons, case_id=case_id)
        self.decisions[out["decision"]] += 1
        out["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
        return out

    def _display(self, k, v):
        """Human-readable value for a reason chip."""
        if v is None: return "none yet"
        if k == "fmt_code": return {c: n for n, c in self.cfg["fmt_codes"].items()}.get(int(v), str(v))
        if k == "cur_code": return {c: n for n, c in self.cfg["cur_codes"].items()}.get(int(v), str(v))
        if k in ("log_usd", "s_out_mean_usd", "s_usd_24h", "d_in_usd_log"): return f"${math.expm1(v):,.0f}"
        if k in ("s_gap_out", "d_gap_in", "s_mins_since_in", "s_age_out"):
            v = float(v); return f"{v:.0f} min" if v < 120 else f"{v/60:.1f} h" if v < 2880 else f"{v/1440:.1f} days"
        if k in ("s_out_over_last_in", "s_passthrough"): return f"{float(v):.2f}×"
        if k in ("new_pair", "reciprocal", "new_bank_pair", "cross_bank", "cross_cur", "self_loop", "round_100"): return "yes" if v else "no"
        if k == "hour": return f"{int(v):02d}:00 UTC"
        return f"{int(v):,}" if float(v).is_integer() else f"{float(v):.2f}"

    # ------------------------------------------------------------ L4 ring linker
    def _find(self, a):
        while self.parent.setdefault(a, a) != a:
            self.parent[a] = self.parent[self.parent[a]]; a = self.parent[a]
        return a

    def _link(self, p):
        a = f'{p["from_bank"]}_{p["from_account"]}'; b = f'{p["to_bank"]}_{p["to_account"]}'
        ra, rb = self._find(a), self._find(b)
        if ra != rb:
            self.parent[ra] = rb
            if ra in self.case_alerts: self.case_alerts[rb].extend(self.case_alerts.pop(ra))
        root = self._find(b); self.case_alerts[root].append((a, b, float(p["amount_paid"]), p["payment_format"]))
        return "CASE-" + str(abs(hash(root)) % 10 ** 8)

    def cases(self, limit=50):
        out = []
        for root, al in self.case_alerts.items():
            accts = {x for e in al for x in e[:2]}
            out.append({"case_id": "CASE-" + str(abs(hash(root)) % 10 ** 8), "alerts": len(al), "accounts": len(accts),
                        "total_amount": round(sum(e[2] for e in al), 2), "formats": sorted({e[3] for e in al}),
                        "edges": [{"from": e[0], "to": e[1], "amount": e[2]} for e in al[:40]]})
        return sorted(out, key=lambda c: -c["alerts"])[:limit]

    # ------------------------------------------------------------ L0 onboarding
    def score_application(self, app: dict) -> dict:
        c = self.l0cfg; cats = c["categorical"]; x = []
        for k in c["features"]:
            v = app.get(k)
            if k in cats:                                   # LightGBM categorical = index in training categories
                lst = c["categories"][k]; x.append(float(lst.index(v)) if v in lst else float("nan"))
            else:
                x.append(float("nan") if v is None else float(v))
        p = float(self.l0.predict(np.array([x]))[0])
        thr = c["age_50_plus"] if (app.get("customer_age") or 0) >= 50 else c["age_under_50"]
        return {"risk": round(p, 5), "threshold": round(thr, 5), "flag": p >= thr,
                "decision": "Enhanced KYC / video verification" if p >= thr else "Open account"}
