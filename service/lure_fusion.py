"""
Joins the message side (L-1 lure detector) to the money side (L0-L4 cascade), and decides who actually gets alerted.

A scam is a workflow: a lure message or call, pressure, then a payment to a mule. A text model alone either misses
scams or nags people; a payment model alone sees an ordinary-looking transfer. Together:

  * The victim gets one warning, at the moment it matters: when a payment to a NEW payee follows a risky message
    within 72 hours (cool-off + step-up verification), or when a payment goes to a UPI ID / account / phone that
    appeared in a risky message (hold).
  * Alert-fatigue controls: a push notification only for the high band, at most one per (account, workflow) per 24h
    and at most 3 per account per day; caution-band messages are never pushed, only shown if a payment follows.
    Every suppressed alert is counted with its reason, so the fatigue budget is measurable (GET /api/v2/alerts/summary).
"""
from __future__ import annotations
from collections import defaultdict, deque

LOOKBACK_MIN = 72 * 60          # a lure stays "live" for 72 hours
DEDUP_MIN = 24 * 60             # one push per (account, workflow) per 24h
DAILY_CAP = 3                   # pushes per account per rolling 24h

RANK = {"Approve": 0, "Step-up verification": 1, "Hold for analyst": 2, "Block + STR draft": 3}
BAND = {0: "safe", 1: "med", 2: "med", 3: "high"}


def _norm(x) -> str:
    return str(x or "").strip().lower()


class LureMemory:
    def __init__(self):
        self.reset()

    def reset(self):
        self.events = defaultdict(deque)          # account -> deque[(tmin, band, workflow_id, workflow_label, risk)]
        self.watch = {}                           # entity -> (tmin, workflow_label, risk)
        self.pushed = defaultdict(deque)          # account -> deque[(tmin, workflow_id)]
        self.stats = defaultdict(int)

    # ---------------------------------------------------------------- messages
    def record_message(self, account: str | None, tmin: int, scored: dict) -> dict:
        """Remember a scored message; return the notification decision."""
        band = scored["band"]; wf = scored.get("workflow") or {}
        wid, wlab = wf.get("id", "unknown"), wf.get("label", "Scam pattern")
        self.stats["messages"] += 1; self.stats[f"messages_{band}"] += 1
        if band == "safe":
            return {"notify": False, "reason": "Low risk"}
        ents = scored.get("entities", {})
        for e in ents.get("upi_ids", []) + ents.get("phones", []) + ents.get("accounts", []):
            self.watch[_norm(e)] = (tmin, wlab, scored["risk"])
        if not account:
            self.stats["suppressed_no_account"] += 1
            return {"notify": False, "reason": "No account given; message kept only for its entities (payee watchlist)"}
        acct = _norm(account); ev = self.events[acct]
        ev.append((tmin, band, wid, wlab, scored["risk"]))
        while ev and ev[0][0] < tmin - LOOKBACK_MIN:
            ev.popleft()
        if band != "high":
            self.stats["suppressed_caution"] += 1
            return {"notify": False, "reason": "Caution band: shown only if a payment follows (no push)"}
        P = self.pushed[acct]
        while P and P[0][0] < tmin - DEDUP_MIN:
            P.popleft()
        if any(w == wid for _, w in P):
            self.stats["suppressed_duplicate"] += 1
            return {"notify": False, "reason": f"Already warned about '{wlab}' in the last 24h"}
        if len(P) >= DAILY_CAP:
            self.stats["suppressed_daily_cap"] += 1
            return {"notify": False, "reason": f"Daily cap of {DAILY_CAP} warnings reached; queued for in-app inbox"}
        P.append((tmin, wid)); self.stats["notified"] += 1
        return {"notify": True, "reason": "High-risk scam message", "message": f"This matches a known scam pattern ({wlab}). Do not pay, share an OTP or install any app. "
                "Banks and police never ask for this."}

    # ---------------------------------------------------------------- payments
    def apply_to_payment(self, p: dict, res: dict) -> dict:
        """Escalate a cascade result using recent lures. Never lowers a decision."""
        tmin = int(p["tmin"]); payer = _norm(f'{p["from_bank"]}:{p["from_account"]}')
        payee_keys = {_norm(p["to_account"]), _norm(f'{p["to_bank"]}:{p["to_account"]}')}
        new_payee = bool((res.get("features") or {}).get("new_pair", 1))
        target, reasons = RANK.get(res["decision"], 0), []
        recent = [e for e in self.events.get(payer, ()) if 0 <= tmin - e[0] <= LOOKBACK_MIN]
        if recent and new_payee:
            t, band, _, wlab, risk = max(recent, key=lambda e: e[4])
            hrs = (tmin - t) / 60
            target = max(target, 1)
            reasons.append({"feature": "Payer received a scam-like message", "display": f"{wlab}, {hrs:.0f} h ago", "impact": round(risk, 3), "value": None})
        hit = next((self.watch[k] for k in payee_keys if k in self.watch and 0 <= tmin - self.watch[k][0] <= LOOKBACK_MIN), None)
        if hit:
            target = max(target, 2)
            reasons.append({"feature": "Payee named in a scam message", "display": hit[1], "impact": round(hit[2], 3), "value": None})
        if not reasons:
            res["lure_context"] = None
            return res
        if target > RANK.get(res["decision"], 0):
            self.stats["payments_escalated"] += 1
            res["decision_before_lure"] = res["decision"]
            res["decision"] = {1: "Step-up verification", 2: "Hold for analyst", 3: "Block + STR draft"}[target]
            res["band"] = BAND[target]
            if target == 1:
                res["payer_prompt"] = "Pause: you got a message that matches a known scam a short while ago. Is someone asking you to make this payment?"
        res["reasons"] = reasons + list(res.get("reasons") or [])
        res["lure_context"] = {"recent_lures": len(recent), "payee_on_watchlist": bool(hit)}
        return res

    def summary(self) -> dict:
        s = dict(self.stats); m = max(s.get("messages", 0), 1)
        s["push_rate_per_100_messages"] = round(100 * s.get("notified", 0) / m, 2)
        s["watchlist_entities"] = len(self.watch)
        return s
