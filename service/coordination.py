"""
Flagged sender network rule ("accounts paid by 5+ flagged senders are flagged for every bank").

A payee that keeps receiving money from accounts that RuseTect has itself flagged recently is collecting from many
scam victims or many laundering hops. Unlike the mule signature (which looks at one account's own in and out
timing), this counts how many DIFFERENT flagged accounts paid the payee, so it needs no per account timing quirk.

Measured on the full IBM files (ml/coordination_eval.py, results/coordination_eval.json): inside the real fusion the
rule adds about one laundering payment per 700 extra flags on HI-Small and LI-Small test weeks, because IBM's
laundering is moved by the payment model's own features already. So the LOCAL count is OFF by default
(RUSETECT_COORDINATION=1 turns it on, e.g. for message led scams where many victims pay one collector, which IBM
data does not contain). Cross bank flags from the consortium ledger stay on: those are payees that two analysts at a
member bank already approved as mules.
"""
from __future__ import annotations
import os
from collections import defaultdict, deque

# least noisy setting in the fusion test (7 days, senders held for an analyst, 3 of them); see results/coordination_eval.json
DEFAULTS = {"window_min": 7 * 24 * 60, "flag_rank": 2, "k": 3}
ENABLED = os.environ.get("RUSETECT_COORDINATION", "0") == "1"


class CoordinationTracker:
    def __init__(self, window_min: int = DEFAULTS["window_min"], flag_rank: int = DEFAULTS["flag_rank"], k: int = DEFAULTS["k"],
                 enabled: bool = ENABLED):
        self.window, self.flag_rank, self.k, self.enabled = window_min, flag_rank, k, enabled
        self.reset()

    def reset(self):
        self.flagged_at = {}                       # sender -> last minute it had a flagged payment
        self.incoming = defaultdict(deque)         # payee -> deque[(minute, sender)]

    def flagged_payers(self, payee, t: int) -> int:
        """Distinct senders that paid this payee in the window and were flagged at or before their payment (read before scoring)."""
        q = self.incoming.get(payee)
        if not q:
            return 0
        while q and q[0][0] <= t - self.window:
            q.popleft()
        lo = t - self.window
        return len({s for (tt, s) in q if self.flagged_at.get(s, -10 ** 12) > lo})

    def coordinated(self, payee, t: int) -> bool:
        return self.enabled and self.flagged_payers(payee, t) >= self.k

    def record(self, sender, payee, t: int, final_rank: int):
        """After scoring: remember the payment and whether its sender was flagged."""
        if final_rank >= self.flag_rank:
            self.flagged_at[sender] = t
        self.incoming[payee].append((t, sender))
