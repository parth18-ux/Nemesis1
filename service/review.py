"""Analyst review, audit trail and evidence packs (finale Part 3).

* Every assessed payment that needs a human (final decision above Approve) becomes an ALERT with an id.
* Analysts act on alerts: release, verify (ask the customer), hold, or approve_block (two different analysts must
  approve a block: four-eyes). The model never blocks on its own: "Block + STR draft" from the engine is shown as
  "Hold + recommend block" until two analysts approve.
* Every event is appended to a hash-chained audit log (each entry carries the hash of the previous one), so an edit
  to history is detectable. In memory here; a bank would write it to append-only storage.
* An evidence pack bundles what the model saw and what people did, with its own content hash, for an STR draft.
"""
from __future__ import annotations
import hashlib, itertools, json, time

ACTIONS = {"release", "verify", "hold", "approve_block"}


def _hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class ReviewDesk:
    def __init__(self):
        self.reset()

    def reset(self):
        self.alerts, self.log, self._ids = {}, [], itertools.count(1)

    def _append(self, event: dict) -> dict:
        prev = self.log[-1]["hash"] if self.log else "0" * 64
        e = {"seq": len(self.log) + 1, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **event, "prev": prev}
        e["hash"] = _hash(e)
        self.log.append(e)
        return e

    def open_alert(self, payment_in: dict, assessed: dict) -> dict | None:
        if assessed["final_decision"] == "Approve":
            return None
        aid = f"A{next(self._ids):05d}"
        shown = "Hold + recommend block" if assessed["final_decision"] == "Block + STR draft" else assessed["final_decision"]
        a = {"alert_id": aid, "status": "open", "shown_decision": shown, "payment": {k: payment_in.get(k) for k in
             ("timestamp", "from_bank", "from_account", "to_bank", "to_account", "amount_paid", "payment_currency", "payment_format")},
             "assessed": assessed, "actions": [], "block_approvals": []}
        self.alerts[aid] = a
        self._append({"type": "alert_opened", "alert_id": aid, "decision": shown, "context_band": assessed["context_band"],
                      "ruleset": assessed["context"]["ruleset"], "inputs_hash": _hash(a["payment"])})
        return a

    def act(self, alert_id: str, analyst: str, action: str, note: str = "") -> dict:
        if alert_id not in self.alerts:
            raise KeyError(alert_id)
        if action not in ACTIONS:
            raise ValueError(f"action must be one of {sorted(ACTIONS)}")
        if not analyst:
            raise ValueError("analyst id is required")
        a = self.alerts[alert_id]
        if action == "approve_block":
            if analyst in a["block_approvals"]:
                raise ValueError("four-eyes: the same analyst cannot approve a block twice")
            a["block_approvals"].append(analyst)
            a["status"] = "blocked" if len(a["block_approvals"]) >= 2 else "block_pending_second_approval"
        else:
            a["status"] = {"release": "released", "verify": "customer_verification", "hold": "held"}[action]
        a["actions"].append({"analyst": analyst, "action": action, "note": note})
        self._append({"type": "analyst_action", "alert_id": alert_id, "analyst": analyst, "action": action, "status": a["status"]})
        return a

    def verify_chain(self) -> bool:
        prev = "0" * 64
        for e in self.log:
            body = {k: v for k, v in e.items() if k != "hash"}
            if e["prev"] != prev or _hash(body) != e["hash"]:
                return False
            prev = e["hash"]
        return True

    def evidence(self, alert_id: str) -> dict:
        a = self.alerts[alert_id]
        linked = [b["alert_id"] for b in self.alerts.values() if b is not a and
                  {b["payment"]["from_account"], b["payment"]["to_account"]} & {a["payment"]["from_account"], a["payment"]["to_account"]}]
        pack = {"alert": a, "linked_alerts": linked, "audit_entries": [e for e in self.log if e.get("alert_id") == alert_id],
                "disclaimer": "Draft for compliance review. Risk points are rule outputs, not fraud probabilities."}
        pack["pack_hash"] = _hash(pack)
        return pack
