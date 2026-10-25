"""
Scam campaigns: link alerts that are the same operation hitting different people (the campaign clustering).

Two alerts join the same campaign when any of these hold:
  * same payee account (many victims paying one collector)
  * a payee identifier hashed out of one message (UPI ID, phone, link, account) appears in the other
  * near identical scripts: Jaccard similarity of 5 character shingles of the REDACTED messages >= 0.5
    (redaction turns amounts, links and phone numbers into placeholders, so one script sent to many people matches)
Union find over those links gives campaigns; only groups of two or more alerts are reported. Deterministic and
explainable (each campaign lists why its alerts are linked), which is why it is used instead of HDBSCAN: there is no
density parameter to tune and no labelled campaign data to tune it on.
"""
from __future__ import annotations
import hashlib
from collections import defaultdict

SIM = 0.5


def shingles(text: str, k: int = 5) -> set:
    t = " ".join((text or "").lower().split())
    return {t[i:i + k] for i in range(max(0, len(t) - k + 1))}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def build(alerts: list[dict]) -> list[dict]:
    """alerts: ReviewDesk alerts (payment + assessed.context with redacted_text and entity_hashes)."""
    n = len(alerts); parent = list(range(n)); why = defaultdict(set)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i

    def union(i, j, reason):
        a, b = find(i), find(j)
        if a != b:
            parent[a] = b
        why[(min(i, j), max(i, j))].add(reason)

    payee = [f'{a["payment"]["to_bank"]}:{a["payment"]["to_account"]}' for a in alerts]
    ctx = [a["assessed"].get("context", {}) for a in alerts]
    ents = [set(c.get("entity_hashes") or []) for c in ctx]
    sh = [shingles(c.get("redacted_text") or "") for c in ctx]
    by_payee = defaultdict(list)
    for i, p in enumerate(payee):
        by_payee[p].append(i)
    for idx in by_payee.values():
        for j in idx[1:]:
            union(idx[0], j, "same payee account")
    for i in range(n):
        for j in range(i + 1, n):
            if ents[i] & ents[j]:
                union(i, j, "same payee identifier in messages")
            if sh[i] and sh[j] and jaccard(sh[i], sh[j]) >= SIM:
                union(i, j, "near identical script")
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for root, idx in groups.items():
        if len(idx) < 2:
            continue
        members = [alerts[i] for i in idx]
        reasons = sorted({r for (i, j), rs in why.items() if find(i) == root for r in rs})
        cid = "CMP-" + hashlib.sha256("|".join(sorted(m["alert_id"] for m in members)).encode()).hexdigest()[:8].upper()
        scripts = [ctx[i].get("redacted_text") for i in idx if ctx[i].get("redacted_text")]
        out.append({"campaign_id": cid, "alerts": [m["alert_id"] for m in members], "size": len(members),
                    "victims": sorted({m["payment"]["from_account"] for m in members}),
                    "payees": sorted({payee[i] for i in idx}), "linked_by": reasons,
                    "total_amount": round(sum(float(m["payment"]["amount_paid"] or 0) for m in members), 2),
                    "first_seen": min(str(m["payment"]["timestamp"]) for m in members),
                    "last_seen": max(str(m["payment"]["timestamp"]) for m in members),
                    "sample_script": scripts[0] if scripts else None,
                    "statuses": sorted({m["status"] for m in members})})
    return sorted(out, key=lambda c: -c["size"])
