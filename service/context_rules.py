"""
RuseTect context layer (finale Part 1): consent, redaction, a keyword baseline and the interpretable
context + transaction + graph risk-point fusion described in the Round 1 PDF.

Contract (Round 1 PDF, Pitch 8 "Fusion contract"):
  * three signal domains, each capped: context 0-40, transaction 0-30, graph 0-30 -> total 0-100
  * bands: CRITICAL >= 85, HIGH >= 60, MEDIUM >= 40, else LOW
  * HIGH and CRITICAL need at least two domains with points (one domain alone caps at MEDIUM)
  * MEDIUM means customer verification; HIGH or CRITICAL means analyst escalation
  * risk points are deterministic rule outputs, NOT calibrated fraud probabilities
  * AI authorship adds zero points: RuseTect scores fraud intent and behaviour, not who wrote the text

Consent and privacy:
  * message context is processed only when the customer opted in (consent=True); otherwise only the
    payment and graph domains are scored
  * text is redacted before anything is stored: UPI IDs, phone numbers, e-mails, links, account numbers
    and OTP-like codes become typed placeholders; entities are kept only as salted hashes so a later
    payment to the same UPI ID / account can still be linked
  * every stored record carries an expiry (RETENTION_DAYS)

Pure Python, no model. Rules are frozen per release: RULESET_VERSION changes whenever a phrase or weight changes.
"""
from __future__ import annotations
import hashlib, os, re, unicodedata
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone

RULESET_VERSION = "ctx-1.0"
RETENTION_DAYS = 30
_SALT = os.environ.get("RUSETECT_HASH_SALT", "rusetect-demo-salt")

CONTEXT_CAP, TXN_CAP, GRAPH_CAP = 40, 30, 30
BANDS = ((85, "CRITICAL"), (60, "HIGH"), (40, "MEDIUM"), (0, "LOW"))
ACTION = {"LOW": "allow", "MEDIUM": "verify customer", "HIGH": "escalate to analyst", "CRITICAL": "escalate to analyst (priority)"}

# ------------------------------------------------------------------ redaction
_PATTERNS = [
    ("upi", re.compile(r"\b[a-z0-9._-]{2,64}@[a-z]{2,20}\b", re.I)),
    ("email", re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.I)),
    ("link", re.compile(r"(?:https?://|www\.)\S+", re.I)),
    ("phone", re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")),
    ("account", re.compile(r"(?<!\d)\d{9,18}(?!\d)")),
    ("code", re.compile(r"(?<!\d)\d{4,8}(?!\d)")),
]


def _h(kind: str, value: str) -> str:
    return kind + ":" + hashlib.sha256((_SALT + "|" + kind + "|" + value.lower()).encode()).hexdigest()[:16]


def redact(text: str) -> tuple[str, list[str]]:
    """Return (redacted text, salted entity hashes). Order matters: e-mail before UPI is not needed because
    UPI handles never contain a dot-TLD; links before accounts so digits inside URLs are not double counted."""
    out, ents = unicodedata.normalize("NFKC", text), []
    for kind, rx in (_PATTERNS[2], _PATTERNS[1], _PATTERNS[0], _PATTERNS[3], _PATTERNS[4], _PATTERNS[5]):
        def sub(m, kind=kind):
            if kind != "code":
                ents.append(_h(kind, m.group(0)))
            return f"[{kind.upper()}]"
        out = rx.sub(sub, out)
    return out, ents


# ------------------------------------------------------------------ keyword baseline (the thing we compare against)
BASELINE_TERMS = ("urgent", "otp", "password", "verify", "account blocked")


def keyword_baseline(text: str) -> dict:
    t = unicodedata.normalize("NFKC", text).lower()
    hits = [w for w in BASELINE_TERMS if re.search(r"\b" + re.escape(w) + r"\b", t)]
    return {"flag": bool(hits), "hits": hits}


# ------------------------------------------------------------------ context cue families (phrase proxy, frozen)
CUES = {
    "authority": (15, [r"\b(?:finance|accounts|payments?) (?:office|team|department|desk)\b", r"\bceo\b", r"\bmanaging director\b",
                       r"\b(?:bank|rbi|police|cbi|customs|income tax|cyber cell|trai|court) (?:officer|official|department|notice)\b",
                       r"\bcustomer (?:care|support) (?:team|executive|agent)\b", r"\bsupport (?:team|desk|executive)\b",
                       r"\b(?:on behalf of|calling from) (?:the )?(?:bank|rbi|police|government)\b"]),
    "redirection": (15, [r"\b(?:revised|updated|new|changed|replacement) (?:vendor |bank |account |payment )?(?:details|beneficiary|account|bank details)\b",
                         r"\bremit (?:the )?(?:balance|amount|payment)\b", r"\btransfer (?:it |the amount |the money )?to (?:this|the following|a safe|the new)\b",
                         r"\bsafe account\b", r"\bpay (?:to|into) (?:this|the new|a different)\b"]),
    "pressure": (10, [r"\b(?:before|ahead of) (?:the )?(?:afternoon |evening |end of day )?(?:cutoff|cut-off|deadline)\b", r"\bwithin (?:\d+|one|two|24) (?:minutes?|hours?)\b",
                      r"\bimmediately\b", r"\bright now\b", r"\blast chance\b", r"\bor (?:else )?(?:your|the) account will\b", r"\bdigital arrest\b", r"\btoday itself\b"]),
    "credential": (15, [r"\b(?:share|send|tell|give|read out) (?:me |us )?(?:the |your )?(?:otp|pin|password|cvv|upi pin)\b", r"\benter (?:your )?upi pin\b"]),
    "remote_access": (15, [r"\b(?:anydesk|teamviewer|quick ?support|rustdesk)\b", r"\bscreen ?shar(?:e|ing)\b"]),
    "refund": (15, [r"\brefund\b.*\b(?:scan|approve|collect request|enter)\b", r"\bcashback\b.*\bscan\b"]),
    "investment": (20, [r"\bguaranteed (?:returns?|profit)\b", r"\bdouble (?:your )?money\b", r"\b\d+% (?:daily|weekly|monthly) (?:returns?|profit)\b", r"\btrading (?:group|tips) (?:vip|premium)\b"]),
    "identity": (15, [r"\b(?:kyc|pan|aadhaar) (?:update|verification|expired|expiry|will expire)\b", r"\bre-?kyc\b"]),
    "secrecy": (10, [r"\bdo not (?:tell|inform|share this with) (?:anyone|your family|the bank)\b", r"\bkeep (?:this|it) confidential\b"]),
}


def context_cues(text: str) -> dict:
    t = unicodedata.normalize("NFKC", text).lower()
    fam = {name: w for name, (w, pats) in CUES.items() if any(re.search(p, t) for p in pats)}
    pts = min(CONTEXT_CAP, sum(fam.values()))
    return {"points": pts, "families": sorted(fam)}


# ------------------------------------------------------------------ transaction and graph checks
@dataclass
class PaymentContext:
    new_beneficiary: bool = False
    amount_deviation: bool = False       # amount outside the payer's usual range (e.g. > 3x their median)
    rapid_pass_through: bool = False     # payee forwards most of what it received within minutes
    fan_in_24h: int = 0                  # distinct senders paying the payee in 24 h
    coordinated: bool = False            # payee or device shared by several flagged accounts
    payee_in_message: bool = False       # payee UPI / account matches an entity hash from the message


def transaction_points(c: PaymentContext) -> dict:
    parts = {}
    if c.new_beneficiary: parts["new_beneficiary"] = 15
    if c.amount_deviation: parts["amount_deviation"] = 15
    if c.payee_in_message: parts["payee_named_in_message"] = 15
    return {"points": min(TXN_CAP, sum(parts.values())), "checks": sorted(parts)}


def graph_points(c: PaymentContext) -> dict:
    parts = {}
    if c.fan_in_24h >= 4: parts["fan_in_4plus"] = 15
    if c.rapid_pass_through: parts["rapid_pass_through"] = 15
    if c.coordinated: parts["coordinated_accounts"] = 15
    return {"points": min(GRAPH_CAP, sum(parts.values())), "checks": sorted(parts)}


# ------------------------------------------------------------------ fusion
def band_for(total: int, domains_with_points: int) -> str:
    b = next(name for floor, name in BANDS if total >= floor)
    if domains_with_points < 2 and b in ("HIGH", "CRITICAL"):
        b = "MEDIUM"
    return b


@dataclass
class ContextResult:
    context: int
    transaction: int
    graph: int
    total: int
    band: str
    action: str
    reasons: list = field(default_factory=list)
    consent: bool = True
    redacted_text: str | None = None
    entity_hashes: list = field(default_factory=list)
    keyword_baseline: dict = field(default_factory=dict)
    ruleset: str = RULESET_VERSION
    expires_at: str | None = None

    def to_dict(self):
        return asdict(self)


def assess(text: str | None, payment: PaymentContext, consent: bool, now: datetime | None = None) -> ContextResult:
    """Score one payment with optional message context. Text is used only if consent is True."""
    now = now or datetime.now(timezone.utc)
    ctx = {"points": 0, "families": []}; red, ents, base = None, [], {}
    if consent and text:
        red, ents = redact(text)
        ctx = context_cues(text)
        base = keyword_baseline(text)
    tx, gr = transaction_points(payment), graph_points(payment)
    total = ctx["points"] + tx["points"] + gr["points"]
    n_dom = sum(1 for p in (ctx["points"], tx["points"], gr["points"]) if p > 0)
    band = band_for(total, n_dom)
    reasons = ([f"message cue: {f}" for f in ctx["families"]] + [f"payment check: {c}" for c in tx["checks"]]
               + [f"account graph: {c}" for c in gr["checks"]])
    if not consent and text:
        reasons.append("message context not used: no consent")
    return ContextResult(ctx["points"], tx["points"], gr["points"], total, band, ACTION[band], reasons, consent, red, ents, base,
                         RULESET_VERSION, (now + timedelta(days=RETENTION_DAYS)).isoformat(timespec="seconds"))


# ------------------------------------------------------------------ combining with the payment cascade
PAYMENT_RANK = {"Approve": 0, "Step-up verification": 1, "Hold for analyst": 2, "Block + STR draft": 3}
CONTEXT_TO_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 2}   # context never auto-blocks; analyst decides


def combine(payment_decision: str, ctx: ContextResult) -> dict:
    """Context veto: a low payment score must not suppress a separately supported context alert.
    The final action is the stricter of the two, and context alone never goes beyond an analyst hold."""
    names = {v: k for k, v in PAYMENT_RANK.items()}
    p, c = PAYMENT_RANK.get(payment_decision, 0), CONTEXT_TO_RANK[ctx.band]
    final = names[max(p, c)]
    return {"payment_decision": payment_decision, "context_band": ctx.band, "final_decision": final,
            "raised_by_context": c > p}
