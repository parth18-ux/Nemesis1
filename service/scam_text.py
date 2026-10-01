"""
RuseTect L-1 lure detector: scores the message side of a scam (SMS, WhatsApp, e-mail, call transcript).

The payment cascade (L0-L4) only sees money. Most Indian scams start earlier, with a message or call that
pushes the victim to pay, share an OTP or install a remote-access app. This module scores that text.

  featurize()   hashed word 1-2 grams + character 3-5 grams, sublinear TF x IDF, L2-normalised.
                Used by both ml/train_message_model.py and the API, so offline and online scores match exactly.
  tag()         rule-based explanation: which scam workflow the text resembles and which pressure tactics it uses.
  entities()    UPI IDs, phone numbers, links, account numbers and amounts mentioned in the text.
  MessageModel  loads service/models/message_model.json (logistic regression weights) and scores a text.

Pure Python + NumPy, no scikit-learn at serve time.
"""
from __future__ import annotations
import json, math, re, unicodedata, zlib
from pathlib import Path

N_BUCKETS = 1 << 18
MODELS = Path(__file__).parent / "models"

_URL = re.compile(r"(?:https?://|www\.)\S+|\b[a-z0-9-]+\.(?:com|in|net|org|co|xyz|top|info|link|ly|me|site|online|app)(?:/\S*)?\b", re.I)
_UPI = re.compile(r"\b[a-z0-9._-]{2,64}@(?:ok[a-z]+|ybl|ibl|axl|paytm|upi|apl|ptyes|ptsbi|pthdfc|ptaxis|sbi|icici|hdfcbank|axisbank|kotak|yesbank|fbl|idfcbank|jupiteraxis|slc|waaxis|freecharge|airtel|jio)\b", re.I)
_PHONE = re.compile(r"(?<!\d)(?:\+?91[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)|(?<!\d)(?:1800|1860)[\s-]?\d{3}[\s-]?\d{4}(?!\d)")
_ACCT = re.compile(r"(?<!\d)\d{9,18}(?!\d)")
_AMOUNT = re.compile(r"(?:rs\.?|inr|₹|£|\$)\s?([\d,]+(?:\.\d+)?)\s*(lakh|lac|crore|k)?", re.I)


def normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    t = _URL.sub(" _url_ ", t)
    t = _UPI.sub(" _upi_ ", t)
    t = _PHONE.sub(" _phone_ ", t)
    t = re.sub(r"(?:rs\.?|inr|₹|£|\$)\s?[\d,]+(?:\.\d+)?", " _money_ ", t)
    t = re.sub(r"\d+", "0", t)
    return re.sub(r"\s+", " ", t).strip()


def _grams(norm: str):
    words = re.findall(r"[a-z_0]+|[ऀ-ॿ]+", norm)
    for w in words:
        yield "w:" + w
    for a, b in zip(words, words[1:]):
        yield "b:" + a + " " + b
    s = " " + norm + " "
    for n in (3, 4, 5):
        for i in range(len(s) - n + 1):
            yield "c:" + s[i:i + n]


def counts(text: str) -> dict:
    """Hashed term counts {bucket: count} for one text."""
    out: dict = {}
    for g in _grams(normalize(text)):
        h = zlib.crc32(g.encode()) & (N_BUCKETS - 1)
        out[h] = out.get(h, 0) + 1
    return out


def featurize(text: str, idf) -> dict:
    """Sparse TF-IDF vector {bucket: weight}, L2-normalised. idf: indexable by bucket."""
    v = {h: (1.0 + math.log(c)) * float(idf[h]) for h, c in counts(text).items()}
    n = math.sqrt(sum(x * x for x in v.values())) or 1.0
    return {h: x / n for h, x in v.items()}


# ---------------------------------------------------------------- explanations
# Each workflow: (label, plain-language description, patterns). Patterns are matched on the lower-cased raw text.
WORKFLOWS = {
    "digital_arrest": ("Digital arrest / fake police", "Caller poses as CBI, ED, customs, police or TRAI and threatens arrest unless money is moved to a 'verification' account.",
                       [r"digital arrest", r"\b(cbi|ed|ncb|crime branch|cyber cell|trai|customs officer|mumbai police|delhi police)\b", r"arrest warrant|money laundering case|fir (has been|is) (registered|filed)", r"stay on (the )?video call|do not disconnect", r"(rbi|safe|secure|verification) account"]),
    "parcel_customs": ("Parcel / courier seizure", "Claims a parcel in your name holds drugs or fake passports, then hands you to a fake officer.",
                       [r"\b(parcel|courier|package|consignment)\b.*\b(seized|drugs|mdma|illegal|passports?|customs)\b", r"\b(fedex|dhl|blue ?dart)\b.*\b(seized|held|returned|illegal)\b"]),
    "kyc_block": ("KYC / account block", "Says your bank account, card, wallet or SIM will be blocked today unless you 'update KYC' through a link or call.",
                  [r"\b(kyc|pan|aadhaa?r)\b.*\b(update|expire|expired|pending|suspend|block)", r"(account|a/c|card|wallet|yono|sim).{0,40}(will be|has been|is) (blocked|suspended|deactivated|closed)", r"(block|suspend)ed (today|within)"]),
    "upi_collect_refund": ("Collect-request / refund trick", "Asks you to 'approve' a request or enter your UPI PIN to receive a refund, prize or payment. A PIN is never needed to receive money.",
                           [r"(enter|put|type).{0,20}(upi )?pin.{0,30}(receive|get|credit|refund)", r"(approve|accept).{0,30}(request|collect).{0,30}(receive|refund|cashback|credit)", r"scan.{0,20}qr.{0,30}(receive|get)"]),
    "remote_access": ("OTP / screen-share / app install", "Asks for an OTP or to install AnyDesk, TeamViewer, QuickSupport or an .apk, which hands control of your phone to the scammer.",
                      [r"\b(anydesk|teamviewer|quick ?support|rustdesk|airdroid)\b", r"\.apk\b", r"(share|send|tell|read out|forward).{0,25}\b(otp|code|pin)\b", r"screen ?shar"]),
    "task_job": ("Task / part-time job", "Offers easy daily earnings for liking videos, rating hotels or 'tasks', then asks for deposits to unlock withdrawals.",
                 [r"part[ -]?time", r"(like|rate|review|subscribe).{0,30}(videos?|hotels?|products?|channels?)", r"(earn|income|salary).{0,30}(daily|per day|/day|from home)", r"\b(prepaid|merchant) task", r"\btelegram\b.*\b(task|job|hr)\b"]),
    "investment": ("Fake investment / trading group", "Promises guaranteed or very high returns via a WhatsApp/Telegram 'stock tips' group, IPO quota or crypto app.",
                   [r"guaranteed (returns?|profit)", r"\d{2,4}\s?% (returns?|profit)", r"(stock|share|ipo|crypto|trading|forex).{0,40}(tips|group|vip|institutional|quota|allotment|profit)", r"(sebi|nse|bse).{0,20}(registered|approved)"]),
    "utility_disconnect": ("Bill / power disconnection", "Says electricity, gas or mobile service will be cut tonight unless you call a number or pay through a link.",
                           [r"(electricity|power|light|bijli|gas|connection).{0,40}(disconnect|cut|kaat|band)", r"(previous|last) month bill (was )?not (updated|paid)", r"tonight.{0,20}(9|10)[:.]?\d{0,2}\s?(pm)?"]),
    "fake_loan": ("Instant loan fee", "Pre-approved loan with no documents, but a 'processing', 'insurance' or 'GST' fee must be paid first.",
                  [r"(loan|credit).{0,40}(approved|sanctioned|pre-?approved)", r"(processing|file|insurance|gst|registration) (fee|charge)", r"(without|no) (cibil|documents?|income proof)"]),
    "lottery_prize": ("Lottery / prize / KBC", "You have 'won' a prize, lottery, car or lucky draw and must pay tax or fees to claim it.",
                      [r"\b(kbc|lottery|lucky draw|jackpot|lucky winner)\b", r"\b(you|u) (have )?(won|win)\b", r"(claim|collect).{0,20}(prize|reward|gift)"]),
    "family_emergency": ("Family emergency / voice clone", "A relative or friend (possibly an AI-cloned voice) is 'in trouble' and needs money urgently and secretly.",
                         [r"(papa|mummy|mom|dad|beta|bhai|uncle|mama|son|daughter).{0,60}(trouble|accident|hospital|police|arrest|caught|pakad)", r"(don'?t|do not|mat) (tell|batana|bata)", r"new number|naya number|this is my new"]),
    "sextortion": ("Sextortion threat", "Threatens to leak a recorded video call or photos unless paid.",
                   [r"(video|photos?|recording|clip).{0,40}(viral|leak|upload|send to your (family|contacts|friends))", r"(nude|obscene|objectionable|private) (video|photos?)"]),
    "govt_scheme": ("Fake government scheme / tax refund", "Fake PM Kisan, subsidy, income-tax refund or free-laptop scheme asking for details or a fee.",
                    [r"(pm kisan|subsidy|yojana|free laptop|income tax refund|it refund|tax refund|epfo|pension)", r"(government|govt|sarkari).{0,30}(scheme|benefit|grant)"]),
    "reward_points": ("Reward points / cashback expiry", "Card reward points or cashback 'expiring today', redeem through a link.",
                      [r"(reward|redeem|loyalty) points?", r"cashback.{0,30}(expir|claim|credited)"]),
    "fake_support": ("Fake customer care", "Fake helpline or refund agent who then asks for OTP, card details or remote access.",
                     [r"customer (care|support|executive)", r"(refund|complaint|order).{0,40}(call|contact|whatsapp).{0,20}(_phone_|\d{10})"]),
}
TACTICS = {
    "urgency": ("Urgency", [r"\b(urgent|immediately|within \d+ (hours?|hrs|minutes?|mins)|today|tonight|last (date|day|chance)|expir|abhi|turant|jaldi|asap)\b"]),
    "authority": ("Impersonates authority", [r"\b(rbi|cbi|police|court|income tax|customs|trai|sebi|npci|government|govt|officer|ministry|uidai)\b"]),
    "threat": ("Threat or fear", [r"\b(arrest|blocked?|suspend|legal action|penalty|jail|fir|warrant|disconnect|seized|viral|leak)\b"]),
    "secrecy": ("Asks for secrecy", [r"(don'?t|do not|mat) (tell|share with|inform)|keep (it|this) (secret|confidential)|kisi ko (mat|na)"]),
    "credential": ("Asks for OTP / PIN / card", [r"(share|send|tell|enter|provide|confirm|read out|forward).{0,25}\b(otp|upi pin|pin|cvv|password|card (number|details))\b"]),
    "remote_app": ("Remote-access app", [r"\b(anydesk|teamviewer|quick ?support|rustdesk|airdroid)\b|\.apk\b"]),
    "payment_ask": ("Asks you to pay", [r"\b(pay|transfer|send|deposit)\b.{0,30}(rs\.?|₹|inr|amount|fee|money|charges?|\d)", r"(processing|registration|gst|tax) (fee|charge)"]),
    "too_good": ("Too good to be true", [r"guaranteed|\bwon\b|\bwinner\b|free gift|\d{2,4}\s?% (profit|returns?)|double (your )?money|per day"]),
    "link": ("Link to click", [_URL.pattern]),
    "off_platform": ("Moves you to WhatsApp / Telegram", [r"\b(whatsapp|telegram|signal)\b"]),
}
_W = {k: (lab, desc, [re.compile(p, re.I) for p in pats]) for k, (lab, desc, pats) in WORKFLOWS.items()}
_T = {k: (lab, [re.compile(p, re.I) for p in pats]) for k, (lab, pats) in TACTICS.items()}
_SAFE_OTP = re.compile(r"(do not|don'?t|never) (share|disclose)|(bank|we) (will )?never (ask|call)", re.I)


# Asks that RBI / NPCI / banks state are never legitimate. Any one of them puts a message at least in the caution band,
# whatever the model says. Kept short on purpose: each entry must be something no genuine sender ever writes.
HARD_SIGNALS = {
    "remote_app": (r"\b(anydesk|teamviewer|quick ?support|rustdesk|airdroid)\b", "Asks you to install a remote-access app"),
    "apk": (r"\.apk\b", "Sends an .apk app file"),
    "pin_to_receive": (r"\b(upi pin|pin|approve|accept)\b.{0,40}\b(to receive|receive|get it|credit)", "Asks for a UPI PIN or approval to RECEIVE money"),
    "digital_arrest": (r"digital arrest|stay on (the )?video call|do not disconnect", "'Digital arrest': no police or agency arrests over video call"),
    "safe_account": (r"(rbi|safe|secure|verification) account", "Asks you to move money to a 'safe' or 'RBI' account"),
    "share_otp": (r"(share|send|tell|read out|forward)( me| us)?( the| your)? .{0,15}\b(otp|code)\b", "Asks you to share an OTP or code"),
}
_H = {k: (re.compile(p, re.I), lab) for k, (p, lab) in HARD_SIGNALS.items()}


def hard_signals(text: str) -> list:
    low = text.lower()
    low = _SAFE_OTP.sub(" ", low)                                       # genuine OTP SMS: "do not share this OTP"
    low = re.sub(r"share (the )?otp \d* ?only with the delivery", " ", low)  # genuine delivery-OTP instruction
    return [{"id": k, "label": lab} for k, (p, lab) in _H.items() if p.search(low)]


def tag(text: str) -> dict:
    """Workflows the text resembles (best first) and the pressure tactics it uses."""
    low = text.lower()
    hits = []
    for k, (lab, desc, pats) in _W.items():
        n = sum(bool(p.search(low)) for p in pats)
        if n:
            hits.append((n, k, lab, desc))
    hits.sort(key=lambda h: -h[0])
    tactics = [{"id": k, "label": lab} for k, (lab, pats) in _T.items() if any(p.search(low) for p in pats)]
    if _SAFE_OTP.search(low):          # "do not share your OTP" is a bank's warning, not a credential ask
        tactics = [t for t in tactics if t["id"] != "credential"]
    return {"workflows": [{"id": k, "label": lab, "description": d, "matches": n} for n, k, lab, d in hits[:3]],
            "tactics": tactics}


def entities(text: str) -> dict:
    amounts = []
    for num, unit in _AMOUNT.findall(text):
        try:
            v = float(num.replace(",", ""))
        except ValueError:
            continue
        v *= {"lakh": 1e5, "lac": 1e5, "crore": 1e7, "k": 1e3}.get((unit or "").lower(), 1)
        amounts.append(v)
    return {"upi_ids": sorted({m.lower() for m in _UPI.findall(text)}),
            "phones": sorted({re.sub(r"\D", "", m)[-10:] for m in _PHONE.findall(text)}),
            "links": sorted({m.rstrip(".,)") for m in _URL.findall(text)}),
            "accounts": sorted(set(_ACCT.findall(_PHONE.sub(" ", text)))),
            "amounts": amounts}


# ---------------------------------------------------------------- model
_STOP = set("a an the to of in on at by for from and or is are was be been your you yr ur u i me my we our us it this that with as "
             "will can has have had do not no so if but please pls sir madam dear hi hello 0 0th".split())
_PRETTY = {"_money_": "₹ amount", "_url_": "link", "_upi_": "UPI ID", "_phone_": "phone number"}


class MessageModel:
    def __init__(self, path: Path = MODELS / "message_model.json"):
        import numpy as np
        m = json.loads(Path(path).read_text())
        self.idf = np.ones(N_BUCKETS, dtype=np.float32)
        self.w = np.zeros(N_BUCKETS, dtype=np.float32)
        idx = np.array(m["buckets"], dtype=np.int64)
        self.idf[idx] = np.array(m["idf"], dtype=np.float32)
        self.idf[np.setdiff1d(np.arange(N_BUCKETS), idx)] = m["idf_unseen"]
        self.w[idx] = np.array(m["coef"], dtype=np.float32)
        self.b = float(m["intercept"]); self.thr = m["thresholds"]; self.meta = m.get("meta", {})

    def prob(self, text: str) -> float:
        v = featurize(text, self.idf)
        z = self.b + sum(x * float(self.w[h]) for h, x in v.items())
        return 1.0 / (1.0 + math.exp(-z))

    def top_terms(self, text: str, k: int = 5) -> list:
        """Hashed n-grams with the largest positive contribution, mapped back to readable words."""
        v = featurize(text, self.idf); norm = normalize(text); seen = {}
        for g in _grams(norm):
            if g.startswith("c:"):
                continue
            h = zlib.crc32(g.encode()) & (N_BUCKETS - 1)
            words = g[2:].split()
            if h in v and not all(w in _STOP or len(w) < 3 for w in words):
                seen[" ".join(_PRETTY.get(w, w) for w in words)] = v[h] * float(self.w[h])
        return [t for t, c in sorted(seen.items(), key=lambda x: -x[1])[:k] if c > 0]

    def score(self, text: str) -> dict:
        p = self.prob(text); tg = tag(text); hard = hard_signals(text)
        if p >= self.thr["high"] or (hard and p >= self.thr["caution"]) or len(hard) >= 2:
            band, action = "high", "Warn the user now and add a cool-off to payments to new payees for 24h"
        elif p >= self.thr["caution"] or hard:
            band, action = "caution", "Show an inline caution; escalate only if a payment follows"
        else:
            band, action = "safe", "No action"
        return {"risk": round(max(p, self.thr["caution"]) if hard else p, 4), "band": band, "action": action, "workflow": tg["workflows"][0] if tg["workflows"] and band != "safe" else None,
                "other_workflows": tg["workflows"][1:] if band != "safe" else [], "tactics": tg["tactics"],
                "key_phrases": self.top_terms(text) if band != "safe" else [], "hard_signals": hard,
                "model_risk": round(p, 4), "entities": entities(text)}
