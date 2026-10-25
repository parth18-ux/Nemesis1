"""
What the customer sees when RuseTect asks for verification, and the duress PIN.

Pause screen: one question chosen from the evidence (a call or screen share, a "digital arrest" or police threat,
a KYC or refund lure, an investment promise), a one line fact, two buttons: stop the payment, or continue after a
30 minute cool off. English and Hindi; more languages are just more strings. The analyst still sees the alert.

Duress PIN: a second PIN the customer sets in advance. Entered under pressure, the app shows "payment successful",
the money is held, trusted contacts and the fraud desk are alerted, and the alert is raised to the top of the queue.
PINs are stored only as salted PBKDF2 hashes.
"""
from __future__ import annotations
import hashlib, hmac, os, time

COOL_OFF_MIN = 30

TEXT = {
    "call": {
        "en": ("Is someone on the phone or sharing your screen asking you to make this payment?",
               "Banks never ask you to move money or share your screen to \"verify\" anything."),
        "hi": ("क्या कोई फ़ोन पर है या आपकी स्क्रीन देख रहा है और यह भुगतान करने को कह रहा है?",
               "बैंक कभी भी पैसे भेजने या स्क्रीन शेयर करने को नहीं कहते।")},
    "arrest": {
        "en": ("Has anyone said you are under investigation or \"digital arrest\" and must pay?",
               "Police, CBI and courts never ask for money on a call. There is no such thing as digital arrest."),
        "hi": ("क्या किसी ने कहा कि आप जाँच या \"डिजिटल अरेस्ट\" में हैं और पैसे देने होंगे?",
               "पुलिस, सीबीआई या अदालत कभी फ़ोन पर पैसे नहीं माँगते। डिजिटल अरेस्ट जैसी कोई चीज़ नहीं होती।")},
    "kyc": {
        "en": ("Did someone ask you to pay to unblock your account, update KYC or receive a refund?",
               "You never pay or enter your PIN to receive money or to update KYC."),
        "hi": ("क्या किसी ने खाता चालू करने, केवाईसी अपडेट करने या रिफ़ंड पाने के लिए पैसे भेजने को कहा?",
               "पैसे पाने या केवाईसी अपडेट के लिए कभी भुगतान या पिन डालना नहीं पड़ता।")},
    "investment": {
        "en": ("Were you promised guaranteed or very high returns for this payment?",
               "Guaranteed returns are a common scam. Check the company on SEBI's website first."),
        "hi": ("क्या इस भुगतान पर आपको पक्का या बहुत ज़्यादा मुनाफ़ा देने का वादा किया गया है?",
               "पक्के मुनाफ़े का वादा अक्सर धोखा होता है। पहले सेबी की वेबसाइट पर कंपनी जाँचें।")},
    "family": {
        "en": ("Has someone claiming to be family or a friend asked for money urgently?",
               "Hang up and call them back on a number you already know before paying."),
        "hi": ("क्या परिवार या दोस्त बनकर किसी ने तुरंत पैसे माँगे हैं?",
               "पैसे भेजने से पहले फ़ोन काटें और उन्हें अपने पास पहले से मौजूद नंबर पर वापस कॉल करें।")},
    "default": {
        "en": ("Do you know and trust the person you are paying?",
               "If anyone is rushing you or telling you to keep this secret, stop and call your bank."),
        "hi": ("क्या आप उस व्यक्ति को जानते हैं और उस पर भरोसा करते हैं जिसे आप भुगतान कर रहे हैं?",
               "अगर कोई जल्दी करवा रहा है या इसे छिपाने को कह रहा है, तो रुकें और अपने बैंक को फ़ोन करें।")},
}
BUTTONS = {"en": ("Yes, stop this payment", f"No, continue in {COOL_OFF_MIN} min"),
           "hi": ("हाँ, यह भुगतान रोकें", f"नहीं, {COOL_OFF_MIN} मिनट बाद जारी रखें")}
TITLE = {"en": "Pause for a moment", "hi": "एक पल रुकिए"}


WORKFLOW_QUESTION = {   # message model workflow -> question (service/scam_text.py)
    "digital_arrest": "arrest", "parcel_customs": "arrest", "sextortion": "arrest",
    "remote_access": "call", "fake_support": "call",
    "kyc_block": "kyc", "upi_collect_refund": "kyc", "reward_points": "kyc", "govt_scheme": "kyc", "utility_disconnect": "kyc",
    "lottery_prize": "kyc", "fake_loan": "kyc",
    "investment": "investment", "task_job": "investment", "family_emergency": "family",
}


def pick_question(reasons: list[str], workflow: str | None = None) -> str:
    """The most specific evidence wins: a named scam workflow from the message model, then live call or screen share
    cues from the session, then message cue families."""
    if workflow in WORKFLOW_QUESTION:
        return WORKFLOW_QUESTION[workflow]
    r = " ".join(reasons).lower()
    if "pressure" in r and ("authority" in r or "arrest" in r):
        return "arrest"
    if any(k in r for k in ("on_call", "screen_share", "remote_access", "long_or_video_call")):
        return "call"
    if "investment" in r:
        return "investment"
    if any(k in r for k in ("identity", "refund", "credential")):
        return "kyc"
    return "default"


def plan(final_decision: str, reasons: list[str], identity: dict | None = None, workflow: str | None = None) -> dict:
    """What the bank app should show for this decision."""
    if identity and identity.get("decision") == "deny_otp_reverify":
        return {"show": "reverify", "otp": "withheld",
                "message": {"en": "For your safety we could not send an OTP to this device. Please re-verify by video KYC or at your branch.",
                            "hi": "आपकी सुरक्षा के लिए इस डिवाइस पर ओटीपी नहीं भेजा गया। कृपया वीडियो केवाईसी या शाखा में दोबारा सत्यापन करें।"}}
    if final_decision == "Approve":
        return {"show": "none"}
    key = pick_question(reasons, workflow)
    screen = {lang: {"title": TITLE[lang], "question": TEXT[key][lang][0], "fact": TEXT[key][lang][1],
                     "stop": BUTTONS[lang][0], "continue": BUTTONS[lang][1]} for lang in ("en", "hi")}
    held = final_decision != "Step-up verification"
    return {"show": "pause", "question_type": key, "cool_off_minutes": COOL_OFF_MIN, "screen": screen,
            "step_up": ["voice or face check"] if not held else [],
            "funds": "held for analyst review" if held else "released after the cool off unless the customer stops it"}


class DuressVault:
    def __init__(self):
        self.reset()

    def reset(self):
        self.users, self.notifications = {}, []

    @staticmethod
    def _hash(pin: str, salt: bytes) -> bytes:
        return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt, 120_000)

    def register(self, account: str, normal_pin: str, duress_pin: str, trusted_contacts: list[str]):
        if normal_pin == duress_pin:
            raise ValueError("the duress PIN must differ from the normal PIN")
        if not (4 <= len(normal_pin) <= 6 and normal_pin.isdigit() and 4 <= len(duress_pin) <= 6 and duress_pin.isdigit()):
            raise ValueError("PINs are 4 to 6 digits")
        salt = os.urandom(16)
        self.users[account] = {"salt": salt, "normal": self._hash(normal_pin, salt), "duress": self._hash(duress_pin, salt),
                               "contacts": list(trusted_contacts)[:5]}
        return {"account": account, "trusted_contacts": len(self.users[account]["contacts"])}

    def confirm(self, account: str, pin: str) -> dict:
        """Returns what the customer sees and what really happens. Constant time comparison."""
        u = self.users.get(account)
        if not u:
            return {"result": "not_registered"}
        h = self._hash(pin, u["salt"])
        if hmac.compare_digest(h, u["duress"]):
            note = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "account": account,
                    "to": u["contacts"], "message": "Possible coercion: a payment was confirmed with the duress PIN. The money is on hold."}
            self.notifications.append(note)
            return {"result": "duress", "shown_to_customer": "Payment successful", "funds": "held",
                    "trusted_contacts_alerted": len(u["contacts"]), "fraud_desk": "priority alert"}
        if hmac.compare_digest(h, u["normal"]):
            return {"result": "ok"}
        return {"result": "wrong_pin"}
