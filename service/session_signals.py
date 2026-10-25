"""
Session and identity domain: is the customer being coached right now, and is it really them?

Signals come from the bank's app (web, Android or iOS SDK) at payment time. Nothing here needs raw biometrics or
raw device identifiers: the device arrives as a salted hash, the SIM swap as "hours since the last swap" from the
telecom check the bank already runs.

Three parts, all readable rules:
  1. coaching cues   live call (long or video), screen share, remote access app, payee pasted in, long hesitation,
                     banking app sent to the background mid payment, typing rhythm unlike the customer's own
  2. identity gate   enrolled device hash vs this device, recent SIM swap: decided BEFORE the OTP is sent
  3. attack chains   ordered account events inside a window: password reset -> new device -> new payee -> large
                     transfer (full chain) or new device -> new payee -> transfer within an hour (short chain)

They produce SESSION points (cap 40) that join message, payment and graph points in context_rules.assess as a
fourth domain, so the two domain rule still holds: coaching cues alone can ask for verification (pause screen),
never an analyst hold. A coercion index in [0, 1] is also reported (weights follow the original design:
screen share 0.30, paste 0.20, hesitation 0.20, SIM swap 0.15, app hidden 0.10, network 0.05); at 0.65 or more it
forces at least verification, as the step up did.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict

SESSION_CAP = 40
LONG_CALL_MIN = 20          # a "digital arrest" keeps the victim on a call for a long time
HESITATION_S = 20
HIDDEN_S = 30
SIM_SWAP_RECENT_H = 72
CHAIN_WINDOW_MIN = 24 * 60
SHORT_CHAIN_MIN = 60
COERCION_STEP_UP = 0.65


@dataclass
class SessionSignals:
    session_id: str | None = None
    on_call: bool = False
    call_minutes: float = 0.0
    video_call: bool = False
    screen_share: bool = False
    remote_access_app: bool = False
    payee_pasted: bool = False
    hesitation_s: float = 0.0
    app_hidden_s: float = 0.0
    typing_anomaly: bool = False          # from behaviour.TypingProfile (experimental)
    device_hash: str | None = None
    enrolled_device_hashes: list = field(default_factory=list)
    sim_swap_hours: float | None = None   # hours since the last SIM swap; None = no swap known
    events: list = field(default_factory=list)   # [{"type": "password_reset"|"new_device"|"new_payee"|"limit_raise"|"transfer", "minute": int}]
    network_flag: bool = False            # payee is on the consortium mule list (set by the server)


@dataclass
class SessionResult:
    points: int
    cues: list
    identity: dict
    chain: dict
    coercion_index: float
    force_verify: bool

    def to_dict(self):
        return asdict(self)


def identity_check(s: SessionSignals) -> dict:
    """Run before the OTP is sent. A new, unenrolled device right after a SIM swap is the SIM swap takeover pattern:
    deny the OTP and require re-verification (video KYC or branch). This is an authentication step, not a payment block."""
    new_device = bool(s.device_hash) and bool(s.enrolled_device_hashes) and s.device_hash not in s.enrolled_device_hashes
    swap = s.sim_swap_hours is not None and s.sim_swap_hours < SIM_SWAP_RECENT_H
    if new_device and swap:
        return {"decision": "deny_otp_reverify", "new_device": True, "recent_sim_swap": True,
                "reason": f"unenrolled device within {SIM_SWAP_RECENT_H} h of a SIM swap"}
    if swap:
        return {"decision": "step_up", "new_device": new_device, "recent_sim_swap": True, "reason": "recent SIM swap"}
    if new_device:
        return {"decision": "step_up", "new_device": True, "recent_sim_swap": False, "reason": "unenrolled device"}
    return {"decision": "allow", "new_device": False, "recent_sim_swap": False, "reason": ""}


def attack_chain(events: list) -> dict:
    """Longest ordered chain among the events (by minute). Full: password_reset -> new_device -> new_payee -> transfer
    within 24 h. Short: new_device -> new_payee -> transfer within 60 min. limit_raise anywhere inside strengthens it."""
    ev = sorted((e for e in events or [] if "type" in e and "minute" in e), key=lambda e: e["minute"])

    def find(seq, window):
        for i, e in enumerate(ev):
            if e["type"] != seq[0]:
                continue
            k, t0, path = 1, e["minute"], [e["type"]]
            for f in ev[i + 1:]:
                if f["minute"] - t0 > window:
                    break
                if k < len(seq) and f["type"] == seq[k]:
                    path.append(f["type"]); k += 1
                    if k == len(seq):
                        return path, f["minute"] - t0
        return None, None

    full, span = find(["password_reset", "new_device", "new_payee", "transfer"], CHAIN_WINDOW_MIN)
    if full:
        return {"chain": "full", "steps": full, "minutes": span, "limit_raised": any(e["type"] == "limit_raise" for e in ev)}
    short, span = find(["new_device", "new_payee", "transfer"], SHORT_CHAIN_MIN)
    if short:
        return {"chain": "short", "steps": short, "minutes": span, "limit_raised": any(e["type"] == "limit_raise" for e in ev)}
    return {"chain": None, "steps": [], "minutes": None, "limit_raised": False}


def coercion_index(s: SessionSignals) -> float:
    w = (0.30 * (s.screen_share or s.remote_access_app) + 0.20 * s.payee_pasted + 0.20 * (s.hesitation_s >= HESITATION_S)
         + 0.15 * (s.sim_swap_hours is not None and s.sim_swap_hours < SIM_SWAP_RECENT_H) + 0.10 * (s.app_hidden_s >= HIDDEN_S)
         + 0.05 * s.network_flag)
    return round(min(1.0, w), 2)


def assess_session(s: SessionSignals | None) -> SessionResult | None:
    if s is None:
        return None
    cues = {}
    if s.remote_access_app:
        cues["remote_access_app"] = 20
    if s.screen_share:
        cues["screen_share"] = 15
    if s.on_call or s.video_call:
        cues["on_call_while_paying"] = 10
        if s.video_call or s.call_minutes >= LONG_CALL_MIN:
            cues["long_or_video_call"] = 5
    if s.payee_pasted:
        cues["payee_pasted"] = 10
    if s.hesitation_s >= HESITATION_S:
        cues["long_hesitation"] = 5         # weak alone: older and first time users also pause (see the fairness slices)
    if s.app_hidden_s >= HIDDEN_S:
        cues["app_hidden_mid_payment"] = 5
    if s.typing_anomaly:
        cues["typing_rhythm_unlike_customer"] = 5
    idn = identity_check(s)
    if idn["decision"] == "deny_otp_reverify":
        cues["sim_swap_takeover"] = 25
    elif idn["recent_sim_swap"]:
        cues["recent_sim_swap"] = 10
    elif idn["new_device"]:
        cues["unenrolled_device"] = 5
    ch = attack_chain(s.events)
    if ch["chain"] == "full":
        cues["attack_chain_full"] = 20
    elif ch["chain"] == "short":
        cues["attack_chain_short"] = 10
    ci = coercion_index(s)
    return SessionResult(points=min(SESSION_CAP, sum(cues.values())), cues=sorted(cues), identity=idn, chain=ch,
                         coercion_index=ci, force_verify=ci >= COERCION_STEP_UP)
