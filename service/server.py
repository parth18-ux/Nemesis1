"""
RuseTect API: layered filtration cascade for mule accounts and scam money flows.

  /                    landing page (links to the console, report and API docs)
  /console             Live Console: replays recorded IBM AMLworld (synthetic) test-period payments through the live path
  /report              benchmark report (21 AML models, 8 onboarding models)
  /api/health          engine status
  /api/v2/message      L-1 lure detector: score an SMS / WhatsApp / e-mail / call transcript, decide whether to warn
  /api/v2/payment      score one payment (L1 -> L2 -> decision band -> L4 case), escalated by recent lures
  /api/v2/alerts/summary  alert-fatigue counters (pushed vs suppressed, and why)
  /api/v2/payments/batch
  /api/v2/cases        current ring cases
  /api/v2/onboard      L0 onboarding gate for a new account application (Feedzai BAF schema)
  /api/v2/model-card   all benchmark numbers
  /docs                OpenAPI docs
"""
import calendar, csv, gzip, logging, os, random, time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from branding import NAME, SLUG, TAGLINE, VERSION
from lure_fusion import LureMemory
from rusetect_engine import Engine
from scam_text import MessageModel
import context_rules as CTX
from review import ReviewDesk
from session_signals import SessionSignals, assess_session, identity_check
from intervention import DuressVault, plan as intervention_plan
from coordination import CoordinationTracker
from security import DEMO_MODE, DEMO_KEYS, NonceCache, RateLimiter, Signer, allowed, analyst_for
from consortium import Consortium
from risk_score import display_score
from behaviour import TypingProfile
import campaigns as CAMPAIGNS
import str_report

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
log = logging.getLogger(SLUG)

app = FastAPI(title=f"{NAME} API", version=VERSION,
              description=f"{TAGLINE}. Live path: L1 XGBoost gatekeeper, L2 XGBoost + LightGBM average, L4 ring linker; "
                          "L0 LightGBM scores account applications. Trained and evaluated on synthetic datasets (IBM AMLworld, Feedzai BAF).")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
api = APIRouter(prefix="/api")

try:
    ENGINE = Engine(MODELS)
except Exception as e:  # pragma: no cover
    log.warning("%s engine unavailable: %s", NAME, e)
    ENGINE = None


try:
    MESSAGES = MessageModel(MODELS / "message_model.json")
except Exception as e:  # pragma: no cover
    log.warning("%s message model unavailable: %s", NAME, e)
    MESSAGES = None
LURES = LureMemory()
DESK = ReviewDesk()
COORD = CoordinationTracker()
DURESS = DuressVault()
SIGNER = Signer()
NONCES = NonceCache()
RATE = RateLimiter(limit=int(os.environ.get("RUSETECT_RATE_LIMIT", "240")))   # requests per 10 s per client on POST /api
CONSORTIUM = Consortium()
THIS_BANK = "BANK-A"
PROFILES: dict = {}


@app.middleware("http")
async def rate_limit(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        client = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0].strip()
        if not RATE.allow(client):
            return JSONResponse({"detail": "rate limit: too many requests, slow down"}, status_code=429)
    return await call_next(request)


def need_engine() -> Engine:
    if ENGINE is None:
        raise HTTPException(503, f"{NAME} models not loaded")
    return ENGINE


# ---------------------------------------------------------------- pages



@app.get("/")
async def landing():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/console")
async def console():
    return FileResponse(ROOT / "static" / "console.html")


@app.get("/investigate", include_in_schema=False)
async def investigate():
    return FileResponse(ROOT / "static" / "investigate.html")


@app.get("/favicon.svg")
async def favicon():
    return FileResponse(ROOT / "static" / "favicon.svg", media_type="image/svg+xml")


@app.get("/report")
async def report():
    return FileResponse(ROOT / "static" / "report.html")


@api.get("/")
@api.get("/health")
async def health():
    return {"service": f"{NAME} API", "name": NAME, "id": SLUG, "version": VERSION, "status": "ok", "engine_loaded": ENGINE is not None,
            "ruleset": CTX.RULESET_VERSION, "auth": "demo keys (set RUSETECT_ANALYST_KEYS)" if DEMO_MODE else "configured keys",
            "signing": {"alg": "Ed25519", "key_id": SIGNER.key_id, "ephemeral": SIGNER.ephemeral}}


# ---------------------------------------------------------------- scoring
class PaymentIn(BaseModel):
    timestamp: str = Field(description="YYYY-MM-DD HH:MM (UTC)", examples=["2022-09-09 10:00"])
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_paid: float = Field(gt=0)
    payment_currency: str = "US Dollar"
    receiving_currency: Optional[str] = None
    payment_format: str = Field("ACH", description="ACH, Bitcoin, Cash, Cheque, Credit Card, Reinvestment, Wire")


def to_payload(p: PaymentIn) -> dict:
    ts = datetime.strptime(p.timestamp.replace("/", "-"), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
    d = p.model_dump()
    d["receiving_currency"] = p.receiving_currency or p.payment_currency
    d["tmin"] = int(ts.timestamp()) // 60
    return d


def score_with_context(eng: Engine, p: PaymentIn) -> dict:
    d = to_payload(p)
    return LURES.apply_to_payment(d, eng.score_payment(d))


@api.post("/v2/payment")
async def v2_payment(p: PaymentIn):
    """Score one payment and update account state. Send payments in time order.
    A recent risky message to the payer, or a payee named in one, can raise the decision (never lower it)."""
    return score_with_context(need_engine(), p)


class MessageIn(BaseModel):
    text: str = Field(min_length=1, max_length=5000, examples=["Your SBI account will be blocked today. Update PAN at http://sbi-kyc-update.in"])
    channel: str = Field("sms", description="sms, whatsapp, email, call_transcript")
    timestamp: Optional[str] = Field(None, description="YYYY-MM-DD HH:MM (UTC); defaults to now")
    bank: Optional[str] = Field(None, description="Recipient's bank code, to link the message to their later payments")
    account: Optional[str] = Field(None, description="Recipient's account id (same id as from_account in /v2/payment)")


class EventIn(BaseModel):
    type: str = Field(description="password_reset, new_device, new_payee, limit_raise or transfer")
    minute: int = Field(description="minutes since epoch (or any consistent clock)")


class SessionIn(BaseModel):
    """Signals from the bank app at payment time (session and person levels). All optional."""
    session_id: Optional[str] = None
    on_call: bool = False
    call_minutes: float = 0
    video_call: bool = False
    screen_share: bool = False
    remote_access_app: bool = False
    payee_pasted: bool = False
    hesitation_s: float = 0
    app_hidden_s: float = 0
    typing_anomaly: bool = False
    device_hash: Optional[str] = Field(None, description="salted device hash, never the raw IMEI")
    enrolled_device_hashes: List[str] = []
    sim_swap_hours: Optional[float] = Field(None, description="hours since the last SIM swap, from the telecom check")
    events: List[EventIn] = []


def to_session(si: Optional[SessionIn]) -> Optional[SessionSignals]:
    if si is None:
        return None
    d = si.model_dump(); d["events"] = [dict(e) for e in d["events"]]
    return SessionSignals(**d)


class AssessIn(PaymentIn):
    message: Optional[str] = Field(None, max_length=5000, description="Message or call transcript the payer received (optional)")
    consent: bool = Field(False, description="Payer opted in to message context; without it the text is ignored and not stored")
    use_intent_model: bool = Field(True, description="Lift the context score with the trained L-1 message model (default since ctx-1.2; see results/fusion_oof.json)")
    session: Optional[SessionIn] = Field(None, description="Session and device signals from the bank app (optional)")


@api.post("/v2/assess")
async def v2_assess(a: AssessIn):
    """Payment cascade + context layer (ruleset in the response). The final decision is the stricter of the two;
    context alone never goes beyond an analyst hold. Message text is redacted before it is returned or stored."""
    eng = need_engine()
    t0 = time.perf_counter()
    d = to_payload(a)
    sender, payee = f'{a.from_bank}:{a.from_account}', f'{a.to_bank}:{a.to_account}'
    sf = CTX.sender_fan_in_24h(eng, d)
    flagged_payers = COORD.flagged_payers(payee, d["tmin"])
    shared = CONSORTIUM.check("mule_account", payee)
    pay = LURES.apply_to_payment(d, eng.score_payment(d))
    mm = MESSAGES.score(a.message) if (a.consent and a.message and MESSAGES) else None
    band = mm["band"] if (mm and a.use_intent_model) else None
    pc = CTX.from_engine_features(pay["features"], sf)
    pc.coordinated = (COORD.enabled and flagged_payers >= COORD.k) or shared["flagged"]
    ss = to_session(a.session)
    if ss is not None:
        ss.network_flag = shared["flagged"]
    sres = assess_session(ss)
    ctx = CTX.assess(a.message, pc, consent=a.consent, model_band=band, session=sres)
    comb = CTX.combine(pay["decision"], ctx)
    COORD.record(sender, payee, d["tmin"], CTX.PAYMENT_RANK[comb["final_decision"]])
    out = {"payment": {k: pay.get(k) for k in ("l1_score", "layer_exit", "risk", "decision", "reasons", "case_id", "latency_ms")},
           "context": ctx.to_dict(), **comb,
           "network": {"flagged_payers_in_window": flagged_payers, "local_rule_enabled": COORD.enabled,
                       "flagged_by_consortium": shared["banks"]},
           "message_model": {k: mm.get(k) for k in ("band", "workflow", "probability") if k in mm} if mm else None}
    out["risk_score"] = display_score(comb["final_decision"], pay.get("risk"), ctx.total, eng.cfg)
    wf = (mm or {}).get("workflow") or {}
    out["intervention"] = intervention_plan(comb["final_decision"], ctx.reasons, sres.identity if sres else None,
                                            wf.get("id") if isinstance(wf, dict) else None)
    alert = DESK.open_alert(a.model_dump(exclude={"message"}), out)
    out["alert_id"] = alert["alert_id"] if alert else None
    out["shown_decision"] = alert["shown_decision"] if alert else "Approve"
    out["time_to_decision_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    core = {"alert_id": out["alert_id"], "final_decision": comb["final_decision"], "shown_decision": out["shown_decision"],
            "risk_score": out["risk_score"]["score"], "ruleset": ctx.ruleset, "inputs": a.model_dump(exclude={"message"})}
    out["signature"] = {**SIGNER.sign(core), "signed_fields": core}
    return out


class ReviewIn(BaseModel):
    action: str = Field(description="release, verify, hold or approve_block (a block needs two different authenticated analysts)")
    note: str = Field("", max_length=500)
    analyst: Optional[str] = Field(None, description="ignored: the analyst is identified by the X-Analyst-Key header")


@api.get("/v2/alerts")
async def v2_alerts(status: Optional[str] = None):
    """Alerts awaiting or after human review, newest first."""
    al = [a for a in DESK.alerts.values() if status is None or a["status"] == status]
    return list(reversed(al))


@api.post("/v2/alerts/{alert_id}/review")
async def v2_review(alert_id: str, r: ReviewIn, x_analyst_key: Optional[str] = Header(None), x_nonce: Optional[str] = Header(None)):
    """Analyst action. Needs X-Analyst-Key (who you are) and a fresh X-Nonce (no replays). Two different analysts
    must approve a block; the block then publishes the payee to the consortium mule registry (hashed)."""
    who = analyst_for(x_analyst_key)
    if not who:
        raise HTTPException(401, "missing or unknown X-Analyst-Key")
    if not allowed(who["role"], r.action):
        raise HTTPException(403, f"role {who['role']} cannot {r.action}")
    if not NONCES.use(x_nonce):
        raise HTTPException(409, "X-Nonce missing or already used")
    try:
        a = DESK.act(alert_id, who["analyst"], r.action, r.note)
    except KeyError:
        raise HTTPException(404, "unknown alert")
    except ValueError as e:
        raise HTTPException(400, str(e))
    if a["status"] == "blocked" and not a.get("published"):
        p = a["payment"]
        e = CONSORTIUM.publish(THIS_BANK, "mule_account", f'{p["to_bank"]}:{p["to_account"]}', "block approved by two analysts",
                               DESK.evidence(alert_id)["pack_hash"])
        a["published"] = e["seq"]
        DESK.system_event(alert_id, "consortium_published", "system", f"ledger entry {e['seq']}")
    return a


@api.get("/v2/alerts/{alert_id}/evidence")
async def v2_evidence(alert_id: str):
    if alert_id not in DESK.alerts:
        raise HTTPException(404, "unknown alert")
    pack = DESK.evidence(alert_id)
    pack["signature"] = SIGNER.sign({"pack_hash": pack["pack_hash"], "alert_id": alert_id})
    return pack


@api.get("/v2/alerts/{alert_id}/str")
async def v2_str(alert_id: str):
    """STR draft built from the evidence pack (template, not a language model). For analyst review; never filed automatically."""
    if alert_id not in DESK.alerts:
        raise HTTPException(404, "unknown alert")
    camp = next((c for c in CAMPAIGNS.build(list(DESK.alerts.values())) if alert_id in c["alerts"]), None)
    return str_report.draft(DESK.evidence(alert_id), camp)


@api.get("/v2/campaigns")
async def v2_campaigns():
    """Alerts linked into scam campaigns by shared payee, payee identifiers in messages, or near identical scripts."""
    return CAMPAIGNS.build(list(DESK.alerts.values()))


class CustomerIn(BaseModel):
    answer: str = Field(description="stop (customer stops the payment) or continue (released after the cool off unless an analyst acts)")


@api.post("/v2/alerts/{alert_id}/customer")
async def v2_customer(alert_id: str, c: CustomerIn):
    """The customer's answer on the pause screen."""
    if alert_id not in DESK.alerts:
        raise HTTPException(404, "unknown alert")
    if c.answer not in ("stop", "continue"):
        raise HTTPException(400, "answer must be stop or continue")
    status = "stopped_by_customer" if c.answer == "stop" else "cooling_off"
    return DESK.system_event(alert_id, status, "customer", "pause screen answer: " + c.answer, set_status=True)


class ReverifyIn(BaseModel):
    method: str = Field("video_kyc", description="video_kyc or branch")


@api.post("/v2/alerts/{alert_id}/reverify")
async def v2_reverify(alert_id: str, r: ReverifyIn):
    """Customer recovery: a re-verified customer goes back to an analyst for release."""
    if alert_id not in DESK.alerts:
        raise HTTPException(404, "unknown alert")
    return DESK.system_event(alert_id, "customer_reverified", "customer", f"re-verified by {r.method}", set_status=True)


@api.post("/v2/identity/check")
async def v2_identity(si: SessionIn):
    """Run before the OTP is sent: unenrolled device plus a recent SIM swap means no OTP and re-verification."""
    res = identity_check(to_session(si))
    return {**res, "intervention": intervention_plan("Approve", [], res)}


class DuressRegisterIn(BaseModel):
    account: str
    normal_pin: str
    duress_pin: str
    trusted_contacts: List[str] = []


@api.post("/v2/duress/register")
async def v2_duress_register(r: DuressRegisterIn):
    try:
        return DURESS.register(r.account, r.normal_pin, r.duress_pin, r.trusted_contacts)
    except ValueError as e:
        raise HTTPException(400, str(e))


class ConfirmIn(BaseModel):
    account: str
    pin: str
    alert_id: Optional[str] = None


@api.post("/v2/payments/confirm")
async def v2_confirm(c: ConfirmIn):
    """Confirm a payment with a PIN. The duress PIN shows success to the customer but holds the money and alerts
    trusted contacts and the fraud desk. The response says what the customer sees and what really happened."""
    r = DURESS.confirm(c.account, c.pin)
    if r["result"] == "duress" and c.alert_id and c.alert_id in DESK.alerts:
        DESK.system_event(c.alert_id, "duress_hold", "customer", "duress PIN used: funds held, trusted contacts alerted", set_status=True)
    return r


class DigraphIn(BaseModel):
    account: str
    digraphs: dict = Field(description='key pair -> milliseconds, e.g. {"th": 112, "he": 98}')


@api.post("/v2/behaviour/enrol")
async def v2_behaviour_enrol(b: DigraphIn):
    prof = PROFILES.setdefault(b.account, TypingProfile()); prof.enrol(b.digraphs)
    return {"account": b.account, "sessions": len(prof.sessions), "ready": prof.ready}


@api.post("/v2/behaviour/check")
async def v2_behaviour_check(b: DigraphIn):
    """Typing rhythm check (verifier trained and tested on real keystrokes, see results/aalto_eval.json);
    use the result as SessionIn.typing_anomaly."""
    prof = PROFILES.get(b.account)
    return prof.check(b.digraphs) if prof else {"ready": False, "score": None, "anomaly": False}


@api.get("/v2/consortium")
async def v2_consortium():
    """Signed, hash chained cross bank ledger (mule accounts, fraud devices, SIM swap takeovers); identifiers are HMAC tokens."""
    return {"members": {b: k.public_b64 for b, k in CONSORTIUM.banks.items()}, "entries": CONSORTIUM.ledger,
            "valid": CONSORTIUM.verify(), "merkle_root": CONSORTIUM.merkle_root()}


class ConsortiumCheckIn(BaseModel):
    registry: str = "mule_account"
    identifier: str = Field(description="for mule_account: bank:account")


@api.post("/v2/consortium/check")
async def v2_consortium_check(c: ConsortiumCheckIn):
    return CONSORTIUM.check(c.registry, c.identifier)


class ConsortiumPublishIn(BaseModel):
    bank: str = Field("BANK-B", description="publishing member (demo: BANK-A, BANK-B, BANK-C)")
    registry: str = "mule_account"
    identifier: str
    reason: str = "confirmed mule"


@api.post("/v2/consortium/publish")
async def v2_consortium_publish(c: ConsortiumPublishIn, x_analyst_key: Optional[str] = Header(None)):
    """Demo: a member bank publishes a flag (needs an analyst key)."""
    if not analyst_for(x_analyst_key):
        raise HTTPException(401, "missing or unknown X-Analyst-Key")
    if c.bank not in CONSORTIUM.banks:
        raise HTTPException(400, "unknown member bank")
    try:
        return CONSORTIUM.publish(c.bank, c.registry, c.identifier, c.reason)
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/v2/keys/public")
async def v2_public_key():
    return {"alg": "Ed25519", "key_id": SIGNER.key_id, "public_key_b64": SIGNER.public_b64, "ephemeral": SIGNER.ephemeral}


class VerifyIn(BaseModel):
    signed_fields: dict
    signature: str


@api.post("/v2/verify")
async def v2_verify(v: VerifyIn):
    return {"valid": Signer.verify(v.signed_fields, v.signature, SIGNER.public_b64)}


@api.get("/v2/demo/analysts")
async def v2_demo_analysts():
    """Demo keys (only when RUSETECT_ANALYST_KEYS is not set)."""
    return [{"key": k, **v} for k, v in DEMO_KEYS.items()] if DEMO_MODE else []


@api.get("/v2/fixtures/mule")
async def v2_fixture_mule():
    """The fictional mule replay fixture (fixtures/mule_replay.json) used by the investigator demo."""
    import json as _json
    return _json.loads((Path(__file__).resolve().parents[1] / "fixtures" / "mule_replay.json").read_text())


@api.get("/v2/audit")
async def v2_audit():
    return {"entries": DESK.log, "chain_valid": DESK.verify_chain()}


@api.post("/v2/message")
async def v2_message(m: MessageIn):
    """L-1 lure detector. Scores the text, explains which scam workflow and tactics it matches, and applies the
    alert-fatigue policy (push only high-risk, deduplicated, capped). Entities in risky messages join a payee watchlist."""
    if MESSAGES is None:
        raise HTTPException(503, "Message model not loaded")
    ts = datetime.strptime(m.timestamp.replace("/", "-"), "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc) if m.timestamp else datetime.now(timezone.utc)
    res = MESSAGES.score(m.text)
    acct = f"{m.bank}:{m.account}" if m.account else None
    res["alert"] = LURES.record_message(acct, int(ts.timestamp()) // 60, res)
    res["channel"] = m.channel
    return res


@api.get("/v2/alerts/summary")
async def v2_alert_summary():
    """How many messages were scored, pushed, or suppressed (and why); how many payments a lure escalated."""
    return LURES.summary()


@api.get("/v2/message-model")
async def v2_message_model():
    import json
    return json.loads((ROOT.parent / "results" / "message_model_eval.json").read_text()) if (ROOT.parent / "results" / "message_model_eval.json").exists() else {}


@api.post("/v2/payments/batch")
async def v2_batch(items: List[PaymentIn]):
    if len(items) > 20000:
        raise HTTPException(413, "Batch too large (max 20000)")
    eng = need_engine()
    out = [score_with_context(eng, p) for p in sorted(items, key=lambda x: x.timestamp)]
    flagged = [dict(i=i, **{k: r[k] for k in ("risk", "decision", "case_id", "reasons")}) for i, r in enumerate(out) if r["band"] != "safe"]
    flagged.sort(key=lambda f: -f["risk"])
    exits = {"L1": sum(r["layer_exit"] == "L1" for r in out), "L2": sum(r["layer_exit"] == "L2" for r in out)}
    return {"scored": len(out), "layer_exit": exits, "flagged": flagged[:500], "cases": eng.cases(20)}


@api.get("/v2/cases")
async def v2_cases(limit: int = 50):
    return need_engine().cases(max(1, min(limit, 500)))


class ApplicationIn(BaseModel):
    """Account-opening application in the Feedzai BAF schema; omitted fields are treated as missing."""
    model_config = {"extra": "allow"}
    customer_age: int


@api.post("/v2/onboard")
async def v2_onboard(app_in: ApplicationIn):
    return need_engine().score_application(app_in.model_dump())


@api.get("/v2/model-card")
async def v2_model_card():
    import json
    return json.loads((MODELS / "benchmark_results.json").read_text())


@api.get("/v2/stats")
async def v2_stats():
    eng = need_engine()
    return {"payments_seen": eng.n_seen, "accounts_tracked": len(eng.accts), "decisions": dict(eng.decisions), "open_cases": len(eng.case_alerts)}


# ---------------------------------------------------------------- demo replay
DEMO = {"rows": None, "ptr": 0, "warm": 0, "truth": {"tp": 0, "fp": 0, "fn": 0, "pos": 0}, "feed": []}
DEMO_WARM = 100_000      # first 100K rows of demo_stream.csv.gz are pre-test history


def demo_rows():
    if DEMO["rows"] is None:
        with gzip.open(MODELS / "demo_stream.csv.gz", "rt") as fh:
            DEMO["rows"] = list(csv.DictReader(fh))
    return DEMO["rows"]


def demo_payload(r):
    return dict(from_bank=r["fb"], from_account=r["fa"], to_bank=r["tb"], to_account=r["ta"], amount_paid=float(r["amt_paid"]),
                payment_currency=r["cur_paid"], receiving_currency=r["cur_recv"], payment_format=r["fmt"],
                tmin=calendar.timegm(time.strptime(r["ts"], "%Y/%m/%d %H:%M")) // 60)


def full_history_rows():
    """All payments before the demo window from the full IBM HI-Small CSV(s), if RUSETECT_HISTORY_GLOB is set."""
    pattern = os.environ.get("RUSETECT_HISTORY_GLOB")
    if not pattern:
        return None
    import glob
    import polars as pl
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    cols = ["ts", "fb", "fa", "tb", "ta", "amt_recv", "cur_recv", "amt_paid", "cur_paid", "fmt", "y"]
    df = pl.concat([pl.read_csv(f, has_header=True, infer_schema=False, new_columns=cols) for f in files]).with_row_index("rid")
    df = (df.with_columns(pl.col("ts").str.strptime(pl.Datetime, "%Y/%m/%d %H:%M").alias("_t")).sort(["_t", "rid"])
            .with_columns((pl.col("_t").dt.epoch("s") // 60).alias("tmin"), pl.col("amt_paid").cast(pl.Float64)))
    return df.head(int(0.8 * df.height))


@api.post("/v2/demo/reset")
async def demo_reset(warm: bool = True):
    """Clear state, then ingest earlier payments as account history (features only, no scoring)."""
    eng = need_engine(); eng.reset(); LURES.reset(); DESK.reset(); COORD.reset(); DURESS.reset(); CONSORTIUM.reset(); PROFILES.clear()
    rows = demo_rows(); t = time.perf_counter(); source = "bundled sample"
    hist = full_history_rows() if warm else None
    if hist is not None:
        for fb, fa, tb, ta, ap, cp, cr, fm, tm in hist.select("fb", "fa", "tb", "ta", "amt_paid", "cur_paid", "cur_recv", "fmt", "tmin").iter_rows():
            eng.features(dict(from_bank=fb, from_account=fa, to_bank=tb, to_account=ta, amount_paid=ap, payment_currency=cp,
                              receiving_currency=cr, payment_format=fm, tmin=tm))
        n = hist.height; source = "full IBM history"
    else:
        n = DEMO_WARM if warm else 0
        for r in rows[:n]:
            eng.features(demo_payload(r))
    DEMO.update(ptr=DEMO_WARM, warm=DEMO_WARM, truth={"tp": 0, "fp": 0, "fn": 0, "pos": 0}, feed=[])
    return {"history_ingested": n, "source": source, "seconds": round(time.perf_counter() - t, 2), "remaining": len(rows) - DEMO_WARM}


@api.post("/v2/demo/next")
async def demo_next(n: int = 2000):
    """Score the next n demo payments. Labels are only used to report accuracy, never by the model."""
    eng = need_engine(); rows = demo_rows(); n = max(1, min(n, 20000))
    if DEMO["ptr"] < DEMO_WARM:
        raise HTTPException(409, "Load account history first (POST /api/v2/demo/reset)")
    batch = rows[DEMO["ptr"]:DEMO["ptr"] + n]; DEMO["ptr"] += len(batch)
    exits = {"L1": 0, "L2": 0}; bands = {"safe": 0, "med": 0, "high": 0}; lat = []
    for r in batch:
        res = eng.score_payment(demo_payload(r)); y = r["y"] == "1"
        exits[res["layer_exit"]] += 1; bands[res["band"]] += 1; lat.append(res["latency_ms"])
        blocked = res["band"] == "high"; T = DEMO["truth"]
        T["pos"] += y; T["tp"] += blocked and y; T["fp"] += blocked and not y; T["fn"] += (not blocked) and y
        if res["band"] != "safe" and (blocked or res["decision"].startswith("Hold") or random.random() < 0.15):
            DEMO["feed"].insert(0, {"ts": r["ts"], "from": f'{r["fb"]}·{r["fa"]}', "to": f'{r["tb"]}·{r["ta"]}',
                                    "amount": round(float(r["amt_paid"]), 2), "currency": r["cur_paid"], "format": r["fmt"],
                                    "risk": res["risk"], "decision": res["decision"], "reasons": res["reasons"][:3],
                                    "case_id": res["case_id"], "truth": y})
    DEMO["feed"] = DEMO["feed"][:60]
    lat.sort()
    return {"scored": len(batch), "position": DEMO["ptr"] - DEMO["warm"], "total": len(rows) - DEMO["warm"],
            "done": DEMO["ptr"] >= len(rows), "exits": exits, "bands": bands, "truth": DEMO["truth"],
            "latency_p50_ms": lat[len(lat) // 2] if lat else None, "decisions": dict(eng.decisions),
            "feed": DEMO["feed"][:25], "cases": eng.cases(8), "open_cases": len(eng.case_alerts),
            "sim_time": batch[-1]["ts"] if batch else None}


app.include_router(api)
