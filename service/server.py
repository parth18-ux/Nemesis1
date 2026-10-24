"""
RuseTect API — layered filtration cascade for mule accounts and scam money flows.

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

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from branding import NAME, SLUG, TAGLINE, VERSION
from lure_fusion import LureMemory
from rusetect_engine import Engine
from scam_text import MessageModel
import context_rules as CTX
from review import ReviewDesk

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
    return {"service": f"{NAME} API", "name": NAME, "id": SLUG, "version": VERSION, "status": "ok", "engine_loaded": ENGINE is not None}


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


class AssessIn(PaymentIn):
    message: Optional[str] = Field(None, max_length=5000, description="Message or call transcript the payer received (optional)")
    consent: bool = Field(False, description="Payer opted in to message context; without it the text is ignored and not stored")
    use_intent_model: bool = Field(False, description="Also lift the context score with the trained L-1 message model")


@api.post("/v2/assess")
async def v2_assess(a: AssessIn):
    """Payment cascade + context layer (ruleset in the response). The final decision is the stricter of the two;
    context alone never goes beyond an analyst hold. Message text is redacted before it is returned or stored."""
    eng = need_engine()
    d = to_payload(a)
    sf = CTX.sender_fan_in_24h(eng, d)
    pay = LURES.apply_to_payment(d, eng.score_payment(d))
    band = MESSAGES.score(a.message)["band"] if (a.use_intent_model and a.consent and a.message and MESSAGES) else None
    ctx = CTX.assess(a.message, CTX.from_engine_features(pay["features"], sf), consent=a.consent, model_band=band)
    comb = CTX.combine(pay["decision"], ctx)
    out = {"payment": {k: pay.get(k) for k in ("l1_score", "layer_exit", "risk", "decision", "reasons", "case_id", "latency_ms")},
           "context": ctx.to_dict(), **comb}
    alert = DESK.open_alert(a.model_dump(exclude={"message"}), out)
    out["alert_id"] = alert["alert_id"] if alert else None
    out["shown_decision"] = alert["shown_decision"] if alert else "Approve"
    return out


class ReviewIn(BaseModel):
    analyst: str = Field(min_length=1, max_length=64)
    action: str = Field(description="release, verify, hold or approve_block (a block needs two different analysts)")
    note: str = Field("", max_length=500)


@api.get("/v2/alerts")
async def v2_alerts(status: Optional[str] = None):
    """Alerts awaiting or after human review, newest first."""
    al = [a for a in DESK.alerts.values() if status is None or a["status"] == status]
    return list(reversed(al))


@api.post("/v2/alerts/{alert_id}/review")
async def v2_review(alert_id: str, r: ReviewIn):
    try:
        return DESK.act(alert_id, r.analyst, r.action, r.note)
    except KeyError:
        raise HTTPException(404, "unknown alert")
    except ValueError as e:
        raise HTTPException(400, str(e))


@api.get("/v2/alerts/{alert_id}/evidence")
async def v2_evidence(alert_id: str):
    if alert_id not in DESK.alerts:
        raise HTTPException(404, "unknown alert")
    return DESK.evidence(alert_id)


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
    """All payments before the demo window from the full IBM HI-Small CSV(s), if RUSETECT_HISTORY_GLOB is set.
    The legacy name RUSETECT_HISTORY_GLOB is still read as a fallback."""
    pattern = os.environ.get("RUSETECT_HISTORY_GLOB") or os.environ.get("RUSETECT_HISTORY_GLOB")
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
    eng = need_engine(); eng.reset(); LURES.reset(); DESK.reset(); rows = demo_rows(); t = time.perf_counter(); source = "bundled sample"
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
