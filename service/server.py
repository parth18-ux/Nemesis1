"""
RuseTect API — layered filtration cascade for mule accounts and scam money flows.

  /                    landing page (links to the console, report and API docs)
  /console             Live Console: replays real IBM AMLworld test-period payments through the cascade
  /report              benchmark report (21 AML models, 8 onboarding models)
  /api/health          engine status
  /api/v2/payment      score one payment (L1 -> L2 -> decision band -> L4 case)
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
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from rusetect_engine import Engine

ROOT = Path(__file__).parent
MODELS = ROOT / "models"
log = logging.getLogger("rusetect")

app = FastAPI(title="RuseTect API", version="2.1.0",
              description="Layered AI filtration cascade: L0 onboarding gate, L1 gatekeeper, L2 graph-boost, L4 ring linker.")
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])
api = APIRouter(prefix="/api")

try:
    ENGINE = Engine(MODELS)
except Exception as e:  # pragma: no cover
    log.warning("RuseTect engine unavailable: %s", e)
    ENGINE = None


def need_engine() -> Engine:
    if ENGINE is None:
        raise HTTPException(503, "RuseTect models not loaded")
    return ENGINE


# ---------------------------------------------------------------- pages
LANDING = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RuseTect</title><style>
:root{--bg:#f5f7f9;--ink:#15202b;--ink2:#4a5866;--acc:#0b6e7a;--rule:#d9e0e6;--panel:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#0d1419;--ink:#e6edf2;--ink2:#a9b6c1;--acc:#4fc3cf;--rule:#24313b;--panel:#152029;color-scheme:dark}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 system-ui,sans-serif;padding:48px 16px}
main{max-width:720px;margin:0 auto;display:flex;flex-direction:column;gap:20px}
h1{font-size:40px;line-height:1.1;margin:0}p{color:var(--ink2);margin:0}
.k{font:600 12px ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--acc)}
.links{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
a.c{display:block;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;color:var(--ink);text-decoration:none}
a.c b{display:block;color:var(--acc)}a.c span{font-size:13px;color:var(--ink2)}a.c:hover{border-color:var(--acc)}
</style></head><body><main>
<span class="k">RuseTect</span>
<h1>Layered AI filtration for mule accounts and scam money flows</h1>
<p>L0 stops mule accounts at opening; L1 clears ~80% of payments in 0.3 ms; L2 boosted graph models block laundering with ~0.92 precision; L4 links blocks into ring cases.</p>
<div class="links">
<a class="c" href="/console"><b>Live Console →</b><span>Replay real test-period payments through the cascade</span></a>
<a class="c" href="/report"><b>Benchmark report →</b><span>21 AML models + 8 onboarding models compared</span></a>
<a class="c" href="/docs"><b>API docs →</b><span>Score payments and applications yourself</span></a>
</div></main></body></html>"""


@app.get("/", response_class=HTMLResponse)
async def landing():
    return LANDING


@app.get("/console")
async def console():
    return FileResponse(ROOT / "static" / "console.html")


@app.get("/report")
async def report():
    return FileResponse(ROOT / "static" / "report.html")


@api.get("/")
@api.get("/health")
async def health():
    return {"service": "RuseTect API", "version": "2.1.0", "status": "ok", "engine_loaded": ENGINE is not None}


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


@api.post("/v2/payment")
async def v2_payment(p: PaymentIn):
    """Score one payment and update account state. Send payments in time order."""
    return need_engine().score_payment(to_payload(p))


@api.post("/v2/payments/batch")
async def v2_batch(items: List[PaymentIn]):
    if len(items) > 20000:
        raise HTTPException(413, "Batch too large (max 20000)")
    eng = need_engine()
    out = [eng.score_payment(to_payload(p)) for p in sorted(items, key=lambda x: x.timestamp)]
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
    eng = need_engine(); eng.reset(); rows = demo_rows(); t = time.perf_counter(); source = "bundled sample"
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
