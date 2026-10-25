# Project audit

The state of RuseTect against its goals.

> **Goal.** AI-driven scam pattern recognition: detect emerging AI-enabled scam workflows in UPI,
> banking or wallet contexts, and deliver timely, useful alerts without creating alert fatigue.

## Coverage of the scam workflow

| Stage of a scam | What covers it | Status |
|---|---|---|
| Lure: SMS, WhatsApp, call (incl. AI voice clone scripts) | L-1 lure detector, `/api/v2/message` | **New.** Synthetic Indian corpus + real UCI SMS; needs real Indian data |
| Pressure → victim pays a new payee | Lure → payment fusion (`service/lure_fusion.py`): cool-off and step-up | **New.** Rule-based join, unit tested |
| Mule account opening | L0 LightGBM on Feedzai BAF | Live; synthetic data |
| Payment into or through a mule | L1 → L2 cascade on IBM AMLworld | Live; synthetic data |
| Drain / layering through rings | L4 ring linker | Live (in memory) |
| Alert fatigue | Push budget, dedup, daily cap, new payee only interruptions, ring grouping | **New** counters at `/api/v2/alerts/summary` |

## Findings

1. **Mismatched block threshold (open, decision needed).** `ml/cascade.py` tunes `block_threshold` (0.829) on the
   offline L3 blend score, but the live engine applies it to the L2 XGBoost + LightGBM average. Measured in the
   previous session on the test period: block precision 0.605 / recall 0.671 at the committed threshold. A
   threshold retuned on validation for the live score (0.950) gives 0.920 / 0.570. The served threshold is
   unchanged until `ml/audit_live_path.py` is rerun on the full data and the trade-off is chosen.
2. **All training data is synthetic or foreign.** IBM AMLworld and Feedzai BAF are generator outputs. The Indian
   message corpus is template based, and UCI SMS is UK-era spam. The pipeline is built to retrain; [datasets.md](datasets.md)
   lists replacements.
3. **The rules only message scores are circular.** The hard signal list and the Indian templates came from the same
   authors. The list is kept deliberately short (6 patterns) and fired on 0 of 5,593 legitimate messages; that
   false alarm count is the number to trust, not its recall on our own corpus.
4. **Refund notices are the main message false alarm.** Genuine IT and PF refund SMS look like scam bait in text. The
   fix is sender verification (TRAI DLT headers), which needs a field the API does not take yet.
5. **State lives in memory.** Account history, ring cases, lure events, the payee watchlist and alert budgets all
   reset on restart, and are not shared between serverless instances. On Vercel, a message and the following
   payment can land on different instances, so the fusion can silently do nothing. Production needs Redis or
   similar; the demo should run on one always on process.
6. **Offline-only layers are labelled.** Random Forest, the L3 sequence blend and the GNN do not run live; the landing
   page and report say so.
7. **Fresh model comparison is half done** (7/14). See [model-selection.md](model-selection.md) for the remaining runs.

## Changes on this branch

- Handoff work from `rusetect-rename`: rename, report build step, `verify_threshold` export, comparison scripts.
- L-1 lure detector: `service/scam_text.py` (featuriser, workflow and tactic tagger, entity extraction, scorer),
  `ml/scam_corpus.py`, `ml/train_message_model.py`, `service/models/message_model.json`, `results/message_model_eval.json`.
- Fusion and alert fatigue policy: `service/lure_fusion.py`; `/api/v2/message`, `/api/v2/alerts/summary`,
  `/api/v2/message-model`; `/api/v2/payment` and `/payments/batch` now apply lure context (they can only raise a decision).
- Console: a message check panel with samples, and "then pay" buttons that show the escalation live.
- Landing page: L-1 card. Tests: `tests/test_messages.py`.

## To do, in priority order

1. Obtain real or better data (see datasets.md) and retrain both sides.
2. Finish the fresh comparison and the sequence runs; run `ml/audit_live_path.py`; decide finding 1.
3. Persistent state store (Redis) for accounts, lures, watchlist and alert budgets.
4. Sender-ID input on `/api/v2/message` (DLT header, WhatsApp business verification) to cut refund notice false alarms.
5. Hindi and regional script data for L-1; the featuriser already keeps Devanagari tokens.
6. Call audio: speech to text upstream of L-1, plus deepfake voice detection if a labelled set is available.
