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

1. **Mismatched block threshold (open, decision needed).** The served `block_threshold` (0.823) keeps the alert rate of
   the threshold `ml/cascade.py` tuned on the offline L3 blend score, but the live engine applies it to the L2 XGBoost
   and LightGBM average. Measured by `ml/audit_live_path.py` with the current models on the full HI-Small test period
   (`results/live_path_audit.json`): block recommendations are 0.638 precise with 0.653 recall (1,840 alerts). A
   threshold retuned on validation for the live score (0.935) would give 0.938 precision and 0.553 recall (1,059
   alerts). Blocks always need two analysts, so this is a choice between analyst workload and recall; the served
   threshold stays as it is until that choice is made.
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
   payment can land on different instances, so the fusion can silently do nothing. `service/scale` keeps this
   state in Redis for production; the demo should run on one always on process.
6. **Offline-only layers are labelled.** Random Forest, the L3 sequence blend and the GNN do not run live; the landing
   page and report say so.
7. **Fresh model comparison is complete.** All 14 tabular models (`results/model_comparison_tabular.json`) and the four
   sequence models (`results/model_comparison_sequence.json`); see [model-selection.md](model-selection.md).
8. **The LightGBM half of L2 is now the weaker one.** Since the retrain with LI-Small, LightGBM alone scores 0.604
   average precision on the HI-Small test period against 0.711 for XGBoost alone (0.208 against 0.344 on LI-Small).
   The served average still scores slightly above XGBoost alone on validation (0.633 against 0.622 on HI-Small, 0.298
   against 0.292 on LI-Small) and matches it on the test periods, so it is kept; LightGBM's settings are the first
   place to look in the next retrain (`results/l2_halves_check.json`).
