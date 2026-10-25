# Context layer: how it was built and measured

Every number below comes from a script in `ml/` and a file in `results/`; all data is synthetic or fictional.

## Run it

```bash
pip install -r requirements.txt polars
uvicorn index:app --port 8001
# console      http://localhost:8001/console
# investigator http://localhost:8001/investigate
pytest tests/
```

## What was built

| Stage | Built | Evidence |
| --- | --- | --- |
| 1 | Context layer: consent, redaction, keyword baseline, risk point fusion, context veto; 12 reference fixtures | `ml/context_validation.py`: 12 of 12 match the expected table |
| 2 | Negation aware cues, sender mule signature, intent model option, `POST /api/v2/assess` | `ml/ablation_context.py`, `ml/mule_signature_check.py` |
| 3 | Review desk: alerts, four eyes block approval, hash chained audit log, evidence packs, `/investigate` page | `tests/test_review.py` |
| 4 | Holdout (412 unseen fills, 144 templates), payment ablation on 60,000 IBM payments, out of fold model fusion | `ml/holdout_eval.py`, `ml/fusion_oof.py` |
| 5 | Load test, unseen scam replay demo | `ml/load_test.py`, `ml/demo_unseen.py` |

## Key results

- Phrase rules alone generalise poorly: on unseen message templates they escalate 3% of scams and send 30% to verification.
- With the out of fold intent model fused in (default since ctx-1.2): 28% of scams escalated, 77% verify or more; legitimate messages 0% escalated, 6% verify or more.
- Payment side on 60,000 IBM payments without messages: 34 verification steps (0.06%), 2 of 97 laundering payments; removing either the transaction or the graph domain drops this to 0.
- Mule replay fixture: 11 of 24 payments reach verification with ctx-1.1 (was 2 with ctx-1.0); the payment engine alone approves 23 of 24.
- Load test: 2,000 requests from 8 threads, 220 requests/s, p95 72 ms, p99 85 ms on a 2 CPU machine (target p95 below 100 ms).

## Demo

1. Open `/investigate`, press **Reset**, then **Replay mule fixture**: 24 payments flow; the payment model approves almost all, the context layer raises the mule's onward transfers to verification once it has 4 senders.
2. Press **Assess** with the S09 message: payment model says Approve, context says HIGH, the veto raises it to Hold for analyst.
3. Open the alert: points per domain, plain reasons, redacted message, linked accounts.
4. Press **Approve block** as one analyst, then again as the same analyst: refused (four eyes). Sign in as a second analyst: blocked. The audit log shows a valid hash chain; **evidence pack** downloads the STR draft.

## Honest limits

- Fixtures were co-designed with the rules; probes were seen while designing ctx-1.1; the message corpus is synthetic template text.
- The served message model was trained on the whole corpus, so in demo scores on corpus messages are optimistic; use `results/fusion_oof.json`.
- No real bank data, no real victim outcomes, no deepfake or voice detection.
