# RuseTect

**Layered AI filtration for mule accounts and scam money flows.**
RuseTect (AI-driven scam pattern recognition).

Every payment passes through a cascade. Each layer approves what it is confident is clean and
sends the rest to a heavier model, so analysts only see a small, high-precision set of ring cases.

| Layer | What it does | Held-out result |
|---|---|---|
| **L0 Onboarding gate** | LightGBM on the account application, with age-group thresholds for fairness | Recall 0.534 at 5% FPR; FPR ratio (50+ vs <50) 0.88, up from 0.31 |
| **L1 Gatekeeper** | 30-tree XGBoost on causal account-history features | Auto-approves 79.5% of payments in 0.3 ms, keeps 99.2% of laundering |
| **L2 Graph-boost** | XGBoost + LightGBM + Random Forest | PR-AUC 0.720 |
| **L3 Behaviour** | + LSTM / Transformer over sender and receiver event sequences | PR-AUC 0.725, F1 0.702, block precision 0.921 |
| **L4 Ring linker** | Union-find over blocked payments that share accounts | 1,105 alerts → 318 cases; 82–89% of the 192 ground-truth rings caught |

Data: IBM AMLworld HI-Small (5,078,345 payments, 5,177 laundering, 370 labelled rings) and
Feedzai BAF Base (1,000,000 account applications). Every metric is measured on a later time
period than the model was trained on.

## What was compared

21 AML model variants: XGBoost, LightGBM, CatBoost, AdaBoost, Logistic Regression, Decision Tree,
Random Forest, Linear SVM, RBF-SVM, KNN, Isolation Forest, MLP, 1D-CNN, LSTM, Transformer,
GAN augmentation and a directed GNN, with and without graph features. On the onboarding side:
8 models on BAF. The full table is in `report/rusetect_report.html`, served at `/report`.

Findings:
- **Account-history features matter most.** XGBoost scores 0.172 PR-AUC on raw payment fields
  and 0.715 with causal graph features.
- **Gradient boosting wins.** Sequence models help a little when blended in.
- **The GNN loses under a strict time split** (0.516). GAN augmentation does not help (0.711 vs 0.715).
- **No payment-format shortcut.** Removing payment format costs XGBoost only 0.025 PR-AUC.

## Run it

```bash
pip install -r requirements.txt polars
uvicorn index:app --port 8001          # open http://localhost:8001
```
On Windows: `.\run_local.ps1`.

Pages:
- `/` is the landing page.
- `/console` is the Live Console: load account history, then stream real test-period payments.
- `/report` is the benchmark report.
- `/docs` is the interactive API.

For benchmark-level recall in the console, set `RUSETECT_HISTORY_GLOB` to the IBM HI-Small
transaction CSV chunks. The engine then ingests all 4.06M pre-test payments as account history
(about 85 s). Without it, the console uses the bundled 100K-payment sample.

### API

```bash
curl -X POST localhost:8001/api/v2/payment -H 'Content-Type: application/json' -d '{
  "timestamp": "2022-09-09 10:00", "from_bank": "070", "from_account": "A1",
  "to_bank": "011", "to_account": "B2", "amount_paid": 9300, "payment_format": "ACH"}'
```
The response carries the decision (Approve / Step-up verification / Hold for analyst / Block + STR draft),
the layer it exited at, the top reasons (XGBoost SHAP contributions in plain words) and a ring case id.
`POST /api/v2/onboard` scores an account application in the BAF schema.

## Deploy on Vercel

1. Import this repo in Vercel. No settings are needed: `index.py` exposes the FastAPI `app`, and
   `requirements.txt` holds only runtime dependencies.
2. LightGBM models are served from JSON by `service/lgbm_lite.py`, a pure-NumPy evaluator that
   exactly matches the lightgbm package. The lightgbm wheel needs a system libgomp that serverless
   runtimes lack. `xgboost-cpu` avoids the 190 MB GPU dependency.

Serverless instances do not share memory, so the console replay can reset between requests on
Vercel. For a live demo, run it locally or on an always-on host (Render, Railway, Fly).

## Reproduce the models

```bash
pip install -r requirements-ml.txt
export RUSETECT_DATA=/path/to/data      # aml_HI-Small_part0*.csv.gz, HI-Small_Patterns.txt, baf_Base_part*.csv.gz
python ml/aml_features.py              # causal features (~70 s)
python ml/extract_patterns.py          # ground-truth rings
python ml/bench_tabular.py             # boosting + classical ML + IsolationForest
python ml/bench_deep.py                # MLP, 1D-CNN, LSTM, Transformer, GAN augmentation
python ml/bench_gnn.py                 # GNN, strictly causal
python ml/bench_baf.py                 # onboarding models
python ml/cascade.py                   # build the cascade, export service/models/*
python ml/l0_onboarding.py             # L0 with fairness thresholds
python ml/parity_check.py              # streaming features == offline features
python ml/online_parity.py             # live scores == offline scores (100% same block decisions)
pytest tests/
```
Hardware used: 2 CPUs, 7 GB RAM.

Protocol:
- **AML:** time-ordered 60/20/20 split; every feature uses only earlier payments; thresholds chosen on validation.
- **GNN:** embeddings come from the past graph only.
- **BAF:** train on months 0–4, set thresholds on month 5, test on months 6–7.

## Limits

- Both datasets are synthetic, so thresholds must be re-tuned on real UPI/IMPS data.
- The lure and pressure stages of a scam (calls, chats) are not in either dataset. This system
  covers the money side: mule account opening, payment and drain.
- The engine keeps account state in memory. Production needs a persistent feature store
  (e.g. Redis) and a history replay at start-up.
