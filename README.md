# RuseTect

**Layered AI filtration for mule accounts and scam money flows.**

RuseTect scores bank payments and new account applications for money-mule activity. A small model clears
most payments quickly; the rest are scored by gradient-boosted trees, explained per feature, and linked into ring
cases for an analyst. It is a research prototype trained and evaluated on **synthetic** datasets
(IBM AMLworld, Feedzai BAF); nothing here shows performance on real banking traffic.

## What runs live

| Layer | Model (file) | What it does |
|---|---|---|
| **L0 onboarding** | LightGBM (`service/models/l0_onboarding_lgbm.json`) | Scores an account application; separate thresholds for applicants 50+ and under 50 |
| **L1 gatekeeper** | XGBoost, 30 trees, depth 3 (`l1_gatekeeper.json`) | Approves most payments; passes the rest on |
| **L2 graph-boost** | XGBoost (`l2_xgb.json`) + LightGBM (`l2_lgbm.json`), averaged | Risk score → Approve / Step-up / Hold / Block + STR draft, with XGBoost TreeSHAP reasons |
| **L4 ring linker** | Union-find over blocked payments | Merges blocks that share accounts into one case |

Random Forest and LSTM/Transformer models appear only in the **offline** benchmark blend ("L3"). The live engine
(`service/rusetect_engine.py`) never loads them. LightGBM models are served from JSON by `service/lgbm_lite.py`,
a pure-NumPy evaluator tested against the lightgbm package, because the lightgbm wheel needs a system library that
serverless runtimes lack.

Payment features: 31 per payment, each computed only from earlier payments: amounts, format, sender fan-out and
receiver fan-in over 24 h, pass-through ratio, minutes since money came in, new counterparties, money flowing back.
See `ml/aml_features.py` (offline) and `Engine.features` (live; parity checked by `ml/parity_check.py`).

## Results

- **Committed benchmark results** (`service/models/benchmark_results.json`, produced earlier by `ml/bench_*.py`,
  `ml/cascade.py`, `ml/l0_onboarding.py`): XGBoost + account-history features 0.715 PR-AUC on payments, against
  0.172 for XGBoost on raw fields. Random Forest 0.706, CatBoost 0.700, LightGBM 0.699. Offline L3 blend 0.725.
- **Fresh comparison** (`results/model_comparison_tabular.json`, `results/model_comparison_sequence.json`): 14 tabular
  models plus 4 sequence models (TCN, LSTM, Transformer, 1D-CNN), re-run with one protocol.
  See `docs/model-selection.md`.
- **Live path audit** (`results/live_path_audit.json`): what the served configuration actually does on the test period.
  See `docs/project-audit.md`.

Why gradient-boosted trees, and what the alternatives showed: `docs/model-selection.md`.

## Run it

```bash
pip install -r requirements.txt polars
uvicorn index:app --port 8001          # open http://localhost:8001
```
On Windows: `.\run_local.ps1`, or `.\run_local.ps1 -HistoryGlob "D:\data\aml_HI-Small_part0*.csv.gz"`.

Pages:
- `/` is the landing page.
- `/console` replays recorded test-period payments from the synthetic IBM dataset through the live path.
- `/report` is the benchmark report.
- `/docs` is the interactive API.

`RUSETECT_HISTORY_GLOB` points the console at the full IBM HI-Small transaction CSVs, so every account has history
(about 85 s to load 4.06M payments). Without it, the console uses a bundled 100K-payment sample and recall is lower.

### API

```bash
curl -X POST localhost:8001/api/v2/payment -H 'Content-Type: application/json' -d '{
  "timestamp": "2022-09-09 10:00", "from_bank": "070", "from_account": "A1",
  "to_bank": "011", "to_account": "B2", "amount_paid": 9300, "payment_format": "ACH"}'
```
Other routes: `POST /api/v2/payments/batch`, `POST /api/v2/onboard`, `GET /api/v2/cases`,
`GET /api/v2/model-card`, `GET /api/health`.

## Reproduce the models and results

The raw datasets are **not in this repository**. Download them and point `RUSETECT_DATA` at a folder containing:
- `aml_HI-Small_part0*.csv.gz`: IBM AMLworld HI-Small transactions (Kaggle `ealtman2019/ibm-transactions-for-anti-money-laundering-aml`, file `HI-Small_Trans.csv`, split into gzip chunks)
- `HI-Small_Patterns.txt`: from the same dataset
- `baf_Base_part*.csv.gz`: Feedzai Bank Account Fraud `Base.csv` (Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022`)

```bash
pip install -r requirements-ml.txt
export RUSETECT_DATA=/path/to/data RUSETECT_WORK=/path/to/work
python ml/aml_features.py              # causal payment features -> $RUSETECT_WORK/aml_feat.parquet
python ml/extract_patterns.py          # ground-truth rings
python ml/bench_tabular.py             # committed AML tabular benchmark
python ml/bench_deep.py                # MLP, 1D-CNN, TCN, LSTM, Transformer, GAN augmentation
python ml/bench_gnn.py                 # GNN (strictly causal)
python ml/bench_baf.py                 # onboarding models
python ml/cascade.py                   # build the cascade, export service/models/*
python ml/l0_onboarding.py             # L0 with age-group thresholds
python ml/compare_models.py            # fresh tabular comparison -> results/model_comparison_tabular.json
python ml/export_sequence_results.py   # sequence models -> results/model_comparison_sequence.json
python ml/audit_live_path.py           # live path audit -> results/live_path_audit.json
python ml/parity_check.py              # streaming features == offline features
python ml/build_report.py              # regenerate service/static/report.html from report/report_template.html
pytest tests/
```
The report has one source: edit `report/report_template.html` and run `python ml/build_report.py`.
`tests/test_branding.py` fails if the built page is stale.

## Deploy

Vercel: `index.py` exposes the FastAPI `app` and `requirements.txt` holds runtime dependencies only. Serverless
instances do not share memory, so the console replay can reset between requests there; run locally or on an
always-on host for demos.

## Migrating from the previous name

This project was previously called RuseTect.
- `RUSETECT_DATA`, `RUSETECT_WORK` and `RUSETECT_HISTORY_GLOB` are still read as fallbacks; prefer `RUSETECT_*`.
- `service/rusetect_engine.py` is now `service/rusetect_engine.py`.
- API routes, model files and the response format are unchanged.

Renaming the source does not change the GitHub repository name or the Vercel project/domain; those are separate
settings (see `docs/project-audit.md`).

## Limits

- Both datasets are synthetic, so thresholds and models must be re-validated on a bank's own data.
- Scam calls and chats (lure and pressure stages) are not modelled. RuseTect covers account opening and money movement.
- The engine keeps account state in memory; production needs a persistent feature store and a history replay at start-up.
