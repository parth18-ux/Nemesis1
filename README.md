# RuseTect

RuseTect detects scam payments and money mule activity in digital banking. It follows a scam from the first message a
victim receives to the accounts that launder the money, and it answers with the lightest intervention that fits the
evidence: a question to the customer, a short pause, an analyst review, or a block approved by two analysts.

Live demo: [rusetect.vercel.app](https://rusetect.vercel.app) · Author: Parth Sharma

> RuseTect is a research project. Its models are trained and evaluated on public datasets that are synthetic or
> collected outside banking (IBM AMLworld, Feedzai BAF, PaySim, Aalto keystrokes, Balabit mouse data). The results below show
> how the methods behave on that data, not performance on real bank traffic.

## How it works

A payment passes through a cascade. A small gatekeeper model approves most payments quickly; the rest are
scored by two gradient boosted models built on 31 account history features, each computed only from earlier payments.
Around that core, independent evidence domains raise or lower the response, and no single domain can hold money on its
own.

| Layer | What it does | Code |
|---|---|---|
| Message (L-1) | Scores an SMS, WhatsApp message or call transcript, names the scam workflow (digital arrest, KYC block, task job, voice clone emergency) and extracts UPI IDs, phone numbers and links | `service/scam_text.py` |
| Message to payment | Asks for confirmation on a payment to a new payee within 72 hours of a risky message, and holds payments to an identifier named in one | `service/lure_fusion.py` |
| Account opening (L0) | Scores new account applications for mule risk, with separate thresholds for applicants under and over 50 | `service/models/l0_onboarding_lgbm.json` |
| Payment cascade (L1, L2) | XGBoost gatekeeper, then XGBoost and LightGBM averaged into a risk score with per feature reasons | `service/rusetect_engine.py` |
| Ring linking (L4) | Merges blocked payments that share accounts into a single case | `service/rusetect_engine.py` |
| Context rules | Combines message, payment, account graph and session evidence; escalation needs two domains | `service/context_rules.py` |
| Session and identity | Live call, screen sharing, remote access apps, pasted payee details, hesitation, SIM swap and device checks before the OTP, account takeover sequences | `service/session_signals.py` |
| Person | Typing rhythm and mouse dynamics compared with the customer's own profile | `service/behaviour.py`, `service/mouse_dynamics.py` |
| Intervention | Pause screen in English and Hindi with a question chosen from the evidence, cool off period, duress PIN | `service/intervention.py` |
| Network | Links victims of the same scam campaign; a signed, hash chained ledger lets banks share mule flags without sharing personal data | `service/campaigns.py`, `service/consortium.py` |
| Review desk | Analyst keys and roles, two analyst approval for blocks, signed verdicts and evidence packs, STR drafts | `service/review.py`, `service/security.py`, `service/str_report.py` |

LightGBM models are served from JSON by a pure NumPy evaluator (`service/lgbm_lite.py`), tested against the lightgbm
package, so the API runs on serverless platforms without native libraries.

## Results

Every number comes from a script in `ml/` and a file in `results/`.

| Area | Data | Result |
|---|---|---|
| Payment models | IBM AMLworld HI-Small, every payment streamed through the live engine, last 20% of time as test | PR-AUC 0.711, ROC-AUC 0.985 |
| Payment models, lower laundering rate | IBM AMLworld LI-Small, same protocol | PR-AUC 0.345, ROC-AUC 0.967 |
| Account opening | Feedzai BAF Base and Variants I to V, months 6 to 7 | mean ROC-AUC 0.888 over six populations; false alarm parity between age groups 0.93 on Base |
| Messages | Synthetic Indian scam messages from templates held out of training | 0.95 precision at 68% recall for push warnings; no false alarms on genuine OTP, bank, bill and delivery messages |
| Typing rhythm | Aalto 136M Keystrokes, 2,868 people never seen in training | ROC-AUC 0.890 from one sentence, 0.936 from three; 64% of impostors caught at 5% false alarms, equal across age groups |
| Mouse dynamics | Balabit Mouse Dynamics Challenge, labelled impostor sessions | median per user ROC-AUC 0.915 from 20 mouse actions |
| End to end scenarios | 480 authored, randomised runs through the API | 280 of 280 scam payments stopped before money moved; 1 of 200 genuine payments interrupted |
| Mobile money | PaySim, 6.36M transactions, last 20% of time as test | PR-AUC 0.52 from account history alone; near perfect once balances are used, because every simulated fraud empties the account ([note](docs/datasets.md)) |
| Throughput | Full HI-Small file, 5.08M payments, one CPU core | 41,459 payments per second with compact account state |

Details: [model selection](docs/model-selection.md), [retraining with the extra data](docs/RETRAINING.md),
[person, session and network layer](docs/SESSION_AND_NETWORK.md), [context layer](docs/CONTEXT_LAYER.md),
[scaling](docs/SCALING.md), [datasets](docs/datasets.md), [project audit](docs/project-audit.md).

## Quick start

```bash
pip install -r requirements.txt polars
uvicorn index:app --port 8001
```

Open `http://localhost:8001`. The site has four pages: the overview (`/`), the investigator (`/investigate`) where you
can replay scam scenarios and review alerts, a live console (`/console`) that replays test period payments through the
engine, and the benchmark report (`/report`). The interactive API reference is at `/docs`. On Windows you can also run
`.\run_local.ps1`.

Score a payment:

```bash
curl -X POST localhost:8001/api/v2/payment -H 'Content-Type: application/json' -d '{
  "timestamp": "2022-09-09 10:00", "from_bank": "070", "from_account": "A1",
  "to_bank": "011", "to_account": "B2", "amount_paid": 9300, "payment_format": "ACH"}'
```

Score a message and link it to the payer, so that their next payment to a new payee is escalated:

```bash
curl -X POST localhost:8001/api/v2/message -H 'Content-Type: application/json' -d '{
  "text": "Your SBI account will be blocked today. Update PAN at http://sbi-kyc-update.in/verify",
  "timestamp": "2022-09-09 09:00", "bank": "070", "account": "A1"}'
```

The full assessment with session signals is `POST /api/v2/assess`. Other routes include `POST /api/v2/onboard`,
`GET /api/v2/cases`, `GET /api/v2/campaigns`, `GET /api/v2/model-card` and `GET /api/health`.

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest tests/
```

The Redis scale test starts its own Redis, so it needs the `redis-server` program on your PATH (Linux:
`apt install redis-server`, macOS: `brew install redis`, Windows: run the tests in WSL). Without it, that test file is
skipped.

For the console to use full account history, point `RUSETECT_HISTORY_GLOB` at the IBM HI-Small CSV files (about 85
seconds to load 4.06M payments); otherwise it uses a bundled sample of 100,000 payments.

## Reproducing the results

The datasets are not stored in this repository. Download them, then point `RUSETECT_DATA` at the folder that holds them.

| Dataset | Source |
|---|---|
| IBM AMLworld HI-Small and LI-Small | [Kaggle: ealtman2019/ibm-transactions-for-anti-money-laundering-aml](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) |
| Feedzai Bank Account Fraud (Base, Variants I to V) | [Kaggle: sgpjesus/bank-account-fraud-dataset-neurips-2022](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022) |
| Aalto 136M Keystrokes | [userinterfaces.aalto.fi/136Mkeystrokes](https://userinterfaces.aalto.fi/136Mkeystrokes/) |
| Balabit Mouse Dynamics Challenge | [github.com/balabit/Mouse-Dynamics-Challenge](https://github.com/balabit/Mouse-Dynamics-Challenge) |
| PaySim | [Kaggle: ealaxi/paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |

```bash
pip install -r requirements-ml.txt
export RUSETECT_DATA=/path/to/data RUSETECT_WORK=/path/to/work

python ml/aml_features.py              # payment features
python ml/cascade.py                   # build the payment cascade and export service/models
python ml/l0_onboarding.py             # account opening model with age group thresholds
python ml/train_message_model.py       # message model
python ml/compare_models.py            # 14 tabular models, one protocol
python ml/bench_deep.py tcn cnn lstm transformer && python ml/export_sequence_results.py   # sequence models
python ml/audit_live_path.py           # live payment path with the served models and thresholds
python ml/full_dataset_eval.py HI      # stream a full dataset through the live engine (also LI)
python ml/retrain_extended.py          # retraining with LI-Small (see docs/RETRAINING.md)
python ml/retrain_l0_variants.py       # retraining with the BAF variants
python ml/aalto_eval.py SAMPLE.zip     # typing rhythm on Aalto keystrokes
python ml/balabit_eval.py              # mouse dynamics on Balabit
python ml/paysim_eval.py PAYSIM.csv    # payment recipe on PaySim
python ml/scenario_suite.py            # end to end scenarios through the API
python ml/build_report.py              # rebuild the benchmark report page
pytest tests/
```

The benchmark report has a single source, `report/report_template.html`; the tests fail if the built page is stale.

## Deployment

`index.py` exposes the FastAPI app and `requirements.txt` lists runtime dependencies only, which is all Vercel needs.
Deploy from a Git integration or from the project folder with the Vercel CLI: `npx vercel deploy --prod`.
Serverless instances do not share memory, so demo state such as the console replay can reset between requests. For a
bank deployment, `service/scale` keeps account state in Redis and spreads payments over workers by payer, and
`deploy/` has a Dockerfile and a compose file; see [scaling](docs/SCALING.md).

Demo mode publishes its analyst keys so the review desk can be tried. Set `RUSETECT_ANALYST_KEYS` and
`RUSETECT_SIGNING_KEY` before using it with anything real.

## Project layout

```
service/          API, live engine, risk layers, review desk, static pages
service/models/   served models and thresholds
service/scale/    Redis state, stream workers and replay for scale out
ml/               feature building, training, benchmarks and evaluations
results/          measured results, one JSON file per evaluation
report/           source of the benchmark report page
docs/             design notes and results
tests/            API, model, context, review, scale and branding tests
```

## Limitations

The payment and account opening data are synthetic, and the message corpus was written from public advisories, so
models and thresholds must be validated again on a bank's own data before use. Call audio is not modelled; the message
layer needs a transcript. Session signals have no public benchmark, and the typing and mouse checks were validated on
desktop and typing test data rather than inside a banking app. The demo keeps state in memory; production needs the
Redis state layer or a feature store.

## Acknowledgements

Datasets by IBM Research (AMLworld, Altman et al., NeurIPS 2023), Feedzai (Bank Account Fraud, Jesus et al., NeurIPS
2022), Aalto University (Dhakal, Feit, Kristensson and Oulasvirta, "Observations on Typing from 136 Million
Keystrokes", CHI 2018), Balabit (Mouse Dynamics Challenge) and Lopez-Rojas, Elmir and Axelsson (PaySim, 2016).
