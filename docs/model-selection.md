# Model selection

Why each RuseTect layer uses the model it does, what it was compared against, and what is still unmeasured.
All payment numbers are on the IBM AMLworld HI-Small **test period** (1,015,669 payments, 1,797 laundering, 0.18%),
with features computed only from earlier payments and thresholds picked on the validation period. All message
numbers are out-of-fold scores on templates the model never saw. Every dataset involved is synthetic or
non-Indian (see [datasets.md](datasets.md)), so treat the thresholds as starting points.

## What runs live

| Layer | Live model | Why this one |
|---|---|---|
| L-1 Lure detector | Logistic regression on hashed TF-IDF (word 1–2 grams + char 3–5 grams), Indian rows weighted ×4, plus a 6-rule "never legitimate" floor | Best average precision on Indian messages among the models tried; weights are a flat vector, so it serves in pure Python/NumPy and every score can be traced to phrases |
| L0 Onboarding gate | LightGBM, separate thresholds for age 50+ and under 50 | Highest recall at 5% FPR of 8 models on Feedzai BAF; per-group thresholds bring the age FPR ratio from 0.31 to 0.88 |
| L1 Gatekeeper | 30-tree XGBoost | Cheap (0.3 ms); approves ~80% of payments while passing on 99.2% of laundering |
| L2 Graph-boost | Mean of XGBoost and LightGBM | Top two of the benchmark, near-identical accuracy; averaging steadies scores; XGBoost gives SHAP reasons |
| L4 Ring linker | Union-find over blocked payments | Deterministic, explainable; turns 1,105 alerts into 318 cases offline |

Random Forest, the LSTM/Transformer blend (L3) and the GNN exist only in the offline benchmark.

## Payment side: the comparisons

### Committed benchmark (21 variants, `service/models/benchmark_results.json`, shown at `/report`)

| Model | PR-AUC | Note |
|---|---|---|
| L3 blend (XGB + LGBM + RF + LSTM/Transformer) | 0.725 | offline only |
| XGBoost + graph features | 0.715 | live (in L2) |
| GAN-augmented XGBoost | 0.711 | augmentation did not help |
| Random Forest + graph | 0.706 | 53 ms/row model latency |
| CatBoost + graph | 0.700 | |
| LightGBM + graph | 0.699 | live (in L2) |
| LSTM / Transformer / 1D-CNN (sequence) | 0.662 / 0.655 / 0.644 | alone; help a little when blended |
| MLP | 0.574 | |
| Decision tree | 0.533 | |
| GNN + graph features (strictly causal) | 0.516 | can't embed accounts first seen at test time |
| AdaBoost + graph | 0.501 | |
| KNN (k=50, subsample) | 0.449 | |
| XGBoost on raw fields only | 0.172 | **graph features are worth +0.54** |
| Logistic regression / linear SVM / RBF-SVM + graph | 0.13 / 0.13 / 0.12 | |
| Isolation Forest (unsupervised) | 0.004 | laundering looks normal row by row |

### Fresh comparison (`ml/compare_models.py`): 7 of 14 models finished

Same split; each model gets its own small tuning grid scored on validation; the test set is used once.
Partial file: `results/model_comparison_tabular.partial.json`.

| Model | Avg. precision | Precision | Recall | False alarms | Missed |
|---|---|---|---|---|---|
| Random Forest | 0.706 | 0.904 | 0.548 | 104 | 813 |
| Decision tree | 0.580 | 0.844 | 0.471 | 157 | 950 |
| KNN (40K-row fit, re-weighted test) | 0.438 | 0.669 | 0.393 | ~350 | ~1,090 |
| Logistic regression | 0.169 | 0.312 | 0.489 | 1,939 | 918 |
| Linear SVM | 0.162 | 0.300 | 0.496 | 2,077 | 905 |
| Gaussian Naive Bayes | 0.013 | 0.015 | 0.698 | 84,165 | 542 |
| Dummy (class prior) | 0.002 | — | 0 | 0 | 1,797 |

**Still to run** (needs the IBM data, which is not in this repo):
Extra Trees, AdaBoost, HistGradientBoosting, XGBoost, LightGBM, CatBoost, MLP; then the sequence models
(`python ml/bench_deep.py tcn cnn lstm transformer`, then `ml/export_sequence_results.py`); then
`ml/audit_live_path.py` and `ml/build_report.py`. When `results/model_comparison_tabular.json` exists, delete the
`.partial.json`.

### "Why not TCN, Gaussian, or …?"

- **TCN**: added to `ml/bench_deep.py`; not yet run. Expect it near the LSTM/1D-CNN (0.64–0.66), since all three
  read the same 12-event sequences; it only goes live if it lifts the blend on validation.
- **Gaussian Naive Bayes**: measured, 0.013. Its independence assumption fails badly on correlated
  account-history counts.
- **Gaussian Process classifier**: not run. Training is O(n³) time and O(n²) memory: infeasible on 307K training
  rows. A sparse/variational GP would be the only option, and kernel methods (RBF-SVM 0.12) give no reason to expect a win.
- **GNN**: 0.516 under a strict time split. Published GNN numbers on this data build the test graph with future
  edges.

## Message side (L-1): the comparisons

`python ml/train_message_model.py` → `results/message_model_eval.json`. 7,365 messages: 5,574 real UCI SMS (747 spam)
and 1,791 synthetic Indian messages (1,025 scam) from 144 templates (15 scam workflows, 8 legitimate kinds, including hard cases
such as "do not share your OTP", genuine bill reminders and relatives asking for money). Five-fold cross-validation
holds out whole templates.

| Model | Avg. precision, Indian | Avg. precision, UCI | Indian @ high threshold: precision / recall |
|---|---|---|---|
| **Logistic regression, India-weighted (served)** | **0.944** | 0.982 | **0.950 / 0.668** |
| Logistic regression, unweighted | 0.931 | 0.984 | 0.942 / 0.588 |
| Blend: LR + Naive Bayes | 0.941 | 0.984 | 0.856 / 0.800 |
| Multinomial Naive Bayes | 0.955 | 0.949 | 0.827 / 0.878 (below the 0.95 precision bar) |
| Rules only | 0.983\* | 0.363 | — |

\* The rules and the Indian templates were written by the same people, so this number is circular. On real UK SMS
the same rules score 0.363, which shows how they generalise.

**Served decision** (model bands + hard-signal floor), Indian messages, out-of-fold:

| Band | Precision | Recall | Legitimate messages flagged |
|---|---|---|---|
| High (push warning) | 0.951 | 0.677 | 4.7% (all 36 are genuine IT/PF refund or similar service notices) |
| Caution or high | 0.917 | 0.821 | 9.9% |

The hard-signal list (remote-access app, `.apk`, PIN to *receive*, "digital arrest", "safe/RBI account",
"share the OTP") fired on 0 of 5,593 legitimate messages (Indian and UCI). Marketing SMS land in caution 75% of the time; they are
never pushed. Fixing the refund-notice false alarms needs the sender ID (TRAI DLT header such as `VM-ITDEPT`), which
the text alone does not carry.

Why not a transformer (e.g. IndicBERT, MuRIL)? With 144 templates it would memorise the phrasing, and it would
need a GPU-backed host instead of a 3 MB JSON file. Revisit once a real labelled corpus exists (see datasets.md).

## Alert fatigue: the decision rules

The problem statement asks for "timely, useful alerts without alert fatigue". RuseTect alerts at three points, each
with a budget:

1. **Push warning to the user**: only the high message band (precision 0.95). At most one per (account, workflow) per
   24 h and three per account per day. Caution messages are never pushed.
2. **Interruption at payment time**: a payment to a **new** payee within 72 h of a risky message gets step-up
   verification plus a one-line prompt. Payments to known payees are not touched.
3. **Analyst queue**: only Block + STR alerts and payments to a UPI ID or account named in a scam message; L4
   groups them into ring cases (3.5× fewer items).

`GET /api/v2/alerts/summary` counts pushed vs suppressed and why, so the budget can be monitored.
