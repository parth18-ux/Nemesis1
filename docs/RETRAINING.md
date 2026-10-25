# Training on the data that was left unused

Two parts of the data had never been used to train anything: **IBM AMLworld LI-Small** (6,924,049 payments, a lower
laundering intensity population) and **Feedzai BAF Variants I to V** (5 million account applications with shifted
bias). This page records how they were added to training, how the result was checked, and what changed.

## Payment models (L1 gatekeeper, L2 XGBoost + LightGBM)

**Preprocessing.** `ml/build_live_features.py` streams every payment of each file, in time order, through
`service/compact_state.py`, the serving code's own feature state (tested equal to the live engine). So training uses
exactly the numbers the API computes: same currency rates, same format and currency codes, same past only windows.
Each file is split by time into 60% train, 20% validation, 20% test.

**Training.** `ml/retrain_extended.py`, same recipe as the original cascade: all laundering payments in the train
weeks plus 10% of the others, L1 = 30 shallow trees with a threshold keeping 99% of validation laundering, L2 = mean
of XGBoost (400 trees) and LightGBM (600 trees).

**Validation.** Three candidates scored exactly like the live path:

| Model | Validation PR-AUC HI / LI | Test PR-AUC HI | Test PR-AUC LI | Test ROC-AUC HI / LI | Caught at verify volume HI / LI | Caught at hold volume HI / LI |
|---|---|---|---|---|---|---|
| Previous live models (HI-Small only) | 0.628 / 0.283 | 0.716 | 0.336 | 0.984 / 0.960 | 1585 / 645 | 1461 / 529 |
| Retrained, HI-Small only (control) | 0.623 / 0.277 | 0.707 | 0.328 | 0.985 / 0.963 | 1598 / 660 | 1461 / 542 |
| Retrained, HI-Small + LI-Small (now live) | 0.633 / 0.298 | 0.711 | 0.345 | 0.985 / 0.967 | 1595 / 664 | 1432 / 533 |

Test weeks: 1,797 laundering payments in HI-Small, 925 in LI-Small. Alert volumes: verify 15,900 / 19,305, hold 5,925 / 6,683 payments.

The swap rule was written into the script before any test number was read: validation PR-AUC on HI-Small at most
0.005 lower and at least 0.01 higher on LI-Small. The HI + LI model passed, so it is now live, with verify, hold and
block thresholds set to keep the previous alert rates on HI-Small validation weeks.

**What it means.** Adding LI-Small beats the fair control (the same recipe on HI-Small alone) on test PR-AUC for both datasets, and
at the "verify" level it catches more laundering on both datasets for the same number of alerts. At the "analyst
hold" level it gains on LI-Small and loses a little on HI-Small. The previous models are in git history and in
`work/previous_*` if that trade off should go the other way.

After the swap, the full streaming runs with the new live models (`ml/full_dataset_eval.py`, every payment through the API's
engine) give PR-AUC 0.711 on HI-Small and 0.345 on LI-Small test weeks, matching the table, and every one of the 60
labelled LI-Small rings in its test weeks gets at least one flag (59 of 60 before).

The training features were checked twice: the previous live models score the same PR-AUC on these features (0.716 on
HI-Small, 0.336 on LI-Small test weeks) as in the full streaming runs (`results/full_eval_*.json`).

## Account opening gate (L0)

**Preprocessing.** Same columns and category order as the served model (`service/models/l0_config.json`); Variants
III and V's leaked features x1 and x2 are dropped. Each population uses the same time split as the original gate:
months 0 to 4 train, month 5 sets thresholds (5% false alarms separately for applicants 50+ and under 50), months 6
to 7 test.

**Training.** `ml/retrain_l0_variants.py`: Base months 0 to 4 plus every variant's months 0 to 4 (all fraud cases,
30% of the other applications per variant), same LightGBM settings as before. A control retrained on Base alone
reproduces the previous model exactly, so the recipe is deterministic.

**Validation.** Rule fixed in advance on month 5: mean ROC-AUC over the six populations at least 0.003 higher, Base no
more than 0.003 lower, worst age parity at least 0.8. Month 5 mean ROC-AUC went from 0.8905 to
0.8945; the model passed and is live.

| Population (months 6 to 7) | ROC-AUC before → after | Recall at 5% false alarms before → after | Recall at the gate's thresholds before → after | Age parity before → after |
|---|---|---|---|---|
| Base | 0.892 → 0.897 | 0.541 → 0.555 | 0.534 → 0.559 | 0.88 → 0.93 |
| Variant I | 0.882 → 0.891 | 0.510 → 0.528 | 0.535 → 0.561 | 0.88 → 0.91 |
| Variant II | 0.897 → 0.899 | 0.549 → 0.549 | 0.523 → 0.541 | 0.89 → 0.92 |
| Variant III | 0.870 → 0.880 | 0.475 → 0.489 | 0.529 → 0.547 | 0.89 → 0.92 |
| Variant V | 0.872 → 0.882 | 0.486 → 0.507 | 0.541 → 0.557 | 0.89 → 0.94 |

Mean test ROC-AUC over the six populations: 0.8805 → 0.8882. Every population
improves; age parity improves everywhere. The false alarm rate on Base test months moves from 5.4% to 5.8%
(thresholds are set on month 5, and month 6 to 7 applications differ slightly). Variant IV is left out of the table
because its test months repeat Variant III's applications.

## Reproduce

```bash
python ml/build_live_features.py HI && python ml/build_live_features.py LI
python ml/retrain_extended.py --adopt          # writes service/models only if the rule passes
python ml/retrain_l0_variants.py --adopt
python -m pytest -q tests
```
All data synthetic. Results: `results/retrain_extended.json`, `results/retrain_l0_variants.json`.
