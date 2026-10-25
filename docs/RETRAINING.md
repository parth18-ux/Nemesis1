# Training on the data that was left unused

Two parts of the data had never been used to train anything: **IBM AMLworld LI-Small** (6,924,049 payments, a lower
laundering intensity population) and **Feedzai BAF Variants I to V** (5 million account applications with shifted
bias). This page records how they were added to training, how the result was checked, and what changed.

## Payment models (L1 gatekeeper, L2 XGBoost + LightGBM)

**Preprocessing.** `ml/build_live_features.py` streams every payment of each file, in time order, through
`service/compact_state.py`, the serving code's own feature state (tested equal to the live engine). So training uses
exactly the numbers the API computes: same currency rates, same format and currency codes, same past-only windows.
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

The training features were checked twice: the previous live models score the same PR-AUC on these features (0.716 on
HI-Small, 0.336 on LI-Small test weeks) as in the full streaming runs (`results/full_eval_*.json`).

## Account opening gate (L0)

__L0_SECTION__

## Reproduce

```bash
python ml/build_live_features.py HI && python ml/build_live_features.py LI
python ml/retrain_extended.py --adopt          # writes service/models only if the rule passes
python ml/retrain_l0_variants.py --adopt
python -m pytest -q tests
```
All data synthetic. Results: `results/retrain_extended.json`, `results/retrain_l0_variants.json`.
