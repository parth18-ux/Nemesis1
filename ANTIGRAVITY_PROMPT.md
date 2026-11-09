# Prompt for the Antigravity agent

Copy everything below the line into the agent chat.

---

You are working on RuseTect, my open source scam payment and money mule detection system (FastAPI service, XGBoost and LightGBM models, trained and evaluated on public synthetic datasets). I am Parth Sharma, the sole author.

Before changing anything, read these files:
1. README.md
2. docs/research/RuseTect research audit and roadmap.md (also as a PDF in the same folder): a strict audit of this repo, a dataset catalogue, 20 jury questions and a four level roadmap
3. docs/project-audit.md and docs/model-selection.md
4. results/live_path_audit.json, results/full_eval_HI.json, results/l2_halves_check.json

Facts to respect:
- The live payment path is: L1 XGBoost gatekeeper (30 trees), then L2 = mean of XGBoost and LightGBM on 31 past-only features (service/rusetect_engine.py). Features for training are computed by the serving code, so training and serving match.
- Every evaluation is chronological (time ordered 60/20/20 split, thresholds chosen on validation, test read once). Model swaps follow adoption rules written before test numbers are read. Keep this discipline.
- Never fabricate metrics. Every number in docs must come from a file in results/ produced by a script in ml/.
- Large datasets stay outside Git (data/ and work/ are ignored). Point scripts at them with RUSETECT_DATA and RUSETECT_WORK.
- Commits are authored by me: git config user.name "Parth Sharma". Do not add AI co-author lines to commits.
- Writing style: plain English, no em dashes or en dashes, minimal hyphenated compounds.

Task: carry out Level 1 of the roadmap in the audit, in this order, committing after each step with all tests passing (pip install -r requirements-dev.txt, then pytest tests/):

1. Hygiene and claims (roadmap 1.7): fix stale docs (results/full_eval_LI.json notes and docs/SCALING.md still say LI-Small was never trained on; it now is). Replace the silent "ACH" default for a missing payment_format with an explicit missing indicator, and add a test. Compute reason codes over the full L2 score, not only the XGBoost half.
2. Calibrate and fix the block band (roadmap 1.1): calibrate XGBoost and LightGBM separately on validation (isotonic or Platt), average calibrated scores, refit the verify, hold and block thresholds on validation, report the before and after on test in a new results file, and update cascade_config.json only if a rule written beforehand passes.
3. Improved validation (roadmap 1.3): add bootstrap confidence intervals to headline metrics, run the same pipeline with a shuffled random split to measure how much a random split overstates performance, and simulate label delay.
4. Better benchmarking (roadmap 1.5): replace the degenerate rules baseline with expert rules tuned on validation; report precision at a fixed daily alert budget, recall at fixed false positive rates, and an amount weighted metric.
5. Error analysis (roadmap 1.6): false negatives and positives by typology, amount and account age; ring level precision.

If a step needs data that is not present (for example the IBM HI-Small CSVs), stop that step, tell me exactly which files to download and where to put them, and continue with the steps that do not need it.

After each step, summarise in a few lines what changed, what was measured, and which files were committed.
