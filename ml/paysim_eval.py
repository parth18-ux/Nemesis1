"""RuseTect's payment recipe on PaySim (synthetic mobile money, Lopez-Rojas, Elmir and Axelsson, 2016).

Data: https://www.kaggle.com/datasets/ealaxi/paysim1 (one CSV, 6,362,620 transactions over 743 hourly steps, 8,213
fraud cases). In PaySim's story a fraudster takes over a customer's account and empties it with a TRANSFER, then a
CASH_OUT. This script asks whether RuseTect's approach (past only account features, time ordered split, XGBoost and
LightGBM averaged, thresholds set by alert budget on validation) transfers to this different kind of data.

Protocol: rows in time order; first 60% train, next 20% validation, last 20% test.
Features a bank has when the payment is requested (all past only or from the payer's own account):
  type, amount, hour of day, payer balance before, share of the balance being sent, whether it empties the account,
  how often the payer and the payee were seen before, how many different payers the payee had, money the payee
  received before, hours since the payee was first seen, whether the payee is a merchant.
Left out on purpose: the payee's balances and the payer's balance after, which PaySim fills in a way that gives fraud
away (a known simulator artefact). A model with every balance column is reported separately as a leaky upper bound.
A third run drops every balance feature (type, amount, hour and account history only), to see how much of the result
comes from PaySim's balance pattern rather than from account behaviour.
Baselines: PaySim's own rule (isFlaggedFraud: transfers above 200,000) and a rule "the payment empties the account".

Usage: python ml/paysim_eval.py path/to/PS_20174392719_1491204439457_log.csv     writes results/paysim_eval.json
"""
import json, os, sys, time
import numpy as np, pandas as pd
import lightgbm as lgb, xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T0 = time.time()
log = lambda *a: print(f"[{time.time() - T0:5.0f}s]", *a, flush=True)

df = pd.read_csv(sys.argv[1], dtype={"type": "category", "nameOrig": "string", "nameDest": "string"})
df = df.sort_values("step", kind="stable").reset_index(drop=True)
log(f"{len(df):,} transactions, {int(df.isFraud.sum()):,} fraud")

# ------------------------------------------------------------------ past only features
f = pd.DataFrame(index=df.index)
for t in ("CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"):
    f[f"type_{t}"] = (df["type"] == t).astype(np.int8)
f["amount"] = df["amount"]
f["log_amount"] = np.log1p(df["amount"])
f["hour"] = (df["step"] % 24).astype(np.int8)
f["payer_balance_before"] = df["oldbalanceOrg"]
f["share_of_balance"] = np.where(df["oldbalanceOrg"] > 0, df["amount"] / df["oldbalanceOrg"].clip(lower=1e-9), -1.0)
f["empties_account"] = ((df["oldbalanceOrg"] > 0) & (df["amount"] >= df["oldbalanceOrg"])).astype(np.int8)
f["payer_seen_before"] = df.groupby("nameOrig", sort=False).cumcount()
f["payee_seen_before"] = df.groupby("nameDest", sort=False).cumcount()
f["payee_received_before"] = df.groupby("nameDest", sort=False)["amount"].cumsum() - df["amount"]
first = df.groupby("nameDest", sort=False)["step"].transform("min")
f["payee_hours_known"] = df["step"] - first
first_pair = ~df.duplicated(["nameDest", "nameOrig"])
f["payee_distinct_payers_before"] = first_pair.groupby(df["nameDest"], sort=False).cumsum() - first_pair.astype(int)
f["payee_is_merchant"] = df["nameDest"].str.startswith("M").astype(np.int8)
FEATS = list(f.columns)
leak = f.copy()
leak["payer_balance_after"] = df["newbalanceOrig"]; leak["payee_balance_before"] = df["oldbalanceDest"]; leak["payee_balance_after"] = df["newbalanceDest"]
leak["payer_balance_error"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
leak["payee_balance_error"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
LEAK = list(leak.columns)
y = df["isFraud"].to_numpy()
log("features built")

n = len(df); a, b = int(0.6 * n), int(0.8 * n)
tr, va, te = slice(0, a), slice(a, b), slice(b, n)
rng = np.random.default_rng(0)
keep = (y[tr] == 1) | (rng.random(a) < 0.10)          # every fraud case and 10% of the rest, as in the payment cascade
log({"train fraud": int(y[tr].sum()), "validation fraud": int(y[va].sum()), "test fraud": int(y[te].sum()),
     "test steps": [int(df.step.iloc[b]), int(df.step.iloc[-1])]})


def fit_blend(X):
    Xtr, ytr = X.iloc[tr][keep], y[tr][keep]
    mx = xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           tree_method="hist", n_jobs=2, eval_metric="aucpr").fit(Xtr, ytr)
    ml = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63, min_child_samples=50, subsample=0.8,
                            subsample_freq=1, colsample_bytree=0.8, n_jobs=2, verbose=-1).fit(Xtr, ytr)
    return lambda Z: 0.5 * mx.predict_proba(Z)[:, 1] + 0.5 * ml.predict_proba(Z)[:, 1]


def budget_metrics(score_va, score_te, y_va, y_te):
    """Thresholds set on validation for alert budgets (share of all payments), measured on test."""
    out = {}
    for share in (0.001, 0.005, 0.01):
        thr = float(np.quantile(score_va, 1 - share))
        flag = score_te > thr
        out[f"budget_{share:.1%}"] = {"alerts_share_test": round(float(flag.mean()), 4), "recall": round(float(flag[y_te == 1].mean()), 4),
                                      "precision": round(float(y_te[flag].mean()) if flag.any() else 0.0, 4)}
    return out


res = {"dataset": "PaySim (synthetic mobile money), https://www.kaggle.com/datasets/ealaxi/paysim1",
       "citation": "Lopez-Rojas, Elmir, Axelsson. PaySim: A financial mobile money simulator for fraud detection. EMSS 2016.",
       "rows": n, "fraud": int(y.sum()), "split": "time ordered 60 / 20 / 20", "test_fraud": int(y[te].sum()), "features": FEATS, "models": {}}
HIST = [c for c in FEATS if c not in ("payer_balance_before", "share_of_balance", "empties_account")]
for name, X in (("history_only_no_balances", f[HIST]), ("rusetect_recipe_bank_visible_features", f), ("leaky_upper_bound_all_balance_columns", leak)):
    pred = fit_blend(X)
    s_va, s_te = pred(X.iloc[va]), pred(X.iloc[te])
    res["models"][name] = {"test_pr_auc": round(float(average_precision_score(y[te], s_te)), 4), "test_roc_auc": round(float(roc_auc_score(y[te], s_te)), 4),
                           "validation_pr_auc": round(float(average_precision_score(y[va], s_va)), 4), **budget_metrics(s_va, s_te, y[va], y[te])}
    log(name, res["models"][name]["test_pr_auc"])
for name, rule in (("paysim_isFlaggedFraud_rule", df["isFlaggedFraud"].to_numpy()), ("rule_payment_empties_account", f["empties_account"].to_numpy())):
    r = rule[te].astype(bool)
    res["models"][name] = {"test_alert_share": round(float(r.mean()), 4), "recall": round(float(r[y[te] == 1].mean()), 4),
                           "precision": round(float(y[te][r].mean()) if r.any() else 0.0, 4)}
res["notes"] = ["PaySim is synthetic; fraud only appears in TRANSFER and CASH_OUT, and each fraud case empties an account.",
                "Payee balance columns are left out of the main model because PaySim fills them in a way that reveals fraud.",
                "PaySim's account ids rarely repeat for payers, so payer history features carry little signal here."]
res["runtime_seconds"] = round(time.time() - T0, 1)
json.dump(res, open(os.path.join(ROOT, "results", "paysim_eval.json"), "w"), indent=1)
print(json.dumps(res["models"], indent=1))
