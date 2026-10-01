"""Copy the fresh sequence-model results from $RUSETECT_WORK/results_deep.json into results/model_comparison_sequence.json
in the same shape as results/model_comparison_tabular.json, so ml/build_report.py can show them side by side."""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import WORK, REPO
src = json.load(open(os.path.join(WORK, "results_deep.json")))
keep = [k for k in ("TCN (sequence)", "LSTM (sequence)", "Transformer (sequence)", "1D-CNN (sequence)") if k in src and "tp" in src[k]]
models = {}
for k in keep:
    r = src[k]
    models[k] = dict(family="sequence (neural)", evaluation="full test set", train_seconds=r["train_secs"], latency_ms_p50=r.get("latency_ms_p50"),
                     model_size_bytes=r.get("model_size_bytes"),
                     test=dict(average_precision=r["pr_auc"], roc_auc=r["roc_auc"], precision=r["precision"], recall=r["recall"], f1=r["f1"],
                               tp=r["tp"], fp=r["fp"], fn=r["fn"], threshold=r["threshold"],
                               recall_at_top_0_2pct=r["recall_at_budget"], precision_at_top_0_2pct=r["precision_at_budget"]))
os.makedirs(os.path.join(REPO, "results"), exist_ok=True)
json.dump(dict(source="ml/bench_deep.py (same split, training rows and F1-on-validation threshold as ml/compare_models.py)",
               architecture="encoder over each sender's and receiver's last 12 events (amount, direction, time gap, format, cross-bank, same counterparty) + MLP head over the 31 payment features",
               models=models), open(os.path.join(REPO, "results", "model_comparison_sequence.json"), "w"), indent=1)
print("exported", keep)
