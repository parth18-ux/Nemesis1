"""
Build service/static/report.html from report/report_template.html.

The template is the only hand-edited copy. Data comes from:
  * service/models/benchmark_results.json        committed results of ml/bench_*.py, ml/cascade.py, ml/l0_onboarding.py
  * results/model_comparison_tabular.json        fresh run of ml/compare_models.py (optional)
  * results/model_comparison_sequence.json       fresh run of ml/bench_deep.py, exported by ml/export_sequence_results.py (optional)
Usage: python ml/build_report.py            (tests/test_report.py fails if the built page is stale)
"""
import json, os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE = os.path.join(REPO, "report", "report_template.html")
OUT = os.path.join(REPO, "service", "static", "report.html")


def benchmark_data():
    b = json.load(open(os.path.join(REPO, "service", "models", "benchmark_results.json")))
    return dict(aml=b["aml_models"], baf=b["baf_models"], l0=b["l0_final"], cascade=b["cascade"], online=b.get("online_parity"))


def fresh_data():
    models, notes = {}, []
    tab = os.path.join(REPO, "results", "model_comparison_tabular.json")
    seq = os.path.join(REPO, "results", "model_comparison_sequence.json")
    if os.path.exists(tab):
        t = json.load(open(tab))
        for k, v in t["models"].items():
            models[k] = {key: v.get(key) for key in ("test", "train_seconds", "latency_ms_p50", "model_size_bytes", "evaluation", "family")}
        ds = t.get("dataset")
        if ds:
            notes.append(f"Fresh run on {ds['name']}: {ds['test_rows']:,} test payments, {ds['test_positives']:,} labelled laundering. "
                         "Same split, training rows and threshold rule for every model.")
        else:
            notes.append("Fresh run in progress: partial results.")
    if os.path.exists(seq):
        s = json.load(open(seq))
        for k, v in s["models"].items():
            models[k] = {key: v.get(key) for key in ("test", "train_seconds", "latency_ms_p50", "model_size_bytes", "evaluation", "family")}
        notes.append("Sequence models (TCN, LSTM, Transformer, 1D-CNN) read each sender's and receiver's last 12 events plus the payment's features.")
    if not models:
        return None
    notes.append("* KNN is fitted on a 40,000-row subsample and scored on a re-weighted test subsample. Gaussian Process and RBF-SVM: not evaluated (cost grows too fast for this data size).")
    return dict(models=models, note=" ".join(notes))


def build():
    html = open(TEMPLATE).read()
    html = html.replace("__BENCHMARK_DATA__", json.dumps(benchmark_data(), separators=(",", ":")))
    html = html.replace("__FRESH_DATA__", json.dumps(fresh_data(), separators=(",", ":")))
    return html


if __name__ == "__main__":
    open(OUT, "w").write(build())
    print("wrote", OUT)
