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


def _load(name):
    path = os.path.join(REPO, "results", name)
    return json.load(open(path)) if os.path.exists(path) else None


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def full_eval_html():
    """Static section built from the full dataset runs (ml/full_dataset_eval.py, tune_context.py, baf_variant_eval.py, scale benches)."""
    hi, tu, li, lif, baf = (_load("full_eval_HI.json"), _load("context_tuning_HI.json"), _load("full_eval_LI.json"),
                            _load("context_tuning_LI_fixed.json"), _load("baf_variant_eval.json"))
    sb, so = _load("scale_bench.json"), _load("scale_out_bench.json")
    if not hi:
        return ""
    out = ['<section id="full">', '<div class="eyebrow">Full dataset evaluation · every payment streamed through the live path</div>',
           "<h2>All 5.08 million HI-Small payments, in time order</h2>",
           f"<p>Each payment goes through the live engine's past-only account state, the live L1 gatekeeper and the L2 average, "
           f"exactly as the API scores it. Metrics are on the last 20% ({hi['scored_payments']:,} payments, {hi['scored_laundering']:,} laundering); "
           f"the 60 to 80% weeks are used only to tune the context rules.</p>"]
    L = hi["levels"]["payment_model"]; R = hi["rings"]["payment_model"]
    out.append('<div class="kpis">'
               f'<div class="kpi"><span class="v">{hi["live_model_pr_auc"]:.3f}</span><span class="k">PR-AUC of the live streaming score (offline benchmark 0.716)</span></div>'
               f'<div class="kpi"><span class="v">{pct(L["verify_or_more"]["recall"])}</span><span class="k">of laundering sent to verification or more, at {pct(L["verify_or_more"]["flag_rate"], 2)} of payments</span></div>'
               f'<div class="kpi"><span class="v">{pct(L["hold_or_more"]["recall"])}</span><span class="k">held for an analyst, at {pct(L["hold_or_more"]["flag_rate"], 2)} of payments (precision {pct(L["hold_or_more"]["precision"])})</span></div>'
               f'<div class="kpi"><span class="v">{R["verify_or_more"]["rings_caught"]}/{R["verify_or_more"]["rings_in_scored_period"]}</span><span class="k">labelled rings in the test weeks with at least one payment flagged</span></div></div>')
    rows = "".join(f"<tr><td>{t}</td><td>{v['rings']}</td><td>{v['caught']}</td><td>{pct(v['rate'], 0)}</td></tr>"
                   for t, v in R["block"]["by_typology"].items())
    out.append('<div class="tbl"><table><thead><tr><th>Typology</th><th>Rings in test weeks</th><th>Caught at block level</th><th>Rate</th></tr></thead>'
               f"<tbody>{rows}</tbody></table></div>")
    out.append(f'<p class="foot">Owner linking from the accounts file merges held payments into {hi["l4_cases_from_holds_entity_merged"]:,} cases '
               f'instead of {hi["l4_cases_from_holds"]:,} by shared accounts alone.</p>')
    if tu:
        a, b = tu["test"]["signature"]["shipped_ctx_1_2"], tu["test"]["signature"]["chosen"]
        c1, c2 = tu["test"]["context_layer_alone_verify_or_more"]["shipped_ctx_1_2"], tu["test"]["context_layer_alone_verify_or_more"]["chosen"]
        out.append('<div class="card fix"><h3>The mule signature was too loose; ctx-1.3 fixes it</h3><p>'
                   f"On the test weeks the ctx-1.2 signature fired on {pct(a['flag_rate'], 2)} of payments (precision {a['precision']:.3f}). "
                   f"A stricter rule chosen on the validation weeks only (payer collected from 4+ accounts in 24 h, forwards at least 95% "
                   f"of the latest deposit within 2 minutes, has paid at most 10 accounts) fires on {pct(b['flag_rate'], 2)} (precision {b['precision']:.3f}). "
                   f"The context layer alone now asks for verification on {pct(c2['flag_rate'], 2)} of payments instead of {pct(c1['flag_rate'], 2)}. "
                   "IBM data has no messages, so on these payments the context layer adds little beyond the payment model; its value is on "
                   "message-led scams, which this dataset cannot measure.</p></div>")
    if li:
        Ll = li["levels"]["payment_model"]
        txt = (f"<h3>LI-Small: {li['payments_total']:,} payments the models never saw</h3><p>A lower laundering intensity population. "
               f"Live streaming PR-AUC {li['live_model_pr_auc']:.3f}; at verification or more the live path catches {pct(Ll['verify_or_more']['recall'])} "
               f"of laundering at {pct(Ll['verify_or_more']['flag_rate'], 2)} of payments; held for an analyst {pct(Ll['hold_or_more']['recall'])} "
               f"(precision {pct(Ll['hold_or_more']['precision'])}). ")
        rr = li["rings"]["payment_model"]["verify_or_more"]
        if rr["rings_in_scored_period"]:
            txt += f"{rr['rings_caught']} of {rr['rings_in_scored_period']} labelled rings in its test weeks get at least one flag. "
        if lif:
            s2 = lif["test"]["signature"]["chosen"]
            txt += f"The ctx-1.3 signature, applied unchanged, fires on {pct(s2['flag_rate'], 3)} of LI payments (precision {s2['precision']:.3f})."
        out.append(f'<div class="card">{txt}</p></div>')
    if baf:
        rows = [("Base (benchmark test)", baf["base_test_months_6_7"])] + [(k, v["months_6_7"]) for k, v in baf["variants"].items()]
        body = "".join(f"<tr><td>{n}</td><td>{r['roc_auc']:.3f}</td><td>{pct(r['with_age_thresholds']['recall'])}</td>"
                       f"<td>{pct(r['with_age_thresholds']['fpr'])}</td><td>{r['with_age_thresholds']['fpr_parity_ratio']:.2f}</td></tr>" for n, r in rows)
        ctl = baf.get("control_trained_on_variant_iii", {})
        out += ['<h2>Account opening gate on all five BAF variants</h2>',
                "<p>The live L0 model was trained on BAF Base months 0 to 4 only. Each variant shifts the population's bias; months 6 to 7 are scored "
                "with the gate's age group thresholds. Parity is the lower age group false alarm rate divided by the higher one (1.0 is equal).</p>",
                '<div class="tbl"><table><thead><tr><th>Population</th><th>ROC-AUC</th><th>Recall</th><th>False alarms</th><th>Parity, 50+ vs under 50</th></tr></thead>'
                f"<tbody>{body}</tbody></table></div>"]
        if isinstance(ctl, dict) and "with_bias_features_x1_x2" in ctl:
            w, wo = ctl["with_bias_features_x1_x2"], ctl["without_bias_features"]
            out.append(f'<p class="foot">Control: a model trained on Variant III that may use its leaked features x1 and x2 reaches {w["roc_auc"]:.3f} ROC-AUC '
                       f'but its false alarm parity falls to {w["fpr_parity_ratio"]:.2f} (people 50+ falsely flagged {w["fpr_50_plus"] / max(w["fpr_under_50"], 1e-9):.1f}x as often); '
                       f'without them {wo["roc_auc"]:.3f} at parity {wo["fpr_parity_ratio"]:.2f}. L0 does not read x1 or x2. '
                       "Variant IV's test months are the same applications as Variant III's, reordered, so its row repeats Variant III.</p>")
    if sb and "exact" in sb:
        e, x, bl = sb["engine"], sb["exact"], sb["bloom"]
        out += ["<h2>Scaling the account state</h2>",
                f"<p>Streaming all {e['payments']:,} payments in one Python process: the prototype's object state holds {e['state_mb']:,} MB; the compact "
                f"layout (integer ids, typed arrays, counters instead of sets) holds {x['state_mb']:,} MB with identical features, and {bl['state_mb']:,} MB with a "
                f"fixed size Bloom pair index. Throughput {e['payments_per_second']:,} vs {x['payments_per_second']:,} vs {bl['payments_per_second']:,} payments per second per core.</p>"]
        ag = sb.get("agreement_bloom_vs_exact")
        if ag:
            out.append(f'<p class="foot">Bloom vs exact on the test weeks: {pct(ag["rows_with_any_feature_difference"], 3)} of payments see any feature '
                       f'change; live decisions agree on {pct(ag["decision_agreement"], 3)}.</p>')
    if so and "throughput" in so:
        t = so["throughput"]; lat = so.get("latency", {})
        out.append(f"<p>Redis state with stateless workers partitioned by payer: {t[0]['payments_per_second']:,} payments per second with one worker, "
                   f"{t[-1]['payments_per_second']:,} with {t[-1]['workers']} on this {so['cores']} core machine (Redis and workers share the cores); "
                   f"publish to decision latency p50 {lat.get('p50_ms')} ms, p99 {lat.get('p99_ms')} ms at {lat.get('target_rate_per_second'):,} per second. "
                   f"Redis holds about {t[-1]['redis_bytes_per_account']:,} bytes per account.</p>")
    rx, rl, bb, ss = _load("retrain_extended.json"), _load("retrain_l0_variants.json"), _load("balabit_eval.json"), _load("scenario_suite.json")
    if rx:
        C = rx["candidates"]
        def cell(n, s_, k):
            t = C[n][s_]["test"]
            return t["pr_auc"] if k == "pr" else t[f"caught_at_{k}_volume"]["caught"]
        body = "".join(f"<tr><td>{lab}</td><td>{C[n]['HI']['validation']['pr_auc']:.3f} / {C[n]['LI']['validation']['pr_auc']:.3f}</td>"
                       f"<td>{cell(n, 'HI', 'pr'):.3f}</td><td>{cell(n, 'LI', 'pr'):.3f}</td><td>{cell(n, 'HI', 'verify')} / {cell(n, 'LI', 'verify')}</td>"
                       f"<td>{cell(n, 'HI', 'hold')} / {cell(n, 'LI', 'hold')}</td></tr>"
                       for n, lab in (("current", "Previous live models (HI-Small only)"), ("retrain_HI", "Retrained on HI-Small (control)"),
                                      ("retrain_HI_LI", "Retrained on HI-Small + LI-Small" + (" (now live)" if rx.get("adopt") else ""))))
        out += ["<h2>Training on the data that was left unused</h2>",
                "<p>LI-Small (6.9 million payments) was added to the payment models' training weeks. Features come from the serving code itself, "
                "so training and the live API see identical numbers. The swap rule was fixed before reading test numbers: validation PR-AUC on HI-Small "
                "no more than 0.005 lower, and at least 0.01 higher on LI-Small. Caught counts compare models at the same alert volume as the previous live models.</p>",
                '<div class="tbl"><table><thead><tr><th>Model</th><th>Validation PR-AUC HI / LI</th><th>Test PR-AUC HI</th><th>Test PR-AUC LI</th>'
                f"<th>Caught at verify volume HI / LI</th><th>Caught at hold volume HI / LI</th></tr></thead><tbody>{body}</tbody></table></div>",
                '<p class="foot">Adding LI-Small beats the fair control (a retrain on HI-Small alone) on test PR-AUC for both datasets, and catches more at the verify level on both. '
                'Against the previous live models it trades a small loss on HI-Small at the hold level for gains on LI-Small.</p>']
    if rl:
        M, T = rl["mean_validation_auc"], rl["mean_test_auc"]
        out.append(f'<p>Account opening gate with the five BAF variants added to training: mean ROC-AUC over the six populations '
                   f'{T.get("current", 0):.3f} (previous) vs {T.get("retrain_variants", 0):.3f} (with variants) on months 6 to 7; '
                   f'{"adopted" if rl.get("adopt") else "not adopted, the previous model stays live"} under a rule fixed in advance on month 5 '
                   f'(validation {M.get("current", 0):.3f} vs {M.get("retrain_variants", 0):.3f}).</p>')
    if bb or ss:
        out.append("<h2>Person and session signals</h2>")
    if bb:
        k = bb["summary"]
        out.append(f'<p>Mouse dynamics on the real Balabit data (10 users, labelled impostor sessions): median user ROC-AUC '
                   f'{k["first_10_actions"]["median_user_auc"]:.2f} after 10 mouse actions and {k["first_50_actions"]["median_user_auc"]:.2f} after 50.</p>')
    if ss:
        su = ss["summary"]
        out.append(f'<p>Authored scenario suite through the live API: {su["scam_payments_stopped"]} scam payments stopped before money moved, '
                   f'{su["genuine_false_alarms"]} genuine payments with a false alarm, {su["genuine_wrongly_held"]} wrongly held. These scenarios test mechanics, not real world rates.</p>')
    out.append('<p class="foot">Code: <code>ml/full_dataset_eval.py</code>, <code>ml/tune_context.py</code>, <code>ml/baf_variant_eval.py</code>, '
               '<code>ml/scale_bench.py</code>, <code>ml/scale_out_bench.py</code>, <code>ml/retrain_extended.py</code>, <code>ml/retrain_l0_variants.py</code>, '
               '<code>ml/balabit_eval.py</code>, <code>ml/scenario_suite.py</code>, <code>service/compact_state.py</code>, <code>service/scale/</code>. '
               'IBM and BAF data are synthetic; Balabit is real mouse data.</p></section>')
    return "\n".join(out)


def build():
    html = open(TEMPLATE).read()
    html = html.replace("<!--FULL_EVAL-->", full_eval_html())
    html = html.replace("__BENCHMARK_DATA__", json.dumps(benchmark_data(), separators=(",", ":")))
    html = html.replace("__FRESH_DATA__", json.dumps(fresh_data(), separators=(",", ":")))
    return html


if __name__ == "__main__":
    open(OUT, "w").write(build())
    print("wrote", OUT)
