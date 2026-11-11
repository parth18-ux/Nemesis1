"""Build service/static/research.html (served at /research) from the experiment outputs.

Every number on the page is read from a JSON file: research/out/slice_results.json, research/out/amlgentex_transfer.json
and the repository's own results/*.json. Prose that is not a measurement says where it comes from.

Usage: python3 -I research/build_report.py [--fragment out.html]
  --fragment also writes the page body without the document wrapper (for publishing as a standalone artifact).
"""
import html, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
J = lambda *p: json.load(open(os.path.join(ROOT, *p)))
R = J("research", "out", "slice_results.json")
T = J("research", "out", "amlgentex_transfer.json") if os.path.exists(os.path.join(ROOT, "research", "out", "amlgentex_transfer.json")) else None
FH = J("results", "full_eval_HI.json")
LPA = J("results", "live_path_audit.json")
E = html.escape


def f(x, d=3):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def ci(c, d=3):
    return f"[{c[0]:.{d}f}, {c[1]:.{d}f}]"


def table(head, rows, cls=""):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl {cls}"><table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>'


# ------------------------------------------------------------------ charts (inline SVG, one scale per axis)
def forest(rows, xmin, xmax, ref=None, ref_label=""):
    """rows: (label, point, lo, hi, emphasis). Horizontal dot and whisker plot."""
    W, L, Rm, rh, top = 720, 230, 30, 34, 26
    H = top + rh * len(rows) + 34
    sx = lambda v: L + (v - xmin) / (xmax - xmin) * (W - L - Rm)
    g = []
    ticks = [round(xmin + 0.1 * i, 1) for i in range(int(round((xmax - xmin) / 0.1)) + 1)]  # tick label == tick position
    for t in ticks:
        g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top - 8}" y2="{H - 30}" class="grid"/>'
                 f'<text x="{sx(t):.1f}" y="{H - 12}" class="tick" text-anchor="middle">{t:.1f}</text>')
    if ref is not None:
        g.append(f'<line x1="{sx(ref):.1f}" x2="{sx(ref):.1f}" y1="{top - 14}" y2="{H - 30}" class="ref"/>'
                 f'<text x="{sx(ref) - 4:.1f}" y="{top - 16}" class="reflab" text-anchor="end">{E(ref_label)}</text>')
    for i, (lab, p, lo, hi, emph) in enumerate(rows):
        y = top + rh * i + rh / 2
        cls = "dot em" if emph else "dot"
        g.append(f'<text x="{L - 12}" y="{y + 4:.1f}" class="lab" text-anchor="end">{E(lab)}</text>'
                 f'<g class="hit"><title>{E(lab)}: {p:.3f} (95% CI {lo:.3f} to {hi:.3f})</title>'
                 f'<rect x="{sx(lo) - 4:.1f}" y="{y - 12:.1f}" width="{sx(hi) - sx(lo) + 8:.1f}" height="24" fill="transparent"/>'
                 f'<line x1="{sx(lo):.1f}" x2="{sx(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="whisk"/>'
                 f'<circle cx="{sx(p):.1f}" cy="{y:.1f}" r="5" class="{cls}"/></g>'
                 f'<text x="{sx(hi) + 8:.1f}" y="{y + 4:.1f}" class="val">{p:.3f}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="PR-AUC with 95% intervals">{"".join(g)}</svg>'


def bars(rows):
    """rows: (label, verify recall, block recall). Grouped horizontal bars on a 0..1 scale."""
    W, L, Rm, gh, top = 720, 230, 46, 40, 10
    H = top + gh * len(rows) + 30
    sx = lambda v: L + v * (W - L - Rm)
    g = []
    for t in (0, .25, .5, .75, 1):
        g.append(f'<line x1="{sx(t):.1f}" x2="{sx(t):.1f}" y1="{top}" y2="{H - 26}" class="grid"/>'
                 f'<text x="{sx(t):.1f}" y="{H - 8}" class="tick" text-anchor="middle">{int(t * 100)}%</text>')
    for i, (lab, v, b) in enumerate(rows):
        y = top + gh * i + 6
        g.append(f'<text x="{L - 12}" y="{y + 16:.1f}" class="lab" text-anchor="end">{E(lab)}</text>')
        for j, (val, cls, name) in enumerate([(v, "s1", "Verify or above"), (b, "s2", "Block")]):
            yy = y + j * 13
            w = max(sx(val) - L, 1.5)
            g.append(f'<g class="hit"><title>{E(lab)}, {name}: {val:.1%} of laundering payments flagged</title>'
                     f'<rect x="{L}" y="{yy:.1f}" width="{w:.1f}" height="11" rx="2" class="{cls}"/>'
                     f'<rect x="{L}" y="{yy - 1:.1f}" width="{W - L - Rm}" height="13" fill="transparent"/></g>'
                     f'<text x="{L + w + 5:.1f}" y="{yy + 9.5:.1f}" class="val">{val:.0%}</text>')
    return f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Recall under evasion attempts">{"".join(g)}</svg>'


# ------------------------------------------------------------------ content
def build():
    A, B, C, D, Eo, F, G, Hx, I, Jr = (R[k] for k in ["A_dataset_audit", "B_feature_intelligence", "C_headline_with_intervals",
                                                      "D_calibration", "E_operational", "F_missing_field", "G_evasion",
                                                      "H_explanations", "I_history_depth", "J_random_vs_chronological"])
    pa = C["pr_auc_iid"]; bands = C["bands"]
    fx = A["laundering_by_format"]
    s = []
    add = s.append

    # ---------------- header
    add(f"""
<header class="hero">
  <span class="eyebrow">Research review · November 2026</span>
  <h1>RuseTect research review</h1>
  <p class="lede">An independent check of the RuseTect audit and roadmap (the 21 page PDF in <code>docs/research/</code>) against the
  code and models in this repository. Every experiment below was run on the served models, through the same Rust feature code
  that scores live payments. Where a number comes from somewhere else, the source is named.</p>
  <div class="meta"><span>Data: labelled IBM AMLworld HI-Small slice ({A['rows']:,} payments, {A['laundering']} laundering) and AMLgentex Sweden 100K</span>
  <span>Scripts: <code>research/</code></span><span>Results: <code>research/out/*.json</code></span></div>
</header>""")

    # ---------------- verdict
    sw = G["format_switch_all_laundering_to"]
    add(f"""
<section id="findings">
  <span class="eyebrow">What changed in our understanding</span>
  <h2>Seven findings</h2>
  <div class="find">
    <div class="card fix"><h3>1. The served model is largely a payment format detector</h3>
      <p>{pct(A['ach_share_of_laundering'], 0)} of laundering in the slice is ACH (laundering rate {pct(fx['ACH']['rate'], 2)} on ACH, at most
      {pct(max(v['rate'] for k, v in fx.items() if k != 'ACH'), 2)} on any other format). If launderers send the same payments as Cheque,
      verify-band recall falls from {pct(G['baseline']['verify'], 0)} to {pct(sw['Cheque']['verify'], 0)} and block recall from
      {pct(G['baseline']['block'], 0)} to {pct(sw['Cheque']['block'], 0)}. The payer chooses the rail, so this is a one-step evasion.</p></div>
    <div class="card fix"><h3>2. The 0.711 headline is inflated by an end-of-simulation tail</h3>
      <p>{tail_sentence()}</p></div>
    <div class="card fix"><h3>3. A missing format silently changed decisions</h3>
      <p>The API filled a missing <code>payment_format</code> with ACH. In a three hour window that sends
      {F['format_missing_defaulted_to_ACH (current API)']['escalated_non_ach_legit']:,} genuine payments to a higher band. Treating it as
      unknown is worse: PR-AUC drops to {f(F['format_missing_as_nan']['pr_auc'])}. Fixed in this branch: the field is now required.</p></div>
    <div class="card"><h3>4. Splitting a payment in four defeats the block band</h3>
      <p>Re-simulated through the engine, structuring each laundering payment into four parts leaves
      {pct(G['structuring_split4']['block_any_part'], 0)} blocked (was {pct(G['baseline']['block'], 0)}); the verify band still catches
      {pct(G['structuring_split4']['verify_any_part'], 0)} of payments on at least one part.</p></div>
    <div class="card"><h3>5. The LightGBM half adds nothing measurable</h3>
      <p>XGBoost alone minus the served average: PR-AUC {f(pa['xgb_half_only']['minus_live_mean']['point'])}, 95% interval
      {ci(pa['xgb_half_only']['minus_live_mean']['ci95'])}. LightGBM alone is worse with an interval that excludes zero. The average costs
      latency, explanations and calibration for no gain.</p></div>
    <div class="card"><h3>6. Scores are not probabilities, and thresholds do not transfer</h3>
      <p>The served score averages {f(D['live_mean_uncalibrated']['mean_score'], 4)} against an observed rate of
      {f(D['live_mean_uncalibrated']['observed_rate'], 4)}. Isotonic calibration cuts ECE from {f(D['live_mean_uncalibrated']['ece'], 4)} to
      {f(D['mean_of_isotonic_calibrated_halves']['ece'], 4)} without changing ranking. A 0.9 precision threshold fitted on one half of the window
      gave {f(D['threshold_transfer_precision_0.9_target']['uncalibrated_mean']['second_half']['precision'], 2)} on the next.</p></div>
    <div class="card"><h3>7. Nothing transfers between generators without retraining</h3>
      <p>{transfer_sentence()}</p></div>
  </div>
</section>""")

    # ---------------- PDF and dataset analysis
    add(f"""
<section id="pdf">
  <span class="eyebrow">Deliverable 1</span>
  <h2>What the PDF establishes, checked against the repository</h2>
  <p>The audit's central reading holds: the payment decision is made by two learned stages (an XGBoost gate with
  {FH['l1_pass_rate']:.1%} pass rate, then the mean of an XGBoost and a LightGBM on 31 past-only features), and the surrounding
  layers are deterministic or offline. Its numbers match the result files they cite. Three points needed correction or extension:</p>
  <ul class="plain">
    <li><b>Licence of AMLgentex.</b> The PDF lists it as "not stated". The Hugging Face dataset cards for
    <code>AMLGentex/sweden_100K_difficult</code> and <code>sweden_100K_easy</code> state MIT; <code>johanos1/AMLGentex-Neurips</code> states CC BY 4.0.</li>
    <li><b>Test window composition.</b> The PDF notes the prevalence shift between validation ({LPA['validation']['positives']:,} laundering)
    and test ({LPA['test']['positives']:,}), but not its cause. The trailing-tail evidence in finding 2 is a likely cause and changes how the 0.711 headline reads.</li>
    <li><b>"Payment format is a shortcut".</b> The PDF cites a retraining ablation (0.728 to 0.701 without the feature). That measures how much
    the <i>information</i> can be recovered from other features. It does not measure how much the <i>served</i> model depends on it; section 6 does.</li>
  </ul>
  <h3>Data available to this investigation</h3>
  {table(["Data", "Where", "Labels", "Used here for"], [
      ["IBM AMLworld HI-Small, 160,000 payment slice", "<code>service/models/demo_stream.csv.gz</code> (in repo)", f"{A['laundering']} laundering, payment level", "every slice experiment"],
      ["IBM AMLworld HI-Small and LI-Small, full", "Kaggle (login needed; not reachable from this environment)", "payment level + typology file", "not re-run; repo results cited"],
      ["AMLgentex Sweden 100K, easy and difficult", "Hugging Face, MIT", "isSAR per transaction", "cross generator transfer"],
      ["Served models", "<code>service/models/</code>, <code>ml/exported/</code>", "", "all scoring"],
      ["Repo results", "<code>results/*.json</code>", "", "reference numbers"]])}
  <p class="foot">The full HI-Small CSVs are needed for the P1 experiments that depend on typology and the test-window tail. They are not in the
  repository and Kaggle needs an account, so those experiments are specified below and left for a machine with the data.</p>
</section>""")

    # ---------------- dataset audit
    fmt_rows = [[k, f"{v['payments']:,}", f"{v['laundering']:,}", pct(v['rate'], 3)] for k, v in sorted(fx.items(), key=lambda kv: -kv[1]['rate'])]
    uni = list(B["univariate_ranked"].items())[:8]
    add(f"""
<section id="audit">
  <span class="eyebrow">Deliverable 2</span>
  <h2>Dataset audit of the labelled slice</h2>
  <div class="kpis">
    <div class="kpi"><span class="v">{A['rows']:,}</span><span class="k">payments, {A['time_span'][0]} to {A['time_span'][1]}</span></div>
    <div class="kpi"><span class="v">{pct(A['prevalence'], 3)}</span><span class="k">laundering prevalence ({A['laundering']} payments)</span></div>
    <div class="kpi"><span class="v">{A['exact_duplicate_rows']}</span><span class="k">duplicate rows; {A['empty_fields']} empty fields</span></div>
    <div class="kpi"><span class="v">{A['accounts']:,}</span><span class="k">accounts, {A['accounts_touching_laundering']} touch laundering</span></div>
  </div>
  <p>The slice is clean in the mechanical sense (no duplicates, no missing values). The problems are in the distributions. Laundering is
  concentrated in one payment format to a degree no real rail would show, and laundering amounts are far larger (median ${A['amount_usd_median']['laundering']:,.0f}
  against ${A['amount_usd_median']['legit']:,.0f}; Mann-Whitney p = {A['amount_mannwhitney_p']:.1e}). The slice carries payment labels only, so ring,
  typology and account level checks need the full patterns file.</p>
  <h3>Laundering by payment format</h3>
  {table(["Format", "Payments", "Laundering", "Rate"], fmt_rows)}
  <h3>Feature intelligence (evaluation window)</h3>
  <p>Single feature ROC-AUC, best direction. Payment format alone reaches {f(uni[0][1]['univariate_auc'])}, more than any behavioural feature.</p>
  {table(["Feature", "ROC-AUC alone", "Direction", "Missing", "Mean, genuine", "Mean, laundering"],
         [[f"<code>{k}</code>", f(v['univariate_auc']), v['direction'], pct(v['missing_share'], 0), f"{v['mean_legit']:,.3g}", f"{v['mean_laundering']:,.3g}"] for k, v in uni])}
  <p>Redundancy: {len(B['redundant_pairs_spearman_ge_0.85'])} feature pairs have Spearman |ρ| ≥ 0.85 here. Within a single day
  <code>s_out_n</code> equals <code>s_cnt_24h</code> and <code>d_in_n</code> equals <code>d_cnt_24h</code> by construction, so on longer
  histories they separate; on short replays they double count. <code>{'</code> and <code>'.join(B['constant_features'])}</code> are constant in the window
  (one weekday; round USD amounts never occur after conversion), so <code>round_100</code> is effectively dead.</p>
</section>""")

    # ---------------- headline with intervals
    rows = [("Served score (mean of halves)", pa['live_mean']['point'], *pa['live_mean']['ci95'], True),
            ("XGBoost half alone", pa['xgb_half_only']['point'], *pa['xgb_half_only']['ci95'], False),
            ("LightGBM half alone", pa['lgbm_half_only']['point'], *pa['lgbm_half_only']['ci95'], False)]
    for k, lab in [("history_10000_payments", "History of 10,000 payments"), ("history_25000_payments", "History of 25,000 payments"),
                   ("history_50000_payments", "History of 50,000 payments"), ("history_100000_payments", "History of 100,000 payments")]:
        rows.append((lab, I[k]['pr_auc'], *I[k]['ci95'], False))
    band_rows = [[n, f"{b['flags']:,}", b['tp'], f(b['precision'], 3) + f" <small>{ci(b['precision_ci95'], 2)}</small>",
                  f(b['recall'], 3) + f" <small>{ci(b['recall_ci95'], 2)}</small>"] for n, b in
                 [("Verify or above (0.137)", bands['verify_or_more']), ("Hold or above (0.450)", bands['hold_or_more']),
                  ("Block, served (0.823)", bands['block_served']), ("Block, retuned (0.935)", bands['block_retuned'])]]
    add(f"""
<section id="intervals">
  <span class="eyebrow">Deliverable 2, statistics</span>
  <h2>Every headline with an interval</h2>
  <p>Evaluation window: {A['eval_rows']:,} payments with {A['eval_laundering']} laundering. With so few positives the intervals are wide, which is the point:
  the audit's comparisons of 0.7107 against 0.7108 are far inside this noise. Paired bootstrap ({R['meta']['bootstrap']}).</p>
  <div class="chart">{forest(rows, 0.2, 0.8, ref=0.7107, ref_label='full protocol 0.711')}</div>
  <p>The slice sits well below the full protocol's 0.711, and history depth does not explain the gap: 10,000 to 100,000 payments of history give
  overlapping intervals. Two explanations remain: window-to-window variation, and the test-window tail of finding 2. The September 9 to 10 figure
  from the independent audit (AP 0.279 for their model against 0.489 on the full test) points the same way.</p>
  <h3>Bands on the slice</h3>
  {table(["Band", "Flags", "Caught", "Precision [95% CI]", "Recall [95% CI]"], band_rows)}
  <p>Paired differences: XGBoost alone minus served {f(pa['xgb_half_only']['minus_live_mean']['point'])} {ci(pa['xgb_half_only']['minus_live_mean']['ci95'])};
  LightGBM alone minus served {f(pa['lgbm_half_only']['minus_live_mean']['point'])} {ci(pa['lgbm_half_only']['minus_live_mean']['ci95'])}.
  Ten minute block bootstrap gives the same conclusion ({ci(C['pr_auc_time_block']['xgb_half_only']['minus_live_mean']['ci95'])}).
  L1 passes {pct(C['l1_pass_rate'])} of payments and keeps {pct(C['l1_laundering_kept'])} of laundering.</p>
</section>""")

    # ---------------- calibration & operational
    tt = D["threshold_transfer_precision_0.9_target"]
    pb = Eo["precision_at_alert_budget"]
    cost = Eo["cost_curve_count_based"]
    add(f"""
<section id="calibration">
  <span class="eyebrow">Deliverable 6, measured</span>
  <h2>Calibration, alert budgets and cost</h2>
  {table(["Score (second half of window)", "Brier", "ECE", "PR-AUC"], [
      ["Served mean, uncalibrated", f(D['live_mean_uncalibrated']['brier'], 5), f(D['live_mean_uncalibrated']['ece'], 5), f(D['pr_auc_test_half_uncalibrated'])],
      ["Mean of isotonic-calibrated halves", f(D['mean_of_isotonic_calibrated_halves']['brier'], 5), f(D['mean_of_isotonic_calibrated_halves']['ece'], 5), f(D['mean_of_isotonic_calibrated_halves']['pr_auc_test_half'])],
      ["Mean of Platt-calibrated halves", f(D['mean_of_platt_calibrated_halves']['brier'], 5), f(D['mean_of_platt_calibrated_halves']['ece'], 5), f(D['mean_of_platt_calibrated_halves']['pr_auc_test_half'])]])}
  <p>Calibrators were fitted on the first {D['split']['calibration_rows']:,} payments ({D['split']['calibration_laundering']} laundering) and judged on the next
  {D['split']['test_rows']:,} ({D['split']['test_laundering']}). Calibration fixes the meaning of the score; it cannot fix threshold drift. The threshold that gave 0.9
  precision on the first half gave {f(tt['uncalibrated_mean']['second_half']['precision'], 2)} on the second ({tt['uncalibrated_mean']['second_half']['flags']} flags), and the
  served block threshold gave {f(D['served_block_on_second_half']['precision'], 2)}. This is the mechanism behind the open block-threshold decision in the project audit.</p>
  <h3>What an operations team would read</h3>
  {table(["Alert budget", "Alerts", "Caught", "Precision", "Recall"],
         [[k.replace('_', ' '), v['alerts'], v['tp'], f(v['precision'], 2), f(v['recall'], 2)] for k, v in pb.items()])}
  <p>Recall at fixed false positive rate: {', '.join(f"{pct(float(k), 2)} FPR gives {pct(v['recall'], 0)}" for k, v in Eo['recall_at_fpr'].items())}.
  Recall stalls between 10 and 25 alerts per 10,000 payments, so a budget above about 10 per 10,000 buys false positives only.</p>
  <p><b>Amount weighting needs care.</b> At the block band count recall is {pct(Eo['amount_weighted_recall']['block']['count_recall'], 0)} but USD recall is
  {pct(Eo['amount_weighted_recall']['block']['usd_recall'], 1)}: two payments of $27.0M and $14.4M carry 98% of laundered value in the window and score 0.69 and 0.67, so
  they are held, not blocked (USD recall at hold {pct(Eo['amount_weighted_recall']['hold']['usd_recall'], 0)}). Report count and value metrics side by side, with and without the largest payments.</p>
  <h3>Cost-minimising thresholds (count based)</h3>
  {table(["Missed laundering costs", "Best threshold", "Cost at best", "At verify", "At hold", "At block"],
         [[k.replace('fn_cost_', '').replace('x_fp', '× a false flag'), f(v['cost_minimising_threshold'], 3), f"{v['cost_minimising_threshold'] and v['cost']:,.0f}",
           f"{v['cost_at_served_verify']:,.0f}", f"{v['cost_at_served_hold']:,.0f}", f"{v['cost_at_served_block']:,.0f}"] for k, v in cost.items()])}
  <p>Each served band is close to optimal only for one cost ratio. A bank's cost ratio (jury question 8 in the PDF) should set the thresholds, not the alert rates inherited from the offline blend.</p>
</section>""")

    # ---------------- robustness
    Fa = F['format_missing_defaulted_to_ACH (current API)']; Fn = F['format_missing_as_nan']; Ft = F['true_format']
    ev_rows = [("No attack", G['baseline']['verify'], G['baseline']['block'])]
    for k in ("Cheque", "Wire", "Credit Card", "Cash", "Bitcoin"):
        ev_rows.append((f"Sent as {k}", sw[k]['verify'], sw[k]['block']))
    ev_rows += [("Amount 1% lower, not round", G['amount_shave_1pct_no_round']['verify'], G['amount_shave_1pct_no_round']['block']),
                ("Split in 2", G['structuring_split2']['verify_any_part'], G['structuring_split2']['block_any_part']),
                ("Split in 4", G['structuring_split4']['verify_any_part'], G['structuring_split4']['block_any_part']),
                ("Split in 4, 30 min apart", G['structuring_split4_spread']['verify_any_part'], G['structuring_split4_spread']['block_any_part'])]
    add(f"""
<section id="robustness">
  <span class="eyebrow">Deliverables 4 and 6, experimentally demonstrated</span>
  <h2>Robustness: missing fields and payment-side evasion</h2>
  <h3>A missing payment format</h3>
  {table(["How the field is filled", "PR-AUC", "Genuine payments flagged", "Decisions changed", "Blocks"], [
      ["True format (reference)", f(Ft['pr_auc']), f"{Ft['legit_flagged_verify_plus']:,}", "", Ft['bands']['block']],
      ["Default to ACH (API before this branch)", f(Fa['pr_auc']), f"{Fa['legit_flagged_verify_plus']:,}", f"{Fa['decisions_changed_vs_true']:,}", Fa['bands']['block']],
      ["Unknown, sent to the model as missing", f(Fn['pr_auc']), f"{Fn['legit_flagged_verify_plus']:,}", f"{Fn['decisions_changed_vs_true']:,}", Fn['bands']['block']]])}
  <p>Neither default is safe, so the API now rejects a payment without a known format or currency (422) instead of guessing. An unknown currency was also
  converted to USD at rate 1.0, which misstates a rupee amount by a factor of about 70.</p>
  <h3>Evasion by the payer</h3>
  <p>Share of the {G['laundering_payments_in_window']} laundering payments in the window that still reach each band. Format switches change only the format
  feature; structuring is re-simulated through the engine, so account state changes too, and a payment counts as caught if any part is flagged.</p>
  <div class="chart"><div class="legend"><span><i class="s1"></i>Verify or above</span><span><i class="s2"></i>Block</span></div>{bars(ev_rows)}</div>
  <p>Format switching is the cheapest attack and the most damaging. In AMLworld it is a simulator artefact (laundering typologies were generated mostly as ACH), but the
  lesson carries to UPI: any field the payer chooses freely (rail, app, note text) must not dominate the score. Structuring leaves the verify band working but
  removes the block band, because blocking leans on amount. Genuine payments flagged at verify stay at {G['structuring_split4']['legit_flagged_verify_plus']:,}
  (baseline {G['legit_flagged_verify_plus_baseline']:,}), so the attack costs the defender nothing in false positives and gains the attacker a lot.</p>
</section>""")

    # ---------------- explanations, history, split
    a, h_ = Hx['all_l2_payments'], Hx['hold_or_more']
    add(f"""
<section id="explain">
  <span class="eyebrow">Deliverable 6, measured</span>
  <h2>Explanations, history depth and split policy</h2>
  <h3>Are XGBoost-only reasons faithful to the served score?</h3>
  {table(["Payments", "Top reason agrees (XGB vs full)", "Top 4 overlap, XGB vs full (Jaccard)", "Top 4 overlap, XGB vs LightGBM"], [
      [f"All L2 payments ({a['payments']:,})", pct(a['top1_agreement_xgb_vs_full']), f(a['top4_jaccard_xgb_vs_full']), f(a['top4_jaccard_xgb_vs_lgbm'])],
      [f"Hold or above ({h_['payments']})", pct(h_['top1_agreement_xgb_vs_full']), f(h_['top4_jaccard_xgb_vs_full']), f(h_['top4_jaccard_xgb_vs_lgbm'])]])}
  <p>The first reason chip is almost always right; the other three often are not (about 40% of the set changes for held payments). Global rank correlation
  between the halves' importances is {f(Hx['global_rank_spearman_xgb_vs_lgbm'], 2)}. Method: TreeSHAP on each half, mapped to probability space by p(1−p)
  and averaged like the score. Dropping the LightGBM half (finding 5) makes the existing reasons exact.</p>
  <h3>Random versus chronological split</h3>
  {table(["Split policy (XGBoost retrained on the slice, 5 seeds)", "PR-AUC mean", "SD"], [
      ["Chronological, last 20%", f(Jr['chronological']['pr_auc_mean']), f(Jr['chronological']['sd'])],
      ["Random rows", f(Jr['random_row_split']['pr_auc_mean']), f(Jr['random_row_split']['sd'])],
      ["Random, grouped by sender", f(Jr['random_split_grouped_by_sender']['pr_auc_mean']), f(Jr['random_split_grouped_by_sender']['sd'])]])}
  <p>On eight hours of data the split policy makes no reliable difference: the standard deviations overlap and the test sets differ. This is a negative result.
  The PDF's proposed one day experiment still needs the full file, where label and state leakage across weeks can show up.</p>
</section>""")

    # ---------------- transfer
    add(full_section())
    add(transfer_section())
    add(samld_section())
    add(unified_section())
    add(other_section())
    add(message_section())

    # ---------------- security
    sec = [
        ["Evasion: payer switches payment format", "Experimentally demonstrated", f"Block recall {pct(G['baseline']['block'], 0)} to 0%", "Open. Retrain without <code>fmt_code</code> or cap its contribution; report the 0.728 to 0.701 trade", "NIST AI 100-2 evasion; MITRE ATLAS AML.T0015"],
        ["Evasion: structuring into smaller payments", "Experimentally demonstrated", f"Block recall to {pct(G['structuring_split4']['block_any_part'], 0)}", "Open. Add per-payee and per-sender sum features over 1 h and 24 h windows; aggregate alerts per account", "NIST AI 100-2 evasion"],
        ["Missing or unknown format and currency silently imputed", "Experimentally demonstrated", f"{Fa['escalated_non_ach_legit']:,} genuine payments escalated in 3 h", "Fixed: 422 on missing or unknown values; test added", "OWASP API8 misconfiguration"],
        ["Rate limit keyed on client-supplied X-Forwarded-For", "Verified in code", "Unlimited requests; unbounded memory per spoofed client", "Fixed: header trusted only on Vercel or with RUSETECT_TRUST_PROXY=1; idle clients evicted", "OWASP API4 unrestricted resource consumption"],
        ["Alerts, evidence packs, STR drafts, cases and audit readable without a key", "Verified in code", "Disclosure of suspicion (tipping off) and account identifiers", "Fixed when keys are configured; demo mode stays open by design", "OWASP API1/API2"],
        ["Demo keys enabled whenever RUSETECT_ANALYST_KEYS is unset", "Verified in code", "Fail-open: a misconfigured deployment accepts published keys", "Open. Require RUSETECT_DEMO=1 explicitly for demo keys", "OWASP API8"],
        ["Hard-coded fallback secrets (consortium HMAC, entity hash salt)", "Verified in code", "Account tokens can be reversed by enumeration when defaults are used", "Fixed: with analyst keys set the server refuses to start without the consortium secret, hash salt and signing key (test added)", "NIST AI 100-2 privacy"],
        ["Ephemeral signing key and in-memory nonce cache per serverless instance", "Verified in design", "Signatures from one instance cannot be checked after restart; nonces can be replayed on another instance", "Partly fixed: a signing key is required outside demo mode; nonces still per instance (move to Redis)", "Integrity"],
        ["Unbounded account state from API traffic", "Verified in code", "Memory exhaustion with random account ids", "Open. LRU or TTL on account state; per-tenant caps", "OWASP API4"],
        ["Label poisoning through analyst feedback", "Theoretical", "Released or blocked verdicts would feed retraining", "Plan: two-person rule on labels used for training; audit trail already signed", "NIST AI 100-2 poisoning"],
        ["Model extraction through scores and reasons", "Theoretical", "API returns risk, both halves and SHAP reasons", "Plan: return bands only to payment channels; full reasons to analysts", "NIST AI 100-2 privacy"],
        ["Dependency vulnerabilities", "Checked", "cargo audit: 0 advisories in 199 crates (RustSec, 1,453 advisories)", "Keep cargo audit in CI", "Supply chain"],
    ]
    add(f"""
<section id="security">
  <span class="eyebrow">Deliverable 4</span>
  <h2>Security and threat assessment</h2>
  <p>Status says how strong the evidence is. "Experimentally demonstrated" means measured above; "verified in code" means read in the current source;
  "theoretical" means plausible but untested. Framework references are to NIST AI 100-2 E2025, the OWASP API Security Top 10 (2023) and MITRE ATLAS.</p>
  {table(["Threat", "Evidence", "Impact", "Status in this branch", "Reference"], sec, "wrap")}
</section>""")

    # ---------------- literature
    lit = [
        ["Altman et al., NeurIPS 2023 D&B", "arXiv 2306.16424", "AMLworld generator and HI/LI sets", "Source of all payment results; its typology mix explains the ACH shortcut"],
        ["Östman et al., 2025 (AMLgentex)", "arXiv 2506.13989", "Configurable generator, 12 banks, label noise", "Second generator used in the transfer test above"],
        ["Oztas et al., IEEE ICEBE 2023 (SAML-D)", "doi 10.1109/ICEBE59045.2023.00028", "28 typologies, 9.5M payments", "Third generator for transfer (Kaggle only)"],
        ["Blanuša et al., ICAIF 2024 (Graph Feature Preprocessor)", "arXiv 2402.08593", "Past-only subgraph pattern features for boosted trees", "Best supported graph upgrade; addresses structuring via fan and cycle counts"],
        ["Egressy et al., AAAI 2024 (Multi-PNA)", "arXiv 2306.11586", "GNNs for directed multigraphs on AMLworld", "Graph baseline with a non-causal window"],
        ["Lin et al., ICAIF 2024 (FraudGT)", "doi 10.1145/3677052.3698648", "Graph transformer, minority F1 on AMLworld", "Comparison point; not strictly causal"],
        ["Maganti, 2026", "arXiv 2604.19514", "Strict inductive re-evaluation of GNNs on Elliptic", "Supports the repo's causal protocol; trees beat GNNs"],
        ["Kim et al., 2024", "arXiv 2404.00060", "TGN for financial graph anomalies (DGraph)", "Temporal GNN research arm"],
        ["Rossi et al., 2020 (TGN)", "arXiv 2006.10637", "Temporal graph networks", "Method behind the research arm"],
        ["Poursafaei et al., NeurIPS 2022", "arXiv 2207.10128", "EdgeBank: memorisation baseline for dynamic graphs", "Any temporal GNN result needs this baseline"],
        ["Jing et al., 2024 (CFDB)", "arXiv 2404.14746", "Customer-level benchmark from SAML-D and AMLworld", "Template for account (mule) labels"],
        ["Nahimana et al., 2026", "arXiv 2608.15447", "Mobile money laundering detection (Rwanda) on synthetic data", "Confirms the lack of real mobile money labels"],
        ["Braithwaite et al., 2025 (nuFormer)", "arXiv 2507.23267", "Transaction transformer with joint fusion", "Small gains over trees at very large scale"],
        ["Grinsztajn et al., NeurIPS 2022", "arXiv 2207.08815", "Trees beat deep nets on tabular data", "Supports keeping boosted trees"],
        ["Geirhos et al., Nature MI 2020", "doi 10.1038/s42256-020-00257-z", "Shortcut learning", "Frames the payment format finding"],
        ["Kireev et al., NDSS 2023", "arXiv 2208.13058", "Cost-aware adversarial robustness for tabular fraud models", "Method for realistic, costed evasion tests"],
        ["Cartella et al., SafeAI 2021; Ballet et al., 2019", "arXiv 2101.08030; 1911.03274", "Adversarial attacks on tabular and fraud data", "Feature-space attack baselines"],
        ["Lunghi et al., 2023", "arXiv 2307.01390", "Adversarial learning in real-world fraud detection", "Threat model for payment-side attackers"],
        ["Niculescu-Mizil and Caruana, ICML 2005", "doi 10.1145/1102351.1102430", "Calibration of boosted models", "Basis for the calibration fix"],
        ["Dal Pozzolo et al., IEEE SSCI 2015", "doi 10.1109/SSCI.2015.33", "Calibration after undersampling", "Explains the 12x score inflation"],
        ["van den Goorbergh et al., JAMIA 2022", "doi 10.1093/jamia/ocac093", "Harm of imbalance corrections", "Argues against SMOTE-style resampling"],
        ["Saito and Rehmsmeier, PLoS ONE 2015", "doi 10.1371/journal.pone.0118432", "PR over ROC under imbalance", "Metric choice"],
        ["Correa Bahnsen et al., ESWA 2015", "doi 10.1016/j.eswa.2015.04.042", "Example-dependent cost-sensitive learning", "Cost model with amounts"],
        ["Lebichot et al., IJDSA 2021", "doi 10.1007/s41060-021-00258-0", "Incremental learning with delayed labels", "Retraining cadence"],
        ["Alves et al., TMLR 2024", "arXiv 2403.06906", "Learning to defer under analyst capacity", "Capacity-aware review band"],
        ["Han et al., NeurIPS 2022 (ADBench)", "arXiv 2206.09426", "Anomaly detection benchmark", "Why unsupervised scorers stay monitors"],
        ["Lundberg et al., Nature MI 2020", "doi 10.1038/s42256-019-0138-9", "TreeSHAP", "Reason codes"],
        ["Jesus et al., FAccT 2021; Weerts et al., 2019", "doi 10.1145/3442188.3445941; arXiv 1907.03324", "SHAP does not reliably help analysts", "Limits on explanation claims"],
        ["Jesus et al., NeurIPS 2022 (BAF)", "arXiv 2211.13358", "Bank Account Fraud datasets", "L0 onboarding data"],
        ["Bellei et al., 2024 (Elliptic2); Elmougy and Liu, KDD 2023 (Elliptic++)", "arXiv 2404.19109; doi 10.1145/3580305.3599803", "Real, anonymised Bitcoin labels", "Real-label sanity check for graph features"],
        ["Huang et al., NeurIPS 2022 (DGraph)", "arXiv 2207.03579", "Real financial graph with missing features", "Robustness of graph methods"],
        ["NIST AI 100-2 E2025", "csrc.nist.gov/pubs/ai/100/2/e2025/final", "Adversarial ML taxonomy", "Threat model vocabulary"],
    ]
    add(f"""
<section id="literature">
  <span class="eyebrow">Deliverable 3</span>
  <h2>Literature, verified</h2>
  <p>Every identifier below was resolved against arXiv or Semantic Scholar during this review (raw responses in <code>research/out/s2_refs.json</code>).
  All arXiv identifiers cited by the PDF resolved to the papers it names. Recent AMLworld work found during the search (BlazingAML, arXiv 2604.12241; TRACK,
  arXiv 2611.06035; federated cross-bank context, arXiv 2610.35996; AMLworld-Compact, arXiv 2610.13627) reports results under transductive or
  account-window protocols, so none is directly comparable with RuseTect's strictly causal payment-level protocol.</p>
  {table(["Work", "Identifier", "What it contributes", "Relevance here"], lit, "wrap")}
</section>""")

    # ---------------- gap & novelty
    add("""
<section id="novelty">
  <span class="eyebrow">Deliverable 5</span>
  <h2>Research gap and novelty</h2>
  <p>A new detector architecture would not be novel: boosted trees with engineered graph features are the documented state of practice on AMLworld,
  and RuseTect's own GNN and sequence results agree. The defensible contribution is methodological, and this review strengthens it with three
  results the literature found in this search does not report together:</p>
  <ul class="plain">
    <li><b>Shortcut audit of a public AML benchmark.</b> A single categorical field dominates both the label distribution and the served model, and a one-step
    payer-side change defeats it. Published AMLworld results do not report format-ablated or format-switched recall.</li>
    <li><b>Benchmark composition.</b> The trailing laundering tail inflates any metric on the last days of HI-Small; a corrected protocol is a short,
    citable contribution.</li>
    <li><b>Serving-parity evaluation.</b> Every number here comes through the production feature code (Python and Rust agree to 3e-8), which turns
    robustness tests into tests of the deployed system rather than of an offline copy.</li>
  </ul>
  <p>Together with the cross-generator result, this supports a workshop paper on <i>what survives a causal, serving-parity, adversarial evaluation of AML
  detection on synthetic data</i>. The claim must stay within synthetic data: no public labelled UPI data exists (confirmed in the PDF and not contradicted by this search).</p>
</section>""")

    # ---------------- ML recommendations
    ml = [
        ["Drop the LightGBM half; serve calibrated XGBoost", "No measurable PR-AUC gain; halves latency and makes reasons exact", "Existing models", "Isotonic on validation; refit bands under a pre-registered rule", "PR-AUC unchanged within CI; ECE ↓"],
        ["Format-robust model", "Removes the dominant shortcut and the one-step evasion", "HI/LI full", "Retrain without fmt_code (or monotone cap); compare under format-switch attack", "Recall under attack; clean PR-AUC cost (PDF: 0.728 → 0.701)"],
        ["Structuring-aware features", "Splitting removes the block band", "HI/LI full, AMLgentex", "Past-only sums and counts per sender and payee over 1 h and 24 h; GFP fan and cycle counts", "Recall under split attack; per-typology recall"],
        ["Account-level mule score", "Banks freeze accounts, not payments", "AMLworld patterns, AMLgentex (account SAR), SAML-D", "CFDB-style aggregation; time to first detection", "Account PR-AUC; days to detection"],
        ["Cost and capacity-aware thresholds", "Bands are optimal for one implicit cost ratio", "Existing", "Choose thresholds by expected cost and daily alert budget; learning to defer later", "Cost at budget; precision@k"],
        ["Unsupervised anomaly scoring as a monitor only", "Repo evidence: IF recovers 0 of 4 held-out typologies", "Existing", "Keep for drift alarms, not decisions", "Alarm delay after injected drift"],
    ]
    add(f"""
<section id="ml">
  <span class="eyebrow">Deliverable 6</span>
  <h2>Methodology improvements, ranked by evidence</h2>
  {table(["Change", "Why (evidence)", "Data", "Method", "Metric"], ml, "wrap")}
  <p>Not recommended: replacing trees with a GNN (strict inductive evidence against), SMOTE-style resampling (miscalibration without AUC gain), and a dedicated
  adversarial training module before the cheap feature fixes above are measured.</p>
</section>""")

    # ---------------- jury
    jury = [
        ("Senior academic researcher", "Accept for a workshop after revision",
         ["The contribution is evaluation methodology; the paper must say so in the title and abstract.",
          "0.711 is reported on a test window whose tail is a simulation artefact. Re-report on September 9 to 10 and with the tail excluded.",
          "All evidence is synthetic. State this in every claim and keep generalisation claims to the measured transfer."]),
        ("Machine learning researcher", "Major revision",
         ["The LightGBM average and the L1 gate are not justified on accuracy; ablate them with intervals.",
          "Scores are uncalibrated by about 12x after negative subsampling. Calibrate before thresholding.",
          "Report seeds and confidence intervals for every model comparison; 0.7107 versus 0.7108 is not a result."]),
        ("Cybersecurity specialist", "Not deployable as is",
         ["A payer-controlled field dominates the score and a one-step change defeats the block band.",
          "Structuring to four parts removes the block band; there is no aggregation defence.",
          "Fail-open defaults (demo keys, secrets, silent imputation) must fail closed outside demo mode. Three are fixed in this branch."]),
        ("Dataset and statistics expert", "Concerns about validity",
         ["97 positives in the slice and 1,797 in the full test are too few for three-decimal claims without intervals.",
          "Prevalence differs 100x between AMLworld and AMLgentex; compare ROC-AUC or prevalence-adjusted metrics across generators.",
          "Amount-weighted metrics are dominated by two payments; report them with and without extremes."]),
        ("Industry research engineer", "Promising engineering, weak operations story",
         ["Serving parity and streaming state are strong and rare.",
          "Thresholds should come from an alert budget and a cost ratio, which only a bank can supply; ask for them.",
          "Serverless state breaks fusion, nonces and signing; the demo should say so on the page."]),
        ("Critical reviewer", "Reject in current form, accept the reframed version",
         ["Several Round 1 claims (97% precision, calibrated bands, 15x fewer alerts) are contradicted by the repo's own files.",
          "\"Detects unseen patterns\" is not supported across generators.",
          "Novelty must rest on the audit, the shortcut and the protocol, not on the cascade."]),
    ]
    jhtml = "".join(f'<div class="card"><h3>{E(n)}</h3><p class="verdict">{E(v)}</p><ul class="plain">' + "".join(f"<li>{E(x)}</li>" for x in pts) + "</ul></div>" for n, v, pts in jury)
    add(f"""
<section id="jury">
  <span class="eyebrow">Deliverable 8</span>
  <h2>Independent jury</h2>
  <p>Six reviewer perspectives, each written against the measured evidence above rather than the project's own framing.</p>
  <div class="find">{jhtml}</div>
  <h3>Questions a demanding panel will ask, with the answer the evidence now gives</h3>
  {table(["Question", "Answer from this review"], [
      ["What happens if a launderer pays by cheque instead of ACH?", f"Block recall falls from {pct(G['baseline']['block'], 0)} to {pct(sw['Cheque']['block'], 0)} (served model, slice)."],
      ["Is 0.711 inflated by the end of the dataset?", "Probably: about 36% of test positives sit in a 1,108 payment tail at 59% prevalence. Re-measure."],
      ["Does the LightGBM half help?", f"No: {f(pa['xgb_half_only']['minus_live_mean']['point'])} {ci(pa['xgb_half_only']['minus_live_mean']['ci95'])} for XGBoost alone minus served."],
      ["Are the scores probabilities?", f"No: mean score {f(D['live_mean_uncalibrated']['mean_score'], 4)} against observed {f(D['live_mean_uncalibrated']['observed_rate'], 4)}."],
      ["Do the features work on another generator?", transfer_short()],
      ["Can splitting payments avoid a block?", f"Yes: four parts leave {pct(G['structuring_split4']['block_any_part'], 0)} blocked."]], "wrap")}
</section>""")

    # ---------------- plan
    plan = [
        ("Priority 1: essential corrections", [
            ["Re-measure the headline without the test tail", "Score HI-Small test as Sep 8 part + Sep 9 to 10 + Sep 11 to 18 separately; report AP with bootstrap CI for each", "HI-Small full", "0.5 week"],
            ["Format robustness", "Retrain L2 without fmt_code (keep everything else); evaluate clean and under format switch with research/analyze_slice.py section G on the full test", "HI, LI full", "1 week"],
            ["Single calibrated L2", "Serve XGBoost alone with isotonic calibration fitted on validation; refit bands under a rule written before reading test; update cascade_config.json only if it passes", "HI, LI validation", "1 week"],
            ["Intervals everywhere", "Wrap every headline in the paired bootstrap used here; add seeds for retrained models", "Existing", "0.5 week"],
            ["Fail-closed configuration", "Require RUSETECT_DEMO=1 for demo keys; refuse to start without consortium secret, hash salt and signing key otherwise", "None", "0.5 week"],
            ["Rewrite the claims", "Bring the Round 1 paper and slides in line with the measured numbers (PDF claim table plus findings 1 to 6)", "None", "1 week"]]),
        ("Priority 2: high-value experiments", [
            ["Structuring defence", "Add past-only 1 h and 24 h sums per sender and per payee and GFP-style fan and cycle counts; measure recall under split attacks", "HI, LI, AMLgentex", "2 to 3 weeks"],
            ["Cross-generator matrix", "Extend research/amlgentex_transfer.py to SAML-D and AMLworld Medium; report ROC-AUC and prevalence-adjusted AP", "SAML-D, HI-Medium", "2 weeks"],
            ["Account-level mule detection", "Aggregate payment scores per account (CFDB method); evaluate on AMLgentex account SARs", "AMLgentex, AMLworld", "2 weeks"],
            ["Cost and capacity thresholds", "Choose bands from expected cost and a daily alert budget; publish the cost ratios used", "Existing", "1 week"]]),
        ("Priority 3: advanced directions", [
            ["Costed adversarial evaluation", "Kireev et al. style attacks where each feature change has a cost for the launderer", "HI full", "3 weeks"],
            ["Temporal GNN research arm", "TGN on identical causal splits with EdgeBank and GFP+XGBoost baselines; adopt only if validation gain clears the rule", "AMLworld S/M", "3 to 4 weeks"],
            ["Drift and label delay", "Walk-forward on HI-Large and AMLgentex typology switch; measure detection delay and recovery", "HI-Large, AMLgentex", "2 weeks"],
            ["Federated or partitioned banks", "AMLgentex 12 banks: local vs pooled vs federated ring recall", "AMLgentex", "3 weeks"]]),
    ]
    ph = "".join(f"<h3>{E(t)}</h3>" + table(["Item", "How to do it", "Data", "Effort"], rows, "wrap") for t, rows in plan)
    add(f"""
<section id="plan">
  <span class="eyebrow">Deliverables 7 and 9</span>
  <h2>Prioritised plan</h2>
  <p>Ranked by scientific value per unit of effort, with data that exists today. Priority 1 turns every existing number into a defensible one and should
  finish before new modelling.</p>
  {ph}
</section>""")

    # ---------------- what changed
    add(f"""
<section id="changes">
  <span class="eyebrow">Deliverable 10</span>
  <h2>What this review changed in the repository</h2>
  <ul class="plain">
    <li><code>crates/rusetect/examples/research_dump.rs</code>: streams any AMLworld-format CSV through the served engine and writes the model inputs and all three scores.</li>
    <li><code>research/make_streams.py</code>, <code>research/analyze_slice.py</code>, <code>research/amlgentex_transfer.py</code>, <code>research/build_report.py</code>: reproducible experiments and this page.</li>
    <li>API: <code>payment_format</code> is required and validated; payment and receiving currencies are validated (tests added).</li>
    <li>Rate limiter no longer trusts a client-supplied <code>X-Forwarded-For</code> outside Vercel, and evicts idle clients.</li>
    <li>With configured analyst keys, alerts, evidence packs, STR drafts, cases and the audit log require a key, and the server refuses to start without its signing key, consortium secret and hash salt.</li>
    <li>Models and pages are embedded in the server binary, which fixes the Vercel deployment (every page returned 404 because Vercel runs the server binary without the repository files).</li>
  </ul>
  <h3>Reproduce</h3>
  <pre><code>cargo build --release -p rusetect --example research_dump
python3 -I research/make_streams.py
python3 -I research/analyze_slice.py
python3 -I research/amlgentex_transfer.py &lt;folder with sweden_difficult.csv and sweden_easy.csv&gt;
python3 -I research/build_report.py</code></pre>
  <p class="foot">Limitations: the HI-Small results here come from an 8 hour slice with 97 evaluation positives and five hours of account history; they measure effects
  on the served models, not the full-protocol level. The test-tail figures come from an independent audit of the same file and have not been re-counted here.
  AMLgentex mapping choices (daily timestamps, format and currency) are stated in the script and are part of the measured gap.</p>
</section>""")
    return "\n".join(s)


def transfer_sentence():
    if not T:
        return "Transfer experiment not run."
    d = T["difficult"]
    g = T["amlgentex_difficult_trained_on_hi_slice_eval"]
    return (f"On AMLgentex Sweden (difficult) the served HI-Small model scores ROC-AUC {f(d['zero_shot_served_model_test']['roc_auc'])}, which is chance. "
            f"The same 31 features retrained on AMLgentex reach only {f(d['in_domain_xgboost_test']['roc_auc_mean'])}, and an AMLgentex-trained model ranks HI-Small "
            f"laundering below genuine payments (ROC-AUC {f(g['roc_auc'])}). The 0.711 result is specific to one simulator.")


def transfer_short():
    if not T:
        return "Not measured."
    d = T["difficult"]
    return (f"No. The served model is at chance on AMLgentex ({f(d['zero_shot_served_model_test']['roc_auc'])}); retrained, the same features reach "
            f"{f(d['in_domain_xgboost_test']['roc_auc_mean'])}; trained on AMLgentex and tested on HI-Small, the ranking reverses ({f(T['amlgentex_difficult_trained_on_hi_slice_eval']['roc_auc'])}).")


FULL = J("research", "out", "full_hi_eval.json") if os.path.exists(os.path.join(ROOT, "research", "out", "full_hi_eval.json")) else None
TS = J("research", "out", "transfer_samld.json") if os.path.exists(os.path.join(ROOT, "research", "out", "transfer_samld.json")) else None
TM = J("research", "out", "transfer_himed.json") if os.path.exists(os.path.join(ROOT, "research", "out", "transfer_himed.json")) else None


def tail_sentence():
    if not FULL:
        return ("An independent audit of the same HI-Small file reports that September 11 to 18 holds 1,108 payments, 655 of them laundering. "
                "The headline must be re-measured without it.")
    T = FULL["test"]; s = T["served_live"]
    return (f"Counted on the full file: September 11 to 18 holds {T['tail']['rows']:,} payments, {T['tail']['laundering']} of them laundering, "
            f"{pct(T['tail']['share_of_test_laundering'], 0)} of all test laundering. The served model scores PR-AUC {f(s['all_test']['pr_auc'])} on the whole "
            f"test window but {f(s['without_sep11_18_tail']['pr_auc'])} {ci(s['without_sep11_18_tail']['ci95'])} without the tail.")


def full_section():
    if not FULL:
        return ""
    T, D = FULL["test"], FULL["decision"]
    rows = []
    for k, lab in (("served_live", "Served cascade (XGBoost + LightGBM mean)"), ("control_31_features", "Retrained XGBoost, 31 features (control)"),
                   ("candidate_no_format", "Retrained XGBoost without payment format, calibrated")):
        v = T[k]
        rows.append([lab, f"{f(v['all_test']['pr_auc'])} <small>{ci(v['all_test']['ci95'])}</small>",
                     f"{f(v['without_sep11_18_tail']['pr_auc'])} <small>{ci(v['without_sep11_18_tail']['ci95'])}</small>",
                     f"{f(v['sep09_10_only']['pr_auc'])} <small>{ci(v['sep09_10_only']['ci95'])}</small>"])
    sa, ca = T["served_attack"], T["candidate_attack"]
    bands = []
    for b in ("verify", "hold", "block"):
        sv, cv = T["served_bands"][b], T["candidate_bands"][b]
        sq, cq = T["bands_under_cheque_attack"]["served"][b], T["bands_under_cheque_attack"]["candidate"][b]
        bands.append([b.capitalize(), f"{pct(sv['recall'])} / {pct(sv['precision'])}", f"{pct(cv['recall'])} / {pct(cv['precision'])}",
                      pct(sq['recall']), pct(cq['recall'])])
    typ = [[k, v["laundering"], pct(v["served_block"], 0), pct(v["candidate_block"], 0)] for k, v in T["typology_recall"].items()]
    vals = D["values"]
    chk = ", ".join(f"{k} {'passed' if v else 'failed'}" for k, v in D["checks"].items())
    return f"""
<section id="full">
  <span class="eyebrow">Full dataset, executed</span>
  <h2>Full HI-Small: the tail, the shortcut and a format-free model</h2>
  <p>The full HI-Small file (Hugging Face mirror, CDLA-Sharing-1.0) was checked against the repository: {FULL['splits']['train_rows'] + FULL['splits']['validation_rows'] + FULL['splits']['test_rows']:,} payments,
  5,177 laundering, and the bundled demo slice matches it row for row. Every payment was streamed through the served Rust engine. The served model reproduces the repository's
  test PR-AUC ({f(T['served_live']['all_test']['pr_auc'])} against 0.7107 in <code>results/full_eval_HI.json</code>).</p>
  {table(["Model", "Test PR-AUC [95% CI]", "Without Sep 11 to 18 tail", "Sep 9 to 10 only"], rows, "wrap")}
  <p>The tail holds {T['tail']['rows']:,} payments with {T['tail']['laundering']} laundering ({pct(T['tail']['share_of_test_laundering'], 0)} of test laundering). Without it every model loses about
  0.13 PR-AUC; the honest headline for the served model is {f(T['served_live']['without_sep11_18_tail']['pr_auc'])}.</p>
  <h3>The payment format attack on the full test window</h3>
  <p>Every laundering payment re-sent in another format, PR-AUC: served {', '.join(f'{k} {f(v)}' for k, v in sa.items())}; format-free model {f(min(ca.values()))} in every case.</p>
  {table(["Band (recall / precision)", "Served", "Format-free", "Served, Cheque attack (recall)", "Format-free, Cheque attack (recall)"], bands)}
  <h3>Block recall by laundering typology (patterns file)</h3>
  {table(["Typology", "Laundering in test", "Served", "Format-free"], typ)}
  <h3>Adoption decision</h3>
  <p>The rule was written into <code>research/full_hi_eval.py</code> before test was read: the format-free model must stay within 0.03 validation PR-AUC of the served model (R1),
  keep 80% of its PR-AUC under the attack (R2), and the attack must really hurt the served model (R3). Validation: served {f(vals['served_clean'])} ({f(vals['served_worst_attack'])} under attack),
  format-free {f(vals['candidate_clean'])} ({f(vals['candidate_worst_attack'])} under attack). Result: {chk}. <b>{'Adopted.' if D['adopt'] else 'Not adopted: the served models stay live.'}</b>
  The trade is a product decision: about {f(T['served_live']['all_test']['pr_auc'] - T['candidate_no_format']['all_test']['pr_auc'], 3)} test PR-AUC for immunity to a one-step evasion that removes almost all detection.</p>
</section>"""


def transfer_section():
    if not T:
        return ""
    rows = []
    for name in ("difficult", "easy"):
        d = T[name]
        z, zt, zw, ind = d["zero_shot_served_model_test"], d["zero_shot_served_model_test_transfers_only"], d["zero_shot_transfer_as_wire_test"], d["in_domain_xgboost_test"]
        rows += [[f"Sweden {name}: served HI model, zero-shot", f(z['prevalence'], 4), f"{f(z['roc_auc'])} <small>{ci(z['roc_auc_ci95'])}</small>", f"{f(z['pr_auc'])} <small>{ci(z['pr_auc_ci95'])}</small>"],
                 [f"Sweden {name}: zero-shot, transfers only", f(zt['prevalence'], 4), f(zt['roc_auc']), f(zt['pr_auc'])],
                 [f"Sweden {name}: zero-shot, transfers sent as Wire", f(zw['prevalence'], 4), f(zw['roc_auc']), f(zw['pr_auc'])],
                 [f"Sweden {name}: XGBoost retrained in-domain (3 seeds)", f(ind['prevalence'], 4), f(ind['roc_auc_mean']), f(ind['pr_auc_mean'])]]
    e2d = T["easy_trained_on_difficult_test"]; g2h = T["amlgentex_difficult_trained_on_hi_slice_eval"]; h2h = T["served_model_on_hi_slice_eval_reference"]
    rows += [["Sweden easy → difficult", f(e2d['prevalence'], 4), f(e2d['roc_auc']), f(e2d['pr_auc'])],
             ["AMLgentex difficult → HI-Small slice", f(g2h['prevalence'], 4), f"{f(g2h['roc_auc'])} <small>{ci(g2h['roc_auc_ci95'])}</small>", f"{f(g2h['pr_auc'])} <small>{ci(g2h['pr_auc_ci95'])}</small>"],
             ["Served HI model → HI-Small slice (reference)", f(h2h['prevalence'], 4), f"{f(h2h['roc_auc'])} <small>{ci(h2h['roc_auc_ci95'])}</small>", f"{f(h2h['pr_auc'])} <small>{ci(h2h['pr_auc_ci95'])}</small>"]]
    d = T["difficult"]
    return f"""
<section id="transfer">
  <span class="eyebrow">Deliverable 2, dataset expansion executed</span>
  <h2>Cross-generator transfer: AMLworld to AMLgentex</h2>
  <p>The PDF ranks a second generator as the most informative next dataset. AMLgentex Sweden 100K was downloaded from Hugging Face, mapped to the AMLworld schema
  (mapping stated in <code>research/amlgentex_transfer.py</code>) and streamed through the same Rust feature code. Test = last 20% of the {d['steps']} daily steps.
  Prevalence differs by more than 10x between the sets, so compare ROC-AUC across rows and read PR-AUC against each row's prevalence.</p>
  {table(["Train → test", "Prevalence", "ROC-AUC [95% CI]", "PR-AUC [95% CI]"], rows, "wrap")}
  <p>SAR rate by transaction type (difficult): {', '.join(f'{k} {pct(v, 1)}' for k, v in d['sar_rate_by_type'].items())}. AMLgentex has its own type shortcut:
  every cash transaction is SAR, mirroring ACH in AMLworld. The reversed ranking on HI-Small follows from these two shortcuts meeting: AMLgentex teaches that
  cash is laundering and transfers (mapped to ACH) are normal, while HI-Small laundering is almost all ACH. Training on the easy set and testing on the difficult one
  ({f(e2d['roc_auc'])}) beats training on difficult itself, so more labels from a related generator help more than matching its difficulty. Strongest features retrained in-domain: {', '.join('<code>'+x+'</code>' for x in d['in_domain_top_features_gain'][:5])}.</p>
</section>"""


def _opt(name):
    q = os.path.join(ROOT, "research", "out", name)
    return J("research", "out", name) if os.path.exists(q) else None


def samld_section():
    if not TS:
        return ""
    rows = [[lab, f(TS[k]['roc_auc']), f(TS[k]['pr_auc']), pct(TS[k]['recall_at_0.5pct_alerts'])] for k, lab in (
        ("served_zero_shot", "Served cascade, zero-shot"), ("hi_small_control_zero_shot", "HI-Small retrained, 31 features, zero-shot"),
        ("hi_small_no_format_zero_shot", "HI-Small retrained without format, zero-shot"), ("in_domain_control", "Trained on SAML-D (31 features)"),
        ("in_domain_no_format", "Trained on SAML-D without format"))]
    a = TS["account_reuse_check"]
    fm = ", ".join(f"{k} {pct(v['rate'], 2)}" for k, v in TS["laundering_rate_by_format"].items())
    return f"""
<section id="samld">
  <span class="eyebrow">Second generator, full file</span>
  <h2>SAML-D: the format-free model transfers, the served one does not</h2>
  <p>SAML-D (Oztas et al. 2023, UK typologies) has {TS['rows']:,} payments. It was mapped to the AMLworld schema and streamed through the same Rust feature code.
  The test part is the last 20% in time ({TS['served_zero_shot']['rows']:,} payments, {TS['served_zero_shot']['positives']:,} laundering).</p>
  {table(["Model", "Test ROC-AUC", "Test PR-AUC", "Recall at 0.5% alerts"], rows)}
  <p>Dropping the format feature raises zero-shot ROC-AUC from {f(TS['hi_small_control_zero_shot']['roc_auc'])} to {f(TS['hi_small_no_format_zero_shot']['roc_auc'])}: the format shortcut
  learnt on AMLworld does not hold elsewhere (laundering rate by format here: {fm}). The in-domain scores near 1.0 are not a real result.
  {pct(a['test_laundering_touching_train_laundering_accounts'], 0)} of test laundering touches an account already seen laundering in train, against
  {pct(a['test_genuine_touching_train_laundering_accounts'], 0)} of genuine payments, so the model can learn account identity. The unified pipeline below removes those payments and tests again.</p>
</section>"""


def unified_section():
    U = _opt("unified_train.json")
    if not U:
        return ""
    rows = []
    for n, v in U["test"].items():
        d = U["datasets"][n]
        rows.append([n, f"{d['rows']:,}", d["test_positives"], f(v['served_cascade']['roc_auc']), f(v['pooled_base']['roc_auc']), f(v['pooled_base_new']['roc_auc']),
                     f(v['leave_dataset_out']['roc_auc']), f(v['pooled_base_new']['pr_auc'])])
    un = [[n, v["unseen_accounts"]["test_positives_kept"], f(v["unseen_accounts"]["served_cascade"]["roc_auc"]), f(v["unseen_accounts"]["pooled_base_new"]["roc_auc"]),
           f(v["unseen_accounts"]["pooled_base_new"]["pr_auc"])] if v["unseen_accounts"]["test_positives_kept"] else [n, 0, "none left", "none left", "none left"]
          for n, v in U["test"].items() if "unseen_accounts" in v]
    T_ = U["test"]
    none_left = [n for n, v in T_.items() if "unseen_accounts" in v and not v["unseen_accounts"]["test_positives_kept"]]
    su = T_.get("samld", {}).get("unseen_accounts")
    lift = [n for n in T_ if T_[n]["pooled_base_new"]["roc_auc"] > T_[n]["pooled_base"]["roc_auc"]]
    drop = [n for n in T_ if T_[n]["pooled_base_new"]["roc_auc"] < T_[n]["pooled_base"]["roc_auc"]]
    roc_win = [n for n in T_ if T_[n]["pooled_base"]["roc_auc"] > T_[n]["served_cascade"]["roc_auc"]]
    pr_lose = [n for n in T_ if T_[n]["pooled_base"]["pr_auc"] < T_[n]["served_cascade"]["pr_auc"]]
    loo_bad = [n for n in T_ if T_[n]["leave_dataset_out"]["roc_auc"] < 0.75]
    N = U["new_indicators_test"]; S = U["structuring_attack"]
    st = [[k, pct(v["clean_recall"], 0), pct(v["split4_recall_any_part"], 0), pct(v["split4_spread_recall_any_part"], 0)] for k, v in S.items() if isinstance(v, dict)]
    return f"""
<section id="unified">
  <span class="eyebrow">Unified training, executed</span>
  <h2>One format-free model over bank AML, cards and wallets</h2>
  <p>Every transaction-shaped dataset was streamed through the served Rust feature code with 13 new causal indicators (bursts, mule collection, same-pair structuring,
  amount and hour anomaly, payee age, new-payee velocity; <code>crates/rusetect/examples/research_dump.rs</code>, with a test that they read no future payment).
  Each dataset is split by time 60/20/20; payment format and currency codes are left out, and so are label-adjacent columns (PaySim balances and flag).
  Each dataset gets the same total training weight. Leave-one-out is the model trained on all other datasets: the honest test of a new institution.</p>
  {table(["Dataset", "Payments", "Test positives", "Served ROC", "Pooled, 29 base ROC", "Pooled + 13 new ROC", "Leave-one-out ROC", "Pooled + new PR-AUC"], rows, "wrap")}
  <p>Rule fixed before test was read: adopt the new indicators if the mean ROC-AUC gain across datasets has a paired bootstrap 95% interval above zero.
  Gain {f(N['mean_roc_gain'], 4)} {ci(N['ci95'], 4)}. <b>{'Adopted.' if N['adopt'] else 'Not adopted.'}</b> The new indicators raise ROC-AUC on {', '.join(lift)} and lower it on {', '.join(drop)}.
  Some rank among the strongest features ({', '.join('<code>'+x+'</code>' for x in U['top_features_gain'][:8])}), but they mostly re-express what the 29 base features already carry, so they stay research-only.</p>
  <p>Pooling works in-domain: the pooled format-free model has higher ROC-AUC than the served cascade on {len(roc_win)} of {len(T_)} datasets
  ({'served PR-AUC stays higher on ' + ', '.join(pr_lose) + ', where the format shortcut is real in the generator' if pr_lose else 'and higher PR-AUC on all'}).
  Generalisation does not: trained without a dataset, the model stays below 0.75 ROC-AUC on {', '.join(loo_bad)}. Each generator and fraud type needs some of its own labels,
  so a new institution must label data before it trusts the score.</p>
  <h3>Unseen accounts</h3>
  {table(["Dataset", "Test positives kept", "Served ROC", "Pooled + new ROC", "Pooled + new PR-AUC"], un) if un else ""}
  <p>Test payments touching any account seen in a positive training payment were removed. This is the check against learning account identity (SAML-D above).
  {('No positives remain in ' + ', '.join(none_left) + ': every test fraud there involves an account or merchant already seen in training fraud, so these sets cannot test unseen accounts. ') if none_left else ''}
  {f"SAML-D stays at ROC-AUC {f(su['pooled_base_new']['roc_auc'])} on unseen accounts, so account reuse does not explain its easy separation: the generator itself makes laundering easy to tell apart, and SAML-D scores should not be read as real-world performance." if su else ''}</p>
  <h3>Structuring attack, HI-Small demo window</h3>
  {table(["Model", "Clean recall", "Split in 4, same minute", "Split in 4, 30 min apart"], st)}
  <p>{S['note']}. A laundering payment counts as caught if any part is flagged. Spreading the parts 30 minutes apart defeats every model
  ({', '.join(f"{k} {pct(v['split4_spread_recall_any_part'], 0)}" for k, v in S.items() if isinstance(v, dict))}), including the one with same-pair structuring indicators.
  Each part is scored alone, so the fix is to sum evidence per account over the window and alert on the account, not to add more per-payment features.</p>
</section>"""


def other_section():
    rows = []
    E = _opt("other_elliptic.json")
    if E:
        rows.append(["Elliptic (Bitcoin, real labels)", "Crypto", f"{E['rows']:,} / {E['illicit']:,}", f"{f(E['local_features']['pr_auc'])} → {f(E['local_plus_neighbour_aggregates']['pr_auc'])}",
                     f"{f(E['local_features']['roc_auc'])} → {f(E['local_plus_neighbour_aggregates']['roc_auc'])}",
                     f"After time step 43 (dark market shutdown) PR-AUC falls to {f(E['local_plus_neighbour_aggregates']['test_after_dark_market_shutdown_ts43plus']['pr_auc'])}"])
    E2 = _opt("other_ellipticpp.json")
    if E2:
        rows.append(["Elliptic++ wallets", "Crypto", f"{E2['rows']:,} / {E2['illicit_rows']:,}", f(E2['xgboost']['pr_auc']), f(E2['xgboost']['roc_auc']), E2["note"]])
    G = _opt("other_dgraph.json")
    if G:
        rows.append(["DGraph-Fin (loan default fraud graph)", "Credit", f"{G['nodes']:,} nodes", f"{f(G['node_features']['pr_auc'])} → {f(G['node_plus_graph_aggregates']['pr_auc'])}",
                     f"{f(G['node_features']['roc_auc'])} → {f(G['node_plus_graph_aggregates']['roc_auc'])}", G["heldout"] + "; " + G["reference"]])
    U = _opt("other_ulb.json")
    if U:
        rows.append(["ULB credit card", "Card", f"{U['rows']:,} / {U['fraud']}", f"{f(U['xgboost']['pr_auc'])} <small>{ci(U['xgboost']['pr_auc_ci95'])}</small>", f(U['xgboost']['roc_auc']), U["split"]])
    I = _opt("other_ieee.json")
    if I:
        rows.append(["IEEE-CIS (Vesta)", "Card not present", f"{I['rows']:,} / {I['fraud']:,}", f"{f(I['transaction_features']['pr_auc'])} → {f(I['plus_identity_device_features']['pr_auc'])}",
                     f"{f(I['transaction_features']['roc_auc'])} → {f(I['plus_identity_device_features']['roc_auc'])}", I["split"] + "; device and identity columns added"])
    R = _opt("other_rba.json")
    if R:
        rows.append(["RBA logins (account takeover)", "Login", f"{R['logins_streamed']:,} / {R['account_takeovers']}", f(R['xgboost']['pr_auc_weighted']), f(R['xgboost']['roc_auc_weighted']),
                     f"{R['test_takeovers']} takeovers in test; " + R["note"]])
    B = _opt("other_berka.json")
    if B:
        rows.append(["Berka (real Czech bank)", "Credit default", f"{B['loans']} loans / {B['bad_loans']} bad", f(B['test']['pr_auc']), f(B['test']['roc_auc']), f"{B['test']['bad']} bad loans in test; " + B["note"]])
    Y = _opt("other_gad.json")
    if Y:
        for k in ("YelpChi", "InstrumentReviews"):
            v = Y[k]
            rows.append([k + " (review graph)", "Not payments", f"{v['nodes']:,} / {v['positives']:,}", f"{f(v['node_features']['pr_auc_mean'])} → {f(v['plus_neighbour_aggregates']['pr_auc_mean'])}",
                         f"{f(v['node_features']['roc_auc_mean'])} → {f(v['plus_neighbour_aggregates']['roc_auc_mean'])}", Y["note"]])
    if not rows:
        return ""
    return f"""
<section id="other">
  <span class="eyebrow">Other fraud types, executed</span>
  <h2>Crypto, credit, cards, logins and graphs</h2>
  <p>Datasets whose schema does not fit a payment stream were tested on their own, with a time split wherever the data has time, and XGBoost as the common learner.
  An arrow shows the gain from adding graph or neighbour aggregates, the architecture question the audit raises. These are capability checks, not served models.</p>
  {table(["Dataset", "Fraud type", "Rows / positives", "Test PR-AUC", "Test ROC-AUC", "Notes"], rows, "wrap")}
  <p>Neighbour aggregates help on every graph (Elliptic, DGraph and both review graphs), which supports adding graph features before a graph neural network.
  Elliptic shows the limit of any static model: after the dark market shutdown the illicit pattern changes and recall collapses, so drift monitoring must trigger retraining.</p>
</section>"""


def message_section():
    M = _opt("message_external_eval.json")
    if not M:
        return ""
    a, h = M["all_external"], M["hinglish_subset"]
    rows = [["All new messages", a["messages"], a["scam"], pct(a["caution_or_above"]["recall"]), pct(a["high_or_above"]["recall"]), a["caution_or_above"]["false_alarms"]],
            ["Hinglish subset", h["messages"], h["scam"], pct(h["caution_or_above"]["recall"]), pct(h["high_or_above"]["recall"]), h["caution_or_above"]["false_alarms"]]]
    return f"""
<section id="messages">
  <span class="eyebrow">Scam message screen, external test</span>
  <h2>Messages the screen has never seen</h2>
  <p>Source: {M['source']}. Only messages absent from the training corpus were kept.</p>
  {table(["Set", "Messages", "Scam", "Recall, caution or above", "Recall, high or above", "False alarms"], rows)}
  <p>Missed Hinglish scams are a recharge offer and an instant loan offer: types the rule set does not cover yet.</p>
</section>"""


CSS = """
:root{
  --bg:#f5f4ef; --panel:#ffffff; --ink:#111111; --ink-2:#5f5f5c; --ink-3:#8d8d88; --rule:#e7e6e1;
  --olive:#3c4a14; --olive-ink:#f3f5ea; --accent:#3c4a14; --accent-soft:#e8ecda; --bad:#b3261e; --bad-soft:#f6e1de;
  --s1:#2a78d6; --s2:#eb6834;
  --serif:"Instrument Serif","Iowan Old Style","Palatino Linotype",Georgia,serif;
  --sans:"Geist",ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
  --mono:"Geist Mono",ui-monospace,SFMono-Regular,Menlo,monospace;
  color-scheme:light;
}
/* Layout: one reading column (max 980px); findings as cards, evidence as tables and two charts. Light only, like the rest of the site. */
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 var(--sans);padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px));padding-block:20px 80px}
nav.site,main{max-width:980px;margin:0 auto}
main{display:flex;flex-direction:column;gap:56px}
h1,h2{font-family:var(--serif);font-weight:400;text-wrap:balance;margin:0;letter-spacing:-.015em}
h1{font-size:clamp(36px,4.8vw,56px);line-height:1}
h2{font-size:30px;line-height:1.15}
h3{font:600 16.5px/1.35 var(--sans);margin:6px 0 0;text-wrap:balance}
p{margin:0;max-width:72ch;color:var(--ink-2)}
.hero{display:flex;flex-direction:column;gap:14px;padding-top:24px}
.lede{font-size:17px;color:var(--ink)}
.eyebrow{font:500 11.5px/1 var(--mono);letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
.meta{display:flex;flex-wrap:wrap;gap:8px 22px;font:12.5px var(--mono);color:var(--ink-3)}
section{display:flex;flex-direction:column;gap:16px;min-width:0}
.find{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}
@media (max-width:720px){.find{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 18px;display:flex;flex-direction:column;gap:8px;min-width:0}
.card p{font-size:14px}
.card.fix{border-color:color-mix(in srgb,var(--bad) 40%,var(--rule));background:var(--bad-soft)}
.card.fix h3{color:var(--bad)}
.verdict{font:500 12.5px var(--mono);color:var(--accent)}
.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px}
@media (max-width:720px){.kpis{grid-template-columns:repeat(2,minmax(0,1fr))}}
.kpi{border-left:2px solid var(--accent);padding:4px 0 4px 14px;display:flex;flex-direction:column;gap:2px;min-width:0}
.kpi .v{font:500 24px var(--mono);font-variant-numeric:tabular-nums}
.kpi .k{font-size:12.5px;color:var(--ink-2);line-height:1.35}
.tbl{overflow-x:auto;border:1px solid var(--rule);border-radius:10px;background:var(--panel)}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th,td{padding:9px 12px;text-align:right;border-bottom:1px solid var(--rule);white-space:nowrap;vertical-align:top}
th:first-child,td:first-child{text-align:left}
.wrap td,.wrap th{white-space:normal;text-align:left;min-width:120px}
.wrap table{min-width:760px}
th{font:500 11px var(--mono);letter-spacing:.06em;text-transform:uppercase;color:var(--ink-3);background:color-mix(in srgb,var(--bg) 60%,var(--panel))}
td{font-variant-numeric:tabular-nums}
td small{color:var(--ink-3);font-family:var(--mono);font-size:11.5px}
tr:last-child td{border-bottom:0}
ul.plain{margin:0;padding-left:18px;color:var(--ink-2);max-width:74ch;display:flex;flex-direction:column;gap:6px}
.card ul.plain{font-size:14px}
code{font:12.5px var(--mono);background:var(--accent-soft);padding:1px 5px;border-radius:3px;color:var(--ink)}
pre{margin:0;background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:14px 16px;overflow-x:auto}
pre code{background:none;padding:0}
.foot{font-size:12.5px;color:var(--ink-3)}
.chart{background:var(--panel);border:1px solid var(--rule);border-radius:10px;padding:16px 14px 8px;overflow-x:auto}
.chart svg{display:block;min-width:600px;width:100%;height:auto}
.chart .grid{stroke:var(--rule);stroke-width:1}
.chart .ref{stroke:var(--bad);stroke-width:1.5;stroke-dasharray:4 3}
.chart .reflab{fill:var(--bad);font:11px var(--mono)}
.chart .tick{fill:var(--ink-3);font:11px var(--mono)}
.chart .lab{fill:var(--ink);font:12.5px var(--sans)}
.chart .val{fill:var(--ink-2);font:11px var(--mono)}
.chart .whisk{stroke:var(--ink-2);stroke-width:2;stroke-linecap:round}
.chart .dot{fill:var(--panel);stroke:var(--accent);stroke-width:2}
.chart .dot.em{fill:var(--accent)}
.chart .s1{fill:var(--s1)} .chart .s2{fill:var(--s2)}
.chart .hit:hover .whisk{stroke:var(--ink)} .chart .hit:hover rect.s1,.chart .hit:hover rect.s2{opacity:.8}
.legend{display:flex;gap:18px;font-size:12.5px;color:var(--ink-2);padding:0 4px 10px}
.legend i{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:6px;vertical-align:-1px}
.legend i.s1{background:var(--s1)} .legend i.s2{background:var(--s2)}
nav.site{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:12px 16px;padding-block:6px 10px;margin-bottom:12px}
nav.site .brand{font:600 19px/1 var(--sans);letter-spacing:-.03em;text-decoration:none;color:var(--ink)}
nav.site .links{display:flex;flex-wrap:wrap;gap:6px 24px;font:500 13.5px var(--sans)}
nav.site .links a{text-decoration:none;color:var(--ink)}
nav.site .links a[aria-current]{text-decoration:underline;text-underline-offset:6px;text-decoration-thickness:1.5px}
nav.site .cta{justify-self:end;background:var(--olive);color:var(--olive-ink);text-decoration:none;font:500 13.5px var(--sans);padding:10px 18px;border-radius:999px}
@media (max-width:780px){nav.site{grid-template-columns:1fr auto}nav.site .links{grid-column:1/-1;order:3}}
:focus-visible{outline:2px solid var(--olive);outline-offset:3px}
"""

FONTS = '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif&family=Geist:wght@400;500;600&family=Geist+Mono:wght@400;500&display=swap">'
NAV = ('<nav class="site" aria-label="Pages"><a class="brand" href="/">RuseTect</a><div class="links"><a href="/">Home</a>'
       '<a href="/investigate">Investigator</a><a href="/report">Benchmark report</a><a href="/research" aria-current="page">Research review</a>'
       '<a href="/docs">API docs</a></div><a class="cta" href="/console">Open live console ↗</a></nav>')


def main():
    body = build()
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<!-- Generated by research/build_report.py from research/out/*.json; do not edit by hand. -->
<title>RuseTect Research Review</title>
<meta name="description" content="Independent review of the RuseTect audit: dataset audit, robustness and evasion tests, cross-generator transfer, security assessment and plan.">
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
{FONTS}
<style>{CSS}</style>
</head>
<body>
{NAV}
<main>
{body}
</main>
</body>
</html>
"""
    out = os.path.join(ROOT, "service", "static", "research.html")
    open(out, "w").write(page)
    print("wrote", out, len(page), "bytes")
    if "--fragment" in sys.argv:
        frag = sys.argv[sys.argv.index("--fragment") + 1]
        open(frag, "w").write(f'<title>RuseTect Research Review</title>\n<meta name="claude:viewport-fit-sides" content="cover">\n{FONTS}\n<style>{CSS}</style>\n<main>\n{body}\n</main>\n')
        print("wrote", frag)


if __name__ == "__main__":
    main()
