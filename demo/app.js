(function () {
  "use strict";
  var D, outcome = "conversion";
  var css = getComputedStyle(document.documentElement);
  function token(name) { return css.getPropertyValue(name).trim(); }
  function layout(extra) {
    var base = {
      paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
      font: { color: token("--text"), size: 13 }, margin: { t: 40, r: 20, b: 50, l: 60 },
      xaxis: { gridcolor: token("--line"), zerolinecolor: token("--control") },
      yaxis: { gridcolor: token("--line"), zerolinecolor: token("--control") }
    };
    return Object.assign(base, extra || {});
  }
  var CFG = { displayModeBar: false, responsive: true };
  function $(id) { return document.getElementById(id); }
  function pp(x, d) { return (x * 100 >= 0 ? "+" : "") + (x * 100).toFixed(d == null ? 3 : d) + " pp"; }
  function pct(x, d) { return (x * 100).toFixed(d == null ? 2 : d) + "%"; }
  function num(x) { return Math.round(x).toLocaleString("en-US"); }
  function metric(k, v) { return '<div class="metric"><div class="v">' + v + '</div><div class="k">' + k + "</div></div>"; }
  function o() { return D.outcomes[outcome]; }

  function renderNotice() {
    $("notice").textContent = "Real Criteo data: a seeded random sample of " + num(D.sample_size) + " of " + num(D.full_rows) +
      " users (seed " + D.seed + "), plus a validation pass over all " + num(D.full_rows) + ". Results generated " + D.generated +
      ". The dataset pools several tests and was sub-sampled by Criteo, so effect sizes are benchmark values, not forecasts.";
  }

  function renderHeadline() {
    var r = o(), fu = D.full_data.unadjusted[outcome], fa = D.full_data.adjusted[outcome];
    var fp = (D.full_data.propensity || {})[outcome];
    var z = 1.959964;
    var rows = [
      { name: "Sample, raw difference", v: r.ate.ate, lo: r.ate.ci_low, hi: r.ate.ci_high, c: token("--accent") },
      { name: "Sample, covariate-adjusted", v: r.adjusted.adjusted_ate, lo: r.adjusted.ci_low, hi: r.adjusted.ci_high, c: token("--good") },
      { name: "All users, raw difference", v: fu.ate, lo: fu.ci_low, hi: fu.ci_high, c: token("--accent") },
      { name: "All users, covariate-adjusted", v: fa.adjusted_ate, lo: fa.adjusted_ate - z * fa.std_error, hi: fa.adjusted_ate + z * fa.std_error, c: token("--good") }
    ];
    if (fp) rows.push({ name: "All users, propensity-weighted", v: fp.ipw_ate, lo: fp.ci_low, hi: fp.ci_high, c: token("--good") });
    $("headline-metrics").innerHTML = metric("Control rate", pct(r.control_rate, 3)) +
      metric("Raw effect, all users", pp(fu.ate)) + metric("Adjusted effect, all users", pp(fa.adjusted_ate)) +
      metric("Relative uplift (sample)", (r.relative.relative_uplift >= 0 ? "+" : "") + (r.relative.relative_uplift * 100).toFixed(1) + "%");
    Plotly.newPlot("chart-forest", [{
      type: "scatter", mode: "markers", orientation: "h",
      x: rows.map(function (x) { return x.v * 100; }), y: rows.map(function (x) { return x.name; }),
      marker: { size: 11, color: rows.map(function (x) { return x.c; }) },
      error_x: { type: "data", symmetric: false, array: rows.map(function (x) { return (x.hi - x.v) * 100; }),
        arrayminus: rows.map(function (x) { return (x.v - x.lo) * 100; }), color: token("--control") }
    }], layout({ xaxis: { title: "Effect on " + outcome + " (percentage points)", gridcolor: token("--line"), zeroline: true, zerolinecolor: token("--control") },
      yaxis: { autorange: "reversed", automargin: true }, margin: { t: 20, r: 20, b: 50, l: 230 }, height: 340 }), CFG);
    var gap = fu.ate - fa.adjusted_ate;
    $("headline-text").textContent = "On all " + num(D.full_rows) + " users the raw effect is " + pp(fu.ate) + " and the covariate-adjusted effect is " +
      pp(fa.adjusted_ate) + ". In a perfectly randomized experiment these would agree; the gap of " + pp(gap) +
      " reflects the imbalance shown below, so the adjusted figure is the safer reading." +
      (fp ? " A second, independent correction (cross-fitted propensity weighting) gives " + pp(fp.ipw_ate) + ", within " + Math.abs((fp.ipw_ate - fa.adjusted_ate) * 100).toFixed(2) + " pp of the adjusted estimate." : "") +
      " The sample estimate can differ from the all-users truth by chance: 95% intervals from 100 independent samples contained the all-users raw effect " +
      pct(D.sampling_validation.ci_coverage_of_full_data_ate[outcome], 0) + " of the time.";
  }

  function renderTrust() {
    var s = D.sample_ratio, b = D.balance;
    $("srm-text").textContent = "Sample ratio: " + pct(s.observed_share_treated) + " treated versus " + pct(s.expected_share_treated) +
      " designed (p = " + s.p_value.toFixed(2) + "), " + (s.mismatch ? "a mismatch." : "no sign of a problem.");
    var rows = b.rows.slice().sort(function (x, y) { return Math.abs(x.smd) - Math.abs(y.smd); });
    Plotly.newPlot("chart-balance", [{
      type: "bar", orientation: "h", x: rows.map(function (r) { return r.smd; }), y: rows.map(function (r) { return r.feature; }),
      marker: { color: rows.map(function (r) { return Math.abs(r.smd) < b.smd_threshold ? token("--good") : token("--accent"); }) }
    }], layout({ xaxis: { title: "Standardized mean difference, treated minus control (dashed: 0.1)", range: [-0.12, 0.12], gridcolor: token("--line") },
      shapes: [-0.1, 0.1].map(function (x) { return { type: "line", x0: x, x1: x, y0: 0, y1: 1, yref: "paper", line: { dash: "dash", color: token("--control") } }; }),
      height: 360 }), CFG);
    var worst = rows[rows.length - 1];
    $("balance-text").textContent = "Largest difference: " + Math.abs(worst.smd).toFixed(3) + " (" + worst.feature + "); " + b.features_over_threshold +
      " of " + rows.length + " features exceed the usual 0.1 rule of thumb. Yet a classifier predicts treatment from the features with AUC " +
      b.auc.toFixed(4) + " (95% CI " + b.auc_ci_low.toFixed(4) + " to " + b.auc_ci_high.toFixed(4) + "; 0.5 would be perfectly random), so assignment is " +
      (b.joint_imbalance_detected ? "slightly predictable from the features. A passing sample ratio check does not rule this out." : "not predictable from the features.");
  }

  function renderExposure() {
    var c = o().cace;
    $("exposure-metrics").innerHTML = metric("Treated users exposed", pct(c.exposure_rate_treated, 1)) + metric("Intent-to-treat effect", pp(c.itt)) +
      metric("Effect on exposed users (CACE)", pp(c.cace, 2)) + metric("Naive exposed vs control", pp(c.naive_exposed_vs_control, 2));
    var t = "Only " + pct(c.exposure_rate_treated, 1) + " of treated users were actually exposed, and no control user ever was. The effect of being assigned is " +
      pp(c.itt) + "; for users who actually saw the ad it is " + pp(c.cace, 2) + " (95% CI " + (c.cace_ci_low * 100).toFixed(2) + " to " + (c.cace_ci_high * 100).toFixed(2) +
      "). Comparing exposed treated users with all control users gives " + pp(c.naive_exposed_vs_control, 2) + ", which mixes the ad effect with differences between people who get exposed and people who do not.";
    if (c.complier_rate_control != null) {
      t += " Users who get exposed would have converted at " + pct(c.complier_rate_control, 2) + " without the ad, versus " + pct(c.never_taker_rate, 2) + " for treated users who were never exposed.";
    }
    $("exposure-text").textContent = t + " This assumes the ad affects users only through being exposed.";
  }

  function renderSegments() {
    var f = $("feature").value, seg = o().segments[f].segments.filter(function (s) { return s.ate != null; });
    Plotly.newPlot("chart-segments", [{
      type: "scatter", mode: "markers", x: seg.map(function (s) { return s.label; }), y: seg.map(function (s) { return s.ate * 100; }),
      marker: { size: 11, color: seg.map(function (s) { return s.significant ? token("--good") : token("--control"); }) },
      error_y: { type: "data", symmetric: false, array: seg.map(function (s) { return (s.ci_high - s.ate) * 100; }),
        arrayminus: seg.map(function (s) { return (s.ate - s.ci_low) * 100; }), color: token("--control") }
    }], layout({ title: { text: "Effect on " + outcome + " by " + f + " bin (green: significant after FDR)", font: { size: 14 } },
      yaxis: { title: "Percentage points", gridcolor: token("--line"), zeroline: true, zerolinecolor: token("--control") }, height: 340 }), CFG);
    $("segment-table").innerHTML = "<tr><th>Bin</th><th>Users</th><th>Effect (pp)</th><th>Adjusted p</th></tr>" + seg.map(function (s) {
      return "<tr><td>" + s.label + "</td><td>" + num(s.n) + "</td><td>" + (s.ate * 100).toFixed(4) + "</td><td>" + s.p_value_adjusted.toExponential(2) + (s.significant ? " *" : "") + "</td></tr>";
    }).join("");
  }

  function renderUplift() {
    var q = o().qini;
    Plotly.newPlot("chart-qini", [
      { type: "scatter", x: q.curve_x.map(function (x) { return x * 100; }), y: q.curve_qini, name: "Uplift model", line: { color: token("--accent"), width: 3 } },
      { type: "scatter", x: q.curve_x.map(function (x) { return x * 100; }), y: q.curve_random, name: "Random targeting", line: { color: token("--control"), dash: "dash" } }
    ], layout({ title: { text: "Qini curve on held-out users (" + outcome + ")", font: { size: 14 } }, xaxis: { title: "Share of users targeted (%)", gridcolor: token("--line") },
      yaxis: { title: "Cumulative incremental responses", gridcolor: token("--line") }, legend: { orientation: "h", y: -0.25 }, height: 340 }), CFG);
    var finalQ = q.curve_qini[q.curve_qini.length - 1], at = function (share) {
      var i = q.curve_x.findIndex(function (x) { return x >= share; });
      return q.curve_qini[i] / finalQ;
    };
    $("qini-text").textContent = "Qini coefficient " + q.qini_coefficient.toFixed(1) + " (95% bootstrap CI " + q.qini_ci_low.toFixed(1) + " to " + q.qini_ci_high.toFixed(1) +
      "). Targeting the model's top 10% captures " + pct(at(0.1), 0) + " of all incremental " + outcome + "s on the held-out users, the top 30% captures " + pct(at(0.3), 0) + ".";
    var L = o().learners.learners;
    Plotly.newPlot("chart-learners", [{
      type: "scatter", mode: "markers", x: L.map(function (r) { return r.name; }), y: L.map(function (r) { return r.qini_coefficient; }),
      marker: { size: 12, color: token("--accent") },
      error_y: { type: "data", symmetric: false, array: L.map(function (r) { return r.ci_high - r.qini_coefficient; }),
        arrayminus: L.map(function (r) { return r.qini_coefficient - r.ci_low; }), color: token("--accent") }
    }], layout({ title: { text: "Qini coefficient by learner (95% bootstrap CI)", font: { size: 14 } }, yaxis: { title: "Qini coefficient", gridcolor: token("--line"), zeroline: true, zerolinecolor: token("--control") }, height: 320 }), CFG);
    var diff = L.filter(function (r) { return r.differs_from_t_learner; }).map(function (r) { return r.name; });
    $("learners-text").textContent = "Best point estimate: " + o().learners.best_by_point_estimate + ". " +
      (diff.length ? "Significantly different from the T-learner: " + diff.join(", ") + "." : "No learner is significantly different from the T-learner, so the simplest one is kept.");
  }

  function renderPolicy() {
    var q = o().qini, sc = o().policy_scale;
    var audience = Math.max(1000, +$("audience").value || 1000000), value = Math.max(0.01, +$("value").value || 30), cost = Math.max(0, +$("cost").value || 0);
    var scale = audience / (sc.n_test * sc.treated_share_test), finalQ = q.curve_qini[q.curve_qini.length - 1];
    var best = 0, bestProfit = -Infinity, model = [], random = [];
    q.curve_x.forEach(function (x, i) {
      var inc = q.curve_qini[i] * scale, pm = value * inc - cost * x * audience, pr = value * finalQ * scale * x - cost * x * audience;
      model.push(pm); random.push(pr);
      if (pm > bestProfit) { bestProfit = pm; best = i; }
    });
    var all = model[model.length - 1], bx = q.curve_x[best], breakeven = value * finalQ * scale / audience;
    $("policy-metrics").innerHTML = metric("Best share to target", pct(bx, 1)) + metric("Incremental conversions", num(q.curve_qini[best] * scale)) +
      metric("Profit at best share", num(bestProfit)) + metric("Profit if everyone is targeted", num(all));
    Plotly.newPlot("chart-policy", [
      { type: "scatter", x: q.curve_x.map(function (x) { return x * 100; }), y: model, name: "Uplift model", line: { color: token("--accent"), width: 3 } },
      { type: "scatter", x: q.curve_x.map(function (x) { return x * 100; }), y: random, name: "Random targeting", line: { color: token("--control"), dash: "dash" } },
      { type: "scatter", mode: "markers", x: [bx * 100], y: [bestProfit], name: "Best share", marker: { size: 13, color: token("--good") } }
    ], layout({ xaxis: { title: "Share of audience targeted (%)", gridcolor: token("--line") }, yaxis: { title: "Scenario profit", gridcolor: token("--line"), zeroline: true, zerolinecolor: token("--control") },
      legend: { orientation: "h", y: -0.25 }, height: 340 }), CFG);
    $("policy-text").textContent = "Targeting everyone only pays when the cost per contact is below " + breakeven.toFixed(4) + " (this is value x the all-users effect on the held-out split). " +
      "The optimum is chosen on the same held-out users, so it is slightly optimistic, and no interval is shown here; the app computes a bootstrap interval for any scenario.";
  }

  function renderDesign() {
    var lift = (+$("lift").value || 10) / 100, power = +$("power").value / 100, share = +$("share").value / 100;
    $("power-out").textContent = Math.round(power * 100) + "%"; $("share-out").textContent = Math.round(share * 100) + "%";
    var base = o().control_rate, html, text;
    try {
      var r = Power.requiredSampleSize(base, lift, 0.05, power, share);
      html = metric("Users needed in total", num(r.nTotal)) + metric("Treated / control", num(r.nTreated) + " / " + num(r.nControl)) + metric("Baseline (control rate)", pct(base, 3));
      var m = Power.minimumDetectableEffect(base, D.sample_size, 0.05, power, 0.85);
      text = "To detect a " + (lift * 100).toFixed(0) + "% relative change from the control rate with " + Math.round(power * 100) + "% power at the 5% level. " +
        "For reference, the " + num(D.sample_size) + "-user sample at the original 85% treated share could detect a relative lift of about " + (m.relativeEffect * 100).toFixed(1) + "% (same power).";
    } catch (e) { html = metric("Not computable", "-"); text = e.message; }
    $("design-metrics").innerHTML = html; $("design-text").textContent = text;
  }

  function renderValidation() {
    var s = D.sampling_validation, c = D.code_simulations;
    $("validation-text").innerHTML = "<ul><li>The file matches its published SHA-256 and row count, and reproduces the dataset card's treatment ratio, visit rate and conversion rate.</li>" +
      "<li>100 independent samples of " + num(s.sample_size) + " users: the 95% interval contained the all-users effect " + pct(s.ci_coverage_of_full_data_ate.conversion, 0) + " (conversion) and " +
      pct(s.ci_coverage_of_full_data_ate.visit, 0) + " (visit) of the time; the sample ratio check raised no false alarm.</li>" +
      "<li>Simulation of the statistics code (" + num(c.runs) + " runs): effect interval coverage " + pct(c.ate_ci_coverage, 1) + ", relative uplift interval " + pct(c.relative_uplift_ci_coverage, 1) +
      ", type I error " + pct(c.type_i_error_at_5_percent, 1) + ", family-wise false positives across 10 null segments with Benjamini-Hochberg " + pct(c.familywise_false_positive_rate_bh_10_null_segments, 1) + ".</li></ul>";
  }

  function renderAll() { renderHeadline(); renderExposure(); renderSegments(); renderUplift(); renderPolicy(); renderDesign(); }

  fetch("data.json").then(function (r) { return r.json(); }).then(function (data) {
    D = data;
    var sel = $("feature");
    Object.keys(D.outcomes.conversion.segments).forEach(function (f) { var op = document.createElement("option"); op.value = f; op.textContent = f; sel.appendChild(op); });
    renderNotice(); renderTrust(); renderValidation(); renderAll();
    $("outcome").addEventListener("change", function (e) { outcome = e.target.value; renderAll(); });
    $("feature").addEventListener("change", renderSegments);
    ["audience", "value", "cost"].forEach(function (id) { $(id).addEventListener("input", renderPolicy); });
    ["lift", "power", "share"].forEach(function (id) { $(id).addEventListener("input", renderDesign); });
  }).catch(function (e) { $("notice").textContent = "Could not load data.json: " + e.message; });
})();
