# Validation record

Everything below was executed, not assumed. Numbers come from `outputs/validation.json` (reproduce with `make validate`), from the test suite, or from the commands named in each entry. Where verification found a defect, the defect and its fix are recorded.

## Data integrity and fidelity (2026-10-28)

- Downloaded `criteo-research-uplift-v2.1.csv.gz` from the Hugging Face mirror and verified its SHA-256 (`2716e1bf…616dc`) and row count (13,979,592) against the constants pinned in `core/data.py`.
- Recomputed the dataset card's key figures on all 13,979,592 rows: treatment ratio 0.850000 (card 0.85), visit rate 0.046992 (card 0.046992), conversion rate 0.002917 (card 0.00292, rounded).
- Confirmed one-sided non-compliance: no control user is ever exposed (control exposure rate 0.0 exactly), which is what makes the complier analysis valid.
- Sample ratio check on all rows: 85.0000% treated, p = 0.999.

## What the full data says, and how the 500k sample compares (2026-10-28)

All 13,979,592 users, streamed in chunks:

| Estimate | Conversion | Visit |
|---|---|---|
| Raw effect (95% CI) | +0.1152 pp (0.1085 to 0.1219) | +1.0342 pp (1.0056 to 1.0629) |
| Relative uplift | +59.4% | +27.1% |
| Covariate-adjusted effect (SE) | +0.1002 pp (0.0034) | +0.7734 pp (0.0131) |
| Effect on exposed users, CACE (95% CI) | +3.196 pp (3.010 to 3.383) | +28.700 pp (27.911 to 29.488) |
| Same, adjusted for the features (95% CI) | +2.827 pp (2.640 to 3.015) | +21.830 pp (21.107 to 22.553) |

- The seeded 500k sample (seed 42) gives +0.115 pp (conversion) and +1.19 pp (visit) raw. The visit estimate sits about two standard deviations above the all-users value (+1.03 pp): ordinary sampling noise, not a bug. The seed was not changed.
- 100 independent random samples of 500,000 users: the 95% interval contained the all-users raw effect in 97% of samples for conversion and 92% for visit (with 100 draws the standard error of a coverage figure is about 2 points). The sample ratio check raised 0 false alarms.

## Randomization is imperfect, a finding (2026-10-28)

- Standardized mean differences between treated and control are small but not zero: largest 0.049 (`f3`), then 0.040 (`f6`) and 0.031 (`f5`) on all users. None exceeds the common 0.1 rule of thumb.
- On the 500k sample a gradient-boosting classifier predicts treatment from the 12 features with held-out AUC 0.5078 (bootstrap 95% CI 0.5041 to 0.5120). Perfect randomization would give 0.5, so the interval rules it out.
- Consequence: the raw and covariate-adjusted effects disagree on all users (visit +1.034 vs +0.773 pp, conversion +0.115 vs +0.100 pp). Both are consistent under perfect randomization, so the gap points to assignment that depends on the features. This fits Criteo's description of the file as several pooled tests, which would give users different treatment probabilities.
- A third, independent method agrees. Cross-fitted propensity weighting (gradient-boosting propensities fitted on one half of a 4,000,000-user random subset and applied to the other, weights stabilized with the Hajek estimator, no weight clipped, propensities between 0.585 and 0.972) gives +0.736 pp on visits (95% CI 0.679 to 0.792) and +0.101 pp on conversions (0.088 to 0.115). It lands within 0.04 pp of the regression-adjusted visit estimate and within 0.001 pp for conversion, and both sit below the raw +1.034 and +0.115 pp. Its standard error treats the propensity as known, which is conservative. In tests it recovers a planted effect that a raw difference overstates, and it agrees with the raw difference under genuine randomization.
- The app and demo therefore show the adjusted estimate next to the raw one, and the explanations no longer call the raw difference simply causal. A passing sample ratio check does not detect this kind of problem.

## The effect on exposed users, corrected for imperfect randomization (2026-11-09)

- The raw complier estimate divides the raw intent-to-treat effect by the raw exposure share. The section above shows the raw effect is too high because assignment depends on the features, so the raw CACE inherited that bias. This was found in an external review and is corrected here.
- `adjusted_complier_effect` adjusts both parts for the features with the same Lin interacted regression and takes their ratio. The delta-method interval uses the HC1 covariance of the two coefficients, which share one design matrix. `make validate` computes it on all rows in chunks (`full_adjusted_cace`), and a test checks the chunked and direct versions agree.
- On all 13,979,592 users the adjusted exposure share is 3.543% against 3.604% raw, and the CACE is +21.83 pp on visits (95% CI 21.11 to 22.55) against +28.70 raw, and +2.83 pp on conversions (2.64 to 3.02) against +3.20. The visit estimate falls by about a quarter, in line with the intent-to-treat effect.
- In a simulation where assignment and exposure both depend on a feature that also drives the outcome (200 runs), the raw estimate is biased upward and the adjusted interval covers the planted effect at the nominal rate (`tests/test_extensions.py`). With no covariates the adjusted estimate equals the raw one exactly.
- The language-model tool `complier_effect` still returns the raw figure, because changing a tool result requires rerunning the evaluation. The app and the demo show the adjusted figure next to it and say which to read first.

## Correctness of the statistics code (2026-10-28)

Simulation, 2,000 runs unless noted (`core/validate.py`):

- Effect confidence interval coverage 94.5% (nominal 95%). Relative-uplift interval coverage 94.3%. Type I error of the two-proportion test 4.6% (nominal 5%).
- Benjamini-Hochberg across 10 null segments, 300 runs: family-wise false positives 5.3%, mean 0.057 false discoveries per run.

Equivalence and known-answer tests (`tests/`):

- The z-statistic and p-value equal `statsmodels.proportions_ztest`.
- The regression-adjusted estimate, its HC1 robust standard error and p-value equal `statsmodels` OLS to 1e-9.
- `required_sample_size(0.5, absolute_lift=-0.1)` returns the textbook 388 users per arm. The JavaScript calculator on the demo page returns the same integers as the Python function on five parameter sets (`tests/test_demo_power_js.py`).
- A simulation of 400 experiments at the computed sample size must reject between 74% and 86% of the time for a nominal 80% power. The test enforces that band.
- Over 300 simulated experiments the complier-effect interval must cover the true effect between 91% and 99% of the time (the test enforces that band), and the naive exposed-versus-control contrast must overstate the true effect in that simulation.
- Qini curves reproduce a hand-computed four-user example exactly (coefficient 0.75), the targeting policy reproduces a hand-computed profit table, and its break-even cost equals value times the all-users effect.
- The covariate balance check flags a planted assignment leak (AUC above 0.6, the leaking feature identified) and passes a properly randomized sample.

## Defects found by verification, and fixed

- **Sample size 0 silently became the default** (`sample_size or default`). Found by a bounds test. It is now rejected.
- **Policy scale was understated by the treated share.** A Qini curve counts incremental responses per treated-sized population. The profit curve now divides by the treated share. The test asserting break-even cost equals value times the effect would catch a regression.
- **Unadjusted standard error used a binary-outcome formula** inside the regression-adjustment comparison. Replaced by a Welch standard error valid for any outcome.
- **Quantile segmentation collapsed for 7 of 12 features** because one value holds 82% to 99% of users. Dominant values now get their own segment (`is_atom`) and the rest are binned by quantiles.
- **nginx rate limiting returned HTTP 503 for Streamlit's lazily loaded JavaScript chunks** (109 rejected requests in one page load), which broke the chat input. Static assets are now exempt and the limit is raised. CI repeats 300 parallel asset requests and fails on any non-200.
- **Two earlier overstatements were corrected before publication**: a claim that nearly all conversion gain sits in the top 10% of ranked users (it is 75%), and a dataset caveat missing from the first README (pooled tests, non-uniform subsampling, CC BY-NC-SA licence).

## Deployment verification (2026-10-28)

- Built the image with hash-verified installs and a passing `pip-audit --strict` stage. Trivy image scan: 0 HIGH or CRITICAL findings. hadolint and Trivy configuration scan clean. All base images are pinned by digest.
- Ran the stack with the real llama.cpp container and the Qwen3 4B model through nginx: non-root users, empty capability set, read-only root filesystems, `no-new-privileges`, an `internal` network with no DNS or internet egress from the app and the model server, and the published port bound to `127.0.0.1`.
- Drove the running UI in a real browser (Playwright) across every tab, including a real question to the model that selected `targeting_policy` with the right arguments.

## Language-model evaluation

Real Qwen3-4B-Instruct-2507 Q4_K_M on llama.cpp, CPU only, 61 questions (`evals/tool_selection.json`).

- Tool choice: 93.4% (57 of 61) in each of three runs (`evals/results_3runs.json`) and in the final run (`evals/results.json`). Argument accuracy was 96.7% and every question was answered.
- Grounding with the final checker: 72.1% of answers have every number traceable to a tool result. Earlier runs with a stricter checker scored 64% to 71%. The checker was then corrected after review to read numbers inside segment labels and to accept scientific notation rounded to one digit. The evaluation was rerun with stored traces.
- Failures found: four questions answered without calling a tool, 10-fold unit errors ("192 basis points" for 19.2, "+51.7%" for +5.17 pp), an addition error (99,999 for 100,000), and two answers quoting the +59% example from the system prompt. That example was then replaced by a neutral value. The CI run on the new prompt (synthetic data, one run on GitHub's runner) scored 91.8% right tool, 95.1% right arguments and 73.8% fully grounded, within two points of the local run on each metric.
- Limits: three answers (134 to 576 seconds) were slowed by a concurrent test run, so latency is not clean for them.

## Limits of this validation

- No live intervention, no out-of-time holdout and no second dataset were used. The Qini and policy results describe held-out users from the same pooled file.
- The complier analysis assumes the ad affects users only through exposure. This cannot be tested from the data.
- The targeting profit is a scenario with assumed value and cost, not a measured return.
- The best targeting profit is optimistic because the fraction is chosen on the same held-out users. A split-half check on the default sample (500,000 rows, seed 42, 100 random halvings of the holdout) picked the fraction on one half and scored it on the other. The profit fell by about 20% (22,884 chosen against 18,389 scored, at the default value and cost). The halves are half the size of the holdout, so this likely overstates the optimism of the reported figure. The chosen fraction ranged from 0% to 16% across halvings, which shows the optimum is not sharply identified. The reported profit interval holds the fraction fixed and is therefore too narrow. The in-app scenario note still says slightly optimistic, because changing a tool result means rerunning the language model evaluation.
- The language-model evaluation has 61 questions and 3 runs. It indicates behaviour and is not a benchmark.

## Grounding repair, built and not yet evaluated (2026-11-09)

- The grounding check used to flag numbers and leave the answer as written. `LLAMA_GROUNDING_REPAIR=1` now gives a flagged answer one rewrite that names the numbers to remove. If the rewrite is still flagged, the answer is replaced by the code-written explanation from `app/fallback.py`. Three tests drive these paths with a mocked server, including that the mode is off by default.
- A test of the fallback path found a false flag in the checker. The code-written explanation says "significant at the 5% level", and the checker did not know that a confidence level of 0.95 implies 5%. It now does, and only for results that carry a confidence level. Rescoring the stored `evals/results.json` traces with the new checker leaves the grounded rate at 72.1%. `evals/results_3runs.json` stores no traces and cannot be rescored.
- No real-model numbers exist for the repair mode yet. Building llama.cpp locally was not possible in the session that wrote it. The weekly LLM workflow now runs a second, ungated pass with the mode on and uploads `evals/ci-results-repair.json`. The default stays off until those results are read, so the published evaluation still describes what ships.

## Grounding repair, first real-model result (2026-11-09)

The weekly LLM workflow ran both passes on the same runner with the real Qwen3 4B model, synthetic data, one run each (run 42688577028). Repair off: tool choice 93.4%, arguments 95.1%, answered 100%, grounded 73.8%, median 31.8 seconds. Repair on: tool choice 93.4%, arguments 95.1%, answered 100%, first drafts grounded 78.7%, final answers grounded 100%, 13 answers rewritten, 4 of them replaced by the code-written explanation, median 33.4 seconds. The first-draft rate differs from the repair-off rate because the model's sampling varies between passes. Repair removed every ungrounded number at a cost of about 1.6 seconds per answer, and did not change tool choice. The cost is that 4 of 61 answers became the plainer code-written text. One run on synthetic data is indicative, not a benchmark.

## Grounding repair switched on by default (2026-11-10)

On the result above the owner chose to switch repair on. `Settings.grounding_repair` now defaults to true and `LLAMA_GROUNDING_REPAIR=0` turns it off. The weekly workflow now gates on the default and runs an ungated pass with repair off for comparison. Making it the default exposed a bug in the first version: the agent loop granted one extra tool round whenever repair was enabled, even when nothing was repaired, so a model that never answered got one round more than its cap. The extra round is now added only when a repair happens, and a test pins it. The published local evaluation (72.1% grounded, 3 runs, real data) was run without repair and is kept as recorded.

## Qini scores corrected for imperfect randomization, design fixed before the run (2026-11-14)

Written before any corrected score was computed. The effect estimates on the full file are corrected for the fact that assignment is slightly predictable from the features. The Qini curves, the learner comparison and the targeting policy still use raw outcomes. This section fixes how the correction is applied and read.

- **Data and split.** The default sample (500,000 rows, seed 42), the same stratified 70/30 split, the T-learner at its current settings (`max_iter` 100). Outcomes: conversion and visit.
- **Corrected score.** On the held-out 30%, doubly robust scores built from cross-fitted nuisance models. Five folds. The nuisance models are a classifier per arm for the outcome and a classifier for assignment, all `HistGradientBoostingClassifier` with `max_iter` 100 and the sample seed. The propensity is clipped to 0.01 and 0.99. The score of a user is mu1 minus mu0 plus the treated residual over the propensity minus the control residual over one minus the propensity.
- **Corrected Qini curve.** Users ranked by the model's uplift score. The curve is the running sum of corrected scores times the treated share of the held-out users, so it has the same units as the raw curve. The coefficient is the area under it minus the area under the random line, as for the raw curve.
- **Interval.** 200 bootstrap resamples of the held-out users with the scores held fixed, as the raw interval. The nuisance models are not refitted inside the bootstrap, so the interval omits their uncertainty. It is said so in the README.
- **Reported.** The raw and the corrected Qini coefficient with their intervals for both outcomes, and the corrected minus the raw coefficient.
- **How it is read.** If the raw coefficient lies outside the corrected interval, or the reverse, the imbalance matters for the model evaluation and the README says so. If each lies inside the other's interval, the README says the correction does not change the evaluation at this sample size. Nothing is retuned after the run, and the learner comparison and the policy keep their raw version until this result is known.
- **Not done here.** A corrected learner comparison, a corrected policy value and any change to a tool description or to `SYSTEM_PROMPT`. Changing those would require rerunning `make eval`.
