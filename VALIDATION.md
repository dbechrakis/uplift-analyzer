# Validation record

Everything below was executed, not assumed. Numbers come from `outputs/validation.json` (reproduce with `make validate`), from the test suite, or from the commands named in each entry. Where verification found a defect, the defect and its fix are recorded.

## Data integrity and fidelity — 2026-10-28

- Downloaded `criteo-research-uplift-v2.1.csv.gz` from the Hugging Face mirror and verified its SHA-256 (`2716e1bf…616dc`) and row count (13,979,592) against the constants pinned in `core/data.py`.
- Recomputed the dataset card's key figures on all 13,979,592 rows: treatment ratio 0.850000 (card 0.85), visit rate 0.046992 (card 0.046992), conversion rate 0.002917 (card 0.00292, rounded).
- Confirmed one-sided non-compliance: no control user is ever exposed (control exposure rate 0.0 exactly), which is what makes the complier analysis valid.
- Sample ratio check on all rows: 85.0000% treated, p = 0.999.

## What the full data says, and how the 500k sample compares — 2026-10-28

All 13,979,592 users, streamed in chunks:

| Estimate | Conversion | Visit |
|---|---|---|
| Raw effect (95% CI) | +0.1152 pp (0.1085 to 0.1219) | +1.0342 pp (1.0056 to 1.0629) |
| Relative uplift | +59.4% | +27.1% |
| Covariate-adjusted effect (SE) | +0.1002 pp (0.0034) | +0.7734 pp (0.0131) |
| Effect on exposed users, CACE (95% CI) | +3.196 pp (3.010 to 3.383) | +28.700 pp (27.911 to 29.488) |

- The seeded 500k sample (seed 42) gives +0.115 pp (conversion) and +1.19 pp (visit) raw. The visit estimate sits about two standard deviations above the all-users value (+1.03 pp): ordinary sampling noise, not a bug. The seed was not changed.
- 100 independent random samples of 500,000 users: the 95% interval contained the all-users raw effect in 97% of samples for conversion and 92% for visit (with 100 draws the standard error of a coverage figure is about 2 points). The sample ratio check raised 0 false alarms.

## Randomization is imperfect — a finding, 2026-10-28

- Standardized mean differences between treated and control are small but not zero: largest 0.049 (`f3`), then 0.040 (`f6`) and 0.031 (`f5`) on all users. None exceeds the common 0.1 rule of thumb.
- On the 500k sample a gradient-boosting classifier predicts treatment from the 12 features with held-out AUC 0.5078 (bootstrap 95% CI 0.5041 to 0.5120). Perfect randomization would give 0.5, so the interval rules it out.
- Consequence: the raw and covariate-adjusted effects disagree on all users (visit +1.034 vs +0.773 pp, conversion +0.115 vs +0.100 pp). Both are consistent under perfect randomization, so the gap points to assignment that depends on the features. This fits Criteo's description of the file as several pooled tests, which would give users different treatment probabilities.
- A third, independent method agrees. Cross-fitted propensity weighting (gradient-boosting propensities fitted on one half of a 4,000,000-user random subset and applied to the other, weights stabilized with the Hajek estimator, no weight clipped, propensities between 0.585 and 0.972) gives +0.736 pp on visits (95% CI 0.679 to 0.792) and +0.101 pp on conversions (0.088 to 0.115). It lands within 0.04 pp of the regression-adjusted visit estimate and within 0.001 pp for conversion, and both sit below the raw +1.034 and +0.115 pp. Its standard error treats the propensity as known, which is conservative. In tests it recovers a planted effect that a raw difference overstates, and it agrees with the raw difference under genuine randomization.
- The app and demo therefore show the adjusted estimate next to the raw one, and the explanations no longer call the raw difference simply causal. A passing sample ratio check does not detect this kind of problem.

## Correctness of the statistics code — 2026-10-28

Simulation, 2,000 runs unless noted (`core/validate.py`):

- Effect confidence interval coverage 94.5% (nominal 95%); relative-uplift interval coverage 94.3%; type I error of the two-proportion test 4.6% (nominal 5%).
- Benjamini-Hochberg across 10 null segments, 300 runs: family-wise false positives 5.3%, mean 0.057 false discoveries per run.

Equivalence and known-answer tests (`tests/`):

- The z-statistic and p-value equal `statsmodels.proportions_ztest`.
- The regression-adjusted estimate, its HC1 robust standard error and p-value equal `statsmodels` OLS to 1e-9.
- `required_sample_size(0.5, absolute_lift=-0.1)` returns the textbook 388 users per arm; the JavaScript calculator on the demo page returns the same integers as the Python function on five parameter sets (`tests/test_demo_power_js.py`).
- A simulation of 400 experiments at the computed sample size must reject between 74% and 86% of the time for a nominal 80% power; the test enforces that band.
- Over 300 simulated experiments the complier-effect interval must cover the true effect between 91% and 99% of the time (the test enforces that band), and the naive exposed-versus-control contrast must overstate the true effect in that simulation.
- Qini curves reproduce a hand-computed four-user example exactly (coefficient 0.75), the targeting policy reproduces a hand-computed profit table, and its break-even cost equals value times the all-users effect.
- The covariate balance check flags a planted assignment leak (AUC above 0.6, the leaking feature identified) and passes a properly randomized sample.

## Defects found by verification, and fixed

- **Sample size 0 silently became the default** (`sample_size or default`). Found by a bounds test; now rejected.
- **Policy scale was understated by the treated share.** A Qini curve counts incremental responses per treated-sized population; the profit curve now divides by the treated share. The test asserting break-even cost equals value times the effect would catch a regression.
- **Unadjusted standard error used a binary-outcome formula** inside the regression-adjustment comparison. Replaced by a Welch standard error valid for any outcome.
- **Quantile segmentation collapsed for 7 of 12 features** because one value holds 82% to 99% of users. Dominant values now get their own segment (`is_atom`) and the rest are binned by quantiles.
- **nginx rate limiting returned HTTP 503 for Streamlit's lazily loaded JavaScript chunks** (109 rejected requests in one page load), which broke the chat input. Static assets are now exempt and the limit raised; CI repeats 300 parallel asset requests and fails on any non-200.
- **Two earlier overstatements were corrected before publication**: a claim that nearly all conversion gain sits in the top 10% of ranked users (it is 75%), and a dataset caveat missing from the first README (pooled tests, non-uniform subsampling, CC BY-NC-SA licence).

## Deployment verification — 2026-10-28

- Built the image with hash-verified installs and a passing `pip-audit --strict` stage. Trivy image scan: 0 HIGH or CRITICAL findings. hadolint and Trivy configuration scan clean. All base images are pinned by digest.
- Ran the stack with the real llama.cpp container and the Qwen3 4B model through nginx: non-root users, empty capability set, read-only root filesystems, `no-new-privileges`, an `internal` network with no DNS or internet egress from the app and the model server, and the published port bound to `127.0.0.1`.
- Drove the running UI in a real browser (Playwright) across every tab, including a real question to the model that selected `targeting_policy` with the right arguments.

## Language-model evaluation

Real Qwen3-4B-Instruct-2507 Q4_K_M on llama.cpp, CPU only, 61 questions (`evals/tool_selection.json`).

- Tool choice: 93.4% (57 of 61) in each of three runs (`evals/results_3runs.json`) and in the final run (`evals/results.json`); argument accuracy 96.7%; every question answered.
- Grounding with the final checker: 72.1% of answers have every number traceable to a tool result. Earlier runs with a stricter checker scored 64% to 71%; the checker was corrected after review to read numbers inside segment labels and to accept scientific notation rounded to one digit, then the evaluation was rerun with stored traces.
- Failures found: four questions answered without calling a tool; 10-fold unit errors ("192 basis points" for 19.2, "+51.7%" for +5.17 pp); an addition error (99,999 for 100,000); two answers quoting the +59% example from the system prompt, which was then replaced by a neutral value without a local rerun.
- Limits: three answers (134 to 576 seconds) were slowed by a concurrent test run, so latency is not clean for them.

## Limits of this validation

- No live intervention, no out-of-time holdout and no second dataset were used; the Qini and policy results describe held-out users from the same pooled file.
- The complier analysis assumes the ad affects users only through exposure; this cannot be tested from the data.
- The targeting profit is a scenario with assumed value and cost, not a measured return.
- The language-model evaluation has 61 questions and 3 runs; it indicates behaviour and is not a benchmark.
