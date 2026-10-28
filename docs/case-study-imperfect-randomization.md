# A randomized experiment that passed the sample ratio check and was still biased

*What 14 million users taught me about checking an A/B test beyond the first check.*

Every A/B testing checklist starts with the sample ratio check (SRM): did the split between treatment and control match the design? I ran it on the Criteo uplift dataset, 13,979,592 users from randomized incrementality tests. The designed treated share is 85%. The observed share was 85.00% (p = 0.999). Textbook clean.

I almost moved on. Instead I asked a second question: **if assignment is truly random, the features should tell you nothing about who got treated.** So I trained a classifier to predict treatment from the 12 user features (on a 500,000-user random sample, scored on held-out users). Under perfect randomization its AUC is 0.5. Mine was 0.508, with a 95% bootstrap interval of 0.504 to 0.512. That is a small number, and it excludes 0.5.

## Why a small imbalance matters

The standard rule of thumb says a standardized mean difference below 0.1 is fine. The largest here was 0.049, so by the rule of thumb the experiment is balanced. But with this many users, even tiny shifts are measurable, and what matters is whether the shifted features also drive the outcome. They do.

On all 14 million users, the raw difference in visit rate between treated and control is **+1.03 percentage points**. After adjusting for the features it is **+0.77**, a quarter smaller. For conversions the raw effect is +0.115 pp and the adjusted one +0.100 pp.

One adjusted number could be an artifact of one model, so I used a second, different method: inverse-propensity weighting with cross-fitted gradient-boosting propensities. It gives **+0.74 pp** (95% CI 0.68 to 0.79) on visits and +0.101 pp on conversions. Two independent corrections land within 0.04 pp of each other, and both sit well below the raw estimate.

My hypothesis, which I cannot verify from the public file, comes from the dataset documentation: it pools data from several incrementality tests. If treatment probability differs by test, and tests differ in who they reach, the raw comparison mixes experiments. The sample ratio check cannot see this because the overall split is exactly as designed.

## What I would check next time

1. **Sample ratio.** Still the first check.
2. **Covariate balance, in two ways.** Standardized differences per feature, and a classifier predicting treatment from the features, with a confidence interval on its AUC. The classifier catches combinations of small shifts that no single feature shows.
3. **Triangulate.** Compare the raw effect with a regression-adjusted one and a propensity-weighted one. If they agree, report that. If they disagree, investigate assignment before reporting a number.

## A second lesson: who actually saw the ad

Only 3.6% of treated users were ever exposed to the ad. The effect of being assigned to treatment, +1.03 pp on visits, averages over the 96% who could not respond. For users who were actually exposed, an instrumental-variable estimate gives **+28.7 pp**. Comparing exposed treated users with all control users instead would give a much larger number, because the people who get exposed differ from the rest. On my 500k sample that naive contrast was +38.6 pp. Intent-to-treat and effect-on-exposed answer different questions, and mixing them up is easy.

## Limits

- This is a benchmark file, pooled and sub-sampled by Criteo. Its effect sizes are not forecasts for any real campaign.
- Adjustment and weighting correct only for observed features. Imbalance on unobserved variables would remain.
- The exposed-user estimate assumes the ad works only through being seen, which the data cannot test.
- The cause of the imbalance is my hypothesis, not a finding.

Everything above is reproducible. The code, the tests, the validation record that includes these numbers and the defects I found along the way, and an interactive explorer are in the repository: https://github.com/dbechrakis/uplift-analyzer (explorer: https://dbechrakis.github.io/uplift-analyzer/).

*Data: Criteo Uplift Prediction dataset, CC BY-NC-SA 4.0. Diemert, Betlei, Renaudin, Amini. A Large Scale Benchmark for Uplift Modeling. AdKDD 2018.*
