# Sequential testing methods used in the replay

This page explains each stopping rule in `core/sequential.py`, where it comes from, and where it stops being valid. The replay itself is in `core/sequential_study.py` and the numbers are in `outputs/sequential.json`.

All rules test the difference in means between two arms at a two-sided level of 0.05. A look is a moment at which the data seen so far is analysed.

## fixed

One z test at the planned end. It is only valid if nobody acts on earlier looks. It is the reference for every other rule.

## peek

The same z test at every look, stopping at the first p below 0.05. Armitage, McPherson and Rowe (1969) showed that this inflates the false positive rate. The replay measures by how much.

## bonferroni

The z test at level 0.05 divided by the number of looks. It controls the error rate at any number of looks and is conservative, because the looks are strongly correlated. Limit: it loses power as the number of looks grows.

## pocock

A constant critical value on the z statistic at every look, chosen so that the chance of any crossing under the null is 0.05. Reference: Pocock (1977) and Jennison and Turnbull, *Group Sequential Methods with Applications to Clinical Trials* (2000), chapter 2. The code solves for the constant by recursive numerical integration over the joint normal distribution of the statistics, as in chapter 19 of the same book. For five equally spaced looks it gives 2.413. Limit: it spends much of the error budget early and pays for it at the end.

## obf

An O'Brien-Fleming type boundary through the Lan and DeMets (1983) spending function 2 minus 2 times the normal distribution function of z at half alpha divided by the square root of the information fraction. Critical values are found one look at a time so that the chance of first crossing at that look equals the alpha spent there. Reference: Jennison and Turnbull (2000), chapters 2 and 7.

For five equally spaced looks the final critical value is 2.063. The classical O'Brien-Fleming constant in the book's tables is 2.040. The two differ because spending is an approximation to the classical boundary and not the same rule. The test suite checks both. The spending value is the one pinned for the rule, and the classical constant is checked by simulation to keep a 5% level.

Boundaries above 8 are treated as never crossed. Limit: it assumes the looks are planned in terms of information, here equal sample per arm.

## msprt

The mixture sequential probability ratio test with a normal mixing distribution on the effect. With variance V of the estimated difference and mixing variance tau squared, the likelihood ratio is the square root of V over V plus tau squared, times the exponential of tau squared times the squared difference over two V times V plus tau squared. The always-valid p value is the running minimum of one over that ratio. References: Johari, Koomen, Pekelis and Walsh, "Peeking at A/B Tests" (KDD 2017), and "Always Valid Inference" (Operations Research 2022).

The mixing variance is the squared absolute effect that the fixed test at the planned size detects with 80% power, computed by `core.power`. It is fixed before any run and not tuned. The variance of the difference is estimated at each look, so the rule is approximate for small samples. A look with zero estimated variance counts as no evidence. Limit: it is valid at any number of looks and pays for that with low power when the mixing variance does not match the true effect. With a single look it is more conservative than `fixed`.

## confidence sequence

The interval implied by the same mixture, in `confidence_sequence`. It covers the true effect at all looks at once with probability at least 0.95. Reference: Howard, Ramdas, McAuliffe and Sekhon (2021), normal mixture boundary. The replay does not report it. It is tested only.

## CUPED

The outcome is replaced by the outcome minus theta times the covariate minus its centre. Reference: Deng, Xu, Kohavi and Walker (WSDM 2013). With a model prediction as the covariate this is sometimes called CUPAC. The covariate here is a logistic regression prediction from the twelve features, fitted on a holdout fifth of the control pool that no experiment draws from. Theta and the centre are fitted once on that holdout and held fixed. An adjusted look therefore depends only on the rows seen so far.

The file has no pre-period outcome, so the saving here is a lower bound for a product with user history.
