# Motor Insurance Pricing Model

[![CI](https://github.com/liamhavers/motor-pricing-model/actions/workflows/ci.yml/badge.svg)](https://github.com/liamhavers/motor-pricing-model/actions/workflows/ci.yml)

Predicting how often a motor policyholder will claim and how much those claims will cost, turning that into a price, and working out what it would mean commercially to price with a gradient-boosted model (LightGBM) instead of the traditional GLM.

Built on the French motor third-party liability data (freMTPL2, about 678,000 policies) as a portfolio project for insurance pricing roles.

## Summary

An insurer needs to charge each customer enough to cover the claims they are likely to make, and not so much that they take their business elsewhere. The part of the price that covers expected claims is the **pure premium**. This project builds it in the standard way, as **how often a policy claims** (frequency) multiplied by **how much each claim costs** (severity), and compares two kinds of model for each part:

- a **GLM** (generalised linear model), the industry standard, which produces a simple rating table of multipliers such as "drivers aged 18 to 19: x1.25";
- a **GBM** (gradient-boosted trees), which is more flexible and usually more accurate, but harder to explain.

Every choice is made on training data and each model is scored once on a held-out test set of 135,000 policies. The comparison goes beyond accuracy scores to the question a pricing team would ask: if we priced with the GBM instead of the GLM, which customers would we be undercharging or overcharging today, and what would that cost?

## Key findings

<!--
To be written by Liam. Numbers to draw on (all on the test set unless stated):

- Frequency: GBM Gini 0.326 vs GLM 0.293; Poisson deviance 0.4543 vs 0.4625. Bootstrap 95% interval
  for the Gini difference 0.025 to 0.041.
- Severity: neither model reliably beats a single average cost per claim (bootstrap intervals include
  zero). Nearly all differentiation between customers comes from frequency.
- Pure premium: GBM frequency x severity Gini 0.354 vs GLM 0.311 (interval for the difference 0.025
  to 0.063). Tweedie vs frequency x severity: no real difference for the GLM; frequency x severity
  better for the GBM.
- Money (both models scaled to the same total premium, capped claims): about EUR 1.25m of EUR 9.7m
  (13%) of premium moves between customers. Customers the GLM undercharges (GBM > 1.1x GLM, about
  51,000 policies): EUR 2.9m premium against EUR 4.2m claims, a shortfall of about EUR 1.3m.
  Customers it overcharges (GBM < 0.9x GLM, about 47,000): EUR 4.3m premium against EUR 3.1m claims.
- Who: undercharged are more often off the BonusMalus claim-free path (claimed recently) and drivers
  under 50 with maximum discount and newer cars (claims 1.35x GLM premium). Overcharged include
  BonusMalus 51 to 54 on the claim-free path, BonusMalus 100+ with newer cars (claims 0.75x GLM
  premium), and more drivers aged 70+.
- Double lift: where the GBM prices about 1.7x the GLM, observed frequency was 1.9x average; GBM
  predicted 1.8x, GLM 1.1x.
- Feeding back: one "off the discount path" flag in the GLM (relativity 2.37) closes about half the
  GLM-to-GBM gap (54% frequency deviance, 49% pure premium Gini).
- Other: large losses above EUR 50k are 0.3% of claims but 24% of cost; claim counts are
  overdispersed (1.7), so GLM confidence intervals are about 30% too narrow.
-->

*To be written.*

## Recommendation

<!-- To be written by Liam. -->

*To be written.*

## The problem

A motor insurer pricing third-party liability cover needs, for every customer, an estimate of the claims they will cost over the year. That estimate has two parts that behave very differently:

- **Frequency**: most policies (96% here) have no claim in a year. The question is how the chance of a claim varies with the driver, the car and where it is kept.
- **Severity**: when a claim happens, its cost varies enormously, from a few hundred euros to several million.

The price then has to account for the fact that most policies were not on risk for a full year, that a handful of very large claims dominate the total, and that the model has to be explained to underwriters, managers and regulators.

## The data

Two public French datasets from OpenML, joined on policy ID:

- **freMTPL2freq**: one row per policy, with exposure (the fraction of a year on risk), claim count and nine rating factors: driver age, vehicle age, vehicle power, vehicle brand, fuel type, BonusMalus (the French no-claims discount level, where 50 is the best and 100 is a new driver), population density, area and region.
- **freMTPL2sev**: one row per claim, with its cost.

Cleaning decisions, all counted and explained in the [EDA notebook](notebooks/01_eda.ipynb):

- About 9,100 policies record claims with no cost in the severity table. Their claim count is set to zero so that the frequency and cost models describe the same claims. This lowers observed frequency and is listed as a limitation.
- Exposure is capped at one year (1,224 policies above) and claim counts at four (9 policies above, several with 9 to 16 claims in a few weeks).
- Individual claims are capped at EUR 50,000 before modelling (see below).

## Approach

The data is split once, 80% for training and 20% for testing, by policy. Every modelling choice (banding, factor selection, model settings) is made on the training data using cross-validation. The test set is scored once, after all models are fixed, and every later analysis reuses those frozen predictions.

**Frequency** ([notebook](notebooks/02_frequency.ipynb)). A Poisson GLM on banded rating factors, with the log of exposure as an offset so that the model predicts claims per year and scales it by time on risk. The GBM uses the raw factors and includes exposure in the same way. The GLM's relativities show why models are fitted on all factors at once: on its own, driver age suggests 18 to 19 year olds claim 3.4 times as often as drivers in their 40s; after allowing for BonusMalus, the difference is 1.25 times.

![Driver age: GLM relativities against one-way](reports/figures/freq_glm_relativities_drivage.png)

**Severity** ([notebook](notebooks/03_severity.ipynb)). A Gamma GLM on the average capped cost per claim, weighted by the number of claims. Claims are capped at EUR 50,000: that affects 0.3% of claims but removes 24% of cost, and cuts the uncertainty in the average claim cost from about 10% to under 2%. The cost above the cap is added back to every price as a **large-loss loading** (about 31% of capped cost on this training data, though the estimate depends heavily on a single EUR 4.1m claim). With all rating factors the severity GLM did worse on unseen data than a single average, so factors were chosen by cross-validation: BonusMalus, driver age and density.

**Pure premium** ([notebook](notebooks/04_pure_premium.ipynb)). Frequency multiplied by severity, compared with a single **Tweedie** model of cost per year, which handles the mix of zero-cost policies and skewed claim amounts in one distribution. The Tweedie power, 1.8, comes from the shape of our own claim costs and was checked on validation data. Boosting with a Tweedie loss systematically under-predicts the overall level, so the Tweedie GBM is rebased to the training total.

**Evaluation** ([notebook](notebooks/05_evaluation.ipynb)) and **commercial analysis** ([notebook](notebooks/06_commercial_analysis.ipynb)) are described under Results.

## Results

All figures are on the test set. Deviance measures fit in the way that suits each kind of data (lower is better); "vs no factors" is the improvement over charging everyone the training average; the Gini measures how well a model ranks customers from cheapest to most expensive (0 is random); balance is total predicted over total actual.

| Stage | Model | Deviance | vs no factors | Gini | Balance |
|---|---|---|---|---|---|
| Frequency (Poisson) | GLM | 0.4625 | 4.8% | 0.293 | 0.97 |
| | GBM | 0.4543 | 6.5% | 0.326 | 0.97 |
| Severity (Gamma) | GLM | 1.2449 | -0.6% | 0.014 | 0.97 |
| | GBM | 1.2346 | 0.3% | 0.037 | 0.95 |
| Pure premium (Tweedie 1.8) | GLM frequency x severity | 29.92 | 2.7% | 0.311 | 0.94 |
| | GBM frequency x severity | 29.61 | 3.7% | 0.354 | 0.92 |
| | GLM Tweedie | 29.93 | 2.7% | 0.317 | 0.96 |
| | GBM Tweedie | 29.78 | 3.2% | 0.330 | 0.95 |

Balance is below 1 for every model, including charging everyone the average, because the test set happens to be about 6% more expensive than the training set. RMSE is not used: it rated the GLM, the GBM and charging everyone the average within 0.1% of each other, because more than half of the squared error came from ten policies.

**Is the difference larger than chance?** The test policies were resampled 200 times to put a 95% interval around each difference between models. For frequency and pure premium, the interval for the GBM's advantage over the GLM excludes zero. For severity it includes zero, so the two models cannot be told apart.

**Where do the models disagree?** A double-lift chart sorts customers by how much the GBM's price differs from the GLM's. Where the GBM prices highest relative to the GLM, actual claims followed the GBM.

![Double lift: GBM against GLM frequency](reports/figures/eval_double_lift_frequency.png)

**What is it worth?** With both models scaled to charge the same total, this is who would pay more or less under the GBM, and what their claims actually cost:

![Premium under each model against observed cost](reports/figures/com_mispricing_by_ratio.png)

**What did the GBM find?** SHAP values break each GBM prediction into contributions from each factor. The largest difference from the GLM is in BonusMalus. The French scale falls 5% for each claim-free year from 100 (95, 90, 85, 80, 76, ... 54, 51, 50), so a value below 100 that is off that path can only be reached after a claim. The GBM learned from the data alone that these customers claim far more often; the GLM's bands of ten mix them with claim-free customers.

![BonusMalus effect in the GBM, on and off the claim-free path](reports/figures/com_shap_bonusmalus_path.png)

Adding a single "off the discount path" flag to the GLM (relativity 2.4) closes about half of the gap between the GLM and the GBM, while keeping the GLM's rating-table form.

## Regulatory context (UK)

This project uses French data, but the target roles are in UK pricing, where two sets of rules shape how a model like this could be used.

**General Insurance Pricing Practices (FCA, in force since 1 January 2022).** A firm must not charge a renewing customer more than it would charge an equivalent new customer through the same channel. This ended "price walking", where loyal customers paid progressively more than new ones. Neither model here uses tenure or price history, so the choice between them does not affect this rule directly, but any pricing built on top of them (discounts, demand-based adjustments) would have to respect it.

**Fair outcomes and protected characteristics.** Under the Equality Act 2010, sex cannot be used as a rating factor, and age can be used only where supported by relevant and reliable data. Beyond the factors themselves, firms are expected to show that their pricing does not produce unfair outcomes for groups with protected characteristics, and the Consumer Duty (from July 2023) adds a requirement that products offer fair value. Rating factors such as region or vehicle can act as proxies for characteristics that cannot be used directly, and a GBM can build such proxies out of interactions that no one chose explicitly. That makes testing outcomes by group, not just reviewing the factors, a necessary step before using either model, and a harder one for the GBM.

## Limitations

- **French data from one period.** The rating factors and the BonusMalus mechanism are French. There is no time dimension, so the models are not tested on a later year, which is how a pricing model would really be validated.
- **No price, conversion or retention data.** The analysis measures who would be over- or undercharged, but not how customers would react to a change in price. The cost of overcharging (customers leaving for competitors) and any price optimisation are out of reach.
- **Missing claim costs.** About 9,100 policies with recorded claims have no cost data. Setting their claims to zero keeps the models consistent but lowers the observed claim frequency.
- **Large losses.** Claims are capped at EUR 50,000 and the excess is spread as a flat loading estimated from one sample, where a single claim moves the loading from about 20% to 31%. A real loading would use many years of large-loss experience or a separate large-loss model.
- **Weak severity signal.** Once large losses are capped, the rating factors explain almost nothing about claim cost, so these conclusions rest almost entirely on frequency.
- **Model uncertainty.** Claim counts vary more than a Poisson model assumes (dispersion about 1.7), so the GLM's confidence intervals are about 30% too narrow. The results come from a single train/test split, and the GBMs were tuned with modest effort.
- **Pure premium only.** Expenses, commission, profit margin and reinsurance costs are not included.

## Next steps

- Validate on a later period of data before drawing any pricing conclusion.
- Test outcomes by group for fairness, including possible proxies.
- With price and conversion data, build a demand model to estimate the commercial effect of the price changes, including the cost of losing overcharged customers.
- Model large losses separately (for example a Pareto distribution above the cap) and estimate the loading over several years.
- Continue feeding GBM findings into the GLM, starting with the driver age by BonusMalus interaction.

## Reproducing the results

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Then run the notebooks in order, `01_eda` to `06_commercial_analysis`. The data downloads from OpenML on the first run and is cached in `data/raw/`. Each notebook saves its test-set predictions and fitted models to `data/processed/` for the next one. The later notebooks take a few minutes each, and the GLM fits need roughly 6 GB of memory. All random steps use a fixed seed.

## Project structure

```
├── data/                  # gitignored; raw data cached here on first download
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_frequency.ipynb
│   ├── 03_severity.ipynb
│   ├── 04_pure_premium.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_commercial_analysis.ipynb
├── src/pricing/
│   ├── data.py            # loading, cleaning, joining, capping, train/test split
│   ├── features.py        # GLM banding, GBM features, BonusMalus discount path
│   ├── glm.py             # Poisson, Gamma and Tweedie GLMs, relativities
│   ├── gbm.py             # LightGBM frequency, severity and Tweedie models
│   ├── evaluation.py      # deviance, Gini, calibration, double lift, bootstrap
│   └── plots.py
├── reports/figures/
├── tests/
└── LEARNING.md            # checkpoint questions and answers from building the project
```

**Stack:** Python 3.11+, polars, statsmodels, scikit-learn, LightGBM, shap, matplotlib, pytest.
