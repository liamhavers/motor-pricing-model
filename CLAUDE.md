# CLAUDE.md

## Project: Motor Insurance Pricing Model (Frequency-Severity, GLM vs GBM)

Portfolio project for Data Scientist applications in insurance pricing and underwriting (first target: Prima, Pricing & Underwriting). The goal is to show I can do the core job of a pricing data scientist: predict how often a policyholder will claim and how much it will cost, turn that into a price, and explain the commercial consequences of the modelling choices.

The project owner (Liam) is new to insurance pricing. Understanding matters as much as working code, because every part of this project may be questioned in interview. See "Working with me" below.

## Goals (the six core deliverables)

### 1. Frequency model
- Fit a Poisson GLM predicting claim counts (`ClaimNb`) with `log(Exposure)` as an offset.
- Fit a GBM (LightGBM preferred, XGBoost acceptable) with a Poisson objective, using exposure correctly (log-exposure as `init_score`/base margin, or claim rate as target with exposure as sample weight; explain which and why).
- Compare the two on held-out data.

### 2. Severity model
- Fit a Gamma GLM on average claim amount for policies with at least one claim, weighted by number of claims.
- Cap large losses before modelling (choose and justify a threshold, e.g. a high percentile) and note how the capped excess would be handled in practice (spread as a loading across all policies).
- Optionally fit a GBM with a Gamma objective for comparison.

### 3. Pure premium
- Pure premium = predicted frequency x predicted severity, per policy.
- Alternative: fit a Tweedie model (GLM and/or GBM) directly on total claim cost per unit exposure. Explore at least one variance power and justify the choice.
- Compare the two approaches.

### 4. Evaluation, the way pricing teams do it
- Poisson deviance (frequency), Gamma deviance (severity), Tweedie deviance (pure premium). Do not use accuracy or RMSE as headline metrics.
- Lorenz curve and Gini coefficient for risk ranking.
- Calibration: predicted vs observed by decile of predicted risk, exposure-weighted.
- Check that total predicted claims/cost on the test set is close to total observed (overall balance).
- Double-lift chart: sort policies by the ratio of GBM to GLM prediction, bucket, and plot observed vs each model's prediction per bucket. This shows where the models disagree and which one is right.

### 5. Commercial analysis
Answer this question with numbers: "If we priced with the GBM instead of the GLM, which customers would we be undercharging or overcharging today, and roughly how much would that cost us?"
- Identify the segments (rating factor combinations) where the GLM misprices most.
- Estimate the size of the mispricing in money on the test portfolio.
- Interpretability: SHAP values and/or partial dependence plots for the GBM, compared against GLM relativities (exponentiated coefficients).
- Discuss the trade-off: GBM accuracy vs GLM transparency and regulatory acceptability. Optional stretch: feed GBM-discovered interactions back into the GLM and measure how much of the gap closes.

### 6. README for a business reader
- Plain-English summary of the problem, approach, key findings and recommendation, written so a pricing manager could follow it.
- Short note on UK regulatory context: the FCA's General Insurance Pricing Practices rules (in force since January 2022) ban charging renewing customers more than equivalent new customers, and pricing models must not produce unfair outcomes for customers with protected characteristics.
- Limitations section (French data, no price or conversion data, no demand modelling, capped large losses).
- Liam writes the findings and recommendation sections himself. Claude can draft structure and technical sections, but must not write the conclusions.

## Data

French motor third-party liability datasets, loaded via scikit-learn from OpenML:

```python
from sklearn.datasets import fetch_openml
freq = fetch_openml(data_id=41214, as_frame=True).frame   # freMTPL2freq, ~678k policies
sev  = fetch_openml(data_id=41215, as_frame=True).frame   # freMTPL2sev, individual claim amounts
```

- Join on `IDpol`. Severity has one row per claim, so aggregate to policy level (sum and count) before joining.
- Known data issues to handle and document: cap `ClaimNb` (a small number of policies have implausibly high counts), cap `Exposure` at 1 year, and check for policies with claim counts but no matching severity rows.
- Cache raw data locally in `data/raw/` after first download. Do not commit data files.

## Tech stack
- Python 3.11+
- polars for data preparation (convert to pandas only where a library requires it)
- statsmodels for GLMs (offsets, coefficient summaries, relativities), scikit-learn for splitting, metrics and preprocessing
- LightGBM for GBMs
- shap for interpretability
- matplotlib for plots
- pytest for tests on data preparation and metric functions

## Repository structure

```
motor-pricing-model/
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── data/                  # gitignored
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_frequency.ipynb
│   ├── 03_severity.ipynb
│   ├── 04_pure_premium.ipynb
│   └── 05_commercial_analysis.ipynb
├── src/pricing/
│   ├── data.py            # loading, cleaning, joining, capping
│   ├── features.py        # banding and encoding of rating factors
│   ├── glm.py             # GLM fitting and relativities
│   ├── gbm.py             # GBM fitting with exposure handling
│   ├── evaluation.py      # deviance, Gini/Lorenz, calibration, double-lift
│   └── plots.py
├── reports/figures/
└── tests/
```

Logic lives in `src/pricing/`; notebooks call it and show results. Keep notebooks readable, with a short markdown explanation above each major step.

## Modelling conventions
- Fixed random seed throughout. Single train/test split by policy (e.g. 80/20), plus cross-validation on train for GBM tuning. The test set is used once, at the end.
- All metrics and plots are exposure-weighted where it matters. State this explicitly each time.
- GLM features: band continuous variables (driver age, vehicle age, density) into sensible groups, as a pricing team would. Record the banding choices and why.
- GBM features: use raw continuous variables; LightGBM native categorical handling is fine.
- Keep the GBM reasonably simple and tuned with modest effort. The point is the comparison, not squeezing out accuracy.
- Report the same metrics for every model in one comparison table.

## Working with me (important)

I am learning this domain as we build. At the end of each goal, before moving on:

1. Explain in plain language what was done and why, in a few short paragraphs, without jargon that isn't defined.
2. Ask me the checkpoint questions below for that stage and wait for my answers. Correct me if I'm wrong or vague. Do not move on until I've answered.
3. If I ask you to "just do it", do it, but still leave the checkpoint questions in a `LEARNING.md` file so I can come back to them.

Checkpoint questions:
- **After 1:** Why use a log(exposure) offset rather than dividing claims by exposure and modelling the rate? Why Poisson for counts? How did we handle exposure in the GBM, and why is that equivalent?
- **After 2:** Why Gamma for severity? Why weight by claim count? Why cap large losses, and what happens to the capped amount in a real pricing process?
- **After 3:** What is a Tweedie distribution and why does it suit total claim cost? When would you prefer frequency x severity over a single Tweedie model?
- **After 4:** What does deviance measure, and why not RMSE? What does a Gini of 0.3 mean in practice? What does a double-lift chart show that a single metric doesn't?
- **After 5:** The GBM beat the GLM. Would you deploy it? What would an underwriter, a regulator and a pricing manager each want to know first?
- **After 6:** Explain the whole project in two minutes to a non-technical interviewer.

## Writing style
- README and notebook prose: plain, direct British English. No em-dashes. Avoid AI-style tells ("delve", "leverage", "robust", "it's worth noting", "demonstrating", "reflecting", neatly symmetrical lists of three).
- Do not overstate results. Report honestly where the GLM does as well as the GBM, or where findings are uncertain.
- Commit messages: short, imperative, lowercase.

## Out of scope
- Price optimisation, demand or conversion modelling (no price data available). Mention as a limitation and natural next step.
- Deployment (API, Docker). My fraud detection project already covers this.
- Heavy hyperparameter searches or deep learning models.
