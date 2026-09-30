# Motor Insurance Pricing Model

[![CI](https://github.com/liamhavers/motor-pricing-model/actions/workflows/ci.yml/badge.svg)](https://github.com/liamhavers/motor-pricing-model/actions/workflows/ci.yml)

Predicting how often a motor policyholder will claim and how much each claim will cost, turning that into a price, and comparing a traditional GLM with a gradient-boosted model (LightGBM) on the commercial consequences of choosing one over the other.

Built on the French motor third-party liability datasets (freMTPL2freq and freMTPL2sev, about 678,000 policies).

> Work in progress. Sections below are filled in as each stage is completed.

## Problem

## Approach

1. **Frequency**: Poisson GLM with a log(exposure) offset, against a LightGBM Poisson model.
2. **Severity**: Gamma GLM on capped average claim amount, weighted by claim count.
3. **Pure premium**: frequency x severity, compared with a single Tweedie model.
4. **Evaluation**: deviance, Lorenz curve and Gini, exposure-weighted calibration, overall balance and double-lift charts.
5. **Commercial analysis**: where the GLM under- or overcharges relative to the GBM, and what that is worth on the test portfolio.

## Results

## Findings and recommendation

## Regulatory context

## Limitations

## Project structure

```
├── data/                  # gitignored, raw data cached here on first download
├── notebooks/             # 01_eda to 06_commercial_analysis
├── src/pricing/           # data, features, glm, gbm, evaluation, plots
├── reports/figures/
└── tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

The data downloads from OpenML on first run and is cached in `data/raw/`.
