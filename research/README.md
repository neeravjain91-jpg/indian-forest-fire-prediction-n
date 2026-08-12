# Research Experimental Protocol

This directory contains the reproducible research track. It is deliberately separated from the portfolio application.

## Core hypothesis

Antecedent meteorological conditions at multiple timescales (24 h, 72 h, 168 h), combined with instantaneous weather, provide useful information for predicting VIIRS fire-detection occurrence across India and may improve generalization to future years.

## Required experiments

1. Temporal baseline: 2018–2023 train, 2024 validation, 2025 final test.
2. Random split baseline, reported only as a reference and not as the primary claim.
3. Geographic holdout.
4. Weather-feature ablation: instantaneous only vs +24 h vs +72 h vs +168 h.
5. Negative-sampling sensitivity.
6. Calibration analysis.

## No leakage rule

FRP, brightness temperature, confidence, or other variables recorded by FIRMS at the detection event must not be used as predictors of whether that same event occurs. They can be retained for descriptive analysis.

## Interpretation

The model predicts the probability of a VIIRS-observed fire-detection event under historical weather conditions. It does not claim to detect fires operationally or prove physical fire absence when FIRMS has no detection.
