# Novelty statement (to be verified experimentally)

## Proposed contribution

This work investigates an India-wide **FIRMS fire-detection occurrence** prediction framework based on Suomi-NPP VIIRS observations from 2018–2025 and historical meteorological conditions.

The novelty claim is deliberately narrow:

> Existing Indian forest-fire studies establish the usefulness of machine learning, meteorological predictors, and satellite-derived fire observations, but this study evaluates whether a standardized India-wide SNPP-VIIRS FIRMS fire-cell representation combined with multi-timescale antecedent meteorology can generalize across future years and geographically held-out regions of India.

## What we do NOT claim

We do not claim that:

- machine learning for forest-fire prediction is new;
- Random Forest or Logistic Regression is a novel algorithm;
- VIIRS is a new fire-detection source;
- India-wide fire prediction has never been studied;
- FIRMS data have never been used for Indian fire research.

## Conditions required for the claim

The paper should make the novelty claim only if the experiments demonstrate at least one meaningful result beyond a conventional random split, for example:

1. The model retains useful predictive skill on the completely held-out 2025 fire season.
2. Multi-timescale weather features improve PR-AUC/F1 over instantaneous weather alone.
3. Spatial group holdout reveals and quantifies geographic transferability.
4. The proposed sampling/evaluation protocol produces conclusions that differ materially from random train/test evaluation.

## Why this is research rather than an application demo

The project is designed around falsifiable experiments: future-year prediction, spatial transfer, feature ablation, calibration, class-imbalance-aware metrics, and explicit limitations of FIRMS-derived labels.

## Current status

This document states a **candidate novelty hypothesis**, not a claim of priority. A final manuscript must update the related-work matrix with the results of a systematic search immediately before submission.
