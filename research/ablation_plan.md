# Ablation plan

The paper should quantify which part of the proposed framework actually contributes predictive value.

## Experiment A — Weather history

Compare:

- instantaneous weather only;
- + 24 h summaries;
- + 24/72/168 h rainfall;
- full multiscale history.

Primary comparison: PR-AUC on the untouched 2025 test year.

## Experiment B — Split strategy

Compare random row-level split against the primary future-year split. The goal is to quantify how much random splitting overestimates generalization.

## Experiment C — Geographic transfer

Hold out 20% of 2° geographic blocks. Report performance on held-out blocks and compare with the temporal test.

## Experiment D — Model robustness

Run Logistic Regression, Random Forest and Gradient Boosting. A contribution should not depend on one arbitrary classifier.

## Experiment E — Sampling sensitivity

Repeat the negative-sampling process with at least three seeds. Report mean and standard deviation of the primary metrics.

## Interpretation rule

A feature set is considered useful only when the improvement is consistent across repeated runs and is evaluated on data not used for tuning.
