# Research paper outline

## 1. Introduction

- Forest-fire impacts in India.
- Need for occurrence prediction rather than retrospective mapping alone.
- Limits of existing regional susceptibility studies.
- Research gap: nationwide, multi-year SNPP-VIIRS FIRMS occurrence representation and generalization.
- Research questions.

## 2. Related Work

- Satellite active-fire products.
- Indian forest-fire susceptibility and prediction studies.
- Meteorological predictors.
- Machine-learning approaches.
- Generalization and rare-event evaluation.

## 3. Data

### 3.1 SNPP-VIIRS FIRMS

Document acquisition date, product/version, spatial coverage, filtering and fire-cell aggregation.

### 3.2 Meteorology

Document Open-Meteo archive, ECMWF IFS configuration, variables, temporal resolution and matching procedure.

### 3.3 Target construction

Define 0.1° grid/time cells and controlled non-fire sampling. Explicitly discuss detection bias and the difference between non-detection and true absence.

## 4. Methodology

- Data preprocessing.
- Multi-timescale weather features.
- Temporal split.
- Spatial group split.
- Baseline models.
- Evaluation metrics.

## 5. Experiments

- Future-year test.
- Spatial transfer test.
- Weather-history ablation.
- Model comparison.
- Sampling sensitivity.

## 6. Results

Report tables for:

1. 2024 validation.
2. 2025 unseen-year test.
3. geographic holdout.
4. feature ablation.
5. repeated negative-sampling sensitivity.

## 7. Discussion

- Which weather timescale matters?
- How much does random splitting overestimate performance?
- Where does geographic transfer fail?
- Practical interpretation.
- Limitations of FIRMS detection labels and weather reanalysis/archive data.

## 8. Conclusion

State only findings supported by the experiments. Do not claim novelty merely from the algorithm or dataset source.
