# India-Wide Forest Fire Occurrence Prediction (2018–2025)

Research-oriented framework for predicting **forest-fire occurrence cells in India** from Suomi-NPP VIIRS FIRMS active-fire observations and multi-timescale meteorological conditions.

## Research question

> Can an India-wide model trained on historical SNPP-VIIRS FIRMS fire detections and meteorological conditions generalize to future years and geographically distinct parts of India?

## Intended contribution

This project does **not** claim novelty from using machine learning alone. The research contribution is the combination of:

1. India-wide SNPP-VIIRS FIRMS observations from 2018–2025.
2. A reproducible spatial-temporal fire-cell target at 0.1° resolution.
3. Meteorological predictors at the fire time plus antecedent 24 h, 72 h and 168 h windows.
4. Controlled fire/non-fire sampling rather than treating every satellite non-detection as ground truth.
5. Temporal holdout testing on future years.
6. Spatial generalization testing using held-out geographic groups.
7. Ablation experiments measuring the value of antecedent weather windows.
8. Class-imbalance-aware metrics: PR-AUC, ROC-AUC, precision, recall, F1 and Brier score.

## Dataset

Place the raw FIRMS file at:

`data/raw/FIRMS_VIIRS_SNPP_India_2018_2025.csv`

The expected FIRMS columns include `latitude`, `longitude`, `acq_date`, `acq_time`, `confidence`, `frp`, `satellite`, and `instrument`.

Weather data are retrieved by `build_dataset.py` from historical Open-Meteo archive data using ECMWF IFS. The weather source and retrieval configuration are recorded in generated metadata.

**Important:** a FIRMS non-detection is not equivalent to proof that no fire existed. The target is therefore described as **FIRMS fire-detection occurrence**.

## Experimental design

### Primary temporal experiment

- Train: 2018–2023
- Validation: 2024
- Final test: 2025

No 2025 observations are used during model fitting or hyperparameter selection.

### Spatial experiment

A grouped geographic holdout is also supported. Geographic cells are assigned to spatial groups so nearby observations are not randomly scattered between train and test.

### Baselines

- Logistic Regression
- Random Forest
- Gradient Boosting

The baseline models are deliberately conventional. Novelty is evaluated through the dataset construction and generalization experiments, not by claiming a novel classifier.

## Pipeline

```text
Raw SNPP-VIIRS FIRMS
        ↓
0.1° grid/time aggregation
        ↓
Stratified fire-cell sampling
        ↓
Controlled non-fire sampling
        ↓
Historical meteorological matching
        ↓
24 h / 72 h / 168 h antecedent features
        ↓
Temporal + spatial splits
        ↓
Baseline ML models
        ↓
PR-AUC / ROC-AUC / F1 / calibration
        ↓
Ablation + generalization analysis
```

## Reproducible commands

```bash
pip install -r requirements.txt
python prepare_fire_cells.py --input data/raw/FIRMS_VIIRS_SNPP_India_2018_2025.csv --output data/processed/fire_cells.csv
python build_dataset.py --fire-input data/processed/fire_cells.csv --output data/processed/india_fire_weather.csv
python train_and_evaluate.py --data data/processed/india_fire_weather.csv --output results
```

The weather-building stage can be expensive because it requires historical weather retrieval. Use `--max-fire-cells` while testing the pipeline.

## Research-paper evidence

The repository contains:

- `research/novelty_statement.md`
- `research/experimental_protocol.md`
- `research/related_work_matrix.md`
- `research/ablation_plan.md`

These documents distinguish **what is already known** from the narrower contribution being tested here.

## Reproducibility rules

- Fixed random seeds are recorded in configuration.
- The final test year is never used for tuning.
- Metrics are reported separately for temporal and spatial tests.
- Class balance and negative-sampling strategy are reported.
- Feature ablations are mandatory before making a novelty claim.
- Results are not described as a real-time operational fire warning system unless forecast-time inputs are actually available without leakage.
