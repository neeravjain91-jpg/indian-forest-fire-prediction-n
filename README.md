# India-Wide Forest Fire Occurrence Prediction (2018–2025)

Research-oriented framework for predicting **forest-fire occurrence cells in India** from Suomi-NPP VIIRS FIRMS active-fire observations and multi-timescale meteorological conditions.

## Research question

> Can an India-wide model trained on historical SNPP-VIIRS FIRMS fire detections and meteorological conditions generalize to future years and geographically distinct parts of India?

## Final working model

The repository now includes a reproducible **HistGradientBoostingClassifier** pipeline in `train_final_model.py` using the validated 31-feature dataset:

- `grid_lat`, `grid_lon`, `hour`, `year`, `month`
- 1-day weather features
- 3-day weather features
- 7-day weather features

The model is trained with a fixed temporal protocol:

- Train: 2018–2022
- Validation: 2023
- Test: 2024–2025

The script produces a serialized model, test predictions, metrics, and permutation feature importance.

## Research contribution

This project does **not** claim novelty from machine learning alone. The contribution is the combination of:

1. India-wide SNPP-VIIRS FIRMS observations from 2018–2025.
2. A reproducible spatial-temporal fire-cell target at 0.1° resolution.
3. Meteorological predictors at the fire time plus antecedent 24 h, 72 h and 168 h windows.
4. Controlled fire/non-fire sampling rather than treating every satellite non-detection as ground truth.
5. Temporal holdout testing on future years.
6. Spatial generalization testing using held-out geographic groups.
7. Ablation experiments measuring the value of antecedent weather windows.
8. Class-imbalance-aware metrics: PR-AUC, ROC-AUC, precision, recall, F1 and Brier score where applicable.

## Dataset

Place the raw FIRMS file at:

`data/raw/FIRMS_VIIRS_SNPP_India_2018_2025.csv`

The expected FIRMS columns include `latitude`, `longitude`, `acq_date`, `acq_time`, `confidence`, `frp`, `satellite`, and `instrument`.

Weather data are retrieved by `build_dataset.py` from historical Open-Meteo archive data using ECMWF/ERA5-Land configuration. The weather source and retrieval configuration are recorded in generated metadata.

**Important:** a FIRMS non-detection is not equivalent to proof that no fire existed. The target is therefore described as **FIRMS fire-detection occurrence**.

## Experimental design

### Primary temporal experiment

- Train: 2018–2022
- Validation: 2023
- Final test: 2024–2025

No test-period observations are used during training.

### Spatial experiment

A grouped geographic holdout is also supported. Geographic cells are assigned to spatial groups so nearby observations are not randomly scattered between train and test.

### Baselines

- Logistic Regression
- Random Forest
- HistGradientBoosting

The baseline models are conventional. The research contribution is evaluated through dataset construction, feature windows, ablation, and generalization experiments rather than by claiming a novel classifier.

## Reproducible commands

```bash
pip install -r requirements.txt
python prepare_fire_cells.py --input data/raw/FIRMS_VIIRS_SNPP_India_2018_2025.csv --output data/processed/fire_cells.csv
python build_dataset.py --fire-input data/processed/fire_cells.csv --output data/processed/india_fire_weather.csv
python train_final_model.py --data data/processed/india_fire_weather_final.csv --output results/final_model
```

The weather-building stage can be expensive because it requires historical weather retrieval. Use `--max-fire-cells` while testing the pipeline.

## Final model outputs

Running `train_final_model.py` creates:

- `results/final_model/final_hgb_model.joblib` — trained working model
- `results/final_model/metrics.json` — validation and test metrics
- `results/final_model/test_predictions.csv` — test-set probabilities and predictions
- `results/final_model/feature_importance.csv` — permutation feature importance

## Research-paper evidence

The repository contains research notes and the experimental outputs used for the paper.

## Reproducibility rules

- Fixed random seed: 42.
- The final test period is not used for model fitting.
- Metrics are reported separately for temporal and spatial tests.
- Class balance and negative-sampling strategy are reported.
- Feature ablations are mandatory before making a novelty claim.
- Results are not described as a real-time operational fire warning system unless forecast-time inputs are actually available without leakage.
