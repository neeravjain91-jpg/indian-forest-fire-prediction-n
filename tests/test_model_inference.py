"""Tests for model loading, hyperparameters, feature alignment, and inference behavior."""
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "results" / "final_model" / "final_hgb_model.joblib"
METRICS_PATH = BASE_DIR / "results" / "final_model" / "metrics.json"
TEST_PRED_PATH = BASE_DIR / "results" / "final_model" / "test_predictions.csv"

EXPECTED_FEATURES = [
    "grid_lat", "grid_lon", "hour", "year", "month",
    "temp_1d", "rh_1d", "wind_1d", "pressure_1d", "soil_1d", "rain_1d",
    "temp_3d_mean", "temp_3d_max", "temp_3d_min",
    "rh_3d_mean", "rh_3d_min",
    "wind_3d_mean", "wind_3d_max",
    "pressure_3d_mean", "soil_3d_mean", "rain_3d_total",
    "temp_7d_mean", "temp_7d_max", "temp_7d_min",
    "rh_7d_mean", "rh_7d_min",
    "wind_7d_mean", "wind_7d_max",
    "pressure_7d_mean", "soil_7d_mean", "rain_7d_total",
]


@pytest.fixture(scope="module")
def loaded_model():
    """Load the final trained model artifact."""
    assert MODEL_PATH.exists(), f"Model file missing at {MODEL_PATH}"
    model = joblib.load(MODEL_PATH)
    return model


def test_model_artifact_exists():
    """Trained joblib artifact must exist in results/final_model/."""
    assert MODEL_PATH.exists()


def test_model_class_and_hyperparameters(loaded_model):
    """Model must be a HistGradientBoostingClassifier with confirmed hyperparameters."""
    assert isinstance(loaded_model, HistGradientBoostingClassifier)
    params = loaded_model.get_params()
    assert params["max_iter"] == 300
    assert params["learning_rate"] == 0.05
    assert params["max_leaf_nodes"] == 31
    assert params["l2_regularization"] == 1.0
    assert params["random_state"] == 42


def test_model_feature_count(loaded_model):
    """Model must expect exactly 31 features matching the schema."""
    assert hasattr(loaded_model, "n_features_in_")
    assert loaded_model.n_features_in_ == 31
    if hasattr(loaded_model, "feature_names_in_"):
        assert list(loaded_model.feature_names_in_) == EXPECTED_FEATURES


def test_model_inference_single_sample(loaded_model):
    """Model must return valid binary prediction and probabilities for a single row."""
    sample_df = pd.DataFrame([{f: 1.0 for f in EXPECTED_FEATURES}])
    proba = loaded_model.predict_proba(sample_df)
    pred = loaded_model.predict(sample_df)

    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0, 0] <= 1.0
    assert 0.0 <= proba[0, 1] <= 1.0
    assert np.isclose(proba[0, 0] + proba[0, 1], 1.0)
    assert pred.shape == (1,)
    assert pred[0] in (0, 1)


def test_model_inference_batch(loaded_model):
    """Model must correctly handle batch predictions."""
    batch_size = 10
    batch_data = pd.DataFrame(np.random.rand(batch_size, 31), columns=EXPECTED_FEATURES)
    probas = loaded_model.predict_proba(batch_data)
    preds = loaded_model.predict(batch_data)

    assert probas.shape == (batch_size, 2)
    assert preds.shape == (batch_size,)
    assert all(p in (0, 1) for p in preds)
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)


def test_metrics_json_consistency():
    """Saved metrics.json must match published test results."""
    assert METRICS_PATH.exists()
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    test_metrics = metrics.get("test", {})
    assert pytest.approx(test_metrics["accuracy"], abs=1e-4) == 0.700111
    assert pytest.approx(test_metrics["precision"], abs=1e-4) == 0.684654
    assert pytest.approx(test_metrics["recall"], abs=1e-4) == 0.742071
    assert pytest.approx(test_metrics["f1"], abs=1e-4) == 0.712207
    assert pytest.approx(test_metrics["roc_auc"], abs=1e-4) == 0.785174
    assert pytest.approx(test_metrics["pr_auc"], abs=1e-4) == 0.783202

    cm = metrics.get("test_confusion_matrix", [])
    assert cm == [[10373, 5388], [4066, 11698]]


def test_test_predictions_consistency():
    """Saved test_predictions.csv must produce the exact published evaluation metrics."""
    assert TEST_PRED_PATH.exists()
    pred_df = pd.read_csv(TEST_PRED_PATH)
    assert len(pred_df) == 31525

    y_true = pred_df["actual_fire"]
    y_prob = pred_df["fire_probability"]
    y_pred = pred_df["predicted_fire"]

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    pr_auc = average_precision_score(y_true, y_prob)

    assert pytest.approx(acc, abs=1e-5) == 0.700111
    assert pytest.approx(prec, abs=1e-5) == 0.684654
    assert pytest.approx(rec, abs=1e-5) == 0.742071
    assert pytest.approx(f1, abs=1e-5) == 0.712207
    assert pytest.approx(roc_auc, abs=1e-5) == 0.785174
    assert pytest.approx(pr_auc, abs=1e-5) == 0.783202
