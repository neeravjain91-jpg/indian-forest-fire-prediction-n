# Feature Ablation & Generalization Analysis

This document details the ablation study quantifying the predictive contributions of geographic location, temporal seasonality, and multi-timescale meteorology on held-out test data (2024–2025, 31,525 observations).

## Empirical Results Summary

Evaluated on the held-out test split using `HistGradientBoostingClassifier` (max_iter=300, lr=0.05, leaves=31):

| Feature Configuration | Features | Accuracy | F1-Score | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **Coordinates + Temporal** | 5 | 0.697859 | 0.724767 | **0.806952** | **0.813684** |
| **Coordinates Only** | 2 | 0.697542 | 0.725872 | 0.806778 | 0.813427 |
| **Coordinates + Weather** | 28 | 0.693354 | 0.710144 | 0.773743 | 0.775811 |
| **Full 31 Features** | 31 | 0.687803 | 0.702551 | 0.767938 | 0.763533 |
| **Weather Only** | 26 | 0.552641 | 0.576957 | 0.575927 | 0.561100 |
| **Temporal Only** | 3 | 0.500048 | 0.573231 | 0.500159 | 0.500125 |

*(Note: The primary final artifact `results/final_model/final_hgb_model.joblib` trained on full 31 features achieves Accuracy 0.700111, F1 0.712207, ROC-AUC 0.785174, PR-AUC 0.783202).*

## Key Scientific Findings

1. **Spatial Dominance in Occurrence Classification**:
   Geographic coordinates (`grid_lat`, `grid_lon`) account for the vast majority of discriminatory power (ROC-AUC ~0.8068). Active forest fires in India are heavily concentrated in specific ecological biomes (e.g. deciduous forests of Central India, Odisha, Western Ghats, Purvanchal), making spatial coordinates a strong proxy for fuel availability, forest type, and historical anthropogenic land use.

2. **Weather Provides Localized Modulation**:
   Weather alone achieves ROC-AUC ~0.5759 and PR-AUC ~0.5611. Across the whole Indian subcontinent, weather variables alone cannot distinguish a dry desert with zero fuel from a dry forest with heavy fuel load. Meteorology acts as a conditional trigger modulating fire timing within fire-prone geographic zones.

3. **Temporal (Diurnal/Annual) Signals**:
   Temporal features alone (hour, year, month) without spatial context yield ROC-AUC ~0.5002, equivalent to chance. However, combined with coordinates, they capture the dry pre-monsoon fire season (March–May) and afternoon peak thermal hours.

