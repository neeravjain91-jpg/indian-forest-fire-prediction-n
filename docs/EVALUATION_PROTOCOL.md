# Evaluation Protocol & Experimental Splits

## 1. Primary Chronological Evaluation Protocol

To prevent temporal data leakage and rigorously evaluate model generalization across future fire seasons, the dataset is partitioned chronologically by calendar year:

| Partition | Time Horizon | Sample Count | Percentage | Research Function |
| :--- | :--- | :---: | :---: | :--- |
| **Training Set** | 2018–2022 (5 years) | 84,661 | 64.63% | Fitting model parameters and learning baseline representations. |
| **Validation Set** | 2023 (1 year) | 14,814 | 11.31% | Hyperparameter tuning and model selection; completely held out from training. |
| **Test Set** | 2024–2025 (2 years) | 31,525 | 24.06% | Final unbiased prospective evaluation; completely held out until testing. |
| **Total** | **2018–2025 (8 years)** | **131,000** | **100.00%** | Comprehensive nationwide evaluation. |

---

## 2. Leakage Prevention Rationale

Standard random $k$-fold cross-validation is fundamentally flawed for spatiotemporal Earth observation datasets because observations from the same calendar days and adjacent grid cells are randomly mixed between training and test sets. This creates severe spatial and temporal autocorrelation leakage, yielding artificially inflated accuracy.

Our chronological split protocol enforces two strict scientific guarantees:
1. **Zero Future Lookahead**: No observations from 2023 or 2024–2025 are ever visible during model fitting on 2018–2022.
2. **Prospective Realism**: Evaluates how effectively an operational classifier trained on historical multi-year records can generalize to entirely unseen future climatic cycles.

---

## 3. Spatial Generalization Protocol

In addition to chronological evaluation, the project evaluates spatial generalization across distinct geographic blocks:
- **2-Degree Spatial Blocks**: Geographic cells are grouped into $(2^\circ \times 2^\circ)$ latitude-longitude spatial blocks (~220 km × 220 km).
- **Group Holdout**: Entire spatial blocks are randomly partitioned into training (80%) and held-out test regions (20%), preventing adjacent cell memorization and testing regional spatial transferability.

---

## 4. Evaluation Metrics

Because wildfire occurrence exhibits varying operational costs between false positives and false negatives, models are evaluated across a comprehensive suite of discrimination and threshold metrics:

1. **ROC-AUC (Receiver Operating Characteristic Area Under Curve)**:
   Measures ranking ability across all possible classification thresholds independent of decision boundary.
2. **PR-AUC (Precision-Recall Area Under Curve / Average Precision)**:
   Particularly sensitive to positive-class retrieval quality.
3. **Accuracy, Precision, Recall, F1-Score**:
   Evaluated at the canonical 0.50 decision threshold ($P(\text{fire} \ge 0.50)$).
4. **Confusion Matrix**:
   Explicit reporting of True Positives ($TP$), False Positives ($FP$), True Negatives ($TN$), and False Negatives ($FN$) on the held-out test set.
