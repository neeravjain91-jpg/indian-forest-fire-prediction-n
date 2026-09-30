# Scientific Evaluation Protocols & Metric Formulations

## 1. Dual Evaluation Protocols

To prevent the pervasive spatial autocorrelation and temporal leakage flaws documented by **Jain et al. (2020)** and **Meyer et al. (2018)**, the framework establishes two distinct evaluation benchmarks:

### Protocol 1: Chronological Temporal Generalization
* **Purpose**: Evaluate forward predictive generalizability to unseen future calendar years under strict temporal gating.
* **Splits**:
  - **Train**: 2018-01-01 to 2022-12-31 ($N = 84,661$)
  - **Validation / Calibration**: 2023-01-01 to 2023-12-31 ($N = 14,814$)
  - **Test**: 2024-01-01 to 2025-12-31 ($N = 31,525$)
* **Constraint**: Test observations occur chronologically *after* all training and validation data.

### Protocol 2: Spatially Disjoint Ecological Holdout
* **Purpose**: Stress-test model transferability across ecologically and climatically distinct geographic regions of India.
* **Splits**:
  - **Training Regions**: North (Himalayan), Northeast (Purvanchal), Western Ghats, East (Eastern Ghats), Northwest (Semi-arid) ($N = 83,603$).
  - **Held-Out Test Region**: Central India / Deccan Plateau dry deciduous teak/sal belt ($N = 47,397$).
* **Constraint**: The test region is completely disjoint in space from the training regions.

---

## 2. Mathematical Formulations of Metrics

### A. Discrimination Metrics
* **Receiver Operating Characteristic Area Under the Curve (ROC-AUC)**:
  $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$
* **Precision-Recall Area Under the Curve (PR-AUC / Average Precision)**:
  $$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) P_k$$
  Crucial for evaluating highly imbalanced forward lead horizons ($T+24\text{h}$, $T+48\text{h}$).

### B. Probabilistic Calibration & Reliability Metrics
* **Brier Score (Mean Squared Probability Error)**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 \quad \in [0, 1]$$
  Lower is better ($0 = \text{perfect probabilistic prediction}$).
* **Expected Calibration Error (ECE)**:
  Partition predicted probabilities into $M=10$ equal-width bins $B_1, \dots, B_M$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
  where $\text{acc}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} y_i$ and $\text{conf}(B_m) = \frac{1}{|B_m|} \sum_{i \in B_m} p_i$.

### C. Epistemic Uncertainty & Out-Of-Distribution (OOD) Metrics
* **Monte Carlo Dropout Predictive Variance**:
  $$\sigma_{\text{epistemic}} = \sqrt{\frac{1}{K} \sum_{k=1}^K (\hat{p}^{(k)} - \bar{p})^2}$$
  Computed over $K=15$ stochastic forward passes with active dropout ($p=0.15$).
* **Distance-to-Support OOD Score**:
  Normalized Mahalanobis-style feature distance from training distribution centroid:
  $$d_{\text{OOD}}(x) = \frac{1}{\sqrt{D}} \left\| \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}} \right\|_2$$
