# Scientific Evaluation Protocols & Metric Formulations

## 1. Dual Evaluation Protocols

To prevent the pervasive spatial autocorrelation and temporal leakage flaws documented by **Jain et al. (2020)** and **Meyer et al. (2018)**, the framework establishes two distinct evaluation benchmarks:

### Protocol 1: Chronological Temporal Generalization
* **Purpose**: Evaluate forward predictive generalizability to unseen future calendar years under strict temporal gating.
* **Splits**:
  - **Train**: 2018-01-01 to 2022-12-31 ($N = 84,661$)
  - **Validation / Calibration**: 2023-01-01 to 2023-12-31 ($N = 14,814$, used strictly for model selection and calibrator fitting)
  - **Held-Out Test**: 2024-01-01 to 2025-12-31 ($N = 31,525$, untouched during model fitting)
* **Constraint**: Test observations occur chronologically *after* all training and validation data.

### Protocol 2: Leave-One-Geographic-Regime-Out (LOGRO) Spatial Cross-Validation
* **Purpose**: Stress-test model transferability across six predefined geographic fire regimes of India (Central, Western Ghats, Northeast, North, East, Northwest).
* **Validation Gating**: Inside the 5 training regimes, temporal validation (Train $\le 2022$, Val $= 2023$) is enforced to prevent spatial autocorrelation leakage into calibration.
* **Constraint**: The held-out geographic regime is completely unseen and untouched during training and calibration.

---

## 2. Mathematical Formulations of Metrics

### A. Discrimination Metrics
* **Receiver Operating Characteristic Area Under the Curve (ROC-AUC)**:
  $$\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt$$
* **Precision-Recall Area Under the Curve (PR-AUC / Average Precision)**:
  $$\text{PR-AUC} = \sum_{k} (R_k - R_{k-1}) P_k$$
  Crucial for evaluating highly imbalanced forward lead horizons ($T+24\text{h}$, $T+48\text{h}$).
* **Top-$k$ Precision and Recall**:
  Evaluates ranking performance for high-risk resource dispatch:
  $$\text{Precision@}k = \frac{\sum_{i=1}^k y_{(i)}}{k}, \quad \text{Recall@}k = \frac{\sum_{i=1}^k y_{(i)}}{\sum_{j=1}^N y_j}$$

### B. Probabilistic Calibration & Reliability Metrics
* **Brier Score (Mean Squared Probability Error)**:
  $$\text{BS} = \frac{1}{N} \sum_{i=1}^N (p_i - y_i)^2 \quad \in [0, 1]$$
* **Expected Calibration Error (ECE)**:
  Partition predicted probabilities into $M=10$ equal-width bins $B_1, \dots, B_M$:
  $$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
* **Maximum Calibration Error (MCE)**:
  $$\text{MCE} = \max_{m=1 \dots M, |B_m| > 0} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$

### C. Factorial Analysis & Paired Bootstrap Resampling
For a $2 \times 2$ factorial experiment ($A = \text{HGB}_{31}$, $B = \text{HGB}_{39}$, $C = \text{LGBM}_{31}$, $D = \text{LGBM}_{39}$):
* **Feature Main Effect**: $\frac{(B - A) + (D - C)}{2}$
* **Model Family Main Effect**: $\frac{(C - A) + (D - B)}{2}$
* **Factorial Interaction**: $(D - C) - (B - A)$
All effects and 95% Confidence Intervals are calculated via paired non-parametric percentile bootstrap ($B=1,000$) over identical test observations.

### D. Epistemic Uncertainty & Out-Of-Distribution (OOD) Metrics
* **Monte Carlo Dropout Predictive Variance**:
  $$\sigma_{\text{epistemic}} = \sqrt{\frac{1}{K} \sum_{k=1}^K (\hat{p}^{(k)} - \bar{p})^2}$$
* **Distance-to-Support OOD Score**:
  Normalized feature distance from training distribution centroid:
  $$d_{\text{OOD}}(x) = \frac{1}{\sqrt{D}} \left\| \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}} \right\|_2$$
