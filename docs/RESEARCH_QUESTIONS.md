# Research Questions, Hypotheses & Statistical Verification

## 1. Primary Research Question
Does an event-centric spatiotemporal framework—integrating multi-timescale atmospheric drying, causal fire-history recurrence, authoritative digital elevation geomorphology, and environmental fuel moisture deficit proxies—demonstrate measurable predictive advantages over static cell-level occurrence classifiers in:
1. Predicting forward fire risk at defensible satellite overpass horizons ($T+24\text{h}$, $T+48\text{h}$)?
2. Generalizing across six predefined geographic fire regimes of India?
3. Accurately quantifying forecast probability calibration and epistemic uncertainty?

---

## 2. Formal Hypotheses & Empirical Statistical Outcomes

### **Hypothesis 1 (H1): Multi-Timescale Weather History vs. Instantaneous Weather**
* **Statement**: Antecedent multi-day atmospheric drying representations (1-day, 3-day, and 7-day windows) provide higher discriminative power ($\text{PR-AUC}$ and $\text{ROC-AUC}$) than relying solely on instantaneous 1-day weather observations.
* **Empirical Outcome**: **Supported**. In controlled modality ablations on the chronological test set (2024–2025), expanding from 1-day weather (6 features) to multi-timescale weather (26 features) improved ROC-AUC from $56.25\%$ to $57.84\%$ ($\Delta = +1.59\%$) and PR-AUC from $54.37\%$ to $55.90\%$.

---

### **Hypothesis 2 (H2): Event-Centric Representation vs. Isolated Point Formulation**
* **Statement**: Modeling connected-component spatiotemporal event complexes improves the forecasting of forward fire persistence compared to treating cells as isolated independent points.
* **Empirical Outcome**: **Supported**. The connected-component event persistence target achieved **$68.39\%$ ROC-AUC**, **$11.90\%$ PR-AUC** (against a $6.87\%$ test prevalence), and a **Top-100 Precision of $18.0\%$** ($2.6\times$ baseline prevalence). In contrast, predicting arbitrary cell ignitions at $T+24\text{h}$ achieved only $53.13\%$ ROC-AUC.

---

### **Hypothesis 3 (H3): Multimodal Environmental Features vs. Baseline Models**
* **Statement**: Incorporating authoritative DEM terrain geomorphology (elevation, canonical Horn slope, Riley TRI) and atmospheric fuel dryness (VPD proxy, soil moisture deficit proxy) yields statistically distinguishable improvements in discrimination and geographic transferability across India's geographic regimes.
* **Empirical Outcome**: **Supported by 95% Bootstrap Confidence Intervals**.
  - **Factorial Feature Main Effect**: $\Delta \text{ROC-AUC} = +2.89\%$ ($95\% \text{ CI} = [+2.40\%, +3.35\%]$, excludes zero).
  - **Factorial Model-Family Main Effect**: $\Delta \text{ROC-AUC} = +0.35\%$ ($95\% \text{ CI} = [+0.10\%, +0.56\%]$, excludes zero).
  - **Factorial Interaction Effect**: $\Delta \text{ROC-AUC} = +0.35\%$ ($95\% \text{ CI} = [-0.02\%, +0.73\%]$, crosses zero).
  - **Leave-One-Geographic-Regime-Out (LOGRO) Cross-Validation**: Across all 6 predefined geographic fire regimes (with temporal validation inside training regimes), the 39-feature multimodal model achieved a Macro Mean ROC-AUC of **$64.20\% \pm 0.97\%$**, outperforming the 31-feature baseline ($57.46\% \pm 1.66\%$) while cutting cross-regional variance in half.

---

### **Hypothesis 4 (H4): Probability Calibration & Epistemic Uncertainty Tracking**
* **Statement**: Post-hoc probability calibration universally reduces Expected Calibration Error (ECE), and estimated epistemic uncertainty metrics correlate monotonically with empirical prediction errors.
* **Empirical Outcome**: **Refuted in its universal form / Method-Dependent**.
  - **Probability Calibration**: Isotonic regression overfit the 2023 validation distribution and **increased** test-set ECE across all benchmark models ($0.0092 \to 0.0137$ in HGB 31; $0.0150 \to 0.0171$ in LightGBM 39). Parametric Platt scaling (logistic on log-odds) regularized estimates, reducing ECE ($0.0119 \to 0.0088$ in LightGBM 31; $0.0150 \to 0.0143$ in LightGBM 39). Brier score consistently improved under multimodal feature expansion (main effect $\Delta = -0.0047$, $95\%\text{ CI} = [-0.0056, -0.0038]$), establishing that Brier refinement is distinct from ECE degradation under non-parametric calibration.
  - **Uncertainty Correlation**: Monte Carlo Dropout epistemic variance on the multi-scale BiGRU deep model exhibited a weak negative correlation with absolute classification error ($r_s = -0.1355$, $p < 0.001$). Epistemic variance from dropout alone without distance-to-support bounds or conformal prediction is insufficient as a standalone error predictor.
