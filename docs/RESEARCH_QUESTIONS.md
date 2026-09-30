# Research Questions, Hypotheses & Statistical Verification

## 1. Primary Research Question
Does an event-centric spatiotemporal framework—integrating multi-timescale atmospheric drying, causal fire-history persistence, real digital elevation geomorphology, and environmental fuel dryness—demonstrate measurable predictive advantages over static cell-level occurrence classifiers in:
1. Predicting forward fire risk at defensible satellite overpass horizons ($T+24\text{h}$, $T+48\text{h}$)?
2. Generalizing across all major ecological and geographic fire regimes of India?
3. Accurately quantifying forecast uncertainty and out-of-distribution conditions?

---

## 2. Formal Hypotheses & Empirical Statistical Outcomes

### **Hypothesis 1 (H1): Multi-Timescale Weather History vs. Instantaneous Weather**
* **Statement**: Antecedent multi-day atmospheric drying signals (1-day, 3-day, and 7-day windows) provide higher forward discriminative power ($\text{PR-AUC}$ and $\text{ROC-AUC}$) than relying solely on instantaneous 1-day weather.
* **Empirical Outcome**: **Supported**. In controlled modality ablations on the held-out test set (2024–2025), expanding from 1-day weather (6 features) to multi-timescale weather (26 features) improved ROC-AUC from $56.25\%$ to $57.84\%$ ($\Delta = +1.59\%$) and PR-AUC from $54.37\%$ to $55.90\%$.

---

### **Hypothesis 2 (H2): Event-Centric Representation vs. Isolated Point Formulation**
* **Statement**: Modeling connected-component spatiotemporal event complexes improves the forecasting of forward fire persistence compared to treating cells as isolated independent points.
* **Empirical Outcome**: **Supported**. The connected-component event persistence target achieved **$68.39\%$ ROC-AUC**, **$11.90\%$ PR-AUC** (against a $6.87\%$ natural test prevalence), and a **Top-100 Precision of $18.0\%$** ($2.6\times$ random base rate).

---

### **Hypothesis 3 (H3): Multimodal Environmental Features vs. Location Baseline**
* **Statement**: Incorporating authoritative DEM terrain geomorphology (elevation, slope, TRI) and atmospheric fuel dryness (VPD, soil drought index) yields statistically distinguishable improvements in discrimination and geographic transferability across India's biomes.
* **Empirical Outcome**: **Supported by 95% Bootstrap Confidence Intervals**.
  - Controlled Factorial Effect in HGB: $\Delta \text{ROC-AUC} = +2.71\%$ ($95\% \text{ CI} = [+2.16\%, +3.22\%]$, excludes zero).
  - Controlled Factorial Effect in LightGBM: $\Delta \text{ROC-AUC} = +2.95\%$ ($95\% \text{ CI} = [+2.42\%, +3.44\%]$, excludes zero).
  - Leave-One-Ecoregion-Out Cross-Validation: Across all 6 major Indian biomes, the 39-feature multimodal model achieved a Macro Mean ROC-AUC of **$64.58\% \pm 0.95\%$**, outperforming the 31-feature baseline ($57.75\% \pm 1.96\%$) with lower cross-regional variance.

---

### **Hypothesis 4 (H4): Probability Calibration & Epistemic Uncertainty Tracking**
* **Statement**: Post-hoc probability calibration reduces Expected Calibration Error (ECE), and estimated uncertainty metrics correlate with empirical prediction errors.
* **Empirical Outcome**: **Partially Supported**.
  - Calibration: Isotonic regression successfully reduced ECE across all models (down to $0.012 - 0.017$).
  - Uncertainty Correlation: Raw Monte Carlo Dropout epistemic variance on the BiGRU deep model exhibited a negative correlation with classification error residual ($r_s = -0.1355$, $p < 0.001$). This confirms that MC Dropout alone without distance-to-support bounds or conformal prediction does not monotonically track empirical errors in this spatiotemporal domain.
