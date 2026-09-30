# Major Research Questions & Testable Hypotheses

## 1. Primary Research Question
Does an event-centric spatiotemporal framework—integrating multi-timescale atmospheric drying, fire-history persistence, terrain geomorphology, and environmental state—significantly outperform static cell-level occurrence classifiers in:
1. Predicting forward fire risk at defensible horizons ($T+24\text{h}$, $T+48\text{h}$)?
2. Generalizing across spatially disjoint geographic and ecological regimes across India?
3. Accurately quantifying forecast uncertainty and out-of-distribution conditions?

---

## 2. Formal Hypotheses & Experimental Verification Protocols

### **Hypothesis 1 (H1): Temporal Weather History vs. Static Weather**
* **Statement**: Incorporating multi-timescale antecedent weather signals (1-day, 3-day, 7-day cumulative precipitation deficits, temperature extremes, relative humidity minimums, and soil moisture drawdown) yields higher forward forecast discriminative power ($\text{PR-AUC}$ and $\text{ROC-AUC}$) than relying solely on instantaneous 1-day weather.
* **Testing Protocol**: Controlled feature-ablation benchmark on held-out test years (2024–2025):
  - Model A1: Instantaneous 1-day weather features only.
  - Model A2: 1-day + 3-day weather features.
  - Model A3: 1-day + 3-day + 7-day multi-timescale weather features.
* **Falsification Criterion**: H1 is falsified if Model A3 fails to achieve a statistically significant improvement ($p < 0.05$ or $\Delta \text{ROC-AUC} \le 0.005$) over Model A1.

---

### **Hypothesis 2 (H2): Event-Centric Representation vs. Isolated Point Formulation**
* **Statement**: Spatiotemporal event representations (incorporating active cluster proximity, cluster age, event density within 50 km, and antecedent local persistence) improve the prediction of future fire occurrence and continuation over models that treat each grid-cell as an isolated independent entity.
* **Testing Protocol**: Compare baseline models without spatial/event history against event-augmented models under identical split rules, specifically evaluating recall on active multi-day fire clusters.
* **Falsification Criterion**: H2 is falsified if event-aware features do not improve event-level detection rate and Brier score on persistent fire complexes.

---

### **Hypothesis 3 (H3): Multimodal Generalization vs. Location Memorization**
* **Statement**: Incorporating physical environmental covariates (terrain elevation, slope, aspect, and vapor pressure deficit proxies) alongside meteorological dynamics reduces the performance drop when transferring models to geographically and ecologically held-out regions of India compared to baseline coordinate-heavy models.
* **Testing Protocol**: Spatially disjoint holdout evaluation:
  - Protocol: Train on 5 geographic regimes (e.g., Western Ghats, Northeast, North, West, East); test exclusively on held-out Central India deciduous zone.
  - Compare relative performance drop ($\Delta \text{F1}$ and $\Delta \text{PR-AUC}$) between coordinate-reliant baseline vs. multimodal environmental model.
* **Falsification Criterion**: H3 is falsified if the multimodal model experiences an equal or greater degradation in performance upon geographic transfer than the location-only baseline.

---

### **Hypothesis 4 (H4): Probability Calibration & Epistemic Uncertainty Tracking**
* **Statement**: Post-hoc probability calibration (Platt scaling / Isotonic regression) and distance-to-support / ensemble disagreement indicators reliably correlate with empirical prediction error, serving as a valid out-of-distribution (OOD) filter in high-uncertainty conditions.
* **Testing Protocol**: 
  - Evaluate Expected Calibration Error (ECE) and Brier Score before and after calibration.
  - Stratify test predictions by estimated uncertainty deciles and verify that error rate monotonically tracks uncertainty scores.
* **Falsification Criterion**: H4 is falsified if calibration fails to reduce ECE or if prediction error is uncorrelated with the estimated uncertainty metric ($r \le 0.10$).
