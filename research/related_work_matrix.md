# Related-work matrix

| Research direction | Established in literature? | Implication for this project |
|---|---|---|
| ML for Indian forest-fire prediction | Yes | Not a novelty claim |
| Meteorological fire predictors | Yes | Use as baseline inputs |
| MODIS-based Indian susceptibility mapping | Yes | Distinguish satellite product and target |
| SNPP-VIIRS fire data for regional Indian susceptibility mapping | Yes, including Northeast India | Do not claim first use of SNPP-VIIRS in India |
| Nationwide 2018–2025 SNPP-VIIRS FIRMS occurrence representation | Requires exact-match verification | Candidate data/coverage contribution |
| 24/72/168 h antecedent weather features | Requires exact-match verification | Candidate feature contribution |
| Future-year 2025 holdout | Requires exact-match verification | Candidate evaluation contribution |
| Geographic group holdout across India | Requires exact-match verification | Candidate generalization contribution |
| Joint temporal + spatial evaluation of the above | Requires exact-match verification | Candidate methodological contribution |

## Known comparison points

### Indian spatial susceptibility work

Prior Indian studies have used satellite fire products, climatic variables, topography, vegetation and machine learning to produce susceptibility maps. Such work establishes the general feasibility of ML-based fire-risk modeling but does not by itself establish the exact 2018–2025 SNPP-VIIRS FIRMS occurrence protocol used here.

### Northeast India SNPP-VIIRS work

A 2024 study used SNPP-VIIRS fire data from 2018–2019 for susceptibility mapping in Northeast India with multiple machine-learning models and many predictor variables. This is a direct reason not to claim that the present study is the first Indian SNPP-VIIRS ML study.

### Broader fire-prediction literature

Recent reviews report heavy use of meteorological predictors and recurring concerns about regional generalizability. The present experiments therefore prioritize future-year and geographic holdout testing rather than random row-level accuracy.

## Required before publication

The authors should repeat searches using combinations of:

- India + SNPP VIIRS + FIRMS + fire occurrence prediction
- India + VIIRS + meteorological + machine learning
- India + FIRMS + 2018 2019 2020 2021 2022 2023 2024 2025
- India + antecedent rainfall + fire occurrence
- India + spatial generalization + forest fire prediction

The final paper must cite exact prior datasets, years, target definitions and validation methods rather than relying on broad keyword similarity.
