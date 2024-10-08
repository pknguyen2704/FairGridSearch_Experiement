fairness_tradeoff.model

To do: 

1. Datasets:
    - [x] ADULTS
    - [x] COMPAS
    - [x] German Credit

2. Base Estimators:
    - [x] Logistics Regression
    - [x] Random Forest
    - [x] Gradient Boosting
    - [x] Support Vector Machine
    - [x] Gaussian Naive Bayes
    - [x] TabTransformer

- [x] Bias Mitigation:
    - [x] **Pre-processing**
        - [x] Reweighing
        - [x] Learning Fair Representation

    - [x] **In-processing**
        - [x] Learning Fair Representation (in)
        - [x] Adversarial Debiasing
        - [x] Exponentiated Gradient Reduction

    - [x] **Post-processing**
        - [x] Reject Option Classifier
        - [x] Calibrated Equalized Odds
        
    - [x] **Mixed Approach**
        - [x] Reweighing+Reject Option Classifier
        - [x] Reweighing+Calibrated Equalized Odds
