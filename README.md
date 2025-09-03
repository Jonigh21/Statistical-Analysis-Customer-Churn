# Telco Customer Churn — Predictive Modeling & Retention Insights


**Goal:** Predict churn and surface actionable drivers so marketing can prioritize proactive retention.


## Data
- **Rows/Cols:** 7,043 × 21
- **Target:** `Churn` (Yes/No → 1/0)
- **Imbalance:** ~73% stayed vs ~27% churned


## Pipeline
1. **Preprocessing**
- `TotalCharges` → numeric (coerce blanks to NaN) + median imputation
- One‑hot encode categoricals; drop `customerID`
- Stratified train/test split (80/20)
2. **Baseline**
- Logistic Regression (Accuracy 0.80, F1_churn 0.60, Recall_churn 0.56)
3. **Imbalance Handling**
- LR with `class_weight='balanced'` → **F1 0.616**, **ROC‑AUC 0.843**, Recall_churn **0.78**
- LR with SMOTE → F1 0.578, ROC‑AUC 0.809
4. **Tree‑Based Models**
- Random Forest → F1 0.547, ROC‑AUC 0.825
- HistGradientBoosting → F1 0.603, ROC‑AUC 0.822
- XGBoost → F1 0.596, ROC‑AUC 0.825
5. **Model Comparison (sorted by F1)**


| Model | F1 | ROC_AUC |
|---|---:|---:|
| LogReg (class_weight) | **0.616** | **0.843** |
| GradBoost (HGB) | 0.603 | 0.822 |
| LogReg Baseline | 0.601 | 0.842 |
| XGBoost | 0.596 | 0.825 |
| LogReg (SMOTE) | 0.578 | 0.809 |
| RandomForest | 0.547 | 0.825 |


## Key Insights (Top Drivers)
- Higher risk: **Month‑to‑month**, **short tenure**, **higher MonthlyCharges**, **Electronic check**, certain **fiber** setups without support bundles.
- Lower risk: Longer contracts, longer tenure, and service bundles.


## Threshold Tuning
- Swept cutoffs on LR (class‑weighted); best **F1 at threshold = 0.66**
- At 0.66 → Precision 0.59 | Recall 0.66 | F1 0.62 | Accuracy 0.79


## Business Actions
1. Incentivize contract upgrades for short‑tenure, month‑to‑month users.
2. Bundle support/security for fiber customers at risk.
3. Price/loyalty adjustments for high‑charge cohorts.
4. Encourage auto‑pay over electronic check.
