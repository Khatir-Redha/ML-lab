# Customer Churn Prediction

A machine learning pipeline that predicts telecom customer churn from a Kaggle Playground-style dataset (~594K training rows, 21 raw features). The notebook goes from raw data through EDA, feature engineering, and multiple models, ending with a 5-fold cross-validated ensemble (LightGBM + CatBoost + XGBoost) for the final submission.

## Dataset

- **Source:** [Kaggle — Playground Churn Dataset](https://www.kaggle.com/datasets/khatirredha/playground-churn-dataset)
- **Train set:** 594,194 rows × 21 columns
- **Target:** `Churn` (Yes/No)
- **Features:** demographic info (gender, senior citizen, partner, dependents), account info (tenure, contract, payment method, charges), and subscribed services (internet, phone, streaming, security add-ons)
- No missing values in the raw data.

## Approach

### 1. Exploratory Data Analysis
Visualized churn distribution, churn by contract type, monthly charges distribution, tenure by churn status, and churn by payment method to identify patterns (e.g. month-to-month contracts and electronic check payments correlate with higher churn).

### 2. Feature Engineering
- Encoded binary columns (`Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `Churn`) to 0/1
- One-hot encoded categorical columns (internet service, contract type, payment method, etc.)
- Ranked features by mutual information score against the target
- Engineered additional features:
  - `Expected_Total` — tenure × monthly charges
  - `Charge_Difference` — actual vs. expected total charges (flags price changes)
  - `Avg_Monthly_Cost` — total charges normalized by tenure
  - `Total_Services` — count of subscribed add-on services
  - `Is_High_Risk_Combo` — flags the highest-churn segment (Fiber optic + Electronic check)

### 3. Models
| Model | Notes |
|---|---|
| Random Forest | Baseline, `class_weight='balanced'`, trained on a 50K sample |
| LightGBM | Trained on full dataset, `scale_pos_weight` for class imbalance, threshold tuned via precision-recall curve |
| LightGBM (5-fold CV) | Stratified K-Fold with early stopping, out-of-fold evaluation |
| **Ensemble (final)** | 5-fold blend of LightGBM (40%) + CatBoost (35%) + XGBoost (25%), each with early stopping |

## Results

| Model | ROC-AUC | F1 Score |
|---|---|---|
| Random Forest (baseline) | — | 0.697 |
| LightGBM (single split) | 0.916 | 0.684 (0.701 at tuned threshold) |
| LightGBM (5-fold CV) | ~0.913 avg | — |
| **Ensemble (5-fold CV)** | **0.915** | — |

At the tuned decision threshold, the LightGBM model reaches **85% accuracy**, **0.64 precision** and **0.78 recall** on the churn class — prioritizing catching churners (recall) over precision, which fits a churn-prevention use case where missing an at-risk customer is costlier than a false alarm.

## Tech Stack

- **Data processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Random Forest, feature selection, cross-validation), LightGBM, CatBoost, XGBoost

## Project Structure

```
.
├── Predict_Customer_Churn.ipynb   # Main notebook (EDA → feature engineering → modeling → ensemble)
└── README.md
```

## Running the Notebook

This notebook was built in a Kaggle environment and expects the dataset at:
```
../input/datasets/khatirredha/playground-churn-dataset/train.csv
../input/datasets/khatirredha/playground-churn-dataset/test.csv
```

To run it locally or elsewhere:
1. Download the dataset from Kaggle (link above)
2. Update the file paths in the second cell to point to your local `train.csv` / `test.csv`
3. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn lightgbm catboost xgboost matplotlib seaborn
   ```
4. Run all cells top to bottom

## Output

The final cell produces `submission_ensemble.csv` — churn probabilities for the test set from the blended ensemble model.

## Possible Improvements

- Hyperparameter tuning (Optuna/Bayesian search) instead of fixed params for each model
- Feature selection based on the mutual information rankings to reduce dimensionality
- Stacking instead of fixed-weight blending for the ensemble
- SHAP analysis for model interpretability

## Final Score
- **Private:** 0.91382

- **Public:** 0.91253

 