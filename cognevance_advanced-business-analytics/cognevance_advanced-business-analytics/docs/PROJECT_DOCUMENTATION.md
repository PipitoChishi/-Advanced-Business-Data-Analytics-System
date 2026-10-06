# Project Documentation - Advanced Business Data Analytics System

## 1. Objective
Provide a retail business with automated, predictive analytics: forecast store demand, identify customers likely to leave, segment customers and surface KPIs and recommendations.

## 2. Data
| File | Grain | Key columns |
|---|---|---|
| `store_sales_daily.csv` | store x day, 40 stores, 2020-2023 | store attributes, promo, is_holiday, temperature_c, sales, avg_basket, footfall, transactions |
| `customers.csv` | customer, 15,000 | tenure, recency, frequency, monetary, basket, categories, promo share, complaints, loyalty, channel, city tier, churned |

Both are synthetic with injected quality problems (duplicates, negatives, 10x keying errors, missing values).

## 3. Methodology
1. **Cleaning:** dedupe on (store, date); negative sales and sales > 4x centred 15-day rolling median set missing; per-store interpolation; customer medians/mode imputation.
2. **Feature engineering:** calendar & cyclical features, holiday proximity, lag/rolling features (>= 28 days old), spend per order, recency buckets.
3. **Models:** seasonal naive, Ridge, tuned HistGradientBoosting (forecasting); logistic regression and HistGradientBoosting (churn); K-Means (segments).
4. **Evaluation:** MAE, RMSE, MAPE, R² (forecast); ROC-AUC, precision, recall, F1, top-decile lift (churn); silhouette (segments).
5. **Deployment-style artefacts:** joblib models, scored CSVs, dashboard, regenerated report.

## 4. Results
See `reports/predictive_insights_report.md` (regenerated on every run) for current numbers, tables and recommendations.

## 5. How to reproduce
`pip install -r requirements.txt && python run_pipeline.py`

## 6. Limitations and next steps
* Synthetic data: validate on real sales before relying on accuracy numbers.
* Future store-level promotions are unknown; only the chain-wide calendar is used.
* Possible extensions: hierarchical reconciliation across stores, prediction intervals (quantile GBM), uplift modelling for retention offers, a Streamlit front-end.
