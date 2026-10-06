# Advanced Business Data Analytics System (Cognevance - Level 3)

End-to-end retail analytics platform: automated data pipeline, feature engineering, **28-day store sales forecasting**,
**customer churn prediction**, **K-Means segmentation**, KPI tracking, an interactive dashboard and an auto-generated report.

| Capability | Method | Result on test data |
|---|---|---|
| Sales forecasting (40 stores, 58k store-days) | Gradient boosting vs Ridge vs seasonal-naive, time-based split (test = 2023) | ~6.7% MAPE, R² ≈ 0.97 (baseline 15.4%) |
| Churn prediction (15k customers) | Logistic regression vs gradient boosting, best by ROC-AUC | ROC-AUC ≈ 0.77, top-decile lift ≈ 2.8x |
| Segmentation | K-Means on log-scaled RFM features, k by silhouette | 3 segments |

## Quick start
```bash
pip install -r requirements.txt
python run_pipeline.py        # data -> clean -> features -> models -> dashboard -> report
```
Open `outputs/interactive_dashboard.html` in a browser; read `reports/predictive_insights_report.md`.

## Outputs
* `outputs/interactive_dashboard.html` - Plotly dashboard (KPIs, forecast, store ranking, drivers, churn risk, segments)
* `outputs/charts/*.png` - static versions for GitHub
* `outputs/forecast_next_28d.csv` - store-level forecast; `outputs/customers_scored.csv` - churn probability + segment
* `models/sales_forecaster.joblib`, `models/churn_model.joblib`
* `reports/predictive_insights_report.md` - regenerated each run

## Automation
`run_pipeline.py` is one command; `.github/workflows/pipeline.yml` runs it weekly (and on demand) in GitHub Actions and uploads the outputs as artifacts.

## Structure
```
run_pipeline.py            orchestrator
src/config.py              paths, horizon, split date, seed
src/data_generation.py     synthetic retail data (replace data/raw/*.csv with real data of the same schema)
src/preprocessing.py       cleaning + leakage-safe feature engineering
src/forecasting.py         forecasting models, tuning, importance, future forecast
src/customer_analytics.py  churn model + segmentation
src/dashboard.py           dashboard, static charts, markdown report
docs/                      ARCHITECTURE.md, PROJECT_DOCUMENTATION.md
```

## Key design decisions
* **No leakage:** every lag/rolling feature uses data at least 28 days old, matching the forecast horizon; same-day footfall/transactions are excluded because they are outcomes of sales.
* **Time-ordered validation** (`TimeSeriesSplit` for tuning, 2023 held out for testing).
* **Honest baselines:** each model is compared with a naive baseline.

## Limitations
Data is synthetic; accuracy numbers demonstrate the methodology rather than real-world performance.
