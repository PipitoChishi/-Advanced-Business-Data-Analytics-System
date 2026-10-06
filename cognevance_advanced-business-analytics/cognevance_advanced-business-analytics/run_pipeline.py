"""One-command automation: data -> clean -> features -> models -> dashboard -> report.
Usage:  python run_pipeline.py            (uses data/raw/*.csv; generates them if missing)
"""
import pandas as pd
from src import config as cfg
from src.data_generation import main as generate
from src.preprocessing import clean_sales, clean_customers
from src import forecasting, customer_analytics as ca, dashboard as dash


def main():
    if not (cfg.RAW / "store_sales_daily.csv").exists():
        generate()
    sales_raw = pd.read_csv(cfg.RAW / "store_sales_daily.csv")
    cust_raw = pd.read_csv(cfg.RAW / "customers.csv")

    clean, log = clean_sales(sales_raw)
    cust = clean_customers(cust_raw)
    clean.to_csv(cfg.PROC / "store_sales_clean.csv", index=False)

    F = forecasting.run(clean)
    C = ca.churn_model(cust)
    S = ca.segment(C["scored"])
    C["scored"] = S["labelled"].assign(churn_prob=C["scored"]["churn_prob"], risk=C["scored"]["risk"])

    F["forecast"].to_csv(cfg.OUT / "forecast_next_28d.csv", index=False)
    C["scored"][["customer_id", "segment", "churn_prob", "risk", "monetary_12m"]].round(3).to_csv(cfg.OUT / "customers_scored.csv", index=False)
    F["table"].to_csv(cfg.OUT / "forecast_model_comparison.csv"); C["table"].to_csv(cfg.OUT / "churn_model_comparison.csv")

    K = dash.kpis(clean, F["forecast"], C["scored"])
    dash.build_dashboard(clean, F, C, S, K)
    dash.static_charts(clean, F, C, S)
    dash.build_report(log, None, F, C, S, K, clean)
    print(F["table"], "\n", C["table"], "\n", S["profile"])


if __name__ == "__main__":
    main()
