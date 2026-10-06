"""Synthetic retail data generator (store-level daily sales + customer base).

Produces two raw files with realistic structure and deliberately messy records:
  data/raw/store_sales_daily.csv   ~58k rows  (40 stores x 4 years)
  data/raw/customers.csv           15k rows   (RFM-style attributes + churn label)
Replace these files with real retail/finance/healthcare data of the same schema to reuse the pipeline.
"""
import numpy as np
import pandas as pd
from .config import *

HOLIDAYS = pd.to_datetime([
    "2020-01-01", "2020-01-26", "2020-03-10", "2020-08-15", "2020-10-02", "2020-11-14", "2020-12-25",
    "2021-01-01", "2021-01-26", "2021-03-29", "2021-08-15", "2021-10-02", "2021-11-04", "2021-12-25",
    "2022-01-01", "2022-01-26", "2022-03-18", "2022-08-15", "2022-10-02", "2022-10-24", "2022-12-25",
    "2023-01-01", "2023-01-26", "2023-03-08", "2023-08-15", "2023-10-02", "2023-11-12", "2023-12-25",
    "2024-01-01", "2024-01-26"])


def promo_calendar(dates):
    """Chain-wide campaign days: first 7 days of every month + festive window 15 Oct - 15 Nov."""
    d = pd.DatetimeIndex(dates)
    festive = ((d.month == 10) & (d.day >= 15)) | ((d.month == 11) & (d.day <= 15))
    return ((d.day <= 7) | festive).astype(int)


def generate_sales(rng):
    types = rng.choice(["Mall", "High Street", "Express"], N_STORES, p=[.35, .4, .25])
    size = np.where(types == "Mall", rng.integers(9000, 16000, N_STORES),
           np.where(types == "High Street", rng.integers(4000, 9000, N_STORES), rng.integers(1200, 3500, N_STORES)))
    stores = pd.DataFrame({"store_id": [f"S{i:02d}" for i in range(1, N_STORES + 1)],
                           "region": rng.choice(["North", "South", "East", "West"], N_STORES),
                           "store_type": types, "size_sqft": size})
    stores["base"] = stores["size_sqft"] * rng.uniform(1.6, 2.6, N_STORES) * 1.0
    stores["growth"] = rng.normal(.06, .03, N_STORES)                  # yearly store-specific growth

    dates = pd.date_range(START, END)
    df = stores.merge(pd.DataFrame({"date": dates}), how="cross")
    df["date"] = pd.to_datetime(df["date"])
    d = df["date"]
    t_years = (d - d.min()).dt.days / 365.25
    dow = np.array([.86, .88, .9, .95, 1.08, 1.32, 1.25])[d.dt.dayofweek]
    month = np.array([1.0, .93, .98, .98, 1.0, .95, .92, .95, 1.0, 1.2, 1.28, 1.15])[d.dt.month - 1]
    df["promo"] = promo_calendar(d)
    store_promo = rng.random(len(df)) < .04                              # store-specific promos
    df["promo"] = np.maximum(df["promo"], store_promo.astype(int))
    df["is_holiday"] = d.isin(HOLIDAYS).astype(int)
    doy = d.dt.dayofyear
    df["temperature_c"] = (25 + 10 * np.sin(2 * np.pi * (doy - 65) / 365.25) + rng.normal(0, 2.5, len(df))).round(1)
    heat_effect = np.where(df["store_type"] == "Express", 1 + .006 * (df["temperature_c"] - 25), 1)
    covid = np.where((d >= "2020-03-25") & (d < "2020-06-15"), .45, 1.0)   # lockdown dip
    mean = (df["base"] * dow * month * (1 + df["growth"]) ** t_years * (1 + .28 * df["promo"])
            * (1 + .3 * df["is_holiday"]) * heat_effect * covid)
    df["sales"] = (mean * rng.lognormal(0, .07, len(df))).round(0)
    df["avg_basket"] = (rng.normal(780, 60, len(df)) * (1 + .12 * df["promo"])).round(0)
    df["transactions"] = (df["sales"] / df["avg_basket"]).round(0)
    df["footfall"] = (df["transactions"] / rng.uniform(.28, .4, len(df))).round(0)
    df = df.drop(columns=["base", "growth"])

    # -------- inject dirt
    n = len(df)
    df.loc[rng.choice(n, int(n * .006), replace=False), "sales"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "sales"] *= -1
    df.loc[rng.choice(n, 25, replace=False), "sales"] *= 10               # outliers / keying errors
    df.loc[rng.choice(n, int(n * .02), replace=False), "temperature_c"] = np.nan
    df = pd.concat([df, df.sample(40, random_state=1)]).sample(frac=1, random_state=2)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True)


def generate_customers(rng, n=15000):
    c = pd.DataFrame({"customer_id": [f"C{i:05d}" for i in range(1, n + 1)]})
    c["tenure_months"] = rng.integers(1, 61, n)
    c["recency_days"] = np.clip(rng.gamma(1.6, 38, n), 1, 365).round()
    c["frequency_12m"] = np.clip(rng.poisson(np.clip(14 - c["recency_days"] / 25, 1, None)), 0, 60)
    c["avg_basket"] = np.clip(rng.normal(900, 320, n), 150, None).round()
    c["monetary_12m"] = (c["frequency_12m"] * c["avg_basket"] * rng.uniform(.8, 1.2, n)).round()
    c["n_categories"] = np.clip(rng.poisson(3, n) + 1, 1, 8)
    c["promo_share"] = np.clip(rng.beta(2, 3, n), 0, 1).round(2)
    c["complaints_12m"] = rng.poisson(.35, n)
    c["loyalty_member"] = (rng.random(n) < np.clip(.25 + c["tenure_months"] / 120, 0, .9)).astype(int)
    c["channel"] = rng.choice(["Store", "Online", "Both"], n, p=[.5, .2, .3])
    c["city_tier"] = rng.choice(["Tier 1", "Tier 2", "Tier 3"], n, p=[.4, .4, .2])
    z = (-0.3 + .014 * c["recency_days"] - .12 * c["frequency_12m"] + .45 * c["complaints_12m"] - .7 * c["loyalty_member"]
         - .012 * c["tenure_months"] + 1.0 * c["promo_share"] - .15 * c["n_categories"] + rng.normal(0, .5, n))
    c["churned"] = (rng.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    c.loc[rng.choice(n, 300, replace=False), "avg_basket"] = np.nan
    c.loc[rng.choice(n, 150, replace=False), "city_tier"] = np.nan
    return c


def main():
    rng = np.random.default_rng(SEED)
    s = generate_sales(rng); c = generate_customers(rng)
    s.to_csv(RAW / "store_sales_daily.csv", index=False); c.to_csv(RAW / "customers.csv", index=False)
    print(f"store_sales_daily.csv {s.shape} | customers.csv {c.shape} | churn rate {c.churned.mean():.1%}")


if __name__ == "__main__":
    main()
