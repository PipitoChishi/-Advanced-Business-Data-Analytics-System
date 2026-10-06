"""Cleaning and feature engineering for store sales and customer data."""
import numpy as np
import pandas as pd
from .config import *
from .data_generation import HOLIDAYS, promo_calendar

STATIC = ["store_id", "region", "store_type", "size_sqft"]


def clean_sales(raw: pd.DataFrame):
    log = {"raw_rows": len(raw)}
    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    log["duplicates_removed"] = int(df.duplicated(["store_id", "date"]).sum())
    df = df.drop_duplicates(["store_id", "date"]).sort_values(["store_id", "date"]).reset_index(drop=True)

    log["negative_sales"] = int((df["sales"] < 0).sum()); df.loc[df["sales"] < 0, "sales"] = np.nan
    # Outliers: sales > 4x the store's rolling-median (centered, 15 days) -> keying errors
    med = df.groupby("store_id")["sales"].transform(lambda s: s.rolling(15, center=True, min_periods=5).median())
    out = df["sales"] > 4 * med
    log["outliers_removed"] = int(out.sum()); df.loc[out, "sales"] = np.nan
    log["missing_sales_imputed"] = int(df["sales"].isna().sum())
    df["sales"] = df.groupby("store_id")["sales"].transform(lambda s: s.interpolate(limit_direction="both"))
    log["missing_temp_imputed"] = int(df["temperature_c"].isna().sum())
    df["temperature_c"] = df.groupby("store_id")["temperature_c"].transform(lambda s: s.interpolate(limit_direction="both"))
    log["clean_rows"] = len(df)
    return df, log


def _holiday_distance(dates):
    h = np.sort(HOLIDAYS.values)
    idx = np.searchsorted(h, dates.values)
    nxt = np.where(idx < len(h), h[np.minimum(idx, len(h) - 1)], h[-1])
    return np.clip((nxt - dates.values).astype("timedelta64[D]").astype(int), 0, 30)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features usable HORIZON days ahead: calendar, known promo/holiday plan and lags >= HORIZON (no leakage)."""
    df = df.sort_values(["store_id", "date"]).reset_index(drop=True).copy()
    d = df["date"]
    df["dow"], df["month"], df["dom"] = d.dt.dayofweek, d.dt.month, d.dt.day
    df["week"] = d.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    df["days_to_holiday"] = _holiday_distance(d)
    df["doy_sin"] = np.sin(2 * np.pi * d.dt.dayofyear / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * d.dt.dayofyear / 365.25)
    g = df.groupby("store_id")["sales"]
    for lag in (HORIZON, 35, 42, 364):
        df[f"lag_{lag}"] = g.shift(lag)
    shifted = g.shift(HORIZON)
    df["roll_mean_7"] = shifted.groupby(df["store_id"]).transform(lambda s: s.rolling(7).mean())
    df["roll_mean_28"] = shifted.groupby(df["store_id"]).transform(lambda s: s.rolling(28).mean())
    df["roll_std_28"] = shifted.groupby(df["store_id"]).transform(lambda s: s.rolling(28).std())
    for c in ("store_id", "region", "store_type"):
        df[c] = df[c].astype("category")
    return df


FEATURES = ["store_id", "region", "store_type", "size_sqft", "promo", "is_holiday", "temperature_c", "dow", "month", "dom",
            "week", "is_weekend", "days_to_holiday", "doy_sin", "doy_cos", f"lag_{HORIZON}", "lag_35", "lag_42", "lag_364",
            "roll_mean_7", "roll_mean_28", "roll_std_28"]


def future_frame(clean: pd.DataFrame) -> pd.DataFrame:
    """Append HORIZON future days per store with known calendar info (promo plan, holidays, climatological temperature)."""
    last = clean["date"].max()
    fut_dates = pd.date_range(last + pd.Timedelta(days=1), periods=HORIZON)
    stores = clean[STATIC].drop_duplicates()
    fut = stores.merge(pd.DataFrame({"date": fut_dates}), how="cross")
    fut["promo"] = promo_calendar(fut["date"])      # chain-wide campaign calendar is known in advance
    fut["is_holiday"] = fut["date"].isin(HOLIDAYS).astype(int)
    clim = clean.groupby(clean["date"].dt.dayofyear)["temperature_c"].mean()
    fut["temperature_c"] = fut["date"].dt.dayofyear.map(clim).round(1)
    fut["sales"] = np.nan
    return pd.concat([clean[fut.columns.intersection(clean.columns)], fut], ignore_index=True)


def clean_customers(raw: pd.DataFrame):
    c = raw.copy()
    c["avg_basket"] = c["avg_basket"].fillna(c["avg_basket"].median())
    c["city_tier"] = c["city_tier"].fillna(c["city_tier"].mode()[0])
    c["spend_per_order"] = c["monetary_12m"] / c["frequency_12m"].replace(0, np.nan)
    c["spend_per_order"] = c["spend_per_order"].fillna(0)
    c["recency_bucket"] = pd.cut(c["recency_days"], [0, 30, 90, 180, 366], labels=["0-30", "31-90", "91-180", "181+"])
    return c
