"""28-day-ahead store sales forecasting: baseline vs Ridge vs tuned Gradient Boosting."""
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import *
from .preprocessing import FEATURES, build_features, future_frame

CAT = ["store_id", "region", "store_type"]


def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": mean_squared_error(y, p) ** .5,
            "MAPE_%": float(np.mean(np.abs((y - p) / y)) * 100), "R2": r2_score(y, p)}


def make_ridge():
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CAT)],
                            remainder=StandardScaler())
    return TransformedTargetRegressor(Pipeline([("pre", pre), ("m", Ridge(alpha=1.0))]), func=np.log1p, inverse_func=np.expm1)


def make_gbm():
    return TransformedTargetRegressor(HistGradientBoostingRegressor(categorical_features="from_dtype", random_state=SEED),
                                      func=np.log1p, inverse_func=np.expm1)


def run(clean: pd.DataFrame):
    feat = build_features(clean).dropna(subset=["lag_364", "roll_std_28"]).reset_index(drop=True)
    train, test = feat[feat["date"] < TEST_START], feat[feat["date"] >= TEST_START]
    Xtr, ytr, Xte, yte = train[FEATURES], train["sales"], test[FEATURES], test["sales"]
    res = {}

    res["Seasonal naive (same weekday, 4 wks ago)"] = (metrics(yte, test[f"lag_{HORIZON}"]), test[f"lag_{HORIZON}"].values)
    ridge = make_ridge().fit(Xtr, ytr); p = ridge.predict(Xte); res["Ridge regression"] = (metrics(yte, p), p)

    grid = GridSearchCV(make_gbm(), {"regressor__learning_rate": [.05, .1], "regressor__max_iter": [300, 600],
                                     "regressor__max_leaf_nodes": [31]}, cv=TimeSeriesSplit(3),
                        scoring="neg_mean_absolute_error", n_jobs=-1).fit(Xtr, ytr)
    gbm = grid.best_estimator_; p = gbm.predict(Xte); res["Gradient boosting (tuned)"] = (metrics(yte, p), p)

    table = pd.DataFrame({k: v[0] for k, v in res.items()}).T.round(3)
    best = table["MAE"].idxmin()
    pred = test[["store_id", "date", "sales"]].copy(); pred["pred_gbm"] = res["Gradient boosting (tuned)"][1]
    pred["store_id"] = pred["store_id"].astype(str)

    imp = permutation_importance(gbm, Xte.sample(6000, random_state=SEED), yte.loc[Xte.sample(6000, random_state=SEED).index],
                                 n_repeats=3, random_state=SEED, scoring="neg_mean_absolute_error", n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

    # refit on ALL history, then forecast the next HORIZON days
    full = feat
    final = make_gbm().set_params(**grid.best_params_).fit(full[FEATURES], full["sales"])
    joblib.dump({"model": final, "features": FEATURES, "horizon": HORIZON}, MODELS / "sales_forecaster.joblib")
    fut_all = build_features(future_frame(clean))
    fut = fut_all[fut_all["date"] > clean["date"].max()].copy()
    fut["forecast"] = final.predict(fut[FEATURES]).round(0)
    fut["store_id"] = fut["store_id"].astype(str)
    return dict(table=table, best=best, pred=pred, importance=importance, forecast=fut[["store_id", "date", "forecast"]],
                best_params=grid.best_params_)
