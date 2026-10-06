"""Customer churn prediction + behavioural segmentation."""
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .config import *

NUM = ["tenure_months", "recency_days", "frequency_12m", "monetary_12m", "avg_basket", "n_categories", "promo_share",
       "complaints_12m", "loyalty_member", "spend_per_order"]
CATS = ["channel", "city_tier"]


def churn_model(c: pd.DataFrame):
    X, y = c[NUM + CATS], c["churned"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=.25, stratify=y, random_state=SEED)
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATS), ("num", StandardScaler(), NUM)])
    logit = Pipeline([("pre", clone(pre)), ("m", LogisticRegression(max_iter=1000))]).fit(Xtr, ytr)
    gbm = Pipeline([("pre", clone(pre)), ("m", HistGradientBoostingClassifier(learning_rate=.06, max_iter=250, max_depth=4, random_state=SEED))]).fit(Xtr, ytr)
    rows = {}
    for name, m in (("Logistic regression", logit), ("Gradient boosting", gbm)):
        pr = m.predict_proba(Xte)[:, 1]; pred = (pr >= .35).astype(int)
        rows[name] = {"ROC-AUC": roc_auc_score(yte, pr), "Precision": precision_score(yte, pred),
                      "Recall": recall_score(yte, pred), "F1": f1_score(yte, pred)}
    table = pd.DataFrame(rows).T.round(3)
    best = table["ROC-AUC"].idxmax()
    model = {"Logistic regression": logit, "Gradient boosting": gbm}[best]      # production model = best AUC
    pr = model.predict_proba(Xte)[:, 1]
    dec = pd.DataFrame({"p": pr, "y": yte.values}).sort_values("p", ascending=False)
    top_dec = dec.head(len(dec) // 10)["y"].mean() / dec["y"].mean()            # lift in top decile
    imp = permutation_importance(model, Xte, yte, n_repeats=5, random_state=SEED, scoring="roc_auc", n_jobs=-1)
    importance = pd.Series(imp.importances_mean, index=NUM + CATS).sort_values(ascending=False)
    cm = confusion_matrix(yte, (pr >= .35).astype(int))
    joblib.dump(model, MODELS / "churn_model.joblib")
    c = c.copy(); c["churn_prob"] = model.predict_proba(X)[:, 1]
    c["risk"] = pd.cut(c["churn_prob"], [-.01, .25, .5, 1.0], labels=["Low", "Medium", "High"])
    return dict(table=table, best=best, lift=top_dec, importance=importance, cm=cm, scored=c, test_n=len(yte))


def segment(c: pd.DataFrame):
    cols = ["recency_days", "frequency_12m", "monetary_12m", "avg_basket", "promo_share"]
    Z = StandardScaler().fit_transform(np.log1p(c[cols]))
    sil = {}
    for k in range(3, 7):
        km = KMeans(k, n_init=5, random_state=SEED).fit(Z)
        from sklearn.metrics import silhouette_score
        sil[k] = silhouette_score(Z, km.labels_, sample_size=4000, random_state=SEED)
    k = max(sil, key=sil.get)
    labels = KMeans(k, n_init=10, random_state=SEED).fit_predict(Z)
    c = c.copy(); c["cluster"] = labels
    prof = c.groupby("cluster").agg(customers=("customer_id", "size"), recency=("recency_days", "mean"),
                                    frequency=("frequency_12m", "mean"), monetary=("monetary_12m", "mean"),
                                    avg_basket=("avg_basket", "mean"), promo_share=("promo_share", "mean"),
                                    churn_rate=("churned", "mean"))
    # name clusters by value rank (monetary) and recency
    order = prof["monetary"].rank(ascending=False).astype(int)
    names = {}
    for cl, r in order.items():
        if r == 1: names[cl] = "High-value regulars"
        elif prof.loc[cl, "recency"] == prof["recency"].max(): names[cl] = "Lapsing / dormant"
        elif r == len(order): names[cl] = "Low-spend occasional"
        else: names[cl] = f"Mid-value segment {cl}"
    prof["segment"] = prof.index.map(names)
    prof["revenue_share_%"] = prof["monetary"] * prof["customers"] / (prof["monetary"] * prof["customers"]).sum() * 100
    c["segment"] = c["cluster"].map(names)
    return dict(profile=prof.set_index("segment").round(2), k=k, silhouette=sil, labelled=c)
