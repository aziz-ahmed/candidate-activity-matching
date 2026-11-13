"""Pointwise learning-to-rank baselines.

sklearn has no native learning-to-rank model, so we use the standard
pointwise approximation: predict P(meaningful response) for a
candidate-job pair and rank jobs per candidate by that score.
Both rankers use the same model class and hyperparameters — the only
difference is the feature set, so the comparison isolates the value
of the activity signal.
"""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ACTIVITY_FEATURES, RESUME_FEATURES

RESUME_ONLY = RESUME_FEATURES
RESUME_PLUS_ACTIVITY = RESUME_FEATURES + ACTIVITY_FEATURES


def make_ranker(kind: str = "gbm"):
    if kind == "gbm":
        return HistGradientBoostingClassifier(random_state=42, max_iter=200, learning_rate=0.06)
    if kind == "logreg":
        return Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=42))]
        )
    raise ValueError(f"unknown ranker kind: {kind}")


def fit_ranker(train, feature_cols, kind: str = "gbm"):
    model = make_ranker(kind)
    model.fit(train[feature_cols], train["meaningful_response"].astype(int))
    return model


def predict_scores(model, df, feature_cols):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(df[feature_cols])[:, 1]
    return model.predict(df[feature_cols]).astype(float)
