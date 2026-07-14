"""Gradient-boosted next-day cloud-free forecaster with time-ordered CV.

Two heads share the same features:
  * regression head  -- predicts the cloud-free fraction in [0, 1];
  * classification head -- probability that the fraction >= threshold.

Evaluation uses scikit-learn's ``TimeSeriesSplit`` over unique sorted dates so
the test set of every fold lies strictly after its training set in time, which
is the correct protocol for a forecasting problem.

The default learners are scikit-learn's HistGradientBoosting* estimators, which
are always available. If LightGBM is installed it is used instead, matching the
operational model in the paper; the two are interchangeable here.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS, CLOUD_FREE_THRESHOLD


@dataclass
class ModelMetrics:
    cv_mae: float
    cv_rmse: float
    cv_auc: Optional[float]
    cv_brier: Optional[float]
    n_train: int
    n_features: int
    threshold: float
    trained_at: str
    feature_importance: Dict[str, float] = field(default_factory=dict)


def _have_lightgbm() -> bool:
    try:
        import lightgbm as lgb  # noqa: F401
        lgb.LGBMRegressor(n_estimators=2).fit(np.zeros((4, 2)), np.zeros(4))
        return True
    except Exception:
        return False


def _make_regressor():
    if _have_lightgbm():
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                 subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
                                 reg_alpha=0.1, reg_lambda=0.2, random_state=42, n_jobs=-1, verbose=-1)
    return HistGradientBoostingRegressor(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                         min_samples_leaf=20, l2_regularization=0.2, random_state=42)


def _make_classifier():
    if _have_lightgbm():
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=63,
                                  subsample=0.9, subsample_freq=1, colsample_bytree=0.85,
                                  reg_alpha=0.1, reg_lambda=0.2, class_weight="balanced",
                                  random_state=42, n_jobs=-1, verbose=-1)
    return HistGradientBoostingClassifier(max_iter=600, learning_rate=0.03, max_leaf_nodes=63,
                                          min_samples_leaf=20, l2_regularization=0.2,
                                          class_weight="balanced", random_state=42)


@dataclass
class CloudFreeModel:
    regressor: Any
    classifier: Any
    feature_columns: List[str]
    threshold: float
    metrics: ModelMetrics

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = X[self.feature_columns]
        cf = np.clip(self.regressor.predict(X), 0.0, 1.0)
        proba = self.classifier.predict_proba(X)[:, 1]
        return cf, proba


def train_model(df: pd.DataFrame, threshold: Optional[float] = None,
                n_splits: int = 5, verbose: bool = True) -> CloudFreeModel:
    """Train both heads and report time-series cross-validated metrics."""
    threshold = CLOUD_FREE_THRESHOLD if threshold is None else threshold
    if df.empty:
        raise ValueError("Empty training dataframe")

    df = df.sort_values(["date", "aoi_id"]).reset_index(drop=True)
    X = df[FEATURE_COLUMNS]
    y_reg = df["target_cf"].astype(float).values
    y_clf = (y_reg >= threshold).astype(int)

    unique_dates = pd.Series(df["date"].unique()).sort_values().tolist()
    if len(unique_dates) < n_splits + 1:
        n_splits = max(2, len(unique_dates) // 2)
    fold_of_date = {d: i for i, d in enumerate(unique_dates)}
    fold_idx = np.array([fold_of_date[d] for d in df["date"].tolist()])

    tscv = TimeSeriesSplit(n_splits=n_splits)
    maes, rmses, aucs, briers = [], [], [], []
    for tr_d, va_d in tscv.split(unique_dates):
        tr = np.isin(fold_idx, tr_d)
        va = np.isin(fold_idx, va_d)
        if tr.sum() == 0 or va.sum() == 0:
            continue
        reg = _make_regressor(); reg.fit(X[tr], y_reg[tr])
        pred = np.clip(reg.predict(X[va]), 0, 1)
        maes.append(mean_absolute_error(y_reg[va], pred))
        rmses.append(np.sqrt(mean_squared_error(y_reg[va], pred)))
        if len(np.unique(y_clf[tr])) >= 2 and len(np.unique(y_clf[va])) >= 2:
            clf = _make_classifier(); clf.fit(X[tr], y_clf[tr])
            proba = clf.predict_proba(X[va])[:, 1]
            aucs.append(roc_auc_score(y_clf[va], proba))
            briers.append(brier_score_loss(y_clf[va], proba))

    final_reg = _make_regressor(); final_reg.fit(X, y_reg)
    final_clf = _make_classifier()
    if len(np.unique(y_clf)) >= 2:
        final_clf.fit(X, y_clf)

    # Feature importance: native if available, else permutation importance.
    importance: Dict[str, float] = {}
    if hasattr(final_reg, "feature_importances_"):
        for f, v in zip(FEATURE_COLUMNS, final_reg.feature_importances_):
            importance[f] = float(v)
        total = sum(importance.values()) or 1.0
        importance = {k: v / total for k, v in importance.items()}  # normalize to gain share
    else:
        r = permutation_importance(final_reg, X, y_reg, n_repeats=3, random_state=42, n_jobs=1)
        raw = {f: float(max(v, 0.0)) for f, v in zip(FEATURE_COLUMNS, r.importances_mean)}
        total = sum(raw.values()) or 1.0
        importance = {k: v / total for k, v in raw.items()}

    metrics = ModelMetrics(
        cv_mae=float(np.mean(maes)) if maes else float("nan"),
        cv_rmse=float(np.mean(rmses)) if rmses else float("nan"),
        cv_auc=float(np.mean(aucs)) if aucs else None,
        cv_brier=float(np.mean(briers)) if briers else None,
        n_train=int(len(df)),
        n_features=len(FEATURE_COLUMNS),
        threshold=float(threshold),
        trained_at=dt.datetime.utcnow().isoformat() + "Z",
        feature_importance=importance,
    )
    if verbose:
        print(f"[train] n={metrics.n_train}  features={metrics.n_features}  "
              f"MAE={metrics.cv_mae:.4f}  RMSE={metrics.cv_rmse:.4f}  "
              f"AUC={metrics.cv_auc if metrics.cv_auc is None else round(metrics.cv_auc,4)}  "
              f"Brier={metrics.cv_brier if metrics.cv_brier is None else round(metrics.cv_brier,4)}")
    return CloudFreeModel(final_reg, final_clf, list(FEATURE_COLUMNS), float(threshold), metrics)


def metrics_dict(m: ModelMetrics) -> dict:
    return asdict(m)
