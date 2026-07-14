"""Feature engineering and training-table assembly.

For each (AOI, date) row the model sees:

  Target
    target_cf     -- cloud-free fraction on date + 1 (regression target)
    target_label  -- 1 if target_cf >= threshold else 0 (classification target)

  Predictors (FEATURE_COLUMNS, 59 in total)
    * 12 ERA5 variables x {mean,min,max,std} spatial statistics  (48)
    * persistence: 1/3/7-day lagged cloud-free fraction + 7-day rolling
      mean and std                                                (5)
    * seasonality: day-of-year sin/cos and month                  (3)
    * AOI metadata: integer code, latitude, longitude             (3)

This is the atmospheric-persistence design from the paper: the model is
conditioned on recent weather and recent observed clarity, and predicts
tomorrow.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .aoi import AOI, list_aois, get_aoi

CLOUD_FREE_THRESHOLD = 0.70

_VARS = ["tcc", "lcc", "mcc", "hcc", "t2m", "d2m", "msl", "u10", "v10", "tp", "tcwv", "cape"]
_STATS = ["mean", "min", "max", "std"]
WEATHER_FEATURES: List[str] = [f"era5_{v}_{s}" for v in _VARS for s in _STATS]
LAG_FEATURES = ["cf_lag_1", "cf_lag_3", "cf_lag_7", "cf_roll7_mean", "cf_roll7_std"]
SEASONAL_FEATURES = ["doy_sin", "doy_cos", "month"]
META_FEATURES = ["aoi_code", "lat", "lon"]
FEATURE_COLUMNS: List[str] = WEATHER_FEATURES + LAG_FEATURES + SEASONAL_FEATURES + META_FEATURES


def _doy_features(date: dt.date):
    doy = date.timetuple().tm_yday
    return (float(np.sin(2 * np.pi * doy / 365.0)),
            float(np.cos(2 * np.pi * doy / 365.0)),
            date.month)


def _aoi_code(aoi_id: str) -> int:
    return sorted(a.id for a in list_aois()).index(aoi_id)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()
    cf = df["cloud_free_fraction"]
    df["cf_lag_1"] = cf.shift(1)
    df["cf_lag_3"] = cf.shift(3)
    df["cf_lag_7"] = cf.shift(7)
    df["cf_roll7_mean"] = cf.shift(1).rolling(7, min_periods=2).mean()
    df["cf_roll7_std"] = cf.shift(1).rolling(7, min_periods=2).std()
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    s = df["date"].apply(_doy_features)
    df["doy_sin"] = [v[0] for v in s]
    df["doy_cos"] = [v[1] for v in s]
    df["month"] = [v[2] for v in s]
    return df


def add_meta_features(df: pd.DataFrame, aoi: AOI) -> pd.DataFrame:
    df = df.copy()
    df["aoi_id"] = aoi.id
    df["aoi_code"] = _aoi_code(aoi.id)
    df["lat"] = aoi.centroid[1]
    df["lon"] = aoi.centroid[0]
    return df


def build_training_dataset(per_aoi_dfs: Dict[str, pd.DataFrame],
                           threshold: Optional[float] = None) -> pd.DataFrame:
    """Assemble the full training table from per-AOI daily frames.

    Each input frame needs ``date``, ``cloud_free_fraction``, and the ERA5
    weather feature columns (as produced by ``synthetic.synthesize_history``
    or the real data layer).
    """
    threshold = CLOUD_FREE_THRESHOLD if threshold is None else threshold
    parts: List[pd.DataFrame] = []
    for aoi_id, df in per_aoi_dfs.items():
        if df is None or len(df) == 0:
            continue
        aoi = get_aoi(aoi_id)
        d = df.copy()
        d["date"] = pd.to_datetime(d["date"]).dt.date
        d = d.sort_values("date").reset_index(drop=True)
        d = add_lag_features(d)
        d = add_seasonal_features(d)
        d = add_meta_features(d, aoi)
        d["target_cf"] = d["cloud_free_fraction"].shift(-1)
        d["target_label"] = (d["target_cf"] >= threshold).astype("Int64")
        d = d.dropna(subset=["target_cf"]).reset_index(drop=True)
        parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["date", "aoi_id"] + FEATURE_COLUMNS + ["target_cf", "target_label"])
    return pd.concat(parts, ignore_index=True)
