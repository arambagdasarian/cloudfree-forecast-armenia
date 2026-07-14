"""Offline synthetic data generator.

Produces physically plausible daily cloud-free-fraction and ERA5-like weather
series for any AOI so the full pipeline runs end-to-end without API keys or
large downloads. The generative process mirrors the real climatology:

  * daily total cloud cover = seasonal cycle (cloudier in winter) + AR(1)
    synoptic noise + an AOI climatological bias (wetter north, drier south);
  * cloud-free fraction ~= 1 - total cloud cover + small noise, clipped to [0,1];
  * remaining ERA5 variables are made seasonally and physically consistent with
    the cloud state (pressure anti-correlated with cloud, larger dewpoint
    depression on clear days, more water vapour when cloudy, and so on).

For each variable we emit the four spatial statistics (mean/min/max/std) that
the real pipeline computes over the AOI grid, so the synthetic data exercises
the same 48 weather features. The generator is deterministic per AOI (seeded by
AOI id). It is a stand-in for demonstration and analysis only, not a substitute
for the real Sentinel-2 + ERA5 record behind the paper's results.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np
import pandas as pd

from .aoi import AOI

# ERA5 variables and the spatial-statistic suffixes, matching features.build
VARS = ["tcc", "lcc", "mcc", "hcc", "t2m", "d2m", "msl", "u10", "v10", "tp", "tcwv", "cape"]
STATS = ["mean", "min", "max", "std"]


def _seed(aoi: AOI) -> int:
    return int(hashlib.md5(aoi.id.encode()).hexdigest()[:8], 16)


def _stats(mean: np.ndarray, spread: np.ndarray, rng, lo=None, hi=None):
    """Return (mean, min, max, std) columns from a daily mean and spatial spread."""
    std = np.abs(spread)
    mn = mean - 1.3 * std
    mx = mean + 1.3 * std
    if lo is not None:
        mean, mn, mx = np.clip(mean, lo, hi), np.clip(mn, lo, hi), np.clip(mx, lo, hi)
    return mean, mn, mx, std


def synthesize_history(aoi: AOI, start: dt.date, end: dt.date) -> pd.DataFrame:
    rng = np.random.default_rng(_seed(aoi))
    n = (end - start).days + 1
    dates = [start + dt.timedelta(days=i) for i in range(n)]
    doy = np.array([d.timetuple().tm_yday for d in dates])

    lon, lat = aoi.centroid
    lat_bias = (lat - 39.5) * 0.03                       # wetter to the north
    elev_proxy = max(0.0, lat - 40.0) * 0.02 + abs(lon - 45.0) * 0.01
    base_tcc = 0.40 + lat_bias + elev_proxy

    seasonal = 0.15 * np.sin(2 * np.pi * (doy - 30) / 365.0)   # winter peak
    eps = rng.normal(0.0, 0.18, size=n)
    ar = np.zeros(n)
    for i in range(1, n):
        ar[i] = 0.6 * ar[i - 1] + eps[i]                  # AR(1) synoptic noise

    tcc = np.clip(base_tcc + seasonal + ar, 0.02, 0.98)
    cf = np.clip(1.0 - tcc + rng.normal(0, 0.05, size=n), 0.0, 1.0)

    t2m_C = 12.0 - 0.6 * (lat - 39.5) - 4.0 * elev_proxy * 5
    t2m = t2m_C + 12.0 * np.sin(2 * np.pi * (doy - 110) / 365.0) + 273.15  # Kelvin
    d2m = t2m - (4.0 + 6.0 * (1.0 - tcc))
    msl = 101500.0 - 1500.0 * (tcc - 0.5)
    u10 = rng.normal(0.0, 2.5, n)
    v10 = rng.normal(0.0, 2.5, n)
    tp = np.clip(0.002 + 0.01 * tcc, 0, 0.05)
    tcwv = 12.0 + 8.0 * tcc
    cape = np.clip(50.0 + 250.0 * tcc, 0, 3000.0)
    lcc = np.clip(tcc * 0.6, 0, 1)
    mcc = np.clip(tcc * 0.4, 0, 1)
    hcc = np.clip(tcc * 0.3, 0, 1)

    daily = {
        "tcc": (tcc, 0.10 + 0.05 * rng.random(n)),
        "lcc": (lcc, 0.08 * np.ones(n)),
        "mcc": (mcc, 0.07 * np.ones(n)),
        "hcc": (hcc, 0.06 * np.ones(n)),
        "t2m": (t2m, 1.5 * np.ones(n)),
        "d2m": (d2m, 1.5 * np.ones(n)),
        "msl": (msl, 250.0 * np.ones(n)),
        "u10": (u10, 1.2 * np.ones(n)),
        "v10": (v10, 1.2 * np.ones(n)),
        "tp":  (tp, 0.003 * np.ones(n)),
        "tcwv": (tcwv, 1.5 * np.ones(n)),
        "cape": (cape, 60.0 * np.ones(n)),
    }
    clip = {"tcc": (0, 1), "lcc": (0, 1), "mcc": (0, 1), "hcc": (0, 1), "tp": (0, 0.06),
            "tcwv": (0, 60), "cape": (0, 4000)}

    out = pd.DataFrame({"date": dates, "cloud_free_fraction": cf})
    for v, (mean, spread) in daily.items():
        lo_hi = clip.get(v, (None, None))
        m, mn, mx, sd = _stats(mean, spread, rng, lo_hi[0], lo_hi[1])
        out[f"era5_{v}_mean"] = m
        out[f"era5_{v}_min"] = mn
        out[f"era5_{v}_max"] = mx
        out[f"era5_{v}_std"] = sd
    return out
