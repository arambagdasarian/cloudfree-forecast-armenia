# Cloud-Free Imaging Probability Forecasting: reproducibility package

Code accompanying the paper *Cloud-Free Imaging Probability Forecasting for
Optical Earth-Observation Tasking over Armenia* (I. Avasapian, Teevial, 2026).

The system forecasts, one day ahead and per administrative region of Armenia,
the probability that an optical EO satellite pass will return a usable
(cloud-free) image, and issues a **Go / No-Go** tasking decision against a 70%
cloud-free threshold. Training labels come from Sentinel-2 L2A scene
`cloudCover`; predictors come from ERA5 reanalysis; live inference uses the
keyless Open-Meteo forecast (ERA5 lags real time by about five days).

This package reproduces the modelling pipeline and the live forecast. Because
the full Sentinel-2 + ERA5 record needs credentialed downloads, an offline
**synthetic** mode reproduces the entire pipeline end-to-end with no API keys.

## Layout

```
code/
  cloudfree/
    aoi.py            11 Armenian AOIs (Yerevan + 10 provinces)
    synthetic.py      keyless synthetic ERA5 + cloud-free generator
    features.py       59-feature engineering + training-table assembly
    model.py          gradient-boosted regressor + classifier, time-series CV
    data_sources.py   live Open-Meteo fetcher + documented Copernicus/CDS sources
  run_pipeline.py     end-to-end training -> outputs/metrics.json
  forecast_live.py    live next-day Go/No-Go from Open-Meteo (no key)
  make_figures.py     reproduce the feature-importance figure from metrics
  requirements.txt    LICENSE  CITATION.cff
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ . `lightgbm` is optional; if absent the model falls back to
scikit-learn's `HistGradientBoosting*` estimators (interchangeable here).

## Reproduce the pipeline (no credentials)

```bash
python run_pipeline.py --synthetic --years 3
python make_figures.py --metrics outputs/metrics.json
```

`run_pipeline.py` synthesises per-AOI daily histories, builds the 59-feature
table, trains both heads with `TimeSeriesSplit` cross-validation, and writes
`outputs/metrics.json` (CV MAE/RMSE, classifier AUC/Brier, and normalised
feature importance). The synthetic generator is deterministic, so runs are
reproducible. **Synthetic numbers are illustrative**, not the paper's results;
they exist to exercise the identical code path. The paper's figures are built
from the real trained-model metrics by `../make_figures.py`.

## Live forecast (no credentials)

```bash
python forecast_live.py                       # Yerevan
python forecast_live.py --aois yerevan lori syunik
python forecast_live.py --all
```

Fetches tomorrow's mean cloud cover from Open-Meteo for each AOI, converts it
to a cloud-free fraction, and prints the Go/No-Go decision, the same rule shown
on the public dashboard.

## Real data (credentialed, documented in `cloudfree/data_sources.py`)

| Source | Role | Access |
|---|---|---|
| Sentinel-2 L2A `cloudCover` (Copernicus Data Space OData) | training labels | free account |
| ERA5 single levels (Copernicus CDS API) | training predictors | free API key |
| Open-Meteo forecast | live inference | no key |

`data_sources.py` gives the exact OData query construction and the ERA5
variable list used by the full system. The feature schema (`features.py`) and
model (`model.py`) are identical whether fed synthetic or real frames.

## Feature schema (59)

- 48: 12 ERA5 variables × {mean, min, max, std} spatial statistics
- 5: persistence (1/3/7-day lagged cloud-free fraction, 7-day rolling mean/std)
- 3: seasonality (day-of-year sin/cos, month)
- 3: AOI metadata (integer code, latitude, longitude)

Target: next-day cloud-free fraction (regression) and its 0.70 exceedance
(classification).

## License

MIT (see `LICENSE`). If you use this work, please cite the paper (`CITATION.cff`).
