"""Reproducibility package for

    "Cloud-Free Imaging Probability Forecasting for Optical Earth-Observation
     Tasking over Yerevan, Armenia" -- Isabella Avasapian, Teevial.

A self-contained reference implementation of the forecasting pipeline described
in the paper: area-of-interest definitions, an offline synthetic data generator,
the atmospheric-persistence feature design, and a gradient-boosted next-day
cloud-free-fraction model evaluated with time-ordered cross-validation.

The package runs end-to-end with no API keys or large downloads via the
synthetic generator. Pointers to the real data sources (Copernicus Data Space,
ERA5/CDS, Open-Meteo) are documented in ``data_sources.py``.
"""

__version__ = "1.0.0"

from .aoi import AOI, PROVINCES, list_aois, get_aoi
from .features import build_training_dataset, FEATURE_COLUMNS
from .model import train_model, ModelMetrics

__all__ = [
    "AOI",
    "PROVINCES",
    "list_aois",
    "get_aoi",
    "build_training_dataset",
    "FEATURE_COLUMNS",
    "train_model",
    "ModelMetrics",
]
