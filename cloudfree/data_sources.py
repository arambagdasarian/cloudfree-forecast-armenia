"""Real data sources behind the paper, and a working keyless live fetcher.

Three sources are used in the full system:

  1. Sentinel-2 L2A cloud-free labels -- the ``cloudCover`` scene attribute from
     the Copernicus Data Space OData API (no raw tiles downloaded). Requires a
     free Copernicus Data Space account.
     Endpoint: https://catalogue.dataspace.copernicus.eu/odata/v1/Products

  2. ERA5 atmospheric predictors (training) -- ECMWF reanalysis via the
     Copernicus Climate Data Store (CDS) API. Requires a free CDS API key.
     Endpoint: https://cds.climate.copernicus.eu/api

  3. Open-Meteo forecast (live inference) -- free, no API key, ECMWF-based.
     Endpoint: https://api.open-meteo.com/v1/forecast

Acquiring (1) and (2) needs credentials and is documented here rather than
executed. (3) requires no key and is implemented below so the live Go/No-Go
forecast can be reproduced directly. The offline pipeline (``run_pipeline.py
--synthetic``) needs none of them.
"""

from __future__ import annotations

import datetime as dt
from typing import Dict
from urllib.parse import urlencode
from urllib.request import urlopen

from .aoi import AOI, get_aoi

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_open_meteo_next_day(aoi: AOI) -> Dict:
    """Fetch tomorrow's mean cloud cover for an AOI from Open-Meteo (no key).

    Returns a dict with the target date and the implied cloud-free fraction.
    This is the dominant model input; the full system additionally assembles the
    other ERA5-convention features and applies the trained model, but raw cloud
    cover is the leading signal and is sufficient for a first-order Go/No-Go.
    """
    lon, lat = aoi.centroid
    q = urlencode({
        "latitude": round(lat, 4),
        "longitude": round(lon, 4),
        "daily": "cloud_cover_mean",
        "timezone": "auto",
        "forecast_days": 2,
    })
    import json
    with urlopen(f"{OPEN_METEO_URL}?{q}", timeout=30) as r:
        j = json.load(r)
    times = j["daily"]["time"]
    cc = j["daily"]["cloud_cover_mean"]
    i = 1 if len(cc) > 1 else 0          # tomorrow
    cloud_free = max(0.0, min(1.0, 1.0 - cc[i] / 100.0))
    return {"aoi": aoi.id, "date": times[i], "cloud_cover_mean_pct": cc[i],
            "cloud_free_fraction": cloud_free}


def go_no_go(cloud_free_fraction: float, threshold: float = 0.70) -> str:
    return "GO" if cloud_free_fraction >= threshold else "NO-GO"


# ---- Documented (credentialed) sources: references, not executed here ----

def cdse_odata_cloudcover_query(aoi: AOI, start: dt.date, end: dt.date) -> str:
    """Return the Copernicus Data Space OData query URL for L2A cloudCover.

    The full system pages this year-by-year (to respect the 10,000-record skip
    limit), keeps the clearest scene per day, and converts to a cloud-free
    fraction via cf = (100 - cloudCover) / 100. Requires authentication.
    """
    lon_min, lat_min, lon_max, lat_max = aoi.bbox
    poly = (f"POLYGON(({lon_min} {lat_min},{lon_max} {lat_min},"
            f"{lon_max} {lat_max},{lon_min} {lat_max},{lon_min} {lat_min}))")
    flt = (
        "Collection/Name eq 'SENTINEL-2' and "
        "contains(Name,'MSIL2A') and "
        f"OData.CSC.Intersects(area=geography'SRID=4326;{poly}') and "
        f"ContentDate/Start gt {start.isoformat()}T00:00:00.000Z and "
        f"ContentDate/Start lt {end.isoformat()}T23:59:59.999Z"
    )
    return ("https://catalogue.dataspace.copernicus.eu/odata/v1/Products?"
            + urlencode({"$filter": flt, "$expand": "Attributes", "$top": 1000}))


ERA5_VARIABLES = [
    "total_cloud_cover", "low_cloud_cover", "medium_cloud_cover", "high_cloud_cover",
    "2m_temperature", "2m_dewpoint_temperature", "mean_sea_level_pressure",
    "10m_u_component_of_wind", "10m_v_component_of_wind",
    "total_precipitation", "total_column_water_vapour",
    "convective_available_potential_energy",
]
"""ERA5 single-level variables retrieved from the CDS API for training."""
