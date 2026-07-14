"""Areas of interest (AOIs) over Armenia.

Eleven administrative units (ten provinces plus Yerevan) defined by a centroid
and a generous bounding box in WGS84 (EPSG:4326) lon/lat. The single-AOI study
in the paper uses ``yerevan``; the operational model trains across all AOIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

ARMENIA_BBOX: Tuple[float, float, float, float] = (43.45, 38.84, 46.63, 41.30)


@dataclass(frozen=True)
class AOI:
    id: str
    name: str
    name_hy: str
    centroid: Tuple[float, float]  # (lon, lat)
    bbox: Tuple[float, float, float, float]  # (lon_min, lat_min, lon_max, lat_max)
    description: str = ""


def _b(lon: float, lat: float, dlon: float = 0.30, dlat: float = 0.20):
    return (lon - dlon, lat - dlat, lon + dlon, lat + dlat)


PROVINCES: Dict[str, AOI] = {
    "yerevan": AOI("yerevan", "Yerevan", "Երևան", (44.5152, 40.1872),
                   _b(44.5152, 40.1872, 0.20, 0.15), "Capital; primary tasking target."),
    "aragatsotn": AOI("aragatsotn", "Aragatsotn", "Արագածոտն", (44.30, 40.45),
                      _b(44.30, 40.45, 0.30, 0.25), "Mt. Aragats massif; orographic cloud."),
    "ararat": AOI("ararat", "Ararat", "Արարատ", (44.71, 39.95),
                  _b(44.71, 39.95, 0.25, 0.20), "Ararat plain; agricultural belt."),
    "armavir": AOI("armavir", "Armavir", "Արմավիր", (44.04, 40.16),
                   _b(44.04, 40.16, 0.25, 0.18), "Western lowland."),
    "gegharkunik": AOI("gegharkunik", "Gegharkunik", "Գեղարքունիք", (45.35, 40.35),
                       _b(45.35, 40.35, 0.40, 0.30), "Lake Sevan basin."),
    "kotayk": AOI("kotayk", "Kotayk", "Կոտայք", (44.72, 40.39),
                  _b(44.72, 40.39, 0.25, 0.20), "Hrazdan corridor."),
    "lori": AOI("lori", "Lori", "Լոռի", (44.50, 41.00),
                _b(44.50, 41.00, 0.35, 0.25), "Northern highlands; wettest region."),
    "shirak": AOI("shirak", "Shirak", "Շիրակ", (43.84, 40.79),
                  _b(43.84, 40.79, 0.30, 0.25), "Northwestern plateau (Gyumri)."),
    "syunik": AOI("syunik", "Syunik", "Սյունիք", (46.20, 39.50),
                  _b(46.20, 39.50, 0.40, 0.40), "Southern mountains."),
    "tavush": AOI("tavush", "Tavush", "Տավուշ", (45.20, 40.95),
                  _b(45.20, 40.95, 0.40, 0.25), "Northeastern forested mountains."),
    "vayots_dzor": AOI("vayots_dzor", "Vayots Dzor", "Վայոց ձոր", (45.40, 39.75),
                       _b(45.40, 39.75, 0.35, 0.25), "Semi-arid south-central region."),
}


def list_aois() -> List[AOI]:
    return list(PROVINCES.values())


def get_aoi(aoi_id: str) -> AOI:
    if aoi_id not in PROVINCES:
        raise KeyError(f"Unknown AOI '{aoi_id}'. Known: {sorted(PROVINCES)}")
    return PROVINCES[aoi_id]
