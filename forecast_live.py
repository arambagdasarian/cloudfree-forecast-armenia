#!/usr/bin/env python3
"""Live next-day Go/No-Go forecast from Open-Meteo (no API key required).

For each requested AOI this fetches tomorrow's mean cloud cover, converts it to
a cloud-free fraction, and applies the 70% imaging threshold, the same decision
rule shown on the public dashboard.

    python forecast_live.py                 # Yerevan
    python forecast_live.py --aois yerevan lori syunik
    python forecast_live.py --all

This is the operational inference path used when ERA5 reanalysis is not yet
available (it lags real time by about five days).
"""

from __future__ import annotations

import argparse

from cloudfree.aoi import list_aois, get_aoi
from cloudfree.data_sources import fetch_open_meteo_next_day, go_no_go
from cloudfree.features import CLOUD_FREE_THRESHOLD


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--aois", nargs="+", default=["yerevan"], help="AOI ids")
    g.add_argument("--all", action="store_true", help="forecast every AOI")
    p.add_argument("--threshold", type=float, default=CLOUD_FREE_THRESHOLD)
    return p.parse_args()


def main():
    args = _parse_args()
    aois = list_aois() if args.all else [get_aoi(a) for a in args.aois]

    print(f"{'AOI':<14}{'date':<12}{'cloud-free':>11}{'  decision'}")
    print("-" * 49)
    for aoi in aois:
        try:
            r = fetch_open_meteo_next_day(aoi)
        except Exception as e:  # network/HTTP errors should not abort the batch
            print(f"{aoi.name:<14}{'n/a':<12}{'n/a':>11}  ({e})")
            continue
        cf = r["cloud_free_fraction"]
        decision = go_no_go(cf, args.threshold)
        print(f"{aoi.name:<14}{r['date']:<12}{cf*100:>9.0f}%  {decision}")


if __name__ == "__main__":
    main()
