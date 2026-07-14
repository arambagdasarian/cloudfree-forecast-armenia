#!/usr/bin/env python3
"""End-to-end training pipeline for the cloud-free imaging forecaster.

Builds per-AOI daily histories, assembles the 59-feature training table,
trains the gradient-boosted regressor + classifier with time-series CV, and
writes the metrics to ``outputs/metrics.json``.

The ``--synthetic`` mode needs no credentials and runs in seconds:

    python run_pipeline.py --synthetic --years 3

To reproduce the paper's operational figures from these metrics:

    python make_figures.py --metrics outputs/metrics.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os

from cloudfree.aoi import list_aois, get_aoi
from cloudfree.features import build_training_dataset, CLOUD_FREE_THRESHOLD
from cloudfree.model import train_model, metrics_dict
from cloudfree.synthetic import synthesize_history


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--synthetic", action="store_true",
                   help="use the offline synthetic generator (no API keys)")
    p.add_argument("--years", type=int, default=3,
                   help="length of synthetic history per AOI (default 3)")
    p.add_argument("--aois", nargs="*", default=None,
                   help="AOI ids to include (default: all)")
    p.add_argument("--threshold", type=float, default=CLOUD_FREE_THRESHOLD,
                   help=f"Go/No-Go cloud-free threshold (default {CLOUD_FREE_THRESHOLD})")
    p.add_argument("--out", default="outputs/metrics.json",
                   help="path to write the metrics JSON")
    return p.parse_args()


def main():
    args = _parse_args()
    aois = ([get_aoi(a) for a in args.aois] if args.aois else list_aois())

    if not args.synthetic:
        raise SystemExit(
            "Real-data mode requires Copernicus Data Space and CDS credentials.\n"
            "See cloudfree/data_sources.py for the documented fetchers, or run\n"
            "with --synthetic for a keyless end-to-end demonstration.")

    # A fixed end date keeps synthetic runs deterministic and reproducible.
    end = dt.date(2024, 12, 31)
    start = end - dt.timedelta(days=int(round(args.years * 365.25)))
    print(f"[data] synthesising {len(aois)} AOIs, {start} -> {end} "
          f"({args.years}y each)")

    per_aoi = {a.id: synthesize_history(a, start, end) for a in aois}
    df = build_training_dataset(per_aoi, threshold=args.threshold)
    print(f"[data] training table: {len(df)} rows across {df['aoi_id'].nunique()} AOIs")

    model = train_model(df, threshold=args.threshold)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    payload = metrics_dict(model.metrics)
    payload["mode"] = "synthetic"
    payload["years_per_aoi"] = args.years
    payload["aois"] = [a.id for a in aois]
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[done] metrics written to {args.out}")

    top = sorted(model.metrics.feature_importance.items(),
                 key=lambda kv: kv[1], reverse=True)[:8]
    print("[top features] " + ", ".join(f"{k} {v:.3f}" for k, v in top))


if __name__ == "__main__":
    main()
