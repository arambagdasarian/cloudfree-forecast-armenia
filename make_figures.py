#!/usr/bin/env python3
"""Reproduce the feature-importance figure from a trained-model metrics JSON.

    python run_pipeline.py --synthetic            # writes outputs/metrics.json
    python make_figures.py --metrics outputs/metrics.json

Produces ``outputs/figures/fig_importance.pdf`` (vector, LaTeX-ready). This is
the same figure as in the paper; running it on the synthetic model lets a
reviewer confirm the cloud-cover-dominated importance structure end-to-end.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NAVY, TEAL, GREY = "#1f3a93", "#2a9d8f", "#9aa3b2"

LABELS = {
    "era5_tcc_mean": "Total cloud cover (mean)", "era5_tcc_std": "Total cloud cover (std)",
    "era5_tcc_min": "Total cloud cover (min)", "era5_tcc_max": "Total cloud cover (max)",
    "era5_lcc_mean": "Low cloud cover (mean)", "era5_lcc_std": "Low cloud cover (std)",
    "era5_mcc_mean": "Mid cloud cover (mean)", "era5_hcc_std": "High cloud cover (std)",
    "era5_msl_std": "Sea-level pressure (std)", "era5_msl_mean": "Sea-level pressure (mean)",
    "era5_tp_mean": "Total precipitation (mean)", "era5_tcwv_min": "Column water vapour (min)",
    "era5_cape_std": "CAPE (std)", "era5_t2m_std": "2 m temperature (std)",
    "era5_d2m_std": "2 m dewpoint (std)", "cf_lag_7": "Cloud-free, 7-day lag",
    "cf_lag_3": "Cloud-free, 3-day lag", "cf_lag_1": "Cloud-free, 1-day lag",
    "cf_roll7_mean": "Cloud-free, 7-day mean", "cf_roll7_std": "Cloud-free, 7-day std",
    "doy_sin": "Day-of-year (sin)", "doy_cos": "Day-of-year (cos)", "month": "Month",
    "lat": "Latitude", "lon": "Longitude", "aoi_code": "AOI code",
}


def pretty(k):
    return LABELS.get(k, k.replace("era5_", "").replace("_", " "))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--metrics", default="outputs/metrics.json")
    p.add_argument("--out", default="outputs/figures")
    p.add_argument("--top", type=int, default=12)
    args = p.parse_args()

    with open(args.metrics) as f:
        metrics = json.load(f)
    fi = metrics.get("feature_importance", {})
    if not fi:
        raise SystemExit("metrics JSON has no feature_importance; run run_pipeline.py first")

    top = sorted(fi.items(), key=lambda kv: kv[1], reverse=True)[:args.top][::-1]
    names = [pretty(k) for k, _ in top]
    vals = [v for _, v in top]
    cols = [NAVY if "cloud cover" in n.lower()
            else (TEAL if any(s in n.lower() for s in ("day-of-year", "cloud-free", "month"))
                  else GREY) for n in names]

    plt.rcParams.update({"font.family": "serif", "font.size": 10,
                         "axes.axisbelow": True, "savefig.bbox": "tight"})
    os.makedirs(args.out, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.barh(names, vals, color=cols, edgecolor="#222", linewidth=0.5, height=0.7)
    ax.set_xlabel("Gain-based feature importance")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6)
    ax.grid(axis="y", visible=False)
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=8)
    ax.set_xlim(0, max(vals) * 1.15)
    path = os.path.join(args.out, "fig_importance.pdf")
    fig.savefig(path)
    print("wrote", path)


if __name__ == "__main__":
    main()
