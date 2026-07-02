
"""
SAR multi-year visualisations  (Parma, Task 4)

Reads the results CSV produced by sentinel1a_multiyear.py and builds a clean
set of thesis-ready figures:

    1. sar_backscatter_separation.png  - urban vs vegetation VV, with delta-VV
    2. sar_polarization_vv_vh.png      - VV vs VH for urban & vegetation
    3. sar_separation_trend.png        - delta-VV across years (stability + S1C step)
    4. sar_classification_split.png    - urban/veg/water % (honestly labelled)
    5. sar_summary_table.png           - rendered results table
    6. sar_dashboard.png               - all panels in one figure

NOTE: these are SUMMARY figures from the per-year metrics. Spatial backscatter
maps are not possible here because the pipeline deletes the scene rasters after
processing - add an array-export step to the pipeline if you want maps.

"""

# ============================ CONFIG ========================================
CSV_PATH = (r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\sar_multiyear"
            r"\sentinel1a_multiyear_results.csv")
OUT_DIR  = (r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\sar_multiyear\figures")

# colours (consistent across every figure)
C_URBAN = "#C0392B"   # built-up
C_VEG   = "#27AE60"   # vegetation
C_WATER = "#2980B9"   # water / shadow
C_S1C   = "#E67E22"   # highlight for the cross-sensor (S1C) point
C_GREY  = "#7F8C8D"
# ============================================================================


import os
import sys
import subprocess


def _ensure(pkg, importname=None):
    try:
        __import__(importname or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])


for _p, _i in [("pandas", "pandas"), ("numpy", "numpy"), ("matplotlib", "matplotlib")]:
    _ensure(_p, _i)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # write PNGs without opening windows
import matplotlib.pyplot as plt


#  Styling

def apply_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 200,
        "font.size": 11,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "legend.frameon": False,
    })


def _years(df):
    return [str(int(y)) for y in df["year"]]


def _has_vh(df):
    return ("vh_urban_db" in df.columns
            and df["vh_urban_db"].notna().all())


def _annotate_bars(ax, bars, fmt="{:.1f}", dy=0.12, fs=9):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom", fontsize=fs)


#  Figure 1 - urban vs vegetation VV backscatter, with separation

def fig_separation(df, out_dir):
    yrs = _years(df)
    x = np.arange(len(yrs))
    w = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.2))

    b1 = ax.bar(x - w / 2, df["vv_urban_db"], w, label="Urban", color=C_URBAN)
    b2 = ax.bar(x + w / 2, df["vv_veg_db"], w, label="Vegetation", color=C_VEG)
    _annotate_bars(ax, b1)
    _annotate_bars(ax, b2)

    top = max(df["vv_urban_db"]) + 1.6
    for i, d in enumerate(df["dVV_db"]):
        ax.annotate(f"$\\Delta$ = {d:.1f} dB", (i, top), ha="center",
                    fontsize=9.5, fontweight="bold", color=C_GREY)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n({p})" for y, p in zip(yrs, df["platform"])])
    ax.set_ylabel("Mean VV backscatter (dB)")
    ax.set_ylim(min(df["vv_veg_db"]) - 2.5, top + 1.2)
    ax.set_title("Urban vs vegetation backscatter separation (2022\u20132025)")
    ax.legend(loc="lower left", ncol=2)
    fig.tight_layout()
    p = os.path.join(out_dir, "sar_backscatter_separation.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


#  Figure 2 - VV vs VH for urban and vegetation

def fig_polarization(df, out_dir):
    if not _has_vh(df):
        return None
    yrs = _years(df)
    x = np.arange(len(yrs))
    fig, ax = plt.subplots(figsize=(9, 5.2))

    ax.plot(x, df["vv_urban_db"], "-o", color=C_URBAN, lw=2, label="Urban VV")
    ax.plot(x, df["vh_urban_db"], "--o", color=C_URBAN, lw=1.8, alpha=.7, label="Urban VH")
    ax.plot(x, df["vv_veg_db"], "-s", color=C_VEG, lw=2, label="Vegetation VV")
    ax.plot(x, df["vh_veg_db"], "--s", color=C_VEG, lw=1.8, alpha=.7, label="Vegetation VH")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{y}\n({p})" for y, p in zip(yrs, df["platform"])])
    ax.set_ylabel("Backscatter (dB)")
    ax.set_title("Co- and cross-polarisation by surface (VV vs VH)")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    p = os.path.join(out_dir, "sar_polarization_vv_vh.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


#  Figure 3 - separation trend across years (+ cross-sensor note)

def fig_separation_trend(df, out_dir):
    yrs = _years(df)
    x = np.arange(len(yrs))
    d = df["dVV_db"].values
    fig, ax = plt.subplots(figsize=(9, 5.2))

    mean, std = d.mean(), d.std()
    ax.axhspan(mean - std, mean + std, color=C_GREY, alpha=0.12,
               label=f"mean {mean:.2f} \u00b1 {std:.2f} dB")
    ax.axhline(mean, color=C_GREY, lw=1, ls=":")

    # colour S1A vs S1C points so the sensor change is explicit
    is_s1c = df["platform"].str.upper().eq("S1C").values
    ax.plot(x, d, "-", color=C_GREY, lw=1.5, zorder=1)
    ax.scatter(x[~is_s1c], d[~is_s1c], s=120, color=C_URBAN, zorder=3,
               edgecolor="white", linewidth=1.5, label="Sentinel-1A")
    if is_s1c.any():
        ax.scatter(x[is_s1c], d[is_s1c], s=160, color=C_S1C, zorder=3, marker="D",
                   edgecolor="white", linewidth=1.5, label="Sentinel-1C")

    for i, v in enumerate(d):
        ax.annotate(f"{v:.2f}", (i, v + 0.05), ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(yrs)
    ax.set_ylabel("Urban \u2212 vegetation separation  $\\Delta$VV (dB)")
    ax.set_ylim(d.min() - 0.5, d.max() + 0.6)
    ax.set_title("Backscatter separation is stable across four years")
    ax.legend(loc="lower right")
    fig.tight_layout()
    p = os.path.join(out_dir, "sar_separation_trend.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


#  Figure 4 - classification split

def fig_classification(df, out_dir):
    yrs = _years(df)
    x = np.arange(len(yrs))
    fig, ax = plt.subplots(figsize=(9, 5.2))

    u, v, w = df["urban_pct"], df["veg_pct"], df["water_pct"]
    ax.bar(x, u, 0.55, label="Urban", color=C_URBAN)
    ax.bar(x, v, 0.55, bottom=u, label="Vegetation", color=C_VEG)
    ax.bar(x, w, 0.55, bottom=u + v, label="Water / shadow", color=C_WATER)

    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel("Coverage (%)"); ax.set_ylim(0, 100)
    ax.set_title("SAR land-cover split by year")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.text(0.5, 0.5,
            "split is fixed by the 75th / 15th percentile thresholds\n"
            "(constant by construction \u2014 see text)",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, color="white", style="italic")
    fig.tight_layout()
    p = os.path.join(out_dir, "sar_classification_split.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p



#  Figure 5 - rendered summary table

def fig_table(df, out_dir):
    cols = ["year", "date", "platform", "vv_urban_db", "vv_veg_db", "dVV_db",
            "urban_pct", "veg_pct", "water_pct"]
    nice = ["Year", "Date", "Sat", "VV urban", "VV veg", "\u0394VV",
            "Urban %", "Veg %", "Water %"]
    show = df[cols].copy()
    for c in ["vv_urban_db", "vv_veg_db", "dVV_db"]:
        show[c] = show[c].map(lambda v: f"{v:.2f}")
    for c in ["urban_pct", "veg_pct", "water_pct"]:
        show[c] = show[c].map(lambda v: f"{v:.1f}")
    show["year"] = show["year"].astype(int)

    fig, ax = plt.subplots(figsize=(10, 1.0 + 0.45 * len(show)))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=nice,
                   cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    for j in range(len(nice)):                       # header styling
        c = tbl[0, j]; c.set_facecolor("#34495E")
        c.set_text_props(color="white", fontweight="bold")
    for i in range(1, len(show) + 1):                # zebra rows
        if i % 2 == 0:
            for j in range(len(nice)):
                tbl[i, j].set_facecolor("#F4F6F7")
    ax.set_title("Multi-year SAR summary (Parma)", fontweight="bold", pad=14)
    p = os.path.join(out_dir, "sar_summary_table.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


#  Figure 6 - combined dashboard

def fig_dashboard(df, out_dir):
    yrs = _years(df)
    x = np.arange(len(yrs))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Sentinel-1 multi-year SAR over Parma (2022\u20132025)",
                 fontsize=15, fontweight="bold")

    # (a) separation bars
    ax = axes[0, 0]; w = 0.36
    ax.bar(x - w / 2, df["vv_urban_db"], w, label="Urban", color=C_URBAN)
    ax.bar(x + w / 2, df["vv_veg_db"], w, label="Vegetation", color=C_VEG)
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel("VV (dB)"); ax.set_ylim(min(df["vv_veg_db"]) - 2, max(df["vv_urban_db"]) + 1)
    ax.set_title("Urban vs vegetation VV"); ax.legend(ncol=2, loc="lower left")
    ax.grid(True, axis="y", alpha=.25)

    # (b) delta-VV trend
    ax = axes[0, 1]
    d = df["dVV_db"].values
    is_s1c = df["platform"].str.upper().eq("S1C").values
    ax.axhspan(d.mean() - d.std(), d.mean() + d.std(), color=C_GREY, alpha=.12)
    ax.plot(x, d, "-", color=C_GREY, lw=1.5)
    ax.scatter(x[~is_s1c], d[~is_s1c], s=90, color=C_URBAN, zorder=3, edgecolor="white")
    if is_s1c.any():
        ax.scatter(x[is_s1c], d[is_s1c], s=120, color=C_S1C, marker="D", zorder=3, edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel("$\\Delta$VV (dB)"); ax.set_title("Separation stability")
    ax.grid(True, axis="y", alpha=.25)

    # (c) polarisation
    ax = axes[1, 0]
    if _has_vh(df):
        ax.plot(x, df["vv_urban_db"], "-o", color=C_URBAN, label="Urban VV")
        ax.plot(x, df["vh_urban_db"], "--o", color=C_URBAN, alpha=.7, label="Urban VH")
        ax.plot(x, df["vv_veg_db"], "-s", color=C_VEG, label="Veg VV")
        ax.plot(x, df["vh_veg_db"], "--s", color=C_VEG, alpha=.7, label="Veg VH")
        ax.legend(fontsize=8, ncol=2)
    else:
        ax.text(0.5, 0.5, "VH not available", transform=ax.transAxes, ha="center")
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel("dB"); ax.set_title("VV vs VH"); ax.grid(True, axis="y", alpha=.25)

    # (d) classification
    ax = axes[1, 1]
    u, v, wat = df["urban_pct"], df["veg_pct"], df["water_pct"]
    ax.bar(x, u, 0.55, label="Urban", color=C_URBAN)
    ax.bar(x, v, 0.55, bottom=u, label="Vegetation", color=C_VEG)
    ax.bar(x, wat, 0.55, bottom=u + v, label="Water", color=C_WATER)
    ax.set_xticks(x); ax.set_xticklabels(yrs)
    ax.set_ylabel("%"); ax.set_ylim(0, 100); ax.set_title("Land-cover split (percentile-defined)")
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.07))

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(out_dir, "sar_dashboard.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


#  Main

def main():
    apply_style()
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: results CSV not found:\n  {CSV_PATH}\n"
              f"Run sentinel1a_multiyear.py first.")
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH).sort_values("year").reset_index(drop=True)
    print(f"Loaded {len(df)} years from {CSV_PATH}")

    made = [
        fig_separation(df, OUT_DIR),
        fig_polarization(df, OUT_DIR),
        fig_separation_trend(df, OUT_DIR),
        fig_classification(df, OUT_DIR),
        fig_table(df, OUT_DIR),
        fig_dashboard(df, OUT_DIR),
    ]
    print("\nFigures written to:", OUT_DIR)
    for p in made:
        if p:
            print("  -", os.path.basename(p))
    if not _has_vh(df):
        print("\n(VH columns missing/empty - polarisation figure skipped.)")


if __name__ == "__main__":
    main()
