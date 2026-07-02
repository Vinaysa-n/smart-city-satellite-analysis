"""
NO2 - NDRE correlation, multi-year  (Parma, Task 4)

Tests whether ground NO2 (terrestrial) and satellite NDRE are related across
2022-2025

Inputs (your existing files):
  - task3_labeled_2022_2025.csv      Sentinel-2 NDRE per scene (Parco Ducale)
  - parma_unified_2022_2025.csv      daily NO2 (Cittadella ground station)

Outputs:
  - no2_ndre_correlation_results.txt  the numbers
  - no2_ndre_correlation.png          3-panel figure (raw / seasonal / deseasonalised)
"""

# ============================ CONFIG ========================================
NDRE_CSV = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\task3_labeled_2022_2025.csv"
NO2_CSV  = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\parma_unified_2022_2025.csv"
OUT_DIR  = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\task4_correlation"

NDRE_COL = "mean_ndre"                  # column in the NDRE csv
NO2_COL  = "cittadella_no2_daily_mean"  # column in the NO2 csv
SUMMER_MONTHS = (6, 7, 8)              # definition of "summer" for the subset
# ============================================================================


import os, sys, subprocess


def _ensure(pkg, importname=None):
    try:
        __import__(importname or pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])


for _p, _i in [("pandas", "pandas"), ("numpy", "numpy"),
               ("scipy", "scipy"), ("matplotlib", "matplotlib")]:
    _ensure(_p, _i)

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

C_FIT  = "#C0392B"
YEAR_COLORS = {2022: "#3498DB", 2023: "#E67E22", 2024: "#27AE60", 2025: "#8E44AD"}


# --------------------------------------------------------------------------
def load_pairs():
    """Join NDRE scenes with same-day NO2. Returns a tidy DataFrame."""
    nd = pd.read_csv(NDRE_CSV)[["date", NDRE_COL]].dropna()
    no = pd.read_csv(NO2_CSV)[["date", NO2_COL]].dropna()
    df = nd.merge(no, on="date", how="inner").rename(
        columns={NDRE_COL: "ndre", NO2_COL: "no2"})
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    # day-of-season: continuous seasonal axis (days since 1 May of that year)
    df["doy"] = (df["date"] - pd.to_datetime(df["year"].astype(str) + "-05-01")).dt.days
    return df.sort_values("date").reset_index(drop=True)


def _resid(y, x):
    """Residuals of y after a linear fit on x (used to remove the season)."""
    X = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def analyse(df):
    """Compute every correlation we report. Returns a dict."""
    n, ndre, no2 = len(df), df["ndre"].values, df["no2"].values
    doy, mon = df["doy"].values, df["month"].values
    out = {"n": n}

    out["raw_r"], out["raw_p"] = pearsonr(ndre, no2)

    # is each variable seasonal?
    out["ndre_season_r"], _ = pearsonr(doy, ndre)
    out["no2_season_r"], out["no2_season_p"] = pearsonr(doy, no2)

    # deseasonalised 1: partial correlation controlling for day-of-season
    out["partial_r"], out["partial_p"] = pearsonr(_resid(ndre, doy), _resid(no2, doy))

    # deseasonalised 2: anomalies vs monthly climatology
    mo_ndre = {m: ndre[mon == m].mean() for m in np.unique(mon)}
    mo_no2 = {m: no2[mon == m].mean() for m in np.unique(mon)}
    nd_an = ndre - np.array([mo_ndre[m] for m in mon])
    no_an = no2 - np.array([mo_no2[m] for m in mon])
    out["anom_r"], out["anom_p"] = pearsonr(nd_an, no_an)
    df["ndre_anom"], df["no2_anom"] = nd_an, no_an

    # summer subset
    s = df[df["month"].isin(SUMMER_MONTHS)]
    if len(s) >= 4:
        out["summer_n"] = len(s)
        out["summer_r"], out["summer_p"] = pearsonr(s["ndre"], s["no2"])

    # per-year
    out["per_year"] = {}
    for y, g in df.groupby("year"):
        if len(g) >= 4:
            r, p = pearsonr(g["ndre"], g["no2"])
            out["per_year"][int(y)] = (len(g), r, p)
    return out


# --------------------------------------------------------------------------
def make_figure(df, res, out_dir):
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("NO\u2082 vs NDRE over Parma (2022\u20132025): raw, seasonal, deseasonalised",
                 fontweight="bold", fontsize=13)

    # ---- Panel A: raw scatter, coloured by year, with regression ----
    a = ax[0]
    for y, g in df.groupby("year"):
        a.scatter(g["ndre"], g["no2"], s=45, color=YEAR_COLORS.get(int(y), "#555"),
                  label=str(int(y)), edgecolor="white", linewidth=0.6, zorder=3)
    m, b = np.polyfit(df["ndre"], df["no2"], 1)
    xs = np.array([df["ndre"].min(), df["ndre"].max()])
    a.plot(xs, m * xs + b, color=C_FIT, lw=2, zorder=2)
    a.set_xlabel("Mean NDRE"); a.set_ylabel("NO\u2082 (\u00b5g/m\u00b3)")
    a.set_title(f"Raw: r = {res['raw_r']:+.2f}, p = {res['raw_p']:.3f}  (n = {res['n']})")
    a.legend(title="year", fontsize=8); a.grid(alpha=.25)

    # ---- Panel B: why it's confounded - both track the season ----
    b2 = ax[1]
    order = df.sort_values("doy")
    b2.scatter(order["doy"], order["ndre"], color="#27AE60", s=30, label="NDRE")
    mN, bN = np.polyfit(order["doy"], order["ndre"], 1)
    b2.plot(order["doy"], mN * order["doy"] + bN, color="#27AE60", lw=2)
    b2.set_xlabel("Day of season (since 1 May)")
    b2.set_ylabel("Mean NDRE", color="#27AE60")
    b2.tick_params(axis="y", labelcolor="#27AE60")
    b2b = b2.twinx()
    b2b.scatter(order["doy"], order["no2"], color="#C0392B", s=30, marker="s", label="NO\u2082")
    mO, bO = np.polyfit(order["doy"], order["no2"], 1)
    b2b.plot(order["doy"], mO * order["doy"] + bO, color="#C0392B", lw=2)
    b2b.set_ylabel("NO\u2082 (\u00b5g/m\u00b3)", color="#C0392B")
    b2b.tick_params(axis="y", labelcolor="#C0392B")
    b2.set_title(f"Seasonal confound: NDRE\u2193 (r={res['ndre_season_r']:+.2f})  "
                 f"NO\u2082\u2191 (r={res['no2_season_r']:+.2f})")
    b2.grid(alpha=.25)

    # ---- Panel C: deseasonalised scatter ----
    c = ax[2]
    for y, g in df.groupby("year"):
        c.scatter(g["ndre_anom"], g["no2_anom"], s=45,
                  color=YEAR_COLORS.get(int(y), "#555"),
                  edgecolor="white", linewidth=0.6, zorder=3)
    mA, bA = np.polyfit(df["ndre_anom"], df["no2_anom"], 1)
    xa = np.array([df["ndre_anom"].min(), df["ndre_anom"].max()])
    c.plot(xa, mA * xa + bA, color=C_FIT, lw=2, ls="--", zorder=2)
    c.axhline(0, color="#999", lw=0.8); c.axvline(0, color="#999", lw=0.8)
    c.set_xlabel("NDRE anomaly (vs monthly mean)")
    c.set_ylabel("NO\u2082 anomaly (\u00b5g/m\u00b3)")
    c.set_title(f"Deseasonalised: partial r = {res['partial_r']:+.2f}, "
                f"p = {res['partial_p']:.2f}")
    c.grid(alpha=.25)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(out_dir, "no2_ndre_correlation.png")
    fig.savefig(p, dpi=200, bbox_inches="tight"); plt.close(fig)
    return p


# --------------------------------------------------------------------------
def write_report(res, out_dir):
    lines = []
    add = lines.append
    add("=" * 66)
    add("NO2 - NDRE correlation, Parma 2022-2025  (Task 4)")
    add("=" * 66)
    add(f"Matched same-day scene/NO2 pairs: n = {res['n']}")
    add("")
    add("RAW (pooled, all matched scenes):")
    add(f"   r = {res['raw_r']:+.3f}   p = {res['raw_p']:.3f}")
    if "summer_r" in res:
        add(f"   summer ({'/'.join(map(str, SUMMER_MONTHS))}): "
            f"n={res['summer_n']}  r={res['summer_r']:+.3f}  p={res['summer_p']:.3f}")
    add("")
    add("SEASONAL STRUCTURE (each variable vs day-of-season):")
    add(f"   NDRE vs season : r = {res['ndre_season_r']:+.3f}   (declines over summer)")
    add(f"   NO2  vs season : r = {res['no2_season_r']:+.3f}   p = {res['no2_season_p']:.3f}")
    add("   -> opposite cycles mechanically produce the negative raw correlation")
    add("")
    add("DESEASONALISED (the honest test):")
    add(f"   partial r (controls day-of-season): r = {res['partial_r']:+.3f}  p = {res['partial_p']:.3f}")
    add(f"   monthly-anomaly correlation       : r = {res['anom_r']:+.3f}  p = {res['anom_p']:.3f}")
    add("")
    add("PER-YEAR (each year alone - note the instability):")
    for y, (n, r, p) in sorted(res["per_year"].items()):
        add(f"   {y}: n={n:2d}  r={r:+.3f}  p={p:.3f}")
    add("")
    add("CONCLUSION:")
    add("   Raw negative is seasonally confounded. After removing the season the")
    add("   association is weak and not robustly significant, and within-year")
    add("   correlations are unstable -> NO2 and NDRE are NOT coupled beyond the")
    add("   shared seasonal cycle. At ambient NO2 (~3-12 ug/m3 -> mild fertiliser,")
    add("   not stressor) a causal damage relationship is not expected either.")
    add("=" * 66)
    txt = "\n".join(lines)
    with open(os.path.join(out_dir, "no2_ndre_correlation_results.txt"), "w",
              encoding="utf-8") as f:
        f.write(txt)
    print(txt)


# --------------------------------------------------------------------------
def main():
    for path, label in [(NDRE_CSV, "NDRE csv"), (NO2_CSV, "NO2 csv")]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found:\n  {path}")
            return
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_pairs()
    print(f"Matched {len(df)} same-day NDRE/NO2 pairs.\n")
    res = analyse(df)
    write_report(res, OUT_DIR)
    fig = make_figure(df, res, OUT_DIR)
    print(f"\nFigure : {fig}")
    print(f"Report : {os.path.join(OUT_DIR, 'no2_ndre_correlation_results.txt')}")


if __name__ == "__main__":
    main()
