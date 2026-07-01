"""
STEP 1 -- Build parma_unified_2022_2025.csv from raw sources.

Per-file daily logic
  GROUND NO2 : for EACH station file -> parse datetimeUtc -> get the calendar
               date -> group by date -> take the MEAN of that day's readings.
               A day with fewer than MIN_GROUND_READINGS valid readings is set
               to BLANK (its mean would be biased -- this station reports a
               median of ~14 readings/day).
               Cittadella and Montebello stay as TWO separate columns.
  SATELLITE  : pool ALL valid Parma pixels of a date -> spatial MEAN
               (two passes on one date are weighted by pixel count).
  METEO      : the gap-free Open-Meteo daily CSV for all four years.

Missing days stay BLANK on a continuous Apr1-Sep30 spine; never filled with 0.
"""

import os
import glob
import numpy as np
import pandas as pd


# CONFIG

BASE = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis"
OUT_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\Task 1 Improved"
OUTPUT_CSV = os.path.join(OUT_DIR, "parma_unified_2022_2025.csv")

YEARS = [2022, 2023, 2024, 2025]

# Ground NO2: 4 Cittadella + 4 Montebello (note the on-disk 'cittedella' spelling)
GROUND_FILES = {
    "cittadella": {y: os.path.join(BASE, f"April to Sep {y} cittedella.csv") for y in YEARS},
    "montebello": {y: os.path.join(BASE, f"April to Sep {y} montebello.csv") for y in YEARS},
}

# S5P clipped folders (columns: latitude, longitude, no2_trop_column, qa_value).
# Set these to your real clipped folders. Missing folders are skipped (satellite
# stays blank for that year) so the build still completes.
S5P_CLIP_DIRS = {
    2022: r"180 data\parma_no2_clipped_2022",
    2023: r"180 data\parma_no2_clipped_2023",
    2024: r"copernicus 2024\parma_no2_clipped",
    2025: r"copernicus 2025\parma_no2_clipped",
}

# Open-Meteo daily meteo (the .xls that is actually a CSV)
METEO_CSV = os.path.join(BASE, "parma_meteo_openmeteo_2022_2025_daily.xls")

# Quality thresholds (a day below the threshold becomes BLANK)
MIN_GROUND_READINGS = 4    # reject a date only if it has FEWER than 4 readings
MIN_S5P_PIXELS = 3


# GROUND NO2  -> per file: detect date, count, average

def aggregate_ground(filepath, station, year):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[{station} {year}] file not found:\n  {filepath}")
    df = pd.read_csv(filepath)

    if "datetimeUtc" not in df.columns or "value" not in df.columns:
        raise ValueError(
            f"[{station} {year}] expected 'datetimeUtc' and 'value', "
            f"found: {list(df.columns)}")

    # keep only NO2 rows if a parameter column exists (safety)
    if "parameter" in df.columns:
        df = df[df["parameter"].astype(str).str.lower() == "no2"]

    dt = pd.to_datetime(df["datetimeUtc"], errors="coerce", utc=True)
    val = pd.to_numeric(df["value"], errors="coerce")
    n_bad = int(val.isna().sum())

    work = pd.DataFrame({"dt": dt, "val": val}).dropna(subset=["dt"])
    work = work[work["val"].notna() & (work["val"] >= 0)]
    work["date"] = work["dt"].dt.normalize().dt.tz_localize(None)   # UTC calendar day

    daily = work.groupby("date")["val"].agg(mean="mean", n="count").reset_index()

    thin = int((daily["n"] < MIN_GROUND_READINGS).sum())
    daily.loc[daily["n"] < MIN_GROUND_READINGS, "mean"] = np.nan
    kept = int(daily["mean"].notna().sum())
    print(f"  [{station} {year}] days={len(daily)}, readings/day median="
          f"{int(daily['n'].median())}, kept={kept} ({100*kept/max(len(daily),1):.0f}%), "
          f"blanked-thin={thin}, bad-rows={n_bad}")

    return daily[["date", "mean"]].rename(columns={"mean": f"{station}_no2_daily_mean"})


def load_all_ground(station):
    frames = [aggregate_ground(GROUND_FILES[station][y], station, y) for y in YEARS]
    out = pd.concat(frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"]).dt.normalize()
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


# SATELLITE S5P  -> pool pixels per date, spatial mean

def aggregate_s5p(clip_dir, year):
    if not os.path.isdir(clip_dir):
        print(f"  [S5P {year}] folder not set/found -> satellite left BLANK for {year}")
        return pd.DataFrame(columns=["date", "s5p_no2_mean", "s5p_pixel_count"])

    rows = []
    files = sorted(glob.glob(os.path.join(clip_dir, "*.csv")) +
                   glob.glob(os.path.join(clip_dir, "*.xls")))
    for f in files:
        if os.path.getsize(f) == 0:
            continue
        base = os.path.basename(f)
        try:
            date = pd.to_datetime(base.split("____")[1][:8])
        except (IndexError, ValueError):
            continue
        d = pd.read_csv(f)
        if "no2_trop_column" not in d.columns or len(d) == 0:
            continue
        rows.append(pd.DataFrame({"date": date, "no2": d["no2_trop_column"].values}))

    if not rows:
        print(f"  [S5P {year}] no usable clipped files")
        return pd.DataFrame(columns=["date", "s5p_no2_mean", "s5p_pixel_count"])

    allpix = pd.concat(rows, ignore_index=True)
    allpix["no2_umol"] = pd.to_numeric(allpix["no2"], errors="coerce") * 1e6  # mol/m2 -> umol/m2
    allpix = allpix[allpix["no2_umol"].notna()]

    daily = allpix.groupby("date").agg(
        s5p_no2_mean=("no2_umol", "mean"),
        s5p_pixel_count=("no2_umol", "count"),
    ).reset_index()

    thin = int((daily["s5p_pixel_count"] < MIN_S5P_PIXELS).sum())
    daily.loc[daily["s5p_pixel_count"] < MIN_S5P_PIXELS, "s5p_no2_mean"] = np.nan
    print(f"  [S5P {year}] dates={len(daily)}, pixels/day median="
          f"{int(daily['s5p_pixel_count'].median())}, {thin} low-pixel date(s) blanked")
    return daily


def load_all_s5p():
    frames = [aggregate_s5p(S5P_CLIP_DIRS[y], y) for y in YEARS]
    out = pd.concat(frames, ignore_index=True)
    if len(out):
        out["date"] = pd.to_datetime(out["date"]).dt.normalize()
        out = out.sort_values("date").reset_index(drop=True)
    return out



# METEO

def load_meteo():
    if not os.path.exists(METEO_CSV):
        raise FileNotFoundError(f"Meteo file not found:\n  {METEO_CSV}")
    m = pd.read_csv(METEO_CSV)            # .xls name but real CSV
    m["date"] = pd.to_datetime(m["date"]).dt.normalize()
    m["wind_speed_mean_kmh"] = pd.to_numeric(m["wind_mean_ms"], errors="coerce") * 3.6
    m = m.rename(columns={"pressure_mean_hpa": "pressure_hpa",
                          "rain_sum_mm": "precipitation_mm"})
    keep = ["date", "temp_mean_c", "temp_min_c", "temp_max_c", "humidity_mean_pct",
            "wind_speed_mean_kmh", "pressure_hpa", "precipitation_mm"]
    return m[[c for c in keep if c in m.columns]]


# BUILD

def build():
    print("\n[1/4] Ground NO2 (per file: detect date -> count -> average) ...")
    cit = load_all_ground("cittadella")
    mont = load_all_ground("montebello")

    print("\n[2/4] Satellite S5P ...")
    s5p = load_all_s5p()

    print("\n[3/4] Meteo ...")
    meteo = load_meteo()

    print("\n[4/4] Merging onto continuous Apr1-Sep30 spine ...")
    spine = [pd.DataFrame({"date": pd.date_range(f"{y}-04-01", f"{y}-09-30", freq="D")})
             for y in YEARS]
    unified = pd.concat(spine, ignore_index=True)

    unified = unified.merge(cit, on="date", how="left").merge(mont, on="date", how="left")
    if len(s5p):
        unified = unified.merge(s5p, on="date", how="left")
    else:
        unified["s5p_no2_mean"] = np.nan
        unified["s5p_pixel_count"] = np.nan
    unified = unified.merge(meteo, on="date", how="left")

    order = ["date", "cittadella_no2_daily_mean", "montebello_no2_daily_mean",
             "s5p_no2_mean", "s5p_pixel_count",
             "temp_mean_c", "temp_min_c", "temp_max_c", "humidity_mean_pct",
             "wind_speed_mean_kmh", "pressure_hpa", "precipitation_mm"]
    unified = unified[[c for c in order if c in unified.columns]]
    unified = unified.sort_values("date").reset_index(drop=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    unified.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "=" * 62)
    print("UNIFIED 2022-2025 BUILT")
    print(f"  Rows (dates): {len(unified)}   (expected 732)")
    print(f"  Saved: {OUTPUT_CSV}")
    print("  Blank values per column:")
    miss = unified.isna().sum()
    for c in unified.columns:
        if c != "date":
            print(f"    {c:28s} {int(miss[c]):4d} blank")
    print("=" * 62)


if __name__ == "__main__":
    build()
