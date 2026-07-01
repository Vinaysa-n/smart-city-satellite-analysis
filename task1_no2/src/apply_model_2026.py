"""
Apply the SAVED Task 1 model (task1_model.joblib, trained on 2022-2025) to a
fresh 2026 window (April 1 - June 10) and compare to the actual measured NO2.

This loads the model trained by train_and_save_model.py -- it does NOT retrain.
It is a NEXT-DAY predictor: each day's prediction uses the ACTUAL measured NO2
of previous days (lag/rolling features), so the real 2026 data is needed for the
window (it exists, the window is in the past). Data is gathered from mid-March so
the lag features have warm-up before April 1.

Prerequisites (download March 15 -> June 10, 2026):
  1. Ground NO2: Cittadella + Montebello ARPAE CSVs (datetimeUtc, value)
  2. Satellite : S5P clipped CSVs in a folder
Meteo is fetched automatically from Open-Meteo.
"""

import os
import glob
import time
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

OUT_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\Task 1 Improved"
MODEL_PATH = os.path.join(OUT_DIR, "task1_model.joblib")
PRED_OUT = os.path.join(OUT_DIR, "predictions_2026_apr_jun.csv")

# --- 2026 raw inputs downloaded (set these paths) ---
GROUND_2026 = {
    "cittadella": r"SET_PATH_2026_cittedella.csv",
    "montebello": r"SET_PATH_2026_montebello.csv",
}
S5P_CLIP_2026 = r"SET_PATH_2026_clipped_folder"

WARMUP_START = "2026-03-15"   # lag warm-up
EVAL_START = "2026-04-01"     # report metrics from here
EVAL_END = "2026-06-10"

LAT, LON = 44.80, 10.33
TIMEZONE = "Europe/Berlin"
MIN_GROUND_READINGS = 4
MIN_S5P_PIXELS = 3

HOLIDAYS = {
    "2026-04-06", "2026-04-25", "2026-05-01", "2026-06-02",  # Easter Mon 2026 = Apr 6
}


# ---------- 2026 raw -> daily ----------
def ground_daily(filepath, station):
    if not os.path.exists(filepath):
        print(f"  [{station} 2026] file not found -> target left blank")
        return pd.DataFrame(columns=["date", f"{station}_no2_daily_mean"])
    df = pd.read_csv(filepath)
    if "parameter" in df.columns:
        df = df[df["parameter"].astype(str).str.lower() == "no2"]
    dt = pd.to_datetime(df["datetimeUtc"], errors="coerce", utc=True)
    val = pd.to_numeric(df["value"], errors="coerce")
    w = pd.DataFrame({"dt": dt, "val": val}).dropna(subset=["dt"])
    w = w[w["val"].notna() & (w["val"] >= 0)]
    w["date"] = w["dt"].dt.normalize().dt.tz_localize(None)
    d = w.groupby("date")["val"].agg(mean="mean", n="count").reset_index()
    d.loc[d["n"] < MIN_GROUND_READINGS, "mean"] = np.nan
    print(f"  [{station} 2026] days={len(d)}, kept={int(d['mean'].notna().sum())}")
    return d[["date", "mean"]].rename(columns={"mean": f"{station}_no2_daily_mean"})


def s5p_daily(clip_dir):
    if not os.path.isdir(clip_dir):
        print("  [S5P 2026] folder not found -> satellite left blank")
        return pd.DataFrame(columns=["date", "s5p_no2_mean", "s5p_pixel_count"])
    rows = []
    for f in sorted(glob.glob(os.path.join(clip_dir, "*.csv")) +
                    glob.glob(os.path.join(clip_dir, "*.xls"))):
        if os.path.getsize(f) == 0:
            continue
        try:
            date = pd.to_datetime(os.path.basename(f).split("____")[1][:8])
        except (IndexError, ValueError):
            continue
        dd = pd.read_csv(f)
        if "no2_trop_column" in dd.columns and len(dd):
            rows.append(pd.DataFrame({"date": date, "no2": dd["no2_trop_column"].values}))
    if not rows:
        print("  [S5P 2026] no usable files")
        return pd.DataFrame(columns=["date", "s5p_no2_mean", "s5p_pixel_count"])
    allp = pd.concat(rows, ignore_index=True)
    allp["umol"] = pd.to_numeric(allp["no2"], errors="coerce") * 1e6
    allp = allp[allp["umol"].notna()]
    d = allp.groupby("date").agg(s5p_no2_mean=("umol", "mean"),
                                 s5p_pixel_count=("umol", "count")).reset_index()
    d.loc[d["s5p_pixel_count"] < MIN_S5P_PIXELS, "s5p_no2_mean"] = np.nan
    print(f"  [S5P 2026] dates={len(d)}")
    return d


def meteo_daily():
    print("  [meteo 2026] fetching from Open-Meteo ...")
    params = {
        "latitude": LAT, "longitude": LON,
        "start_date": WARMUP_START, "end_date": EVAL_END,
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,precipitation",
        "timezone": TIMEZONE, "wind_speed_unit": "ms",
        "temperature_unit": "celsius", "precipitation_unit": "mm",
    }
    for attempt in range(3):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive",
                             params=params, timeout=60)
            r.raise_for_status()
            h = r.json()["hourly"]
            break
        except Exception as e:
            print(f"    attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("Open-Meteo fetch failed")
    m = pd.DataFrame(h)
    m["time"] = pd.to_datetime(m["time"])
    m["date"] = m["time"].dt.normalize()
    return m.groupby("date").agg(
        temp_mean_c=("temperature_2m", "mean"),
        temp_min_c=("temperature_2m", "min"),
        temp_max_c=("temperature_2m", "max"),
        humidity_mean_pct=("relative_humidity_2m", "mean"),
        wind_speed_mean_kmh=("wind_speed_10m", lambda s: s.mean() * 3.6),
        pressure_hpa=("surface_pressure", "mean"),
        precipitation_mm=("precipitation", "sum"),
    ).reset_index()


def build_2026():
    print("\nBuilding 2026 rows ...")
    spine = pd.DataFrame({"date": pd.date_range(WARMUP_START, EVAL_END, freq="D")})
    cit = ground_daily(GROUND_2026["cittadella"], "cittadella")
    mont = ground_daily(GROUND_2026["montebello"], "montebello")
    s5p = s5p_daily(S5P_CLIP_2026)
    meteo = meteo_daily()
    for d in (cit, mont, s5p, meteo):
        if len(d):
            d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    out = spine.merge(cit, on="date", how="left").merge(mont, on="date", how="left")
    out = out.merge(s5p, on="date", how="left") if len(s5p) else out.assign(
        s5p_no2_mean=np.nan, s5p_pixel_count=np.nan)
    out = out.merge(meteo, on="date", how="left")
    return out


def engineer(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["year"] = df["date"].dt.year
    df["target_real"] = df["cittadella_no2_daily_mean"]
    df["target_filled"] = df.groupby("year")["target_real"].transform(
        lambda s: s.interpolate(limit=3))
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    dstr = df["date"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = dstr.isin(HOLIDAYS).astype(int)
    df["is_august"] = (df["month"] == 8).astype(int)
    if "s5p_no2_mean" in df.columns:
        df["s5p_no2_mean"] = df.groupby("year")["s5p_no2_mean"].transform(
            lambda s: s.interpolate(limit=7))
        df["s5p_no2_mean"] = df.groupby("year")["s5p_no2_mean"].transform(
            lambda s: s.fillna(s.median()))
    df["s5p_pixel_count"] = df.get("s5p_pixel_count", pd.Series(0, index=df.index)).fillna(0)

    def py(col, fn):
        return df.groupby("year")[col].transform(fn)
    for lag in [1, 2, 3, 7, 14]:
        df[f"target_lag_{lag}"] = py("target_filled", lambda x: x.shift(lag))
    df["s5p_lag_1"] = py("s5p_no2_mean", lambda x: x.shift(1))
    df["s5p_lag_7"] = py("s5p_no2_mean", lambda x: x.shift(7))
    df["target_roll_7"] = py("target_filled", lambda x: x.rolling(7, min_periods=4).mean())
    df["target_roll_14"] = py("target_filled", lambda x: x.rolling(14, min_periods=7).mean())
    df["s5p_roll_7"] = py("s5p_no2_mean", lambda x: x.rolling(7, min_periods=3).mean())
    return df


def report(name, y, p):
    print(f"  {name:14s} R2={r2_score(y,p):.3f}  "
          f"RMSE={np.sqrt(mean_squared_error(y,p)):.3f}  MAE={mean_absolute_error(y,p):.3f}")


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Saved model not found. Run train_and_save_model.py first.\n  {MODEL_PATH}")
    bundle = joblib.load(MODEL_PATH)
    feats = bundle["features"]
    print(f"Loaded model trained on {bundle['n_train']} days ({bundle['trained_on']}).")

    df = engineer(build_2026())
    test = df[(df["date"] >= EVAL_START) & (df["date"] <= EVAL_END)]
    test = test.dropna(subset=feats + ["target_real"])
    print(f"\n2026 test rows (Apr1-Jun10 with features + real target): {len(test)}")
    if len(test) == 0:
        print("No usable 2026 test days. Check the 2026 ground/satellite paths and data.")
        return

    Xte = test[feats].values
    yte = test["target_real"].values

    print("\n--- 2026 validation (model trained on 2022-2025) ---")
    out = test[["date", "target_real"]].rename(columns={"target_real": "actual_no2"}).copy()

    rf_pred = np.expm1(bundle["rf"].predict(Xte))
    report("RandomForest", yte, rf_pred)
    out["rf_predicted_no2"] = rf_pred.round(2)

    if "xgb" in bundle:
        xgb_pred = np.expm1(bundle["xgb"].predict(Xte))
        report("XGBoost", yte, xgb_pred)
        out["xgb_predicted_no2"] = xgb_pred.round(2)

    out.to_csv(PRED_OUT, index=False)
    print(f"\nSaved predicted vs actual -> {PRED_OUT}")
    print(f"2026 target std: {yte.std():.2f} ug/m3 "
          f"(spring varies more than the flat 2025 summer, so R2 here is more informative)")


if __name__ == "__main__":
    main()
