"""
Train the FINAL Task 1 model on all clean 2022-2025 days and SAVE it to disk.

Use this once. The saved file (task1_model.joblib) contains the trained models
plus the exact feature list and the log-transform flag, so you can apply it to a
new window (e.g. 2026) later without retraining.

Expected performance (already measured by time-series cross-validation):
  pooled out-of-fold R2 ~ 0.66 (RandomForest), ~ 0.63 (XGBoost).

The RTX 4050 mode
XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
             subsample=0.8, random_state=42, tree_method="gpu_hist")
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

OUT_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\Task 1 Improved"
UNIFIED_CSV = os.path.join(OUT_DIR, "parma_unified_2022_2025.csv")
MODEL_PATH = os.path.join(OUT_DIR, "task1_model.joblib")

HOLIDAYS = {
    "2022-04-18", "2022-04-25", "2022-05-01", "2022-06-02", "2022-08-15",
    "2023-04-10", "2023-04-25", "2023-05-01", "2023-06-02", "2023-08-15",
    "2024-04-01", "2024-04-25", "2024-05-01", "2024-06-02", "2024-08-15",
    "2025-04-21", "2025-04-25", "2025-05-01", "2025-06-02", "2025-08-15",
}

FEATURES = [
    "target_lag_1", "target_lag_2", "target_lag_3", "target_lag_7", "target_lag_14",
    "target_roll_7", "target_roll_14",
    "s5p_no2_mean", "s5p_lag_1", "s5p_lag_7", "s5p_roll_7", "s5p_pixel_count",
    "temp_mean_c", "temp_min_c", "temp_max_c",
    "humidity_mean_pct", "wind_speed_mean_kmh", "pressure_hpa", "precipitation_mm",
    "day_of_week", "is_weekend", "month", "day_of_year", "is_holiday", "is_august",
]


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


def main():
    if not os.path.exists(UNIFIED_CSV):
        raise FileNotFoundError(f"Build the unified file first. Not found:\n  {UNIFIED_CSV}")

    df = engineer(pd.read_csv(UNIFIED_CSV))
    feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feats + ["target_real"]).reset_index(drop=True)

    print(f"Training on {len(df)} clean days (2022-2025), "
          f"by year: {df['year'].value_counts().sort_index().to_dict()}")
    print(f"Using {len(feats)} features.")

    X = df[feats].values
    y_log = np.log1p(df["target_real"].values)   # model learns log(NO2)

    print("\nTraining Random Forest ...")
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1).fit(X, y_log)

    bundle = {
        "rf": rf,
        "features": feats,
        "target_transform": "log1p",   # predictions must be inverted with expm1
        "holidays": sorted(HOLIDAYS),
        "trained_on": "2022-2025",
        "n_train": int(len(df)),
        "target_column": "cittadella_no2_daily_mean",
    }

    if HAVE_XGB:
        print("Training XGBoost ...")
        xgb = XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                           subsample=0.8, random_state=42, n_jobs=-1,
                           tree_method="hist").fit(X, y_log)
        bundle["xgb"] = xgb
    else:
        print("(xgboost not installed -> saving Random Forest only)")

    joblib.dump(bundle, MODEL_PATH)
    print("\n" + "=" * 58)
    print("FINAL MODEL SAVED")
    print(f"  File:     {MODEL_PATH}")
    print(f"  Trained:  {bundle['n_train']} days (2022-2025)")
    print(f"  Models:   {'RandomForest + XGBoost' if HAVE_XGB else 'RandomForest'}")
    print(f"  Features: {len(feats)} (log-transformed target)")
    print("  Apply it to 2026 with apply_model_2026.py")
    print("=" * 58)


if __name__ == "__main__":
    main()
