"""
STEP 2 -- Improved Task 1 NO2 model on parma_unified_2022_2025.csv

What is different from the old single-year model, and why:
  * LOG-TRANSFORM the target  -- NO2 is right-skewed; predicting log(NO2)
    stabilises variance and usually lifts R2 a few points.
  * CALENDAR features added    -- Italian public holidays + an August flag
    (Parma empties out in August; traffic and NO2 drop).
  * LAGS / ROLLING per YEAR    -- features never reach across the Oct-Mar gap
    between growing seasons (a lag from 1 Apr back to 30 Sep is meaningless).
  * REAL labels only           -- the target is never interpolated; only days
    with a genuine measured NO2 are trained/tested on. (A lightly filled copy
    is used ONLY to compute lag/rolling features, never as the label.)
  * TimeSeriesSplit tuning      -- XGBoost tuned without leaking the future.
  * SPLIT: train 2022-2024, test 2025  -- stronger year-ahead validation.

Realistic target for daily single-station NO2: R2 ~ 0.55-0.65.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

OUT_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\Task 1 Improved"
UNIFIED_CSV = os.path.join(OUT_DIR, "parma_unified_2022_2025.csv")

# Italian public holidays falling inside the Apr-Sep window, per year
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

    # label (NEVER interpolated) + a filled copy used ONLY for feature lags
    df["target_real"] = df["cittadella_no2_daily_mean"]
    df["target_filled"] = df.groupby("year")["target_real"].transform(
        lambda s: s.interpolate(limit=3))

    # calendar features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    dstr = df["date"].dt.strftime("%Y-%m-%d")
    df["is_holiday"] = dstr.isin(HOLIDAYS).astype(int)
    df["is_august"] = (df["month"] == 8).astype(int)

    # satellite: short interpolation within year, then year-median fill (keep rows);
    # pixel_count tells the model how reliable each day's satellite value is
    if "s5p_no2_mean" in df.columns:
        df["s5p_no2_mean"] = df.groupby("year")["s5p_no2_mean"].transform(
            lambda s: s.interpolate(limit=7))
        df["s5p_no2_mean"] = df.groupby("year")["s5p_no2_mean"].transform(
            lambda s: s.fillna(s.median()))
    if "s5p_pixel_count" in df.columns:
        df["s5p_pixel_count"] = df["s5p_pixel_count"].fillna(0)
    else:
        df["s5p_pixel_count"] = 0

    # lags / rolling -- computed WITHIN each year so nothing crosses the winter gap
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


def evaluate(name, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"  {name:14s} R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")


def main():
    if not os.path.exists(UNIFIED_CSV):
        raise FileNotFoundError(f"Run STEP 1 first. Not found:\n  {UNIFIED_CSV}")

    df = engineer(pd.read_csv(UNIFIED_CSV))
    feats = [f for f in FEATURES if f in df.columns]
    missing_feats = [f for f in FEATURES if f not in df.columns]
    if missing_feats:
        print(f"NOTE: features not in file, skipped: {missing_feats}")

    df = df.dropna(subset=feats + ["target_real"]).reset_index(drop=True)
    print(f"Usable rows: {len(df)}  by year: "
          f"{df['year'].value_counts().sort_index().to_dict()}")

    train = df[df["year"] <= 2024]
    test = df[df["year"] == 2025]
    print(f"  train 2022-2024: {len(train)}   test 2025: {len(test)}\n")

    Xtr, Xte = train[feats].values, test[feats].values
    ytr_raw, yte_raw = train["target_real"].values, test["target_real"].values
    ytr_log = np.log1p(ytr_raw)   # train on log scale, evaluate on real scale

    print("--- Random Forest (baseline) ---")
    rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
    rf.fit(Xtr, ytr_log)
    evaluate("RandomForest", yte_raw, np.expm1(rf.predict(Xte)))

    if HAVE_XGB:
        print("\n--- XGBoost (tuned, TimeSeriesSplit) ---")
        grid = {
            "n_estimators": [300, 600],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.03, 0.06],
            "subsample": [0.8, 1.0],
        }
        gs = GridSearchCV(
            XGBRegressor(random_state=42, n_jobs=-1, tree_method="hist"),
            grid, cv=TimeSeriesSplit(n_splits=4),
            scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(Xtr, ytr_log)
        print(f"  best params: {gs.best_params_}")
        evaluate("XGBoost", yte_raw, np.expm1(gs.best_estimator_.predict(Xte)))

        imp = pd.Series(gs.best_estimator_.feature_importances_, index=feats)
        imp = imp.sort_values(ascending=False)
        print("\n  Top 10 features (XGBoost):")
        for k, v in imp.head(10).items():
            print(f"    {k:22s} {v:.3f}")
    else:
        print("\n(xgboost not installed -> run:  pip install xgboost  to enable it)")

    print("\nOld single-year RF was R2=0.366 (different meteo source -- not a direct comparison).")
    print("Aim here: R2 ~ 0.55-0.65 for daily single-station NO2.")


if __name__ == "__main__":
    main()
