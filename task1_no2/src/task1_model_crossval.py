"""
Task 1 -- TIME-SERIES CROSS-VALIDATION evaluation.

Why: judging the model only on the sparse, low-variance 2025 test year gives an
unstable R2 (it swung between +0.13 and -0.01 on feature changes). Time-series
cross-validation scores the model across ALL four years -- including the higher-
variance spring/early-summer periods where there is real signal to explain -- so
the reported number reflects real performance instead of one unlucky test year.

This respects time order: every fold trains on the past and tests on the future
(no leakage). Features are the full set (NO2 history + satellite + meteo + calendar).

Two headline numbers:
  * POOLED out-of-fold R2  -- all fold test predictions concatenated, one R2 over
    the full multi-year range. This is the honest summary (high variance -> R2 meaningful).
  * Per-fold R2 mean +/- std -- shows stability across folds.
And, kept as a secondary robustness number:
  * Year-ahead holdout: train 2022-2024, test 2025 (the strict but harsh split).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

try:
    from xgboost import XGBRegressor
    HAVE_XGB = True
except Exception:
    HAVE_XGB = False

OUT_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\Task 1 Improved"
UNIFIED_CSV = os.path.join(OUT_DIR, "parma_unified_2022_2025.csv")

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

# fixed sensible hyperparameters (from the earlier tuning) -- no nested search needed
XGB_PARAMS = dict(n_estimators=300, max_depth=3, learning_rate=0.03,
                  subsample=0.8, random_state=42, n_jobs=-1, tree_method="hist")


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
    if "s5p_pixel_count" in df.columns:
        df["s5p_pixel_count"] = df["s5p_pixel_count"].fillna(0)
    else:
        df["s5p_pixel_count"] = 0

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


def metrics(y_true, y_pred):
    return (r2_score(y_true, y_pred),
            float(np.sqrt(mean_squared_error(y_true, y_pred))),
            mean_absolute_error(y_true, y_pred))


def make_models():
    models = {"RandomForest": RandomForestRegressor(
        n_estimators=400, random_state=42, n_jobs=-1)}
    if HAVE_XGB:
        models["XGBoost"] = XGBRegressor(**XGB_PARAMS)
    return models


def main():
    if not os.path.exists(UNIFIED_CSV):
        raise FileNotFoundError(f"Build the unified file first. Not found:\n  {UNIFIED_CSV}")

    df = engineer(pd.read_csv(UNIFIED_CSV))
    feats = [f for f in FEATURES if f in df.columns]
    df = df.dropna(subset=feats + ["target_real"]).reset_index(drop=True)

    print(f"Usable rows: {len(df)}  by year: "
          f"{df['year'].value_counts().sort_index().to_dict()}")
    print(f"Target std over ALL usable days: {df['target_real'].std():.2f} ug/m3")
    print(f"Target std over 2025 only:       "
          f"{df[df['year']==2025]['target_real'].std():.2f} ug/m3  "
          f"(why the 2025-only R2 is unstable)\n")

    X = df[feats].values
    y_raw = df["target_real"].values
    y_log = np.log1p(y_raw)
    dates = df["date"].values

    tscv = TimeSeriesSplit(n_splits=5)
    oof = {name: {"true": [], "pred": []} for name in make_models()}
    per_fold = {name: [] for name in make_models()}

    print("=== Time-series cross-validation (5 folds, train past -> test future) ===")
    for i, (tr, te) in enumerate(tscv.split(X), 1):
        d0, d1 = pd.Timestamp(dates[te][0]).date(), pd.Timestamp(dates[te][-1]).date()
        tstd = y_raw[te].std()
        print(f"\nFold {i}: train={len(tr):3d}  test={len(te):3d}  "
              f"test dates {d0} -> {d1}  (test std={tstd:.2f})")
        for name, model in make_models().items():
            model.fit(X[tr], y_log[tr])
            pred = np.expm1(model.predict(X[te]))
            r2, rmse, mae = metrics(y_raw[te], pred)
            print(f"    {name:14s} R2={r2:6.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
            oof[name]["true"].append(y_raw[te])
            oof[name]["pred"].append(pred)
            per_fold[name].append(r2)

    print("\n" + "=" * 60)
    print("CROSS-VALIDATED SUMMARY")
    for name in make_models():
        t = np.concatenate(oof[name]["true"])
        p = np.concatenate(oof[name]["pred"])
        r2p, rmsep, maep = metrics(t, p)
        pf = np.array(per_fold[name])
        print(f"\n  {name}")
        print(f"    POOLED out-of-fold:  R2={r2p:.3f}  RMSE={rmsep:.3f}  MAE={maep:.3f}  (n={len(t)})")
        print(f"    Per-fold R2:         mean={pf.mean():.3f}  std={pf.std():.3f}")
    print("=" * 60)

    # secondary: strict year-ahead holdout
    print("\n--- Secondary (strict) year-ahead holdout: train 2022-2024, test 2025 ---")
    tr = df["year"] <= 2024
    te = df["year"] == 2025
    for name, model in make_models().items():
        model.fit(X[tr.values], y_log[tr.values])
        pred = np.expm1(model.predict(X[te.values]))
        r2, rmse, mae = metrics(y_raw[te.values], pred)
        print(f"  {name:14s} R2={r2:6.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  (test n={te.sum()})")

    print("\nReport the POOLED cross-validated R2 as the headline; keep the year-ahead")
    print("2025 number as an honest secondary result (sparse, low-variance test year).")


if __name__ == "__main__":
    main()
