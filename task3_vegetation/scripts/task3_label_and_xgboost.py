"""
task3_label_and_xgboost.py   (Stage C)
--------------------------------------
Reads the per-scene features CSV from Stage B and finishes the rebuild:
  * builds monthly-baseline DEVIATION labels (your exact v2 logic),
  * runs Leave-One-Out XGBoost (robust per-fold remapping, so it copes
    when a class - e.g. 'Above average' - is absent, as in your n=19 run),
  * prints the class distribution, LOO accuracy, classification report and
    feature importance, plus a four-year NDRE trend summary,
  * writes task3_labeled_2022_2025.csv.

This mirrors xgboost_vegetation_health_v2.py - no data leakage: labels come
from the seasonal baseline, never from a raw NDRE threshold, and every scene
is tested while held out.

RUN (after Stage B has built the features CSV):
    python task3_label_and_xgboost.py
"""

from pathlib import Path
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    import subprocess
    subprocess.run(["pip", "install", "xgboost", "--break-system-packages", "-q"])
    import xgboost as xgb

from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, classification_report

# ----------------------------------------------------------------------
OUT_DIR      = Path(r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs")
FEATURES_CSV = OUT_DIR / "task3_features_2022_2025.csv"
LABELED_CSV  = OUT_DIR / "task3_labeled_2022_2025.csv"

MARGIN = 0.010   # your v2 value: ±0.010 = "average" band

# Scenes excluded after QC. 2024-09-07: 13.6% of the Parco Ducale box flagged
# cloud/cirrus/shadow by SCL; NDRE 0.132 vs 0.153 two days later (09-09) -> haze,
# not real stress. The reflectance mask passes thin haze, so it was caught visually.
EXCLUDE_DATES = ['2024-09-07']

FEAT_COLS = ['month', 'year', 'scene_idx',
             'mean_ndre', 'std_ndre', 'median_ndre',
             'q25_ndre', 'q75_ndre',
             'mean_lai', 'std_lai',
             'frac_above_020', 'frac_below_012']

NAMES = {0: 'Above average', 1: 'Average', 2: 'Below average'}


def main():
    if not FEATURES_CSV.exists():
        raise SystemExit(f"Features CSV not found:\n  {FEATURES_CSV}\n"
                         "Run Stage B (download_and_process_task3.py) first.")

    df = pd.read_csv(FEATURES_CSV).sort_values('date').reset_index(drop=True)
    if EXCLUDE_DATES:
        before = len(df)
        df = df[~df['date'].isin(EXCLUDE_DATES)].reset_index(drop=True)
        if before != len(df):
            print(f"  Excluded {before - len(df)} scene(s) for QC: {EXCLUDE_DATES}")
    n = len(df)
    print("=" * 66)
    print(f"Task 3 — four-year vegetation-health classification  (n={n} scenes)")
    print("=" * 66)
    yr_counts = df['year'].value_counts().sort_index()
    print("  scenes per year: " +
          ", ".join(f"{y}:{c}" for y, c in yr_counts.items()))

    # ── STEP 1: LABELS from monthly-baseline deviation (no leakage) ───────────
    monthly_baseline = df.groupby('month')['mean_ndre'].median().to_dict()
    df['baseline_ndre'] = df['month'].map(monthly_baseline)
    df['ndre_deviation'] = df['mean_ndre'] - df['baseline_ndre']

    def deviation_label(dev):
        if dev > MARGIN:
            return 0
        if dev < -MARGIN:
            return 2
        return 1

    df['label'] = df['ndre_deviation'].apply(deviation_label)
    df['label_name'] = df['label'].map(NAMES)

    print("\n  Monthly baselines (median mean_NDRE across all years):")
    for m, v in sorted(monthly_baseline.items()):
        print(f"    Month {m:02d}: NDRE = {v:.4f}")

    print("\n  Label distribution:")
    for lbl in (0, 1, 2):
        print(f"    {NAMES[lbl]:14s}: {(df.label == lbl).sum()} scenes")

    print("\n  Below-average scenes (flagged stress):")
    below = df[df.label == 2]
    if below.empty:
        print("    (none)")
    else:
        for _, r in below.iterrows():
            print(f"    {r['date']}  dev={r['ndre_deviation']:+.4f}  "
                  f"NDRE={r['mean_ndre']:.4f}")

    # ── STEP 2: LEAVE-ONE-OUT CV (robust per-fold remapping) ──────────────────
    X = df[FEAT_COLS].values
    y = df['label'].values
    loo = LeaveOneOut()
    y_pred = np.zeros(len(y), dtype=int)

    for tr, te in loo.split(X):
        ytr = y[tr]
        uc = np.unique(ytr)
        if len(uc) < 2:                       # only one class in training fold
            y_pred[te] = uc[0]
            continue
        remap = {c: i for i, c in enumerate(uc)}
        unmap = {i: c for i, c in enumerate(uc)}
        ytr_r = np.array([remap[c] for c in ytr])
        ncls = len(uc)
        obj = 'binary:logistic' if ncls == 2 else 'multi:softprob'
        params = dict(n_estimators=50, max_depth=3, learning_rate=0.2,
                      subsample=0.8, random_state=42, n_jobs=-1, verbosity=0,
                      eval_metric='logloss' if ncls == 2 else 'mlogloss',
                      objective=obj)
        if ncls > 2:
            params['num_class'] = ncls
        clf = xgb.XGBClassifier(**params)
        clf.fit(X[tr], ytr_r)
        pr = int(clf.predict(X[te])[0])
        y_pred[te] = unmap.get(pr, uc[0])

    acc = accuracy_score(y, y_pred)
    df['pred_label'] = y_pred
    df['pred_label_name'] = df['pred_label'].map(NAMES)
    df['correct'] = df['label'] == df['pred_label']

    print("\n" + "=" * 66)
    print(f"  Leave-One-Out accuracy: {acc:.3f}  "
          f"({int(df.correct.sum())}/{n} scenes correct)")
    present = sorted(np.unique(np.concatenate([y, y_pred])))
    print(classification_report(y, y_pred, labels=present,
          target_names=[NAMES[l] for l in present], digits=3, zero_division=0))

    # ── STEP 3: feature importance (final model on all data) ──────────────────
    uc_all = np.unique(y)
    if len(uc_all) >= 2:
        remap_all = {c: i for i, c in enumerate(uc_all)}
        y_all = np.array([remap_all[c] for c in y])
        ncls = len(uc_all)
        obj = 'binary:logistic' if ncls == 2 else 'multi:softprob'
        p = dict(n_estimators=50, max_depth=3, learning_rate=0.2, subsample=0.8,
                 random_state=42, n_jobs=-1, verbosity=0,
                 eval_metric='logloss' if ncls == 2 else 'mlogloss', objective=obj)
        if ncls > 2:
            p['num_class'] = ncls
        final = xgb.XGBClassifier(**p)
        final.fit(X, y_all)
        imp = final.feature_importances_
        order = np.argsort(imp)[::-1]
        print("  Feature importance (gain):")
        for i in order:
            print(f"    {FEAT_COLS[i]:16s} {imp[i]:.3f}")

    # ── STEP 4: four-year NDRE trend summary ──────────────────────────────────
    print("\n  Mean NDRE by year (four-year trend):")
    for yv, sub in df.groupby('year'):
        print(f"    {yv}: mean_NDRE = {sub['mean_ndre'].mean():.4f}  "
              f"(min {sub['mean_ndre'].min():.4f}, max {sub['mean_ndre'].max():.4f})")

    df.to_csv(LABELED_CSV, index=False)
    print(f"\nLabeled per-scene table -> {LABELED_CSV}")
    print("=" * 66)


if __name__ == "__main__":
    main()
