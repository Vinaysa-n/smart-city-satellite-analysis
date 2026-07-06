"""
task3_visualize_results.py
--------------------------
Turns the Stage C output (task3_labeled_2022_2025.csv) into figures + metrics.

Produces:
  task3_fig_trend.png        four-year NDRE trend, points coloured by class
  task3_fig_seasonal.png     Jun-Sep decline, one line per year + baseline
  task3_fig_year_box.png     NDRE distribution per year (the flat trend)
  task3_fig_deviation.png    deviation from monthly baseline, +/- margin band
  task3_fig_confusion.png    confusion matrix (actual vs predicted)
  task3_fig_importance.png   feature importance (gain)
  task3_fig_dashboard.png    all six panels in one image  <- upload this one

Also re-prints accuracy + per-class precision/recall to the console.

RUN:  python task3_visualize_results.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ----------------------------------------------------------------------
OUT_DIR      = Path(r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs")
LABELED_CSV  = OUT_DIR / "task3_labeled_2022_2025.csv"

MARGIN   = 0.010
FEAT_COLS = ['month', 'year', 'scene_idx', 'mean_ndre', 'std_ndre',
             'median_ndre', 'q25_ndre', 'q75_ndre', 'mean_lai', 'std_lai',
             'frac_above_020', 'frac_below_012']
LABELS    = [0, 1, 2]
NAMES     = {0: 'Above avg', 1: 'Average', 2: 'Below avg'}
COLORS    = {0: '#2ca02c', 1: '#9e9e9e', 2: '#d62728'}

plt.rcParams.update({"font.size": 10, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.axisbelow": True})

# ----------------------------------------------------------------------
df = pd.read_csv(LABELED_CSV)
df['dt'] = pd.to_datetime(df['date'])
df = df.sort_values('dt').reset_index(drop=True)
monthly_baseline = df.groupby('month')['mean_ndre'].median()
cm = confusion_matrix(df['label'], df['pred_label'], labels=LABELS)
acc = accuracy_score(df['label'], df['pred_label'])

# feature importance via BOOTSTRAP (single-fit importance is unstable at this
# sample size, so we resample scenes many times, refit, and report mean +/- SD)
N_BOOT = 300
imp_mean = imp_std = None
try:
    import xgboost as xgb
    Xall = df[FEAT_COLS].values
    yall = df['label'].values
    rng = np.random.default_rng(42)
    boot = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(yall), len(yall))   # resample with replacement
        Xb, yb = Xall[idx], yall[idx]
        uc = np.unique(yb)
        if len(uc) < 2:
            continue
        remap = {c: i for i, c in enumerate(uc)}
        ybr = np.array([remap[c] for c in yb])
        ncls = len(uc)
        p = dict(n_estimators=50, max_depth=3, learning_rate=0.2, subsample=0.8,
                 random_state=42, n_jobs=-1, verbosity=0,
                 eval_metric='logloss' if ncls == 2 else 'mlogloss',
                 objective='binary:logistic' if ncls == 2 else 'multi:softprob')
        if ncls > 2:
            p['num_class'] = ncls
        m = xgb.XGBClassifier(**p)
        m.fit(Xb, ybr)
        boot.append(m.feature_importances_)
    boot = np.array(boot)
    imp_mean, imp_std = boot.mean(0), boot.std(0)
except Exception as e:
    print(f"(feature-importance panel skipped: {e})")

MONTH_LBL = {6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep'}


# ----------------------------------------------------------------------
# Plot functions (each draws onto a given axis)
# ----------------------------------------------------------------------
def plot_trend(ax):
    for yr_, g in df.groupby('year'):
        g = g.sort_values('dt')
        ax.plot(g['dt'], g['mean_ndre'], color='0.7', lw=1, zorder=1)
    for lbl in LABELS:
        s = df[df.label == lbl]
        ax.scatter(s['dt'], s['mean_ndre'], c=COLORS[lbl], s=45,
                   edgecolor='k', linewidth=0.4, label=NAMES[lbl], zorder=3)
    ax.set_title("Four-year NDRE trend (Parco Ducale)")
    ax.set_ylabel("Mean NDRE")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, loc='upper left')


def plot_seasonal(ax):
    for yr_, g in df.groupby('year'):
        mm = g.groupby('month')['mean_ndre'].mean()
        ax.plot([MONTH_LBL[m] for m in mm.index], mm.values,
                marker='o', lw=1.5, label=str(yr_))
    ax.plot([MONTH_LBL[m] for m in monthly_baseline.index],
            monthly_baseline.values, 'k--', lw=2.5, label='Baseline (median)')
    ax.set_title("Seasonal NDRE decline (Jun-Sep)")
    ax.set_ylabel("Mean NDRE")
    ax.legend(fontsize=8)


def plot_year_box(ax):
    years = sorted(df['year'].unique())
    data = [df[df.year == y]['mean_ndre'].values for y in years]
    bp = ax.boxplot(data, patch_artist=True)
    ax.set_xticks(range(1, len(years) + 1))
    ax.set_xticklabels([str(y) for y in years])
    for patch in bp['boxes']:
        patch.set_facecolor('#a6cee3')
    ax.set_title("NDRE distribution by year")
    ax.set_ylabel("Mean NDRE")


def plot_deviation(ax):
    ax.axhspan(-MARGIN, MARGIN, color='0.85', zorder=0,
               label=f'Average zone (\u00b1{MARGIN})')
    ax.axhline(0, color='k', lw=0.8)
    for lbl in LABELS:
        s = df[df.label == lbl]
        ax.scatter(s['dt'], s['ndre_deviation'], c=COLORS[lbl], s=40,
                   edgecolor='k', linewidth=0.4, label=NAMES[lbl], zorder=3)
    ax.set_title("Deviation from monthly baseline")
    ax.set_ylabel("NDRE deviation")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, loc='upper left')


def plot_confusion(ax):
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels([NAMES[l] for l in LABELS], rotation=20, ha='right')
    ax.set_yticklabels([NAMES[l] for l in LABELS])
    thr = cm.max() / 2 if cm.max() else 0.5
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thr else 'black',
                    fontsize=12, fontweight='bold')
    ax.set_title(f"Confusion matrix (LOO acc = {acc:.3f})")
    ax.set_ylabel("Actual"); ax.set_xlabel("Predicted")
    ax.grid(False)


def plot_importance(ax):
    if imp_mean is None:
        ax.text(0.5, 0.5, "xgboost not available", ha='center')
        ax.grid(False); return
    order = np.argsort(imp_mean)
    ax.barh([FEAT_COLS[i] for i in order], imp_mean[order],
            xerr=imp_std[order], color='#6a51a3', ecolor='0.35', capsize=2)
    ax.set_title(f"Feature importance (mean \u00b1 SD, {N_BOOT} bootstraps)")
    ax.set_xlabel("Importance")


PLOTS = [("trend", plot_trend), ("seasonal", plot_seasonal),
         ("year_box", plot_year_box), ("deviation", plot_deviation),
         ("confusion", plot_confusion), ("importance", plot_importance)]


def main():
    # ---- metrics to console ----
    print("=" * 60)
    print(f"Scenes: {len(df)}   |   Leave-One-Out accuracy: {acc:.3f}")
    print(classification_report(df['label'], df['pred_label'], labels=LABELS,
          target_names=[NAMES[l] for l in LABELS], digits=3, zero_division=0))

    if imp_mean is not None:
        print(f"Feature importance (mean \u00b1 SD over {N_BOOT} bootstraps):")
        for i in np.argsort(imp_mean)[::-1]:
            flag = "  (unstable: SD > mean)" if imp_std[i] > imp_mean[i] else ""
            print(f"   {FEAT_COLS[i]:16s} {imp_mean[i]:.3f} \u00b1 {imp_std[i]:.3f}{flag}")
        print()

    # ---- individual figures ----
    for name, fn in PLOTS:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        fn(ax)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"task3_fig_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- combined dashboard ----
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    for (name, fn), ax in zip(PLOTS, axes.flat):
        fn(ax)
    fig.suptitle("Task 3 — Four-year vegetation-health classification (Parco Ducale)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_DIR / "task3_fig_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Saved 6 individual figures + task3_fig_dashboard.png to:")
    print(f"  {OUT_DIR}")
    print("Upload task3_fig_dashboard.png and I'll review them.")


if __name__ == "__main__":
    main()
