# Task 3 — Vegetation Health Classification (Parma, 2022–2025)

Part of the MSc thesis *"Definition and Implementation of Algorithms for Analyzing
Satellite Data in the Smart City Context".

## Description

This task classifies the health of urban vegetation in the **Parco Ducale** park
of Parma from four summers (2022–2025) of **Sentinel-2** Level-2A imagery.

For every acquisition it computes two biophysical indicators over the park —
**NDRE** (Normalized Difference Red Edge, sensitive to canopy chlorophyll/stress)
and an **NDRE-derived LAI** (Leaf Area Index) — summarises each scene into 12
features, and trains an **XGBoost** classifier to label each scene as *above /
average / below* a seasonally-adjusted norm. Labels are built from the
**deviation of a scene from its monthly baseline** (not by thresholding NDRE), so
the model cannot cheat by inverting its own input — i.e. the analysis is free of
the data leakage that produced a fake ~99.9% accuracy in an earlier version.
Every scene is also cloud-screened at the study-area level using the Sentinel-2
Scene Classification Layer (SCL).

## Pipeline (run order)

Scripts are in `scripts/`. Run them in this order:

1. **`check_s2_availability_2022_2023.py`** — probes the public CDSE catalogue to
   confirm enough cloud-free scenes exist per year *(no login needed)*.
2. **`select_task3_scenes_2022_2025.py`** — builds the scene manifest
   (12/year, June–September, ≤20% cloud) → `data/task3_scene_manifest_2022_2025.csv`.
3. **`download_and_process_task3.py`** — *Stage B*: downloads each scene, clips it
   to the Parco Ducale window, computes NDRE + LAI, and writes 12 scene-level
   features. *(CDSE login required — see below.)*
4. **`task3_label_and_xgboost.py`** — *Stage C*: monthly-baseline deviation labels
   + leave-one-out XGBoost → `tables/task3_labeled_2022_2025.csv`.
5. **`task3_visualize_results.py`** — produces the report figures (`figures/`).

Quality-control tools (run any time):

- **`audit_scl_all.py`** — SCL cloud-screen of every scene → `tables/task3_scl_audit.csv`.
- **`verify_scene.py`** — visual check of a single suspect scene vs a clean one.
- **`visualize_bbox_contents.py`** — compares the Parco Ducale window with a wider
  Parma box (the imagery that justified the study-area choice).

### How to run (no prior Python experience needed)

- Environment used: **Windows + Anaconda, Python 3.13**. Install the libraries once:
  `pip install rasterio pyproj numpy pandas xgboost scikit-learn matplotlib requests scipy`
- Each script has a **CONFIG block at the top** with file paths — edit those to match
  your machine before running.
- Copernicus (CDSE) login is asked for **at runtime** (username + password prompt),
  or you can set the environment variables `CDSE_USER` and `CDSE_PASS`. **No
  password is stored in any script.**

## Key results (as reported in the chapter)

- **48** candidate scenes → **1 excluded** for cloud contamination
  (2024-09-07, 13.6% cloud over the park) → **2** clean July scenes pending
  re-acquisition → **45 analysed**. The SCL screen found 47 of 48 scenes at
  exactly 0.0% park cloud.
- **Seasonal decline** of NDRE of ~13% across the summer (June ≈ 0.176 →
  September ≈ 0.157).
- **Flat interannual trend**: annual mean NDRE ≈ 0.160 (2022), 0.160 (2023),
  0.170 (2024), 0.166 (2025) — not statistically distinguishable.
- **Classifier**: leave-one-out accuracy **0.911** against an 84.4% majority
  baseline; 3 above / 38 average / 4 below. Perfect precision on the minority
  classes but low recall — a conservative detector (no false alarms, misses the
  hardest cases).
- **Feature importance** (bootstrapped): only `mean_ndre` is robustly dominant;
  other features are within noise.

## Folder layout

```
task3_vegetation/
├── README.md          this file
├── scripts/           Python pipeline (Stages B–C) + QC tools
├── data/              input scene manifest
├── figures/           the 8 figures used in the report chapter
└── tables/            results CSVs (SCL cloud audit, labelled scenes)
