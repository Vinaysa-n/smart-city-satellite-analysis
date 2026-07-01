Parma NO₂ Analysis: Satellite & Ground Data Pipeline
This repository contains the data engineering and machine learning pipeline developed for my Telecommunication Engineering Master's thesis at the University of Parma (UNIPR). The Task-1 focuses on predicting daily Nitrogen Dioxide (NO₂) concentrations in Parma by fusing surface-level sensor readings, meteorological data, and Copernicus Sentinel-5P satellite imagery.

Project Overview
The core objective is to automate the extraction and analysis of massive environmental datasets to generate actionable smart city metrics. The repository handles a complete end-to-end workflow: from fetching and clipping raw satellite NetCDF files to engineering time-series features and deploying an XGBoost predictive model.

Repository Structure

parma-no2-analysis/
├── data/                  # Contains the engineered parma_unified_2022_2025.csv dataset
├── src/                   # Core Python pipeline scripts
├── models/                # Saved serialized models (task1_model.joblib)
├── report_assets/         # Output tables and figures for thesis documentation
├── requirements.txt       # Python environment dependencies
└── README.md              # Project documentation

## Daily NO₂ Concentration Prediction (Parma)

Predicting daily ground-level NO₂ at the Cittadella station in Parma by combining
Sentinel-5P satellite NO₂ columns, ARPAE ground measurements, and ERA5 (Open-Meteo)
meteorology, over four years (2022–2025), using Random Forest and XGBoost.

## Key results
- Baseline (original single-year model): **R² = 0.366**
- Revised four-year model, pooled time-series cross-validation: **R² = 0.659**, RMSE ≈ 2.6 µg/m³
- Out-of-sample validation on unseen spring 2026: **R² = 0.522**, within ±3 µg/m³ on ~89% of days, unbiased

## Pipeline (run order)
1. `scripts/download_s5p_no2_2022_2023.py` — download & clip Sentinel-5P NO₂ (2022–2023)
2. `scripts/download_meteo_openmeteo_2022_2025.py` — download gap-free meteo (all years)
3. `scripts/build_unified_2022_2025.py` — aggregate to daily & merge into one 4-year table
4. `scripts/train_and_save_model.py` — train the final model on 2022–2025 and save it
5. `scripts/apply_model_2026.py` — apply the saved model to the fresh 2026 window
6. `scripts/generate_task1_report_assets.py` — produce all figures and metric tables

Analysis / evaluation scripts:
- `scripts/task1_model_crossval.py` — time-series cross-validation (headline metric)
- `scripts/task1_model_improved.py` — full model (year-ahead split)
- `scripts/task1_model_no2only.py` — meteo-free variant (feature ablation)
- `scripts/crosscheck_meteo_nasa.py` — independent meteo validation vs NASA POWER

## Folder layout
- `scripts/`  — all Python scripts
- `data/`     — ground NO₂ (ARPAE), meteo, a sample clipped satellite file, 2026 predictions
- `figures/`  — report figures
- `tables/`   — metric tables (CV folds, feature importances, model summary, coverage)
- `report/`   — thesis chapter (`chapter1.tex`) and compiled PDF

## Environment
Windows, Anaconda Python 3.13. Libraries: pandas, numpy, scikit-learn, xgboost,
matplotlib, requests, netCDF4, joblib, Cuda
