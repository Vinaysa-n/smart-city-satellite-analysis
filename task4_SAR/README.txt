Task4_Thesis_Package

--- Figures (used by chapter4.tex) ---
figures/sar_scene_2022.png               Single SAR scene: VV backscatter (dB) plus land-cover classification
figures/sar_polarization_vv_vh.png       Co- vs cross-polarisation (VV vs VH) by surface, 2022-2025
figures/sar_backscatter_separation.png   Urban vs vegetation VV backscatter per year, with VV separation
figures/sar_backscatter_allyears.png     Four-year VV backscatter panels (2022-2025), common grayscale
figures/sar_separation_trend.png         Urban-vegetation backscatter separation across 2022-2025
figures/sar_classification_allyears.png  Four-year land-cover classification panels (2022-2025)
figures/sar_classification_split.png     Land-cover proportions by year (percentile-defined)
figures/no2_ndre_correlation.png         NO2-NDRE relationship: raw, seasonal confound, deseasonalised (3 panels)

--- Python code (produces the analysis and figures) ---
code/sentinel1a_multiyear.py             Core SAR pipeline: downloads Sentinel-1A (2022-2024) + uses local S1C (2025), computes VV backscatter and land-cover classification, writes results CSV + comparison figure
code/sar_visualizations.py               Builds the SAR summary figures (separation trend, backscatter separation, VV/VH polarisation, land-cover split) from the results CSV
code/sar_scene_figures.py                Saves per-scene backscatter + classification arrays and renders the per-year scene images and combined four-year panels
code/no2_ndre_correlation_multiyear.py   Pairs ground NO2 with satellite NDRE and computes raw / seasonal / deseasonalised correlations; produces the three-panel correlation figure
code/sar_interactive_map.py              Geocodes a Sentinel-1 scene to lat/lon and builds an interactive Folium HTML map of backscatter and land cover over Parma

---Some of the satellite data has to be downloaded manually, 
it is not possible to attach here because of the storage constraints in the GitHub.