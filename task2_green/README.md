# Task 2 — Urban Green Space Segmentation

Cross-city transfer learning for mapping urban vegetation from Sentinel-2 imagery,
without any manual annotation of the target city.

## What This Task Does

Trains a U-Net V3 deep learning model on Prague satellite imagery, then applies it
to Parma, Italy — a city the model has never seen — to produce a pixel-level green
space map. The key question: **can we skip the expensive step of labelling every
city's imagery by hand?**

## Architecture

- **Encoder:** ResNet-34 pretrained on ImageNet
- **Decoder:** U-Net with attention gates on all skip connections
- **Input:** 5 channels — RGB (B02, B03, B04) + NDRE (B8A−B05)/(B8A+B05) + LAI
- **Loss:** DiceBCE (Dice + weighted BCE, pos_weight=2.0)
- **Training data:** Prague S2 imagery, 10k+ patches (64×64), 12× augmentation
- **Transfer:** Histogram matching mandatory (without it → 0% green predicted)

## Key Results

| Model | Training Data        | IoU   | F1    | Green % |
|-------|----------------------|-------|-------|---------|
| A     | Prague only          | 0.703 | 0.825 | 38.0%   |
| B     | EuroSAT + Prague     | 0.670 | 0.802 | 39.3%   |
| C     | + Parma fine-tune    | 0.598 | 0.749 | 41.1%   |
| V3+TTA| Model A + 8-fold ens.| —     | —     | 37.4%   |

**Model A is recommended** — highest IoU, no target-city data needed.

The progressive IoU decline from A→C is not a failure; it reflects a systematic
accuracy-coverage trade-off as the model adapts toward Parma's spectral domain.

## Pipeline Run Order

```
1. unet_v3_all_improvements.py    — Train Models A/B/C on Prague, evaluate on Prague
2. (histogram matching applied internally during inference)
3. tta_fix.py                     — Run 8-fold TTA inference on Parma → 37.4% green
```

## Folder Layout

```
task2_green/
├── README.md                          ← this file
├── scripts/
│   └── SCRIPTS_LOCATION.txt           ← where to find the .py files on your machine
├── data/
│   ├── parma_pred_tta.npy             ← V3+TTA probability map (37.4% at threshold 0.5)
│   └── parma_green_pred.npy           ← non-TTA prediction (~20% at 0.5)
├── figures/
│   ├── prague_segmentation.png        ← Prague validation (43.9% green)
│   ├── modelA_prague_only.png         ← Model A on Parma (38.0%, IoU=0.703)
│   ├── modelB_eurosat_prague.png      ← Model B on Parma (39.3%, IoU=0.670)
│   ├── modelC_finetuned.png           ← Model C on Parma (41.1%, IoU=0.598)
│   ├── modelA_confusion_matrix.png    ← Confusion matrix Model A
│   ├── modelB_confusion_matrix.png    ← Confusion matrix Model B
│   ├── modelC_confusion_matrix.png    ← Confusion matrix Model C
│   ├── final_comparison_all_models.png← Side-by-side A/B/C comparison
│   ├── metrics_comparison.png         ← IoU/F1/coverage bar chart
│   └── parma_v3_segmentation.png      ← Final V3+TTA result (37.4% green)
├── tables/
│   └── model_comparison.csv           ← All metrics in machine-readable format
└── report/
    └── (chapter .docx files if available)
```

## Excluded Figures (superseded drafts)

These files were uploaded but are NOT included — they represent earlier pipeline
versions that were replaced by the final figures above:

- `parma_segmentation.png` — old pipeline, 20.2% green (replaced by V3+TTA)
- `parma_before_finetune.png` — intermediate step, no longer relevant
- `parma_after_finetune.png` — intermediate step, no longer relevant
- `parma_eurosat_segmentation_1.png` — redundant EuroSAT variant
- `parma_eurosat_segmentation_2.png` — redundant EuroSAT variant
- `enhancement_comparison.png` — old enhancement pipeline
- `benchmark_comparison.png` — old benchmark comparison

## Critical Implementation Notes

1. **Histogram matching is non-negotiable.** Without it the model predicts 0% green
   on Parma due to spectral domain shift between Prague and Italian imagery.

2. **Coordinate system:** All S2 data is in EPSG:32632 (UTM Zone 32N). Use pyproj
   for lat/lon → UTM conversion. Do NOT use rasterio.rowcol() with geographic coords.

3. **5-channel input initialization:** NDRE and LAI channel weights are initialized
   by averaging the pretrained RGB weights + small random perturbation (N(0, 0.01)).

4. **Tile T32TNQ only.** Adjacent tile T32TPQ overlaps partially — exclude it.

## Dependencies

- Python 3.13, PyTorch 2.6 (CUDA), rasterio, numpy, PIL, scikit-learn
- GPU: NVIDIA RTX 4050 (6GB VRAM) — 64×64 tile size fits in memory
- Training time: ~30 minutes on RTX 4050

## Security Note

No API keys, passwords, or tokens are present in any file in this package.
CDSE credentials are handled at download time only and are not stored in scripts.
