"""
visualize_bbox_contents.py
--------------------------
Shows exactly what falls inside the two candidate study-area boxes, using one
Sentinel-2 scene you already have on disk. For EACH box it renders three panels
- true colour, SCL land-cover, and NDRE - and prints a land-cover breakdown.

  * No download needed: it reuses a zip from your SP2 / SP2_2024 folders.
  * Output:
      - bbox_contents.png   (2 rows = the two boxes, 3 cols = views)
"""

import os
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pyproj import Transformer
import rasterio

# ----------------------------------------------------------------------
BASE = Path(r"C:\Users\vinay\OneDrive\Desktop\Thesis")
OUT_PNG = BASE / "outputs" / "bbox_contents.png"
WORK = BASE / "outputs" / "task3_work"
WORK.mkdir(parents=True, exist_ok=True)

# Folders with scenes already downloaded (any T32TNQ zip will do)
EXISTING_DIRS = [BASE / "files for thesis" / "SP2_2024",
                 BASE / "files for thesis" / "SP2"]
ZIP_PATH = None        # optional: set to r"...\some_scene.zip" to force a scene

# The two boxes: (lon_min, lon_max, lat_min, lat_max)
BOXES = {
    "Parco Ducale (1.3 km2)":      (10.313, 10.328, 44.802, 44.812),
    "Wide Parma box (24.6 km2)":   (10.290, 10.360, 44.790, 44.830),
}

SCL_COLORS = {0:"#000000", 1:"#ff0000", 2:"#2f2f2f", 3:"#643200", 4:"#1f9e1f",
              5:"#ffe65a", 6:"#1414ff", 7:"#7f7f7f", 8:"#b0b0b0", 9:"#e8e8e8",
              10:"#64c8ff", 11:"#ff96ff"}
SCL_NAMES = {0:"no data", 1:"saturated", 2:"dark/shadow", 3:"cloud shadow",
             4:"vegetation", 5:"not vegetated (urban/bare)", 6:"water",
             7:"unclassified", 8:"cloud", 9:"cloud", 10:"cirrus", 11:"snow"}


def find_zip():
    if ZIP_PATH:
        return Path(ZIP_PATH)
    for d in EXISTING_DIRS:
        if d.exists():
            zs = sorted(z for z in d.glob("*.zip") if "T32TNQ" in z.name)
            if zs:
                return zs[0]
    return None


def extract(zf, predicate, tmpname):
    cands = [f for f in zf.namelist() if predicate(f)]
    if not cands:
        return None
    tmp = WORK / tmpname
    with zf.open(cands[0]) as s, open(tmp, "wb") as d:
        d.write(s.read())
    return tmp


def box_window(src, box):
    lon0, lon1, lat0, lat1 = box
    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    xs, ys = tr.transform([lon0, lon1, lon0, lon1], [lat0, lat0, lat1, lat1])
    t = src.transform
    c0 = int((min(xs) - t.c) / t.a); c1 = int((max(xs) - t.c) / t.a)
    r0 = int((max(ys) - t.f) / t.e); r1 = int((min(ys) - t.f) / t.e)
    r0, r1 = sorted((max(0, r0), max(0, r1)))
    c0, c1 = sorted((max(0, c0), max(0, c1)))
    return r0, min(r1, src.height), c0, min(c1, src.width)


def read_clip(tmp_path, box, bands=1):
    with rasterio.open(tmp_path) as src:
        r0, r1, c0, c1 = box_window(src, box)
        if r1 <= r0 or c1 <= c0:
            return None
        if bands == 1:
            return src.read(1, window=((r0, r1), (c0, c1)))
        return src.read(window=((r0, r1), (c0, c1)))   # (bands, h, w)


def true_color(zf, box):
    """TCI (10 m) if present, else a B04/B03/B02 stretch (20 m)."""
    tci = extract(zf, lambda f: f.endswith("_TCI_10m.jp2") and "R10m" in f, "_tci.jp2")
    if tci is not None:
        arr = read_clip(tci, box, bands=3)
        tci.unlink(missing_ok=True)
        if arr is not None:
            return np.transpose(arr, (1, 2, 0)).astype(np.uint8)
    # fallback
    chans = []
    for code in ("B04", "B03", "B02"):
        tmp = extract(zf, lambda f, c=code: f"_{c}_20m.jp2" in f and "R20m" in f, f"_{code}.jp2")
        a = read_clip(tmp, box, bands=1) if tmp else None
        if tmp:
            tmp.unlink(missing_ok=True)
        if a is None:
            return None
        chans.append(a.astype(np.float32) / 10000.0)
    rgb = np.dstack(chans)
    for i in range(3):
        lo, hi = np.percentile(rgb[:, :, i], (2, 98))
        rgb[:, :, i] = np.clip((rgb[:, :, i] - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb


def scl_and_ndre(zf, box):
    scl_t = extract(zf, lambda f: f.endswith("_SCL_20m.jp2") and "R20m" in f, "_scl.jp2")
    b05_t = extract(zf, lambda f: f"_B05_20m.jp2" in f and "R20m" in f, "_b05.jp2")
    b8a_t = extract(zf, lambda f: f"_B8A_20m.jp2" in f and "R20m" in f, "_b8a.jp2")
    scl = read_clip(scl_t, box, 1) if scl_t else None
    b05 = read_clip(b05_t, box, 1) if b05_t else None
    b8a = read_clip(b8a_t, box, 1) if b8a_t else None
    for t in (scl_t, b05_t, b8a_t):
        if t:
            t.unlink(missing_ok=True)
    ndre = None
    if b05 is not None and b8a is not None:
        b5 = b05.astype(np.float32) / 10000.0
        b8 = b8a.astype(np.float32) / 10000.0
        d = b8 + b5
        ndre = np.where(d > 1e-6, (b8 - b5) / d, np.nan)
        ndre = np.where((b5 > 0.001) & (b8 > 0.001) & (b5 < 0.9) & (b8 < 0.9), ndre, np.nan)
    return scl, ndre


def scl_to_rgb(scl):
    h, w = scl.shape
    img = np.zeros((h, w, 3), np.float32)
    for cls, hexc in SCL_COLORS.items():
        m = scl == cls
        if m.any():
            img[m] = [int(hexc[i:i+2], 16) / 255 for i in (1, 3, 5)]
    return img


def main():
    z = find_zip()
    if z is None:
        raise SystemExit("No T32TNQ zip found in SP2 / SP2_2024.\n"
                         "Set ZIP_PATH at the top to one of your scene zips.")
    print(f"Using scene: {z.name}\n")

    fig, axes = plt.subplots(len(BOXES), 3, figsize=(13, 4.6 * len(BOXES)))
    if len(BOXES) == 1:
        axes = axes[None, :]

    for row, (name, box) in enumerate(BOXES.items()):
        with zipfile.ZipFile(z, "r") as zf:
            rgb = true_color(zf, box)
            scl, ndre = scl_and_ndre(zf, box)

        # panel 1: true colour
        ax = axes[row, 0]
        if rgb is not None:
            ax.imshow(rgb)
        ax.set_title(f"{name}\nTrue colour", fontsize=10)
        ax.axis("off")

        # panel 2: SCL land cover
        ax = axes[row, 1]
        if scl is not None:
            ax.imshow(scl_to_rgb(scl))
            present = [c for c in SCL_COLORS if (scl == c).any()]
            ax.legend(handles=[mpatches.Patch(color=SCL_COLORS[c], label=SCL_NAMES[c])
                               for c in present], fontsize=6, loc="upper right",
                      framealpha=0.85)
        ax.set_title("Land cover (SCL)", fontsize=10)
        ax.axis("off")

        # panel 3: NDRE
        ax = axes[row, 2]
        if ndre is not None:
            im = ax.imshow(ndre, cmap="RdYlGn", vmin=-0.05, vmax=0.45)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("NDRE", fontsize=10)
        ax.axis("off")

        # ---- printed stats ----
        print("=" * 60)
        print(f"{name}")
        if scl is not None:
            tot = scl.size
            veg = (scl == 4).sum() / tot * 100
            nonveg = (scl == 5).sum() / tot * 100
            water = (scl == 6).sum() / tot * 100
            other = 100 - veg - nonveg - water
            print(f"   pixels in box: {tot:,}")
            print(f"   vegetation (SCL 4)        : {veg:5.1f} %")
            print(f"   not vegetated/urban (SCL 5): {nonveg:5.1f} %")
            print(f"   water (SCL 6)             : {water:5.1f} %")
            print(f"   other (cloud/shadow/etc.) : {other:5.1f} %")
        if ndre is not None:
            v = ndre[np.isfinite(ndre)]
            if v.size:
                print(f"   mean NDRE                 : {v.mean():.4f}")
                print(f"   pixels NDRE >= 0.20 (green): {(v >= 0.20).mean()*100:5.1f} %")
        print()

    fig.suptitle(f"What is inside each study-area box   ({z.name[:24]}...)",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print("=" * 60)
    print(f"Figure saved -> {OUT_PNG}")
    print("Upload that PNG and/or paste the stats above back to me.")


if __name__ == "__main__":
    main()
