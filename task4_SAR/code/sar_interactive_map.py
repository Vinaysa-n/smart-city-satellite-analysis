
"""
=============================================================================
 Interactive SAR map over Parma  (Task 4)
=============================================================================
Turns one Sentinel-1 scene into an INTERACTIVE web map: the VV backscatter and
the urban/vegetation/water classification, geocoded to real lat/lon and laid
over a street / satellite basemap of Parma. Open the HTML, zoom into the city,
and toggle layers - far clearer for a viva than bar charts.

Workflow (your usual one): extract -> process -> geocode -> SAVE small arrays
-> DELETE the big files. By default it uses your LOCAL S1C 2025 scene, so there
is no 1 GB download. On re-runs it reuses the saved arrays (fast map rebuild).

WHY A SPECIAL GEOCODING STEP: a Sentinel-1 GRD is in radar geometry, not
lat/lon. This script reads the scene's geolocation grid and resamples the
backscatter onto a true lat/lon grid, so it aligns correctly with the map.

make sure LOCAL_S1C_ZIP below points at your S1C .SAFE.zip.

"""


# Source scene (local S1C - no download needed). Point at your .SAFE.zip:
LOCAL_S1C_ZIP = (r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\SAR"
                 r"\S1C_IW_GRDH_1SDV_20250603T052626_20250603T052651_002614_0056AB_BFCF.SAFE.zip")

WORK_DIR = r"D:\thesis download\sar_map_work"          # big temp files (deleted)
OUT_DIR  = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\sar_map"   # small outputs + html

# Parma study box (same as your SAR analysis)
LAT_MIN, LAT_MAX = 44.79, 44.83
LON_MIN, LON_MAX = 10.29, 10.36

GRID_NX, GRID_NY = 600, 500            # lat/lon output grid resolution
REBUILD_MAP_ONLY = False               # True = skip processing, just rebuild html from saved arrays

# Parco Ducale box (your NDRE study area) - drawn on the map
PARCO_LAT = (44.802, 44.812)
PARCO_LON = (10.313, 10.328)

# Optional ground-station markers. Coordinates are APPROXIMATE - set your exact
# ARPAE station coordinates here, or set SHOW_STATIONS = False to hide them.
SHOW_STATIONS = True
STATIONS = {
    "ARPAE Cittadella (NO\u2082)": (44.7905, 10.3290),   # <-- verify / replace
    "ARPAE Montebello (NO\u2082)": (44.7880, 10.3420),   # <-- verify / replace
}

# processing params (locked to your SAR pipeline)
SPECKLE_SIZE = 5
# ============================================================================


import os, sys, json, zipfile, shutil, subprocess
from pathlib import Path
import xml.etree.ElementTree as ET


def _ensure(pkg, importname=None):
    try:
        __import__(importname or pkg)
    except ImportError:
        print(f"  installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])


for _p, _i in [("numpy", "numpy"), ("scipy", "scipy"), ("rasterio", "rasterio"),
               ("matplotlib", "matplotlib"), ("folium", "folium"), ("Pillow", "PIL")]:
    _ensure(_p, _i)

import numpy as np
from scipy.ndimage import uniform_filter
from scipy.interpolate import LinearNDInterpolator
import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
import folium

WORK_DIR = Path(WORK_DIR); OUT_DIR = Path(OUT_DIR)
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

DB_NPY     = OUT_DIR / "sar_db_latlon.npy"
CLASS_NPY  = OUT_DIR / "sar_class_latlon.npy"
BOUNDS_JSON = OUT_DIR / "sar_map_bounds.json"


#  SAR processing

def extract_safe(zip_path, target):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        safes = {n.split("/")[0] for n in names if n.split("/")[0].endswith(".SAFE")}
        if not safes:
            raise RuntimeError("No .SAFE folder inside the zip.")
        safe = sorted(safes)[0]
        out = Path(target) / safe
        if not out.exists():
            print(f"  extracting {safe} ...")
            zf.extractall(target)
        return out


def read_geoloc_grid(safe):
    annot = sorted((Path(safe) / "annotation").glob("*.xml"))
    if not annot:
        raise RuntimeError("No annotation XML in SAFE.")
    root = ET.parse(annot[0]).getroot()
    pts = root.findall(".//geolocationGridPoint")
    lat = np.array([float(p.find("latitude").text) for p in pts])
    lon = np.array([float(p.find("longitude").text) for p in pts])
    line = np.array([int(p.find("line").text) for p in pts])
    pix = np.array([int(p.find("pixel").text) for p in pts])
    return lat, lon, line, pix


def parma_window(lat, lon, line, pix, margin=60):
    corners = np.array([[LAT_MIN, LON_MIN], [LAT_MIN, LON_MAX],
                        [LAT_MAX, LON_MIN], [LAT_MAX, LON_MAX]])
    from scipy.interpolate import griddata
    pr = griddata(np.column_stack([lat, lon]), line, corners, method="linear")
    pc = griddata(np.column_stack([lat, lon]), pix, corners, method="linear")
    if np.any(np.isnan(pr)) or np.any(np.isnan(pc)):
        raise RuntimeError("Parma falls outside this scene.")
    r0 = max(0, int(np.nanmin(pr)) - margin); r1 = int(np.nanmax(pr)) + margin
    c0 = max(0, int(np.nanmin(pc)) - margin); c1 = int(np.nanmax(pc)) + margin
    return r0, r1, c0, c1


def read_vv_window(safe, win):
    import rasterio
    from rasterio.windows import Window
    r0, r1, c0, c1 = win
    meas = Path(safe) / "measurement"
    vv = None
    for t in list(meas.glob("*.tiff")) + list(meas.glob("*.tif")):
        if "vv" in t.name.lower():
            with rasterio.open(t) as src:
                rr1, cc1 = min(r1, src.height), min(c1, src.width)
                vv = src.read(1, window=Window(c0, r0, cc1 - c0, rr1 - r0)).astype(np.float32)
            break
    if vv is None:
        raise RuntimeError("No VV band found.")
    return vv


def to_db(lin, eps=1e-10):
    return 10 * np.log10(np.maximum(lin, eps))


def process_window(vv_lin):
    filt = uniform_filter(vv_lin.astype(np.float64), size=SPECKLE_SIZE).astype(np.float32)
    db = to_db(filt)
    valid = db[db > -40]
    t_urban = np.percentile(valid, 75)
    t_water = np.percentile(valid, 15)
    cls = np.full(db.shape, np.nan, np.float32)        # 0 water, 1 veg, 2 urban
    cls[(db <= t_water)] = 0
    cls[(db > t_water) & (db < t_urban) & (db > -35)] = 1
    cls[(db >= t_urban)] = 2
    cls[db <= -40] = np.nan
    return db, cls



#  Geocoding: resample the radar-geometry window onto a regular lat/lon grid

def geocode(db_win, cls_win, lat, lon, line, pix, win):
    r0, _, c0, _ = win
    # build inverse mapping (lat,lon) -> (line,pixel) from the geolocation grid
    pts = np.column_stack([lat, lon])
    f_line = LinearNDInterpolator(pts, line)
    f_pix = LinearNDInterpolator(pts, pix)

    lats = np.linspace(LAT_MAX, LAT_MIN, GRID_NY)      # row 0 = north
    lons = np.linspace(LON_MIN, LON_MAX, GRID_NX)      # col 0 = west
    LON, LAT = np.meshgrid(lons, lats)

    full_line = f_line(LAT, LON)
    full_pix = f_pix(LAT, LON)
    rr = np.rint(full_line - r0).astype(float)
    cc = np.rint(full_pix - c0).astype(float)

    H, W = db_win.shape
    ok = (~np.isnan(rr)) & (~np.isnan(cc)) & (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
    db_ll = np.full((GRID_NY, GRID_NX), np.nan, np.float32)
    cl_ll = np.full((GRID_NY, GRID_NX), np.nan, np.float32)
    ri = rr[ok].astype(int); ci = cc[ok].astype(int)
    db_ll[ok] = db_win[ri, ci]
    cl_ll[ok] = cls_win[ri, ci]
    return db_ll, cl_ll


#  Colourised overlays (RGBA PNGs)

def db_to_png(db_ll, path):
    finite = db_ll[np.isfinite(db_ll)]
    vmin, vmax = np.percentile(finite, 5), np.percentile(finite, 95)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    rgba = cm.gray(norm(np.nan_to_num(db_ll, nan=vmin)))
    rgba[..., 3] = np.where(np.isfinite(db_ll), 1.0, 0.0)   # transparent outside
    Image.fromarray((rgba * 255).astype(np.uint8)).save(path)


def class_to_png(cl_ll, path):
    colors = {0: (41, 128, 185), 1: (39, 174, 96), 2: (192, 57, 43)}  # water/veg/urban
    rgba = np.zeros((*cl_ll.shape, 4), np.uint8)
    for code, (r, g, b) in colors.items():
        m = cl_ll == code
        rgba[m] = (r, g, b, 255)
    rgba[~np.isfinite(cl_ll), 3] = 0
    Image.fromarray(rgba).save(path)


#  Folium interactive map

def build_map(db_png, class_png, bounds, out_html):
    sw, ne = bounds
    center = [(sw[0] + ne[0]) / 2, (sw[1] + ne[1]) / 2]
    m = folium.Map(location=center, zoom_start=14, tiles=None, control_scale=True)

    folium.TileLayer("OpenStreetMap", name="Street map").add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri World Imagery", name="Satellite").add_to(m)

    folium.raster_layers.ImageOverlay(
        image=str(db_png), bounds=bounds, opacity=0.85,
        name="SAR backscatter (VV dB)").add_to(m)
    folium.raster_layers.ImageOverlay(
        image=str(class_png), bounds=bounds, opacity=0.6,
        name="SAR land cover (urban/veg/water)", show=False).add_to(m)

    # study-area box
    folium.Rectangle([[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]],
                     color="#F1C40F", weight=2, fill=False,
                     tooltip="SAR study area").add_to(m)
    # Parco Ducale box
    folium.Rectangle([[PARCO_LAT[0], PARCO_LON[0]], [PARCO_LAT[1], PARCO_LON[1]]],
                     color="#16A085", weight=2, fill=False,
                     tooltip="Parco Ducale (NDRE study area)").add_to(m)

    if SHOW_STATIONS:
        for name, (la, lo) in STATIONS.items():
            folium.Marker([la, lo], tooltip=name,
                          icon=folium.Icon(color="blue", icon="cloud")).add_to(m)

    # simple legend
    legend = ("<div style='position:fixed;bottom:30px;left:30px;z-index:9999;"
              "background:white;padding:10px 12px;border:1px solid #aaa;border-radius:6px;"
              "font-size:13px;font-family:sans-serif'>"
              "<b>SAR land cover</b><br>"
              "<span style='color:#C0392B'>&#9608;</span> Urban (high backscatter)<br>"
              "<span style='color:#27AE60'>&#9608;</span> Vegetation<br>"
              "<span style='color:#2980B9'>&#9608;</span> Water / shadow</div>")
    m.get_root().html.add_child(folium.Element(legend))

    folium.LayerControl(collapsed=False).add_to(m)
    m.save(str(out_html))



#  Main

def main():
    bounds = [[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]]

    if REBUILD_MAP_ONLY and DB_NPY.exists() and CLASS_NPY.exists():
        print("Reusing saved geocoded arrays (REBUILD_MAP_ONLY).")
        db_ll = np.load(DB_NPY); cl_ll = np.load(CLASS_NPY)
    else:
        zp = Path(LOCAL_S1C_ZIP)
        if not zp.exists():
            print(f"ERROR: scene not found:\n  {zp}")
            return
        safe = None
        try:
            safe = extract_safe(zp, WORK_DIR)
            print("  reading geolocation grid ...")
            lat, lon, line, pix = read_geoloc_grid(safe)
            win = parma_window(lat, lon, line, pix)
            print(f"  Parma window rows {win[0]}-{win[1]}, cols {win[2]}-{win[3]}")
            vv = read_vv_window(safe, win)
            print("  computing backscatter + classification ...")
            db_win, cls_win = process_window(vv)
            print(f"  geocoding to {GRID_NX}x{GRID_NY} lat/lon grid ...")
            db_ll, cl_ll = geocode(db_win, cls_win, lat, lon, line, pix, win)
            np.save(DB_NPY, db_ll); np.save(CLASS_NPY, cl_ll)
            with open(BOUNDS_JSON, "w") as f:
                json.dump({"bounds": bounds}, f)
            print(f"  saved arrays to {OUT_DIR}")
        finally:
            if safe and Path(safe).exists():
                shutil.rmtree(safe, ignore_errors=True)   # delete extracted copy (your zip is untouched)
                print(f"  deleted working copy {Path(safe).name}")

    db_png = OUT_DIR / "overlay_backscatter.png"
    cls_png = OUT_DIR / "overlay_landcover.png"
    db_to_png(db_ll, db_png)
    class_to_png(cl_ll, cls_png)

    out_html = OUT_DIR / "parma_sar_interactive_map.html"
    build_map(db_png, cls_png, bounds, out_html)
    print(f"\nInteractive map: {out_html}")
    print("Open it in a browser - zoom into Parma and toggle the layers (top-right).")


if __name__ == "__main__":
    main()
