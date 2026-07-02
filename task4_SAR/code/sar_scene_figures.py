"""
SAR backscatter scene figures, multi-year  (Parma, Task 4)

Produces the actual backscatter IMAGES per year (2022-2025), in the style of 
original single-scene figures: a grayscale VV (dB) panel + the
urban/vegetation/water classification panel, for each year, plus combined
four-year comparison panels.

Workflow: for each year - download -> extract -> compute ->
SAVE the small dB + classification arrays -> DELETE the big files. 2025 uses
LOCAL S1C scene (already downloaded). On re-runs, set REBUILD_FIGURES_ONLY = True
to skip all downloading and just re-render from the saved arrays.

Same scene selection as your summary pipeline: Sentinel-1A on the SAME relative
orbit + descending pass as your 2025 S1C scene, early June each year.

HOW TO RUN
----------
1. Fill in CDSE_USER / CDSE_PASS (needed to download 2022-2024).
2. Run:  python sar_scene_figures.py   (or exec(...) in a notebook)
3. First run downloads 3 scenes (~1 GB each, deleted after). Then re-runs with
   REBUILD_FIGURES_ONLY = True re-render instantly from saved arrays.

"""

# ============================ CONFIG ========================================
CDSE_USER = "PUT_YOUR_CDSE_EMAIL_HERE"
CDSE_PASS = "PUT_YOUR_CDSE_PASSWORD_HERE"

LOCAL_S1C_ZIP = (r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\SAR"
                 r"\S1C_IW_GRDH_1SDV_20250603T052626_20250603T052651_002614_0056AB_BFCF.SAFE.zip")

WORK_DIR   = r"D:\thesis download\sar_scene_work"                       # big temp files (deleted)
ARRAYS_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\sar_scenes\arrays"   # saved small arrays
OUT_DIR    = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\sar_scenes\figures"  # rendered figures

YEARS_TO_DOWNLOAD = [2022, 2023, 2024]   # 2025 comes from the local S1C scene
REBUILD_FIGURES_ONLY = False             # True = skip downloads, re-render from saved arrays

# Parma study box + scene-selection window (same as your summary pipeline)
LAT_MIN, LAT_MAX = 44.79, 44.83
LON_MIN, LON_MAX = 10.29, 10.36
SEARCH_FROM, SEARCH_TO = "05-15", "07-05"
WIDEN_IF_EMPTY = ("04-15", "08-15")
TARGET_MONTH, TARGET_DAY = 6, 3
SPECKLE_SIZE = 5


import os, sys, json, zipfile, shutil, subprocess, datetime as dt
from pathlib import Path
import xml.etree.ElementTree as ET


def _ensure(pkg, importname=None):
    try:
        __import__(importname or pkg)
    except ImportError:
        print(f"  installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])


for _p, _i in [("requests", "requests"), ("numpy", "numpy"), ("scipy", "scipy"),
               ("rasterio", "rasterio"), ("matplotlib", "matplotlib")]:
    _ensure(_p, _i)

import numpy as np
import requests
from scipy.ndimage import uniform_filter
from scipy.interpolate import griddata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = Path(WORK_DIR); ARRAYS_DIR = Path(ARRAYS_DIR); OUT_DIR = Path(OUT_DIR)
for d in (WORK_DIR, ARRAYS_DIR, OUT_DIR):
    d.mkdir(parents=True, exist_ok=True)

IDENTITY_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token")
ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
PARMA_POLY = (f"POLYGON(({LON_MIN} {LAT_MIN},{LON_MAX} {LAT_MIN},"
              f"{LON_MAX} {LAT_MAX},{LON_MIN} {LAT_MAX},{LON_MIN} {LAT_MIN}))")

CLASS_COLORS = ["#2980B9", "#27AE60", "#C0392B"]      # 0 water, 1 veg, 2 urban
CLASS_NAMES = ["Water / shadow", "Vegetation", "Urban"]



#  CDSE

def get_token():
    r = requests.post(IDENTITY_URL, data={
        "grant_type": "password", "username": CDSE_USER,
        "password": CDSE_PASS, "client_id": "cdse-public"}, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Login failed ({r.status_code}). Check CDSE_USER/CDSE_PASS.")
    return r.json()["access_token"]


def odata_search(start_iso, end_iso, top=200):
    flt = (f"Collection/Name eq 'SENTINEL-1' and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{PARMA_POLY}') and "
           f"ContentDate/Start gt {start_iso}T00:00:00.000Z and "
           f"ContentDate/Start lt {end_iso}T23:59:59.999Z")
    r = requests.get(ODATA_URL, params={
        "$filter": flt, "$expand": "Attributes", "$top": str(top),
        "$orderby": "ContentDate/Start asc"}, timeout=120)
    r.raise_for_status()
    return r.json().get("value", [])


def attr(prod, name, default=None):
    for a in prod.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return default


def scene_date(prod):
    return dt.date.fromisoformat(prod["ContentDate"]["Start"][:10])


def get_reference_orbit():
    prods = odata_search("2025-06-01", "2025-06-06")
    s1c = [p for p in prods if p["Name"].startswith("S1C") and "IW_GRDH" in p["Name"]]
    if not s1c:
        raise RuntimeError("Could not find S1C reference scene in CDSE.")
    target = dt.date(2025, 6, 3)
    s1c.sort(key=lambda p: ("002614" not in p["Name"], abs((scene_date(p) - target).days)))
    ref = s1c[0]
    return int(attr(ref, "relativeOrbitNumber")), str(attr(ref, "orbitDirection"))


def select_scene_for_year(year, ron, direction):
    def search(a, b):
        return odata_search(f"{year}-{a}", f"{year}-{b}")
    prods = search(SEARCH_FROM, SEARCH_TO) or search(*WIDEN_IF_EMPTY)
    cand = [p for p in prods if p["Name"].startswith("S1A") and "IW_GRDH" in p["Name"]]
    if not cand:
        return None
    target = dt.date(year, TARGET_MONTH, TARGET_DAY)
    strict = [p for p in cand
              if str(attr(p, "orbitDirection")) == direction
              and int(attr(p, "relativeOrbitNumber", -1)) == ron and "1SDV" in p["Name"]]
    pool = strict or [p for p in cand if "1SDV" in p["Name"]] or cand
    pool.sort(key=lambda p: abs((scene_date(p) - target).days))
    return pool[0]


def download_product(prod, token):
    out = WORK_DIR / (prod["Name"] + ".zip")
    url = f"{ODATA_URL}({prod['Id']})/$value"
    headers = {"Authorization": f"Bearer {token}"}
    sess = requests.Session()
    r = sess.get(url, headers=headers, allow_redirects=False, stream=True, timeout=120)
    hops = 0
    while r.status_code in (301, 302, 303, 307, 308) and hops < 6:
        r = sess.get(r.headers["Location"], headers=headers,
                     allow_redirects=False, stream=True, timeout=120)
        hops += 1
    r.raise_for_status()
    total = int(r.headers.get("Content-Length", 0)); done = last = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(8 * 1024 * 1024):
            if chunk:
                f.write(chunk); done += len(chunk)
                if total and done - last > 100 * 1024 * 1024:
                    print(f"    {done/1e6:6.0f}/{total/1e6:.0f} MB", end="\r"); last = done
    if out.stat().st_size < 50 * 1e6:
        raise RuntimeError("Download too small - probably an error page.")
    print(f"    downloaded {out.stat().st_size/1e6:.0f} MB            ")
    return out


#  Processing

def find_safe_folder(zip_path, target):
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        safes = {n.split("/")[0] for n in names if n.split("/")[0].endswith(".SAFE")}
        safe = sorted(safes)[0]
        out = Path(target) / safe
        if not out.exists():
            print(f"    extracting {safe} ...")
            zf.extractall(target)
        return out


def derive_parma_window(safe, margin=60):
    annot = sorted((Path(safe) / "annotation").glob("*.xml"))
    root = ET.parse(annot[0]).getroot()
    pts = root.findall(".//geolocationGridPoint")
    lat = np.array([float(p.find("latitude").text) for p in pts])
    lon = np.array([float(p.find("longitude").text) for p in pts])
    line = np.array([int(p.find("line").text) for p in pts])
    pix = np.array([int(p.find("pixel").text) for p in pts])
    corners = np.array([[LAT_MIN, LON_MIN], [LAT_MIN, LON_MAX],
                        [LAT_MAX, LON_MIN], [LAT_MAX, LON_MAX]])
    pr = griddata(np.column_stack([lat, lon]), line, corners, method="linear")
    pc = griddata(np.column_stack([lat, lon]), pix, corners, method="linear")
    r0 = max(0, int(np.nanmin(pr)) - margin); r1 = int(np.nanmax(pr)) + margin
    c0 = max(0, int(np.nanmin(pc)) - margin); c1 = int(np.nanmax(pc)) + margin
    return r0, r1, c0, c1


def load_vv_window(safe, win):
    import rasterio
    from rasterio.windows import Window
    r0, r1, c0, c1 = win
    meas = Path(safe) / "measurement"
    for t in list(meas.glob("*.tiff")) + list(meas.glob("*.tif")):
        if "vv" in t.name.lower():
            with rasterio.open(t) as src:
                rr1, cc1 = min(r1, src.height), min(c1, src.width)
                return src.read(1, window=Window(c0, r0, cc1 - c0, rr1 - r0)).astype(np.float32)
    raise RuntimeError("No VV band found.")


def process_scene(vv_lin):
    filt = uniform_filter(vv_lin.astype(np.float64), size=SPECKLE_SIZE).astype(np.float32)
    db = 10 * np.log10(np.maximum(filt, 1e-10))
    valid = db[db > -40]
    tu = np.percentile(valid, 75); tw = np.percentile(valid, 15)
    cls = np.full(db.shape, np.nan, np.float32)
    cls[db <= tw] = 0
    cls[(db > tw) & (db < tu) & (db > -35)] = 1
    cls[db >= tu] = 2
    cls[db <= -40] = np.nan
    stats = {"vv_urban_db": float(db[db >= tu].mean()),
             "vv_veg_db": float(db[(db > tw) & (db < tu) & (db > -35)].mean())}
    return db, cls, stats


#  Save / load small arrays

def save_arrays(year, db, cls, meta):
    np.save(ARRAYS_DIR / f"db_{year}.npy", db.astype(np.float32))
    np.save(ARRAYS_DIR / f"cls_{year}.npy", cls.astype(np.float32))
    with open(ARRAYS_DIR / f"meta_{year}.json", "w") as f:
        json.dump(meta, f)


def load_arrays(year):
    db = np.load(ARRAYS_DIR / f"db_{year}.npy")
    cls = np.load(ARRAYS_DIR / f"cls_{year}.npy")
    meta = json.load(open(ARRAYS_DIR / f"meta_{year}.json"))
    return db, cls, meta


#  Figures

def fig_one_scene(year, db, cls, meta):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.suptitle(f"Sentinel-1 SAR over Parma \u2014 {meta['date']} ({meta['platform']})",
                 fontweight="bold", fontsize=13)

    finite = db[np.isfinite(db)]
    vmin, vmax = np.percentile(finite, 5), np.percentile(finite, 95)
    im = ax[0].imshow(db, cmap="gray", vmin=vmin, vmax=vmax)
    ax[0].set_title(f"VV backscatter (dB)   urban {meta['vv_urban_db']:.1f} / "
                    f"veg {meta['vv_veg_db']:.1f}")
    ax[0].axis("off")
    cb = fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04); cb.set_label("dB")

    cmap = ListedColormap(CLASS_COLORS)
    ax[1].imshow(cls, cmap=cmap, vmin=0, vmax=2)
    ax[1].set_title("Land-cover classification")
    ax[1].axis("off")
    ax[1].legend(handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)],
                 loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)

    fig.text(0.5, 0.01, "Radar geometry (descending, rel. orbit 168) \u2014 not north-aligned",
             ha="center", fontsize=8, color="#888")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    p = OUT_DIR / f"sar_scene_{year}.png"
    fig.savefig(p, dpi=180, bbox_inches="tight"); plt.close(fig)
    return p


def fig_combined(scenes, kind):
    """kind = 'backscatter' or 'classification'. scenes = {year:(db,cls,meta)}."""
    years = sorted(scenes)
    n = len(years)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    if kind == "backscatter":
        allf = np.concatenate([scenes[y][0][np.isfinite(scenes[y][0])] for y in years])
        vmin, vmax = np.percentile(allf, 5), np.percentile(allf, 95)   # common scale
        fig.suptitle("Sentinel-1 VV backscatter over Parma, 2022\u20132025",
                     fontweight="bold", fontsize=14)
        for k, y in enumerate(years):
            db, _, meta = scenes[y]
            im = axes[k].imshow(db, cmap="gray", vmin=vmin, vmax=vmax)
            axes[k].set_title(f"{y}  ({meta['platform']})"); axes[k].axis("off")
        fig.subplots_adjust(right=0.9)
        cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax).set_label("VV (dB)")
        out = OUT_DIR / "sar_backscatter_allyears.png"
    else:
        cmap = ListedColormap(CLASS_COLORS)
        fig.suptitle("SAR land-cover classification over Parma, 2022\u20132025",
                     fontweight="bold", fontsize=14)
        for k, y in enumerate(years):
            _, cls, meta = scenes[y]
            axes[k].imshow(cls, cmap=cmap, vmin=0, vmax=2)
            axes[k].set_title(f"{y}  ({meta['platform']})"); axes[k].axis("off")
        fig.legend(handles=[Patch(color=c, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)],
                   loc="lower center", ncol=3, fontsize=10, frameon=False)
        out = OUT_DIR / "sar_classification_allyears.png"

    for k in range(len(years), len(axes)):
        axes[k].axis("off")
    fig.tight_layout(rect=[0, 0.04, 0.9 if kind == "backscatter" else 1, 0.95])
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    return out


#  Per-scene lifecycle

def process_local_s1c():
    zp = Path(LOCAL_S1C_ZIP)
    if not zp.exists():
        print(f"  [SKIP] local S1C not found: {zp}"); return None
    safe = None
    try:
        safe = find_safe_folder(zp, WORK_DIR)
        win = derive_parma_window(safe)
        db, cls, stats = process_scene(load_vv_window(safe, win))
        meta = {"date": "2025-06-03", "platform": "S1C", **stats}
        save_arrays(2025, db, cls, meta)
        print(f"  2025 saved  (urban {stats['vv_urban_db']:.1f} / veg {stats['vv_veg_db']:.1f})")
        return 2025
    finally:
        if safe and Path(safe).exists():
            shutil.rmtree(safe, ignore_errors=True)
            print(f"  deleted working copy {Path(safe).name}")


def process_year(year, ron, direction, token_getter):
    print(f"\n----- {year} -----")
    prod = select_scene_for_year(year, ron, direction)
    if prod is None:
        print(f"  [SKIP] no S1A scene found for {year}"); return None
    d = scene_date(prod)
    print(f"  {prod['Name'][:60]}  ({d})")
    zip_path = safe = None
    try:
        print("  downloading ...")
        zip_path = download_product(prod, token_getter())
        safe = find_safe_folder(zip_path, WORK_DIR)
        win = derive_parma_window(safe)
        db, cls, stats = process_scene(load_vv_window(safe, win))
        meta = {"date": d.isoformat(), "platform": prod["Name"][:3], **stats}
        save_arrays(year, db, cls, meta)
        print(f"  {year} saved  (urban {stats['vv_urban_db']:.1f} / veg {stats['vv_veg_db']:.1f})")
        return year
    finally:
        for p in (safe, zip_path):
            if p and Path(p).exists():
                (shutil.rmtree(p, ignore_errors=True) if Path(p).is_dir() else Path(p).unlink())
                print(f"  deleted {Path(p).name}")


#  Main
def main():
    have = []
    if REBUILD_FIGURES_ONLY:
        have = [y for y in (2022, 2023, 2024, 2025)
                if (ARRAYS_DIR / f"db_{y}.npy").exists()]
        if not have:
            print("REBUILD_FIGURES_ONLY set but no saved arrays found - doing a full run.")
    if not have:
        if not REBUILD_FIGURES_ONLY:
            if CDSE_USER.startswith("PUT_YOUR"):
                print("*** Fill in CDSE_USER and CDSE_PASS first. ***"); return
            print("[1] CDSE login + reading reference orbit ...")
            _ = get_token()
            ron, direction = get_reference_orbit()
            print(f"    reference rel.orbit {ron}, {direction}")
            for y in YEARS_TO_DOWNLOAD:
                try:
                    if process_year(y, ron, direction, get_token):
                        have.append(y)
                except Exception as e:
                    print(f"  [ERROR {y}] {e}")
        try:
            if process_local_s1c():
                have.append(2025)
        except Exception as e:
            print(f"  [ERROR 2025] {e}")

    if not have:
        print("\nNo scenes available to plot."); return

    print("\nRendering figures ...")
    scenes = {}
    for y in sorted(set(have)):
        db, cls, meta = load_arrays(y)
        scenes[y] = (db, cls, meta)
        print("  -", fig_one_scene(y, db, cls, meta).name)
    print("  -", fig_combined(scenes, "backscatter").name)
    print("  -", fig_combined(scenes, "classification").name)
    print(f"\nFigures in: {OUT_DIR}")
    print(f"Arrays saved in: {ARRAYS_DIR}  (re-render anytime with REBUILD_FIGURES_ONLY = True)")


if __name__ == "__main__":
    main()
