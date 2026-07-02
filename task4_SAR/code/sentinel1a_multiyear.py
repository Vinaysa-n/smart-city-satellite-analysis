# -*- coding: utf-8 -*-
"""
Sentinel-1A multi-year SAR backscatter pipeline  (Parma, Task 4 extension)


Extends single June-2025 Sentinel-1C SAR analysis backward to 2022, 2023
and 2024 using Sentinel-1A (the only SAR satellite operating in those years).
For each year it:
    1. Searches the Copernicus Data Space (CDSE) catalogue,
    2. Selects the S1A scene on the SAME relative orbit + direction as your
       2025 S1C scene, in early June (closest to June 3),
    3. Downloads it to the D: drive,
    4. Extracts the .SAFE,
    5. Processes it with EXACTLY your S1C recipe
       (downsample -> 5x5 speckle filter -> 10*log10 -> 75/15 thresholds),
    6. Writes a one-line entry to a small text log + a results CSV,
    7. DELETES the big zip and the extracted .SAFE so the disk never fills.

Your 2025 row is RE-COMPUTED from your local S1C scene through the identical
code (no download), so all four years are directly comparable. Nothing is
hardcoded; every number comes out of a real scene.

CLOUDS: irrelevant. SAR is radar, it sees through cloud day and night.
There is no cloud filtering anywhere in here.

"""

# --- 1. CREDENTIALS (fill these two in) ---
CDSE_USER = "PUT_YOUR_CDSE_EMAIL_HERE"
CDSE_PASS = "PUT_YOUR_CDSE_PASSWORD_HERE"

# --- 2. SAFETY SWITCH ---
# True  = only search & list scenes (no download, no processing). Run this first.
# False = full run: download, extract, process, delete.
SEARCH_ONLY = True

# --- 3. PATHS ---
# Big temporary downloads go on D: (deleted after each scene is processed).
DOWNLOAD_DIR = r"D:\thesis download\s1a_multiyear"
# Logs, CSV and figures (small) go with the rest of your outputs.
OUTPUT_DIR   = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs\sar_multiyear"

#existing local S1C scene = the 2025 reference (re-processed, NOT deleted).
LOCAL_S1C_ZIP = (r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\SAR"
                 r"\S1C_IW_GRDH_1SDV_20250603T052626_20250603T052651_002614_0056AB_BFCF.SAFE.zip")
PROCESS_2025_LOCAL = True   # set False to skip re-processing 2025 (uses CSV only)

# --- 4. STUDY AREA (your Parma box, identical to the S1C script) ---
PARMA_LAT_MIN, PARMA_LAT_MAX = 44.79, 44.83
PARMA_LON_MIN, PARMA_LON_MAX = 10.29, 10.36

# --- 5. WHICH YEARS / WHEN ---
YEARS_TO_ADD   = [2022, 2023, 2024]   # 2025 comes from the local S1C scene
TARGET_MONTH   = 6                    # aim for June (match your June-3 scene)
TARGET_DAY     = 3
SEARCH_FROM    = "05-15"              # search window each year: May 15 ...
SEARCH_TO      = "07-05"              #                         ... July 5
WIDEN_IF_EMPTY = ("04-15", "08-15")   # fallback window if nothing matches

# --- 6. PROCESSING PARAMS (locked to S1C script) ---
DOWNSAMPLE_MAXPX = 1000
SPECKLE_SIZE     = 5
WINDOW_MARGIN_PX = 50
PRISMA_NDRE      = 0.2230
S2_NDRE          = 0.1958
# ============================================================================


# ---------------------- dependencies (auto-install if missing) --------------
import sys, subprocess
def _ensure(pkg, importname=None):
    try:
        __import__(importname or pkg)
    except ImportError:
        print(f"  installing {pkg} ...")
        subprocess.run([sys.executable, "-m", "pip", "install", pkg])
for _p, _i in [("requests", "requests"), ("numpy", "numpy"), ("scipy", "scipy"),
               ("rasterio", "rasterio"), ("pyproj", "pyproj"),
               ("matplotlib", "matplotlib"), ("Pillow", "PIL")]:
    _ensure(_p, _i)

import os, re, time, json, zipfile, shutil, datetime as dt
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")            # save figures to disk without popping windows
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from scipy.interpolate import griddata
from pyproj import Transformer
import warnings
warnings.filterwarnings("ignore")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

DOWNLOAD_DIR = Path(DOWNLOAD_DIR); DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR   = Path(OUTPUT_DIR);   OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT_DIR / "sentinel1a_multiyear_log.txt"
CSV_PATH = OUTPUT_DIR / "sentinel1a_multiyear_results.csv"

IDENTITY_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token")
ODATA_URL    = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

PARMA_POLY = (f"POLYGON(({PARMA_LON_MIN} {PARMA_LAT_MIN},"
              f"{PARMA_LON_MAX} {PARMA_LAT_MIN},"
              f"{PARMA_LON_MAX} {PARMA_LAT_MAX},"
              f"{PARMA_LON_MIN} {PARMA_LAT_MAX},"
              f"{PARMA_LON_MIN} {PARMA_LAT_MIN}))")


def log_line(msg):
    """Print to screen and append to the small log file."""
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


#  PART 1 - CDSE AUTHENTICATION

def get_token():
    """Get a fresh CDSE access token (valid ~10 min)."""
    data = {"grant_type": "password", "username": CDSE_USER,
            "password": CDSE_PASS, "client_id": "cdse-public"}
    r = requests.post(IDENTITY_URL, data=data, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"Login failed ({r.status_code}). "
                           f"Check CDSE_USER / CDSE_PASS.\n{r.text[:300]}")
    return r.json()["access_token"]


#  PART 2 - CATALOGUE SEARCH

def odata_search(start_iso, end_iso, top=200):
    """Return all SENTINEL-1 products intersecting Parma in [start, end]."""
    flt = (f"Collection/Name eq 'SENTINEL-1' and "
           f"OData.CSC.Intersects(area=geography'SRID=4326;{PARMA_POLY}') and "
           f"ContentDate/Start gt {start_iso}T00:00:00.000Z and "
           f"ContentDate/Start lt {end_iso}T23:59:59.999Z")
    params = {"$filter": flt, "$expand": "Attributes",
              "$top": str(top), "$orderby": "ContentDate/Start asc"}
    r = requests.get(ODATA_URL, params=params, timeout=120)
    r.raise_for_status()
    return r.json().get("value", [])


def attr(prod, name, default=None):
    """Read one attribute value from an expanded product."""
    for a in prod.get("Attributes", []):
        if a.get("Name") == name:
            return a.get("Value")
    return default


def scene_date(prod):
    s = prod["ContentDate"]["Start"][:10]
    return dt.date.fromisoformat(s)


def get_reference_orbit():
    """
    Find your 2025 S1C scene in the catalogue and read its relative orbit
    and orbit direction. Falls back to the IW/GRDH/S1C descending scene over
    Parma in early June 2025 if the exact name is not matched.
    """
    prods = odata_search("2025-06-01", "2025-06-06")
    s1c = [p for p in prods
           if p["Name"].startswith("S1C") and "IW_GRDH" in p["Name"]]
    if not s1c:
        raise RuntimeError("Could not find your S1C reference scene in CDSE "
                           "(June 2025 over Parma). Cannot match orbit.")
    # prefer the one whose absolute orbit is 2614 (your exact scene), else closest to Jun 3
    target = dt.date(2025, 6, 3)
    s1c.sort(key=lambda p: ("002614" not in p["Name"], abs((scene_date(p) - target).days)))
    ref = s1c[0]
    ron = int(attr(ref, "relativeOrbitNumber"))
    direction = str(attr(ref, "orbitDirection"))
    log_line(f"  Reference S1C scene : {ref['Name'][:62]}")
    log_line(f"  Reference rel.orbit : {ron}   direction: {direction}")
    return ron, direction


def select_scene_for_year(year, ron, direction):
    """
    Pick the best S1A scene for one year: same relative orbit + direction,
    dual-pol IW GRDH, closest to June 3. Returns (product, note) or (None, reason).
    """
    def search(win_from, win_to):
        return odata_search(f"{year}-{win_from}", f"{year}-{win_to}")

    prods = search(SEARCH_FROM, SEARCH_TO)
    if not prods:
        prods = search(*WIDEN_IF_EMPTY)

    # keep S1A, IW GRDH only
    cand = [p for p in prods
            if p["Name"].startswith("S1A") and "IW_GRDH" in p["Name"]]
    if not cand:
        return None, f"{year}: no S1A IW_GRDH scenes found over Parma."

    target = dt.date(year, TARGET_MONTH, TARGET_DAY)

    # 1st choice: same relative orbit AND same direction, dual-pol
    strict = [p for p in cand
              if str(attr(p, "orbitDirection")) == direction
              and int(attr(p, "relativeOrbitNumber", -1)) == ron
              and "1SDV" in p["Name"]]
    if strict:
        strict.sort(key=lambda p: abs((scene_date(p) - target).days))
        return strict[0], "exact orbit + direction match (dual-pol)"

    # 2nd choice: same direction, dual-pol, any relative orbit covering Parma
    same_dir = [p for p in cand
                if str(attr(p, "orbitDirection")) == direction
                and "1SDV" in p["Name"]]
    if same_dir:
        same_dir.sort(key=lambda p: abs((scene_date(p) - target).days))
        chosen = same_dir[0]
        chosen_ron = int(attr(chosen, "relativeOrbitNumber", -1))
        return chosen, (f"FALLBACK: same direction but rel.orbit "
                        f"{chosen_ron} != reference {ron} (geometry caveat - note this)")

    # 3rd choice: anything dual-pol covering Parma
    dual = [p for p in cand if "1SDV" in p["Name"]]
    pool = dual if dual else cand
    pool.sort(key=lambda p: abs((scene_date(p) - target).days))
    chosen = pool[0]
    return chosen, ("FALLBACK: different direction/orbit - geometry differs, "
                    "use with caution and document")


#  PART 3 - DOWNLOAD  (streamed, manual redirect so the token survives hops)

def download_product(prod, token):
    """Download one product zip to DOWNLOAD_DIR. Returns the zip Path."""
    pid  = prod["Id"]
    name = prod["Name"]
    out  = DOWNLOAD_DIR / (name + ".zip")
    url  = f"{ODATA_URL}({pid})/$value"

    headers = {"Authorization": f"Bearer {token}"}
    sess = requests.Session()
    # follow redirects by hand so Authorization is re-sent to the download node
    r = sess.get(url, headers=headers, allow_redirects=False, stream=True, timeout=120)
    hops = 0
    while r.status_code in (301, 302, 303, 307, 308) and hops < 6:
        nxt = r.headers["Location"]
        r = sess.get(nxt, headers=headers, allow_redirects=False, stream=True, timeout=120)
        hops += 1
    r.raise_for_status()

    total = int(r.headers.get("Content-Length", 0))
    done = 0; last = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
            if chunk:
                f.write(chunk); done += len(chunk)
                if total and done - last > 100 * 1024 * 1024:   # progress every ~100 MB
                    print(f"    {done/1e6:7.0f} / {total/1e6:.0f} MB", end="\r")
                    last = done
    size_mb = out.stat().st_size / 1e6
    print(f"    downloaded {size_mb:.0f} MB                       ")
    if size_mb < 50:   # a real S1 GRD is ~1 GB; tiny file => something went wrong
        raise RuntimeError(f"Downloaded file is only {size_mb:.1f} MB - likely an "
                           f"error page, not a product. Aborting this scene.")
    return out


#  PART 4 - PROCESSING

def find_safe_folder(zip_path, extract_to):
    """Extract the .SAFE from a zip into extract_to. Returns the .SAFE Path."""
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        safe_dirs = {n.split("/")[0] for n in names if n.split("/")[0].endswith(".SAFE")}
        if not safe_dirs:
            safe_dirs = {n.split("/")[0] for n in names if "measurement" in n}
        if not safe_dirs:
            raise RuntimeError(f"No .SAFE folder inside {zip_path.name}")
        safe_name = sorted(safe_dirs)[0]
        target = Path(extract_to) / safe_name
        if not target.exists():
            print(f"    extracting {safe_name} ...")
            zf.extractall(extract_to)
        return target


def derive_parma_window(safe_path):
    """
    Per-scene Parma pixel window from the annotation geolocation grid
    (your cells 3+4). Interpolates the Parma corners to row/col, +margin.
    """
    annot = list((Path(safe_path) / "annotation").glob("*.xml"))
    if not annot:
        raise RuntimeError("No annotation XML found in SAFE.")
    root = ET.parse(annot[0]).getroot()
    pts = root.findall(".//geolocationGridPoint")
    lats = np.array([float(p.find("latitude").text)  for p in pts])
    lons = np.array([float(p.find("longitude").text) for p in pts])
    rows = np.array([int(p.find("line").text)        for p in pts])
    cols = np.array([int(p.find("pixel").text)       for p in pts])

    corners = np.array([
        [PARMA_LAT_MIN, PARMA_LON_MIN], [PARMA_LAT_MIN, PARMA_LON_MAX],
        [PARMA_LAT_MAX, PARMA_LON_MIN], [PARMA_LAT_MAX, PARMA_LON_MAX],
    ])
    pr = griddata(np.column_stack([lats, lons]), rows, corners, method="linear")
    pc = griddata(np.column_stack([lats, lons]), cols, corners, method="linear")
    if np.any(np.isnan(pr)) or np.any(np.isnan(pc)):
        raise RuntimeError("Parma falls outside this scene's geolocation grid.")

    r0 = int(np.nanmin(pr)) - WINDOW_MARGIN_PX
    r1 = int(np.nanmax(pr)) + WINDOW_MARGIN_PX
    c0 = int(np.nanmin(pc)) - WINDOW_MARGIN_PX
    c1 = int(np.nanmax(pc)) + WINDOW_MARGIN_PX
    return max(0, r0), r1, max(0, c0), c1


def load_vv_vh(safe_path, window):
    """Windowed read of VV and VH GRD bands (linear DN), Parma only."""
    import rasterio
    from rasterio.windows import Window
    r0, r1, c0, c1 = window
    meas = Path(safe_path) / "measurement"
    tiffs = list(meas.glob("*.tiff")) + list(meas.glob("*.tif"))
    bands = {}
    for t in tiffs:
        nm = t.name.lower()
        pol = "VV" if "vv" in nm else ("VH" if "vh" in nm else None)
        if pol is None:
            continue
        with rasterio.open(t) as src:
            H, W = src.height, src.width
            rr1, cc1 = min(r1, H), min(c1, W)
            win = Window(col_off=c0, row_off=r0, width=cc1 - c0, height=rr1 - r0)
            bands[pol] = src.read(1, window=win).astype(np.float32)
    if "VV" not in bands:
        raise RuntimeError("No VV band found in this scene.")
    return bands


def _downsample(arr, max_px=DOWNSAMPLE_MAXPX):
    f = 1
    while max(arr.shape) // f > max_px:
        f *= 2
    return arr[::f, ::f] if f > 1 else arr


def _to_db(arr, eps=1e-10):
    return 10 * np.log10(np.maximum(arr, eps))


def process_sar(bands):
    """
    Your exact recipe: downsample -> 5x5 boxcar speckle filter -> 10*log10 ->
    valid mask (> -40) -> 75th/15th percentile thresholds -> class means & %.
    Returns a dict of metrics.
    """
    vv_lin = _downsample(bands["VV"])
    vv_filt = uniform_filter(vv_lin.astype(np.float64), size=SPECKLE_SIZE).astype(np.float32)
    vv_db = _to_db(vv_filt)
    valid_vv = vv_db[vv_db > -40]

    has_vh = "VH" in bands
    if has_vh:
        vh_lin = _downsample(bands["VH"])
        vh_filt = uniform_filter(vh_lin.astype(np.float64), size=SPECKLE_SIZE).astype(np.float32)
        vh_db = _to_db(vh_filt)

    thr_urban = np.percentile(valid_vv, 75)
    thr_water = np.percentile(valid_vv, 15)
    urban = vv_db >= thr_urban
    water = vv_db <= thr_water
    veg   = (~urban) & (~water) & (vv_db > -35)

    m = {
        "shape": f"{vv_db.shape[0]}x{vv_db.shape[1]}",
        "n_valid_px": int((vv_db > -40).sum()),
        "vv_mean_db": float(valid_vv.mean()),
        "vv_urban_db": float(vv_db[urban].mean()),
        "vv_veg_db": float(vv_db[veg].mean()),
        "urban_pct": float(urban.mean() * 100),
        "veg_pct": float(veg.mean() * 100),
        "water_pct": float(water.mean() * 100),
    }
    m["dVV_db"] = m["vv_urban_db"] - m["vv_veg_db"]
    if has_vh:
        m["vh_mean_db"]  = float(vh_db[vh_db > -40].mean())
        m["vh_urban_db"] = float(vh_db[urban].mean())
        m["vh_veg_db"]   = float(vh_db[veg].mean())
        m["dVH_db"]      = m["vh_urban_db"] - m["vh_veg_db"]
    else:
        m["vh_mean_db"] = m["vh_urban_db"] = m["vh_veg_db"] = m["dVH_db"] = float("nan")
    return m


#  PART 5 - PER-SCENE LIFECYCLE

def process_one_scene(safe_path):
    win = derive_parma_window(safe_path)
    log_line(f"    Parma window rows {win[0]}-{win[1]}, cols {win[2]}-{win[3]}")
    bands = load_vv_vh(safe_path, win)
    return process_sar(bands)


def handle_year(year, ron, direction, token_getter):
    """Search -> download -> extract -> process -> log -> delete, for one year."""
    log_line(f"\n----- {year} -----")
    prod, note = select_scene_for_year(year, ron, direction)
    if prod is None:
        log_line(f"  [SKIP] {note}")
        return None
    d = scene_date(prod)
    offset = (d - dt.date(year, TARGET_MONTH, TARGET_DAY)).days
    info = {
        "year": year, "date": d.isoformat(), "platform": prod["Name"][:3],
        "rel_orbit": int(attr(prod, "relativeOrbitNumber", -1)),
        "direction": str(attr(prod, "orbitDirection")),
        "day_offset_from_jun3": offset, "match": note, "product_name": prod["Name"],
    }
    log_line(f"  Selected : {prod['Name'][:62]}")
    log_line(f"  Date {d}  (offset {offset:+d} d)  relOrbit {info['rel_orbit']}  "
             f"{info['direction']}")
    log_line(f"  Match    : {note}")

    if SEARCH_ONLY:
        log_line("  [SEARCH_ONLY] not downloading.")
        return info

    zip_path = None; safe_path = None
    try:
        log_line("  downloading ...")
        zip_path = download_product(prod, token_getter())
        safe_path = find_safe_folder(zip_path, DOWNLOAD_DIR)
        log_line("  processing ...")
        info.update(process_one_scene(safe_path))
        log_line(f"  VV urban={info['vv_urban_db']:.2f} dB  veg={info['vv_veg_db']:.2f} dB  "
                 f"dVV={info['dVV_db']:.2f} dB")
        log_line(f"  Class urban={info['urban_pct']:.1f}%  veg={info['veg_pct']:.1f}%  "
                 f"water={info['water_pct']:.1f}%")
    finally:
        # delete the big files no matter what, so the disk never fills
        for p in (safe_path, zip_path):
            try:
                if p and Path(p).exists():
                    if Path(p).is_dir():
                        shutil.rmtree(p, ignore_errors=True)
                    else:
                        Path(p).unlink()
                    log_line(f"  deleted {Path(p).name}")
            except Exception as e:
                log_line(f"  (could not delete {p}: {e})")
    return info


def handle_local_2025():
    """Re-process the local S1C scene (no download, never deleted)."""
    log_line("\n----- 2025 (local S1C) -----")
    zp = Path(LOCAL_S1C_ZIP)
    if not zp.exists():
        log_line(f"  [SKIP] local S1C zip not found: {zp}")
        return None
    info = {"year": 2025, "date": "2025-06-03", "platform": "S1C",
            "rel_orbit": -1, "direction": "", "day_offset_from_jun3": 0,
            "match": "local reference scene", "product_name": zp.stem}
    if SEARCH_ONLY:
        log_line("  [SEARCH_ONLY] not processing local scene.")
        return info
    # extract a working copy onto D:, process, then delete only the copy
    safe_path = None
    try:
        safe_path = find_safe_folder(zp, DOWNLOAD_DIR)
        info.update(process_one_scene(safe_path))
        log_line(f"  VV urban={info['vv_urban_db']:.2f} dB  veg={info['vv_veg_db']:.2f} dB  "
                 f"dVV={info['dVV_db']:.2f} dB  (compare to your 26.4/23.1/3.3)")
        log_line(f"  Class urban={info['urban_pct']:.1f}%  veg={info['veg_pct']:.1f}%  "
                 f"water={info['water_pct']:.1f}%")
    finally:
        if safe_path and Path(safe_path).exists():
            shutil.rmtree(safe_path, ignore_errors=True)   # delete the COPY on D:, not your C: zip
            log_line(f"  deleted working copy {Path(safe_path).name}")
    return info


#  PART 6 - OUTPUTS (CSV + comparison figure)

def write_csv(rows):
    cols = ["year", "date", "platform", "rel_orbit", "direction",
            "day_offset_from_jun3", "vv_mean_db", "vv_urban_db", "vv_veg_db",
            "dVV_db", "vh_mean_db", "vh_urban_db", "vh_veg_db", "dVH_db",
            "urban_pct", "veg_pct", "water_pct", "n_valid_px", "shape",
            "match", "product_name"]
    import csv
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["year"]):
            w.writerow(r)
    log_line(f"\nResults CSV: {CSV_PATH}")


def make_figure(rows):
    rows = [r for r in sorted(rows, key=lambda x: x["year"]) if "vv_urban_db" in r]
    if not rows:
        return
    yrs = [str(r["year"]) for r in rows]
    x = np.arange(len(yrs))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sentinel-1 multi-year SAR over Parma (S1A 2022-24 + S1C 2025)",
                 fontweight="bold", fontsize=12)

    vu = [r["vv_urban_db"] for r in rows]
    vv_ = [r["vv_veg_db"] for r in rows]
    b1 = ax[0].bar(x - 0.18, vu, 0.34, label="Urban VV", color="#C0392B", alpha=.85)
    b2 = ax[0].bar(x + 0.18, vv_, 0.34, label="Vegetation VV", color="#27AE60", alpha=.85)
    for bars in (b1, b2):
        for bar in bars:
            ax[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                       f"{bar.get_height():.1f}", ha="center", fontsize=8)
    for i, r in enumerate(rows):
        ax[0].text(i, max(vu) + 1.2, f"Δ {r['dVV_db']:.1f}", ha="center",
                   fontsize=8, fontweight="bold", color="#34495E")
    ax[0].set_xticks(x); ax[0].set_xticklabels(yrs)
    ax[0].set_ylabel("Mean VV backscatter (dB)")
    ax[0].set_title("Urban vs vegetation separation by year")
    ax[0].legend(); ax[0].grid(True, axis="y", alpha=.3)

    up = [r["urban_pct"] for r in rows]
    vp = [r["veg_pct"] for r in rows]
    wp = [r["water_pct"] for r in rows]
    ax[1].bar(x, up, 0.55, label="Urban", color="#C0392B", alpha=.85)
    ax[1].bar(x, vp, 0.55, bottom=up, label="Vegetation", color="#27AE60", alpha=.85)
    ax[1].bar(x, wp, 0.55, bottom=np.array(up) + np.array(vp),
              label="Water/shadow", color="#2980B9", alpha=.85)
    ax[1].set_xticks(x); ax[1].set_xticklabels(yrs)
    ax[1].set_ylabel("Coverage (%)"); ax[1].set_ylim(0, 100)
    ax[1].set_title("SAR land-cover split by year")
    ax[1].legend(loc="upper right"); ax[1].grid(True, axis="y", alpha=.3)

    plt.tight_layout()
    out = OUTPUT_DIR / "sar_multiyear_comparison.png"
    plt.savefig(out, dpi=150, bbox_inches="tight"); plt.close()
    log_line(f"Figure: {out}")


def print_table(rows):
    rows = [r for r in sorted(rows, key=lambda x: x["year"]) if "vv_urban_db" in r]
    if not rows:
        log_line("\n(No processed scenes to tabulate - SEARCH_ONLY or all skipped.)")
        return
    log_line("\n" + "=" * 78)
    log_line("MULTI-YEAR SAR SUMMARY")
    log_line("=" * 78)
    log_line(f"{'Year':<6}{'Date':<12}{'Sat':<5}{'VV_urb':>8}{'VV_veg':>8}"
             f"{'ΔVV':>7}{'Urban%':>8}{'Veg%':>7}{'Water%':>8}")
    for r in rows:
        log_line(f"{r['year']:<6}{r['date']:<12}{r['platform']:<5}"
                 f"{r['vv_urban_db']:>8.1f}{r['vv_veg_db']:>8.1f}{r['dVV_db']:>7.1f}"
                 f"{r['urban_pct']:>8.1f}{r['veg_pct']:>7.1f}{r['water_pct']:>8.1f}")


#  MAIN
def main():
    open(LOG_PATH, "w").close()   # fresh log
    log_line("=" * 78)
    log_line("Sentinel-1A multi-year SAR pipeline - Parma (Task 4 extension)")
    log_line(f"Run: {dt.datetime.now():%Y-%m-%d %H:%M}   SEARCH_ONLY={SEARCH_ONLY}")
    log_line("=" * 78)

    if CDSE_USER.startswith("PUT_YOUR"):
        log_line("\n*** Fill in CDSE_USER and CDSE_PASS at the top of the file first. ***")
        return

    log_line("\n[1] Logging in to CDSE & reading your S1C reference orbit ...")
    _ = get_token()                       # verify credentials early
    ron, direction = get_reference_orbit()

    # one fresh token per download keeps us inside the ~10-min token lifetime
    def token_getter():
        return get_token()

    rows = []
    for y in YEARS_TO_ADD:
        try:
            info = handle_year(y, ron, direction, token_getter)
            if info:
                rows.append(info)
        except Exception as e:
            log_line(f"  [ERROR {y}] {e}")

    if PROCESS_2025_LOCAL:
        try:
            info = handle_local_2025()
            if info:
                rows.append(info)
        except Exception as e:
            log_line(f"  [ERROR 2025] {e}")

    if not SEARCH_ONLY and rows:
        write_csv(rows)
        make_figure(rows)
    print_table(rows)

    log_line("\nDone.")
    if SEARCH_ONLY:
        log_line("This was a SEARCH-ONLY preview. If the scenes above look right, "
                 "set SEARCH_ONLY = False and run again to download + process.")


if __name__ == "__main__":
    main()
