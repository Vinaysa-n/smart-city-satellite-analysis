"""
download_and_process_task3.py   (Parco Ducale, matches xgboost)
--------------------------------------------------------------------

Method is lifted from (scene-level, no-leakage) classifier, with the
bounding box set to PARCO DUCALE as you confirmed:
  * bands   : B05 + B8A at 20 m
  * scaling : DN / 10000  (NO -1000 offset)  -> your 0.196 scale
  * NDRE    : (B8A - B05)/(B8A + B05), masked (0.001 < refl < 0.9)
  * LAI     : Beer-Lambert / Clevers average, derived from NDRE
  * 12 scene-level features written per scene (feeds Stage C labeling)

For EACH of the 48 manifest scenes it gets the clipped bands the cheapest way:
  1. from a tiny cached .npz if we've processed this date before, else
  2. from an existing zip already on disk in SP2 / SP2_2024, else
  3. by downloading from CDSE (your login), then deletes the big zip.
So the ~40 GB download happens at most once, and re-runs are free.

TEST_FIRST_N = 2 below -> processes only the first 2 scenes. Confirm the
printed NDRE looks right (~0.19-0.20 over Parco Ducale), then set it to 0
for all 48.

REQUIRES (already installed from earlier tasks; install only if missing):
    conda install -c conda-forge rasterio pyproj
    pip install requests numpy
"""

import os
import re
import csv
import time
import zipfile
import getpass
from pathlib import Path

import numpy as np
import requests
import rasterio
from pyproj import Transformer

# ----------------------------------------------------------------------
# CONFIG - paths already filled in for your machine.
# ----------------------------------------------------------------------
BASE         = Path(r"C:\Users\vinay\OneDrive\Desktop\Thesis")
OUT_DIR      = BASE / "outputs"
MANIFEST     = OUT_DIR / "task3_scene_manifest_2022_2025.csv"
FEATURES_CSV = OUT_DIR / "task3_features_2022_2025.csv"
WORK_DIR     = Path(r"D:\thesis download")     # big temp zips on D: drive (auto-cleaned)
CACHE_DIR    = OUT_DIR / "task3_band_cache"    # tiny clipped arrays, kept

# Folders where your 2024 / 2025 zips already live (reused, never re-downloaded)
EXISTING_DIRS = [
    BASE / "files for thesis" / "SP2_2024",    # 2024 scenes
    BASE / "files for thesis" / "SP2",         # 2025 scenes
]

TILE = "T32TNQ"
TEST_FIRST_N = 0      # 0 = process ALL 48 manifest scenes (already-done ones are skipped)

# Parco Ducale study area (lon/lat)
PARCO_LON_MIN, PARCO_LON_MAX = 10.313, 10.328
PARCO_LAT_MIN, PARCO_LAT_MAX = 44.802, 44.812

MIN_VALID_PX = 500    # skip scene if fewer valid pixels (heavy cloud)

# CDSE endpoints
TOKEN_URL    = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token")
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"


# ----------------------------------------------------------------------
# CDSE token manager (auto-refreshes during long runs)
# ----------------------------------------------------------------------
class CDSEToken:
    def __init__(self, user, pw):
        self._user, self._pw = user, pw
        self._access = self._refresh = None
        self._got, self._ttl = 0.0, 600
        self._login()

    def _login(self):
        data = {"grant_type": "password", "username": self._user,
                "password": self._pw, "client_id": "cdse-public"}
        r = requests.post(TOKEN_URL, data=data, timeout=30)
        if r.status_code != 200:
            raise SystemExit("CDSE login failed - check username/password.\n"
                             f"Server: {r.status_code} {r.text[:200]}")
        j = r.json()
        self._access, self._refresh = j["access_token"], j.get("refresh_token")
        self._ttl, self._got = j.get("expires_in", 600), time.time()

    def _do_refresh(self):
        if not self._refresh:
            return self._login()
        data = {"grant_type": "refresh_token", "refresh_token": self._refresh,
                "client_id": "cdse-public"}
        r = requests.post(TOKEN_URL, data=data, timeout=30)
        if r.status_code != 200:
            return self._login()
        j = r.json()
        self._access = j["access_token"]
        self._refresh = j.get("refresh_token", self._refresh)
        self._ttl, self._got = j.get("expires_in", 600), time.time()

    def token(self):
        if time.time() - self._got > max(60, self._ttl - 60):
            self._do_refresh()
        return self._access


def download_scene(pid, tok, dest_zip):
    """Download one product zip, following CDSE redirects with auth kept."""
    url = DOWNLOAD_URL.format(pid=pid)
    sess = requests.Session()
    sess.headers.update({"Authorization": f"Bearer {tok.token()}"})
    r = sess.get(url, allow_redirects=False, stream=True, timeout=120)
    hops = 0
    while r.status_code in (301, 302, 303, 307, 308) and hops < 6:
        url = r.headers["Location"]
        hops += 1
        r = sess.get(url, allow_redirects=False, stream=True, timeout=120)
    r.raise_for_status()
    total, done = int(r.headers.get("content-length", 0)), 0
    with open(dest_zip, "wb") as f:
        for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
            if not chunk:
                continue
            f.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r      {done/1e6:6.0f}/{total/1e6:6.0f} MB "
                      f"({done/total*100:4.0f}%)", end="", flush=True)
    print()
    return dest_zip


# ----------------------------------------------------------------------
# Band reading / clipping (your v2 method, Parco Ducale bbox)
# ----------------------------------------------------------------------
def _read_band(zf, code):
    """Read one R20m band array + geotransform from the zip."""
    cands = [f for f in zf.namelist()
             if code in os.path.basename(f) and f.endswith(".jp2") and "R20m" in f]
    if not cands:
        cands = [f for f in zf.namelist()
                 if code in os.path.basename(f) and f.endswith(".jp2")]
    if not cands:
        return None, None
    tmp = WORK_DIR / f"_tmp_{code}.jp2"
    with zf.open(cands[0]) as s, open(tmp, "wb") as d:
        d.write(s.read())
    with rasterio.open(tmp) as src:
        arr = src.read(1).astype(np.float32) / 10000.0     # /10000, NO offset
        t = src.transform
        gt = (t.c, t.a, t.b, t.f, t.d, t.e)
    tmp.unlink(missing_ok=True)
    return arr, gt


def _clip_parco(arr, gt):
    """Clip a full-tile band to the Parco Ducale box (pixel math, not rowcol)."""
    tr = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
    xmn, ymn = tr.transform(PARCO_LON_MIN, PARCO_LAT_MIN)
    xmx, ymx = tr.transform(PARCO_LON_MAX, PARCO_LAT_MAX)

    def px(x, y, g):
        return int((y - g[3]) / g[5]), int((x - g[0]) / g[1])

    r0, c0 = px(xmn, ymx, gt)
    r1, c1 = px(xmx, ymn, gt)
    h, w = arr.shape
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(h, r1), min(w, c1)
    return arr[r0:r1, c0:c1]


def bands_from_zip(zip_path):
    """Return Parco-Ducale-clipped (b5, b8a) reflectance arrays from a zip."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        b5_full,  gt5 = _read_band(zf, "B05")
        b8a_full, gt8 = _read_band(zf, "B8A")
    if b5_full is None or b8a_full is None:
        raise RuntimeError("B05 or B8A missing in zip.")
    b5  = _clip_parco(b5_full, gt5)
    b8a = _clip_parco(b8a_full, gt8)
    if b5.shape != b8a.shape:                 # both are 20 m, but be safe
        from PIL import Image as PILImage
        b8a = np.array(PILImage.fromarray(b8a).resize(
            (b5.shape[1], b5.shape[0]), PILImage.BILINEAR))
    return b5, b8a


def compute_ndre_lai(b5, b8a):
    """Your exact v2 NDRE + LAI."""
    d = b8a + b5
    ndre = np.where(d > 1e-6, (b8a - b5) / d, np.nan)
    ndre = np.where((b5 > 0.001) & (b8a > 0.001) & (b5 < 0.9) & (b8a < 0.9),
                    ndre, np.nan)
    ndre_c = np.clip(ndre, 0.01, 0.94)
    lai_beer = -2.0 * np.log(1 - ndre_c / 0.95) / 0.5
    lai_lin  = 3.618 * ndre_c - 0.118
    lai = np.clip((lai_beer + lai_lin) / 2.0, 0, 10)
    lai = np.where(np.isnan(ndre), np.nan, lai)
    return ndre, lai


# ----------------------------------------------------------------------
# Getting bands the cheapest way: cache -> existing zip -> download
# ----------------------------------------------------------------------
def cache_file(date_str):
    return CACHE_DIR / f"{date_str}_b05_b8a.npz"


def find_existing_zip(date_compact):
    """Look for an already-downloaded zip for this date+tile on disk."""
    for d in EXISTING_DIRS:
        if d.exists():
            for z in d.glob("*.zip"):
                if date_compact in z.name and TILE in z.name:
                    return z
    return None


def get_bands(scene):
    """
    Return (b5, b8a, source) for one manifest scene.
    source in {'cache','existing','download'} - tells caller whether to clean up.
    """
    date_str = scene["date"]                  # YYYY-MM-DD
    date_compact = date_str.replace("-", "")  # YYYYMMDD

    cf = cache_file(date_str)
    if cf.exists():
        d = np.load(cf)
        return d["b05"], d["b8a"], "cache"

    z = find_existing_zip(date_compact)
    if z is not None:
        b5, b8a = bands_from_zip(z)
        np.savez_compressed(cf, b05=b5, b8a=b8a)
        return b5, b8a, "existing"

    # download
    zip_path = WORK_DIR / f"{scene['product_name']}.zip"
    try:
        download_scene(scene["product_id"], get_bands.tok, zip_path)
        b5, b8a = bands_from_zip(zip_path)
    finally:
        if zip_path.exists():
            zip_path.unlink()                 # free disk immediately
    np.savez_compressed(cf, b05=b5, b8a=b8a)
    return b5, b8a, "download"


# ----------------------------------------------------------------------
# Scene-level features (your exact v2 set)
# ----------------------------------------------------------------------
FIELDS = ["date", "year", "month", "scene_idx",
          "mean_ndre", "std_ndre", "median_ndre", "q25_ndre", "q75_ndre",
          "mean_lai", "std_lai", "frac_above_020", "frac_below_012",
          "valid_px", "cloud_cover", "product_name"]


def features_for(scene, b5, b8a):
    ndre, lai = compute_ndre_lai(b5, b8a)
    valid = ~np.isnan(ndre)
    if valid.sum() < MIN_VALID_PX:
        return None
    nv, lv = ndre[valid], lai[valid]
    year, month = int(scene["year"]), int(scene["month"])
    return {
        "date": scene["date"], "year": year, "month": month,
        "scene_idx": month + (year - 2024) * 12,
        "mean_ndre": float(np.nanmean(nv)), "std_ndre": float(np.nanstd(nv)),
        "median_ndre": float(np.nanmedian(nv)),
        "q25_ndre": float(np.percentile(nv, 25)),
        "q75_ndre": float(np.percentile(nv, 75)),
        "mean_lai": float(np.nanmean(lv)), "std_lai": float(np.nanstd(lv)),
        "frac_above_020": float((nv >= 0.20).mean()),
        "frac_below_012": float((nv < 0.12).mean()),
        "valid_px": int(valid.sum()),
        "cloud_cover": scene["cloud_cover"],
        "product_name": scene["product_name"],
    }


def load_manifest():
    with open(MANIFEST, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def already_done():
    if not FEATURES_CSV.exists():
        return set()
    with open(FEATURES_CSV, newline="", encoding="utf-8") as fh:
        return {row["date"] for row in csv.DictReader(fh)}


def append_row(row):
    new = not FEATURES_CSV.exists()
    with open(FEATURES_CSV, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    scenes = load_manifest()
    done = already_done()
    if TEST_FIRST_N and TEST_FIRST_N > 0:
        scenes = scenes[:TEST_FIRST_N]
        print(f"*** TEST MODE: first {TEST_FIRST_N} scene(s) only. "
              f"Set TEST_FIRST_N = 0 for all 48. ***\n")

    # Only ask for CDSE login if at least one scene will actually need a download
    need_login = any(
        s["date"] not in done
        and not cache_file(s["date"]).exists()
        and find_existing_zip(s["date"].replace("-", "")) is None
        for s in scenes
    )
    if need_login:
        print("Some scenes need downloading - CDSE login required.")
        user = os.environ.get("CDSE_USER") or input("CDSE username (email): ").strip()
        pw   = os.environ.get("CDSE_PASS") or getpass.getpass("CDSE password: ")
        get_bands.tok = CDSEToken(user, pw)
        print("Login OK.\n")
    else:
        get_bands.tok = None
        print("All scenes available from cache/existing zips - no download needed.\n")

    for i, sc in enumerate(scenes, 1):
        date = sc["date"]
        if date in done:
            print(f"[{i}/{len(scenes)}] {date}  already in features CSV - skipping")
            continue
        print(f"[{i}/{len(scenes)}] {date}  cloud {sc['cloud_cover']}%  "
              f"{sc['product_name'][:46]}")
        try:
            b5, b8a, source = get_bands(sc)
            feats = features_for(sc, b5, b8a)
        except Exception as exc:
            print(f"      ! failed: {exc}")
            continue
        if feats is None:
            print(f"      ! too few valid pixels (heavy cloud) - skipped")
            continue
        append_row(feats)
        print(f"      [{source:8s}] valid_px={feats['valid_px']}  "
              f"mean_NDRE={feats['mean_ndre']:.4f}  "
              f"mean_LAI={feats['mean_lai']:.3f}  "
              f"dev-basis ready")

    print(f"\nDone. Per-scene features -> {FEATURES_CSV}")
    if TEST_FIRST_N and TEST_FIRST_N > 0:
        print("Check mean_NDRE (~0.19-0.20 expected over Parco Ducale). If good,")
        print("set TEST_FIRST_N = 0 and run again for the full 48-scene table.")
        print("Then I'll hand you Stage C: the v2 monthly-baseline labels + LOO-CV.")


if __name__ == "__main__":
    main()
