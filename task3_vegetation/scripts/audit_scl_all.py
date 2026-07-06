"""
audit_scl_all.py
----------------
Cloud-screens EVERY manifest scene by measuring the % of the Parco Ducale box
flagged as cloud / cirrus / cloud-shadow by the Sentinel-2 SCL band - the same
test that exposed 2024-09-07 (13.6%). Lets you find any other contaminated
scenes before they bias the four-year result.

Efficient: for each scene it gets the SCL band the cheapest way -
   1. from a cached .npz if already screened, else
   2. from a full zip already in SP2 / SP2_2024, else
   3. by downloading ONLY the SCL file (~10 MB) via the CDSE node API, else
   4. (fallback) by downloading the full zip to D:\thesis download.

Outputs:
   task3_scl_audit.csv   one row per scene, sorted clean -> dirty
   a console table with a CLEAN / minor / REVIEW / EXCLUDE tier per scene
   a recommendation of which scenes to drop or eyeball.

RUN:  python audit_scl_all.py
"""

import os
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
BASE      = Path(r"C:\Users\vinay\OneDrive\Desktop\Thesis")
OUT_DIR   = BASE / "outputs"
MANIFEST  = OUT_DIR / "task3_scene_manifest_2022_2025.csv"
FEATURES  = OUT_DIR / "task3_features_2022_2025.csv"
AUDIT_CSV = OUT_DIR / "task3_scl_audit.csv"
SCL_CACHE = OUT_DIR / "task3_scl_cache"
WORK      = Path(r"D:\thesis download")

EXISTING_DIRS = [BASE / "files for thesis" / "SP2_2024",
                 BASE / "files for thesis" / "SP2"]

TILE = "T32TNQ"
LON  = (10.313, 10.328)          # Parco Ducale
LAT  = (44.802, 44.812)

# SCL contamination classes: 3 shadow, 8 cloud-med, 9 cloud-high, 10 cirrus
SHADOW, CLOUD_M, CLOUD_H, CIRRUS, VEG = 3, 8, 9, 10, 4

DL = "https://download.dataspace.copernicus.eu/odata/v1"
TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")


# ---- CDSE auth -------------------------------------------------------
class CDSEToken:
    def __init__(self, u, p):
        self.u, self.p = u, p; self.a = self.r = None; self.t0 = 0; self.ttl = 600
        self._login()
    def _login(self):
        d = {"grant_type": "password", "username": self.u, "password": self.p,
             "client_id": "cdse-public"}
        r = requests.post(TOKEN_URL, data=d, timeout=30)
        if r.status_code != 200:
            raise SystemExit(f"CDSE login failed: {r.status_code} {r.text[:150]}")
        j = r.json(); self.a = j["access_token"]; self.r = j.get("refresh_token")
        self.ttl = j.get("expires_in", 600); self.t0 = time.time()
    def token(self):
        if time.time() - self.t0 > self.ttl - 60:
            self._login()
        return self.a


def session_for(tok):
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {tok.token()}"})
    return s


def stream_to(sess, url, dest):
    r = sess.get(url, allow_redirects=False, stream=True, timeout=120)
    h = 0
    while r.status_code in (301, 302, 303, 307, 308) and h < 6:
        r = sess.get(r.headers["Location"], allow_redirects=False, stream=True, timeout=120)
        h += 1
    r.raise_for_status()
    with open(dest, "wb") as f:
        for c in r.iter_content(2 * 1024 * 1024):
            if c:
                f.write(c)


def list_nodes(sess, path):
    r = sess.get(path + "/Nodes", timeout=60)
    r.raise_for_status()
    j = r.json()
    return j.get("value") or j.get("result") or []


def scl_path_via_nodes(sess, pid):
    """Navigate the product tree to the SCL_20m.jp2 node path."""
    base = f"{DL}/Products({pid})"
    safe = next(c["Name"] for c in list_nodes(sess, base) if c["Name"].endswith(".SAFE"))
    p = f"{base}/Nodes({safe})/Nodes(GRANULE)"
    gran = next(c["Name"] for c in list_nodes(sess, p) if c["Name"].startswith("L2A_"))
    p = f"{p}/Nodes({gran})/Nodes(IMG_DATA)/Nodes(R20m)"
    scl = next(c["Name"] for c in list_nodes(sess, p) if c["Name"].endswith("_SCL_20m.jp2"))
    return f"{p}/Nodes({scl})"


# ---- clipping --------------------------------------------------------
def clip_scl(jp2_path):
    with rasterio.open(jp2_path) as src:
        tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        xs, ys = tr.transform([LON[0], LON[1], LON[0], LON[1]],
                              [LAT[0], LAT[0], LAT[1], LAT[1]])
        t = src.transform
        c0 = int((min(xs) - t.c) / t.a); c1 = int((max(xs) - t.c) / t.a)
        r0 = int((max(ys) - t.f) / t.e); r1 = int((min(ys) - t.f) / t.e)
        r0, r1 = sorted((max(0, r0), max(0, r1)))
        c0, c1 = sorted((max(0, c0), max(0, c1)))
        return src.read(1, window=((r0, min(r1, src.height)), (c0, min(c1, src.width))))


def scl_from_zip(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        cand = [f for f in zf.namelist() if f.endswith("_SCL_20m.jp2") and "R20m" in f]
        if not cand:
            raise RuntimeError("no SCL in zip")
        tmp = WORK / "_scl_tmp.jp2"
        with zf.open(cand[0]) as s, open(tmp, "wb") as d:
            d.write(s.read())
    arr = clip_scl(tmp); tmp.unlink(missing_ok=True)
    return arr


def find_existing_zip(date_compact):
    for d in EXISTING_DIRS:
        if d.exists():
            for z in d.glob("*.zip"):
                if date_compact in z.name and TILE in z.name:
                    return z
    return None


def get_scl(scene, tok_holder):
    date = scene["date"]; dc = date.replace("-", "")
    cf = SCL_CACHE / f"{date}_scl.npy"
    if cf.exists():
        return np.load(cf), "cache"

    z = find_existing_zip(dc)
    if z is not None:
        arr = scl_from_zip(z)
        np.save(cf, arr)
        return arr, "existing"

    # need CDSE
    if tok_holder[0] is None:
        print("Some scenes need a CDSE fetch - login required.")
        u = os.environ.get("CDSE_USER") or input("CDSE username: ").strip()
        p = os.environ.get("CDSE_PASS") or getpass.getpass("CDSE password: ")
        tok_holder[0] = CDSEToken(u, p)
    sess = session_for(tok_holder[0])
    pid = scene["product_id"]

    # try SCL-only via node API (~10 MB)
    try:
        spath = scl_path_via_nodes(sess, pid)
        tmp = WORK / f"{dc}_scl.jp2"
        stream_to(sess, f"{spath}/$value", tmp)
        arr = clip_scl(tmp); tmp.unlink(missing_ok=True)
        np.save(cf, arr)
        return arr, "node"
    except Exception as e:
        print(f"      node fetch failed ({e}); falling back to full zip")

    # fallback: full product download
    zp = WORK / f"{scene['product_name']}.zip"
    try:
        stream_to(sess, f"{DL}/Products({pid})/$value", zp)
        arr = scl_from_zip(zp)
    finally:
        if zp.exists():
            zp.unlink()
    np.save(cf, arr)
    return arr, "fullzip"


def tier(contam):
    if contam < 1:  return "CLEAN"
    if contam < 5:  return "minor"
    if contam < 10: return "REVIEW"
    return "EXCLUDE"


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    SCL_CACHE.mkdir(parents=True, exist_ok=True)

    scenes = list(csv.DictReader(open(MANIFEST)))
    ndre = {}
    if FEATURES.exists():
        ndre = {r["date"]: float(r["mean_ndre"]) for r in csv.DictReader(open(FEATURES))}

    tok_holder = [None]
    rows = []
    print(f"Screening {len(scenes)} scenes ...\n")
    for i, sc in enumerate(scenes, 1):
        date = sc["date"]
        try:
            scl, src = get_scl(sc, tok_holder)
        except Exception as e:
            print(f"[{i:2d}/{len(scenes)}] {date}  ! failed: {e}")
            continue
        tot = scl.size
        shadow = int((scl == SHADOW).sum())
        cloud = int(((scl == CLOUD_M) | (scl == CLOUD_H)).sum())
        cirrus = int((scl == CIRRUS).sum())
        contam = (shadow + cloud + cirrus) / tot * 100
        veg = (scl == VEG).sum() / tot * 100
        rows.append({
            "date": date, "year": sc["year"], "month": sc["month"],
            "tile_cloud_pct": sc["cloud_cover"],
            "park_contam_pct": round(contam, 2),
            "park_cloud_pct": round(cloud / tot * 100, 2),
            "park_cirrus_pct": round(cirrus / tot * 100, 2),
            "park_shadow_pct": round(shadow / tot * 100, 2),
            "park_veg_pct": round(veg, 1),
            "mean_ndre": ndre.get(date, ""),
            "tier": tier(contam),
        })
        print(f"[{i:2d}/{len(scenes)}] {date}  [{src:8s}] "
              f"park cloud/cirrus/shadow = {contam:5.1f}%  -> {tier(contam)}")

    rows.sort(key=lambda r: r["park_contam_pct"])
    with open(AUDIT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 68)
    print(f"{'date':12s}{'park cloud%':>12}{'mean_ndre':>11}{'tile%':>8}   tier")
    print("-" * 68)
    for r in sorted(rows, key=lambda r: -r["park_contam_pct"]):
        nd = f"{r['mean_ndre']:.4f}" if r["mean_ndre"] != "" else "  -  "
        print(f"{r['date']:12s}{r['park_contam_pct']:>11.1f}%{nd:>11}"
              f"{r['tile_cloud_pct']:>7}%   {r['tier']}")

    exclude = [r["date"] for r in rows if r["tier"] == "EXCLUDE"]
    review  = [r["date"] for r in rows if r["tier"] == "REVIEW"]
    print("\n" + "=" * 68)
    print(f"CLEAN: {sum(r['tier']=='CLEAN' for r in rows)}   "
          f"minor: {sum(r['tier']=='minor' for r in rows)}   "
          f"REVIEW: {len(review)}   EXCLUDE: {len(exclude)}")
    if exclude:
        print(f"\nRecommend EXCLUDE (>10% cloud over the park): {exclude}")
    if review:
        print(f"Eyeball these (5-10%, run verify_scene.py on each): {review}")
    if not exclude and not review:
        print("\nNo other contaminated scenes - 2024-09-07 was the only offender.")
    print(f"\nAudit table -> {AUDIT_CSV}")


if __name__ == "__main__":
    main()
