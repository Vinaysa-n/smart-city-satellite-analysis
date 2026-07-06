"""
verify_scene.py
---------------
Checks whether the suspect scene 2024-09-07 (low NDRE 0.132, 14.7% cloud) is
REAL vegetation stress or residual haze, by rendering it next to a known-clean
September scene over Parco Ducale.

For each of the two scenes it shows: true colour | SCL land cover | NDRE,
and prints the % of the box flagged as cloud/cirrus/shadow + the mean NDRE.
It also prints the NDRE of all 2024-September scenes from your features CSV
plus the September baseline, so you can see if 09-07 is the anomaly.

Needs the full zip for each scene: it reuses one from SP2/SP2_2024 if present,
otherwise downloads it to D:\thesis download (CDSE login), then deletes it.

RUN:  python verify_scene.py
"""

import os
import csv
import time
import zipfile
import getpass
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import requests
import rasterio
from pyproj import Transformer

# ----------------------------------------------------------------------
BASE     = Path(r"C:\Users\vinay\OneDrive\Desktop\Thesis")
OUT_DIR  = BASE / "outputs"
MANIFEST = OUT_DIR / "task3_scene_manifest_2022_2025.csv"
FEATURES = OUT_DIR / "task3_features_2022_2025.csv"
WORK     = Path(r"D:\thesis download")
OUT_PNG  = OUT_DIR / "verify_2024-09-07.png"

EXISTING_DIRS = [BASE / "files for thesis" / "SP2_2024",
                 BASE / "files for thesis" / "SP2"]

SUSPECT_DATE   = "2024-09-07"   # the scene under question
REFERENCE_DATE = "2025-09-19"   # a clean (0% cloud) September scene for comparison

# Parco Ducale box
LON = (10.313, 10.328)
LAT = (44.802, 44.812)

SCL_COLORS = {0:"#000000",1:"#ff0000",2:"#2f2f2f",3:"#643200",4:"#1f9e1f",
              5:"#ffe65a",6:"#1414ff",7:"#7f7f7f",8:"#b0b0b0",9:"#e8e8e8",
              10:"#64c8ff",11:"#ff96ff"}
SCL_NAMES = {0:"no data",1:"saturated",2:"dark",3:"cloud shadow",4:"vegetation",
             5:"not vegetated",6:"water",7:"unclassified",8:"cloud",9:"cloud",
             10:"cirrus/haze",11:"snow"}
CLOUDY = {3, 8, 9, 10}          # shadow / cloud / cirrus = contamination

TOKEN_URL    = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
                "protocol/openid-connect/token")
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({pid})/$value"


class CDSEToken:
    def __init__(self, u, p):
        self.u, self.p = u, p; self.a = self.r = None; self.t0 = 0; self.ttl = 600
        self._login()
    def _login(self):
        d = {"grant_type":"password","username":self.u,"password":self.p,"client_id":"cdse-public"}
        r = requests.post(TOKEN_URL, data=d, timeout=30)
        if r.status_code != 200:
            raise SystemExit(f"CDSE login failed: {r.status_code} {r.text[:150]}")
        j = r.json(); self.a=j["access_token"]; self.r=j.get("refresh_token")
        self.ttl=j.get("expires_in",600); self.t0=time.time()
    def token(self):
        if time.time()-self.t0 > self.ttl-60:
            self._login()
        return self.a


def download(pid, tok, dest):
    s = requests.Session(); s.headers.update({"Authorization": f"Bearer {tok.token()}"})
    r = s.get(DOWNLOAD_URL.format(pid=pid), allow_redirects=False, stream=True, timeout=120)
    h = 0
    while r.status_code in (301,302,303,307,308) and h < 6:
        r = s.get(r.headers["Location"], allow_redirects=False, stream=True, timeout=120); h += 1
    r.raise_for_status()
    tot, done = int(r.headers.get("content-length",0)), 0
    with open(dest, "wb") as f:
        for c in r.iter_content(4*1024*1024):
            if c:
                f.write(c); done += len(c)
                if tot: print(f"\r    {done/1e6:5.0f}/{tot/1e6:5.0f} MB", end="", flush=True)
    print()


def window(src):
    tr = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
    xs, ys = tr.transform([LON[0],LON[1],LON[0],LON[1]], [LAT[0],LAT[0],LAT[1],LAT[1]])
    t = src.transform
    c0=int((min(xs)-t.c)/t.a); c1=int((max(xs)-t.c)/t.a)
    r0=int((max(ys)-t.f)/t.e); r1=int((min(ys)-t.f)/t.e)
    r0,r1 = sorted((max(0,r0),max(0,r1))); c0,c1 = sorted((max(0,c0),max(0,c1)))
    return r0, min(r1,src.height), c0, min(c1,src.width)


def extract(zf, pred, name):
    cs = [f for f in zf.namelist() if pred(f)]
    if not cs: return None
    tmp = WORK / name
    with zf.open(cs[0]) as s, open(tmp,"wb") as d: d.write(s.read())
    return tmp


def read1(tmp):
    with rasterio.open(tmp) as src:
        r0,r1,c0,c1 = window(src)
        a = src.read(1, window=((r0,r1),(c0,c1)))
    tmp.unlink(missing_ok=True)
    return a


def load_views(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        tci_t = extract(zf, lambda f: f.endswith("_TCI_10m.jp2") and "R10m" in f, "_tci.jp2")
        with rasterio.open(tci_t) as src:
            r0,r1,c0,c1 = window(src); rgb = src.read(window=((r0,r1),(c0,c1)))
        tci_t.unlink(missing_ok=True)
        rgb = np.transpose(rgb,(1,2,0)).astype(np.uint8)
        scl = read1(extract(zf, lambda f: f.endswith("_SCL_20m.jp2") and "R20m" in f, "_scl.jp2"))
        b5  = read1(extract(zf, lambda f: "_B05_20m.jp2" in f and "R20m" in f, "_b05.jp2")).astype(np.float32)/10000
        b8  = read1(extract(zf, lambda f: "_B8A_20m.jp2" in f and "R20m" in f, "_b8a.jp2")).astype(np.float32)/10000
    d = b8 + b5
    ndre = np.where(d > 1e-6, (b8-b5)/d, np.nan)
    ndre = np.where((b5>0.001)&(b8>0.001)&(b5<0.9)&(b8<0.9), ndre, np.nan)
    return rgb, scl, ndre


def scl_rgb(scl):
    h,w = scl.shape; img = np.zeros((h,w,3))
    for c,hx in SCL_COLORS.items():
        m = scl == c
        if m.any(): img[m] = [int(hx[i:i+2],16)/255 for i in (1,3,5)]
    return img


def get_zip(date_str, tok_holder):
    dc = date_str.replace("-","")
    for d in EXISTING_DIRS:
        if d.exists():
            for z in d.glob("*.zip"):
                if dc in z.name and "T32TNQ" in z.name:
                    return z, False
    # need to download -> look up product_id in manifest
    rows = list(csv.DictReader(open(MANIFEST)))
    m = [r for r in rows if r["date"] == date_str]
    if not m:
        raise SystemExit(f"{date_str} not in manifest.")
    if tok_holder[0] is None:
        print("CDSE login required to download a scene.")
        u = os.environ.get("CDSE_USER") or input("CDSE username: ").strip()
        p = os.environ.get("CDSE_PASS") or getpass.getpass("CDSE password: ")
        tok_holder[0] = CDSEToken(u, p)
    dest = WORK / f"{m[0]['product_name']}.zip"
    print(f"  downloading {date_str} ...")
    download(m[0]["product_id"], tok_holder[0], dest)
    return dest, True


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    tok_holder = [None]
    scenes = [(SUSPECT_DATE, "SUSPECT"), (REFERENCE_DATE, "clean reference")]

    fig, axes = plt.subplots(2, 3, figsize=(13, 9))
    for row, (date_str, tag) in enumerate(scenes):
        z, downloaded = get_zip(date_str, tok_holder)
        try:
            rgb, scl, ndre = load_views(z)
        finally:
            if downloaded and z.exists():
                z.unlink()

        cloud_pct = np.isin(scl, list(CLOUDY)).mean() * 100
        nd = ndre[np.isfinite(ndre)]
        mean_ndre = float(nd.mean()) if nd.size else float("nan")

        axes[row,0].imshow(rgb)
        axes[row,0].set_title(f"{date_str}  ({tag})\nTrue colour", fontsize=10)
        axes[row,1].imshow(scl_rgb(scl))
        present = [c for c in SCL_COLORS if (scl==c).any()]
        axes[row,1].legend(handles=[mpatches.Patch(color=SCL_COLORS[c], label=SCL_NAMES[c])
                                    for c in present], fontsize=6, loc="upper right", framealpha=.85)
        axes[row,1].set_title(f"SCL  (cloud/cirrus/shadow = {cloud_pct:.1f}%)", fontsize=10)
        im = axes[row,2].imshow(ndre, cmap="RdYlGn", vmin=-0.05, vmax=0.45)
        plt.colorbar(im, ax=axes[row,2], fraction=.046, pad=.04)
        axes[row,2].set_title(f"NDRE  (mean = {mean_ndre:.4f})", fontsize=10)
        for c in range(3):
            axes[row,c].axis("off")

        print(f"{tag:16s} {date_str}: cloud/cirrus/shadow={cloud_pct:5.1f}%   mean_NDRE={mean_ndre:.4f}")

    # numeric context from the features CSV
    if FEATURES.exists():
        rows = list(csv.DictReader(open(FEATURES)))
        sep24 = sorted((r["date"], float(r["mean_ndre"])) for r in rows
                       if r["year"] == "2024" and r["month"] == "9")
        if sep24:
            print("\n2024 September scenes (from features CSV):")
            for d, v in sep24:
                flag = "  <-- suspect" if d == SUSPECT_DATE else ""
                print(f"   {d}: NDRE = {v:.4f}{flag}")
            print("   September baseline (median across years): 0.1558")

    fig.suptitle("Verify 2024-09-07: suspect vs clean September (Parco Ducale)", fontsize=12, y=0.99)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(OUT_PNG, dpi=140, bbox_inches="tight")
    print(f"\nFigure -> {OUT_PNG}")
    print("Upload it. Decision: if the suspect row shows haze / SCL clouds over the park,")
    print("or its NDRE sits well below the clean reference for no visible reason -> drop it.")
    print("If the park looks crisp and SCL says vegetation -> it's a real stress scene, keep it.")


if __name__ == "__main__":
    main()
