import requests
import os
import time
import csv
import netCDF4

# CREDENTIALS  --  paste your password below, then RUN.

USERNAME = "vinay.sankar@studenti.unipr.it"
PASSWORD = ""   # <-- paste your Copernicus password here


YEARS = [2022, 2023, 2024, 2025]                 # all years run in one go
BASE_DIR = "180 data"                # same parent folder as before
POLYGON = "POLYGON((10.20 44.70, 10.45 44.70, 10.45 44.90, 10.20 44.90, 10.20 44.70))"

LAT_MIN, LAT_MAX = 44.70, 44.90
LON_MIN, LON_MAX = 10.20, 10.45

# Current official Copernicus download endpoint
DOWNLOAD_HOST = "https://download.dataspace.copernicus.eu/odata/v1/Products"


# ============================================================
# AUTH
# ============================================================
def get_token():
    r = requests.post(
        "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token",
        data={
            "client_id": "cdse-public",
            "username": USERNAME,
            "password": PASSWORD,
            "grant_type": "password",
        },
    )
    if r.status_code != 200:
        print(f"Auth failed: {r.text}")
        exit()
    print("Token refreshed.")
    return r.json()["access_token"]


# ============================================================
# CLIP: extract only Parma pixels from a .nc file into a CSV
# ============================================================
def clip_to_parma(nc_path, clip_path):
    try:
        ds = netCDF4.Dataset(nc_path)
        product = ds.groups["PRODUCT"]

        lat = product.variables["latitude"][0, :, :]
        lon = product.variables["longitude"][0, :, :]
        no2 = product.variables["nitrogendioxide_tropospheric_column"][0, :, :]
        qa = product.variables["qa_value"][0, :, :]

        mask = (
            (lat >= LAT_MIN) & (lat <= LAT_MAX) &
            (lon >= LON_MIN) & (lon <= LON_MAX) &
            (qa > 0.75)
        )

        if mask.sum() == 0:
            ds.close()
            return False, "No valid pixels over Parma"

        parma_lat = lat[mask].flatten()
        parma_lon = lon[mask].flatten()
        parma_no2 = no2[mask].flatten()
        parma_qa = qa[mask].flatten()

        with open(clip_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["latitude", "longitude", "no2_trop_column", "qa_value"])
            for j in range(len(parma_lat)):
                writer.writerow([
                    f"{parma_lat[j]:.6f}",
                    f"{parma_lon[j]:.6f}",
                    f"{parma_no2[j]:.6e}",
                    f"{parma_qa[j]:.4f}",
                ])

        ds.close()
        return True, f"{mask.sum()} pixels"

    except Exception as e:
        return False, str(e)



# PROCESS ONE YEAR  (search -> download -> clip -> delete)

def process_year(year, token_holder):
    start_date = f"{year}-04-01T00:00:00.000Z"
    end_date = f"{year}-09-30T23:59:59.000Z"

    raw_dir = os.path.join(BASE_DIR, f"raw_temp_{year}")
    clip_dir = os.path.join(BASE_DIR, f"parma_no2_clipped_{year}")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(clip_dir, exist_ok=True)

    print("\n" + "#" * 55)
    print(f"# YEAR {year}   ({start_date[:10]} -> {end_date[:10]})")
    print("#" * 55)

    # ---- SEARCH ----
    print("Searching for NO2 data...")
    all_results = []
    skip = 0
    while True:
        params = {
            "$filter": (
                f"Collection/Name eq 'SENTINEL-5P' "
                f"and Attributes/OData.CSC.StringAttribute/any("
                f"att:att/Name eq 'productType' and "
                f"att/OData.CSC.StringAttribute/Value eq 'L2__NO2___') "
                f"and OData.CSC.Intersects(area=geography"
                f"'SRID=4326;{POLYGON}') "
                f"and ContentDate/Start gt {start_date} "
                f"and ContentDate/Start lt {end_date}"
            ),
            "$top": 100,
            "$skip": skip,
            "$orderby": "ContentDate/Start asc",
        }
        r = requests.get(
            "https://catalogue.dataspace.copernicus.eu/odata/v1/Products",
            params=params,
        )
        batch = r.json().get("value", [])
        if not batch:
            break
        all_results.extend(batch)
        skip += 100
        print(f"  Found {len(all_results)} products so far...")

    print(f"Total: {len(all_results)} products for {year}")

    # ---- SKIP ALREADY CLIPPED ----
    existing_clips = set(os.listdir(clip_dir))
    to_download = []
    for p in all_results:
        clip_name = p["Name"].replace(".nc", "_parma.csv")
        if clip_name in existing_clips:
            continue
        to_download.append(p)

    print(f"Already clipped: {len(all_results) - len(to_download)}")
    print(f"Remaining:       {len(to_download)}\n")

    if not to_download:
        print(f"{year} already complete.")
        return 0, 0, 0

    # ---- DOWNLOAD -> CLIP -> DELETE ----
    success = no_data = failed = 0

    for i, product in enumerate(to_download, 1):
        pid = product["Id"]
        name = product["Name"]
        raw_path = os.path.join(raw_dir, name)
        clip_name = name.replace(".nc", "_parma.csv")
        clip_path = os.path.join(clip_dir, clip_name)

        # refresh token every ~8 minutes
        if time.time() - token_holder["time"] > 480:
            token_holder["token"] = get_token()
            token_holder["time"] = time.time()

        headers = {"Authorization": f"Bearer {token_holder['token']}"}
        url = f"{DOWNLOAD_HOST}({pid})/$value"

        print(f"[{year}] [{i}/{len(to_download)}] Downloading...")

        try:
            with requests.get(url, headers=headers, stream=True, timeout=300) as r:
                r.raise_for_status()
                with open(raw_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=65536):
                        f.write(chunk)

            size_mb = os.path.getsize(raw_path) / 1_000_000
            print(f"  Downloaded ({size_mb:.0f} MB), clipping to Parma...")

            ok, msg = clip_to_parma(raw_path, clip_path)
            if ok:
                print(f"  Clipped: {msg}")
                success += 1
            else:
                print(f"  Skip (no Parma data): {msg}")
                no_data += 1

            os.remove(raw_path)   # delete the large original

        except Exception as e:
            print(f"  FAILED: {e}")
            if os.path.exists(raw_path):
                os.remove(raw_path)
            failed += 1

        time.sleep(1.5)

    clip_count = len(os.listdir(clip_dir))
    print("\n" + "=" * 50)
    print(f"YEAR {year} COMPLETE")
    print(f"  Clipped to Parma:  {success}")
    print(f"  No Parma pixels:   {no_data}")
    print(f"  Failed:            {failed}")
    print(f"  Files in clip dir: {clip_count}")
    print("=" * 50)

    return success, no_data, failed



# MAIN

if __name__ == "__main__":
    if not PASSWORD:
        print("ERROR: paste your Copernicus password into the PASSWORD field first.")
        exit()

    os.makedirs(BASE_DIR, exist_ok=True)

    token_holder = {"token": get_token(), "time": time.time()}

    totals = {"success": 0, "no_data": 0, "failed": 0}
    for yr in YEARS:
        s, n, f = process_year(yr, token_holder)
        totals["success"] += s
        totals["no_data"] += n
        totals["failed"] += f

    print("\n" + "*" * 55)
    print("ALL YEARS DONE")
    print(f"  Total clipped to Parma: {totals['success']}")
    print(f"  Total no Parma pixels:  {totals['no_data']}")
    print(f"  Total failed:           {totals['failed']}")
    print("*" * 55)
