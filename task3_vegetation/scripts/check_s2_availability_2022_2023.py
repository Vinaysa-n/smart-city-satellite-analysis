"""
check_s2_availability_2022_2023.py
-----------------------------------
Checks whether cloud-free SUMMER Sentinel-2 L2A scenes exist over Parma
(tile T32TNQ, June-September window) for 2022 and 2023, using the
Copernicus Data Space Ecosystem (CDSE) OData catalog.

Why this is safe / easy:
  * NO login, NO token, NO credentials needed.
    The CDSE catalog *search* is open to everyone. You only need a token
    to DOWNLOAD the actual data later, not to list what exists.
  * It downloads nothing big. It just reads scene metadata.

What it does:
  * Lists every L2A acquisition over T32TNQ (T32TPQ is automatically
    excluded - we only match scenes whose name contains 'T32TNQ').
  * Shows each scene's date and tile cloud-cover %.
  * Counts how many scenes fall under 5 / 10 / 20 / 30 % cloud, per year.
  * Saves the full list to a CSV so you have a record.

"""

import os
import time
import csv
import datetime

try:
    import requests
except ImportError:
    raise SystemExit(
        "The 'requests' package is missing. In the Anaconda prompt run:\n"
        "    pip install requests\n"
        "then run this script again."
    )

# ----------------------------------------------------------------------
# Settings - already filled in for your thesis. Change only if you want.
# ----------------------------------------------------------------------
TILE          = "T32TNQ"      # Parma tile (T32TPQ is excluded automatically)
PRODUCT_TYPE  = "MSIL2A"      # Level-2A (surface reflectance) only
YEARS         = [2022, 2023]
MONTH_START   = 6             # June
MONTH_END     = 9             # September (inclusive, through Sep 30)
THRESHOLDS    = [5, 10, 20, 30]   # cloud-cover % buckets to summarise

OUT_DIR = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs"
OUT_CSV = os.path.join(OUT_DIR, "s2_availability_2022_2023.csv")

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


# ----------------------------------------------------------------------
def build_filter(year):
    """Build the OData $filter string for one year's summer window."""
    start = f"{year}-{MONTH_START:02d}-01T00:00:00.000Z"
    if MONTH_END == 12:
        end = f"{year}-12-31T23:59:59.999Z"
    else:
        last_day = datetime.date(year, MONTH_END + 1, 1) - datetime.timedelta(days=1)
        end = f"{last_day.isoformat()}T23:59:59.999Z"
    return (
        "Collection/Name eq 'SENTINEL-2' "
        f"and contains(Name,'{PRODUCT_TYPE}') "
        f"and contains(Name,'{TILE}') "
        f"and ContentDate/Start ge {start} "
        f"and ContentDate/Start le {end}"
    )


def get_cloud_cover(attributes):
    """Pull the cloudCover value out of a product's expanded Attributes."""
    for att in attributes or []:
        if att.get("Name") == "cloudCover":
            try:
                return float(att.get("Value"))
            except (TypeError, ValueError):
                return None
    return None


def query_year(year):
    """Query CDSE for one year and return a de-duplicated list of scenes."""
    params = {
        "$filter": build_filter(year),
        "$expand": "Attributes",
        "$top": "200",
    }
    raw = []
    url = ODATA_URL
    first = True
    while url:
        for attempt in range(3):
            try:
                if first:
                    r = requests.get(url, params=params, timeout=60)
                else:
                    r = requests.get(url, timeout=60)  # nextLink keeps its own params
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    print(f"  ! Network error for {year}: {exc}")
                    return []
                time.sleep(3)
        data = r.json()
        for prod in data.get("value", []):
            name = prod.get("Name", "")
            if TILE not in name:               # safety: never let T32TPQ slip in
                continue
            start = prod.get("ContentDate", {}).get("Start", "")
            raw.append({
                "year": year,
                "date": start[:10],
                "cloud_cover": get_cloud_cover(prod.get("Attributes")),
                "name": name,
            })
        url = data.get("@odata.nextLink")
        first = False

    # Collapse multiple processing baselines of the SAME acquisition (same date)
    # into one entry, keeping the lowest reported cloud cover.
    by_date = {}
    for s in raw:
        d = s["date"]
        if d not in by_date:
            by_date[d] = s
        else:
            cur, new = by_date[d]["cloud_cover"], s["cloud_cover"]
            if cur is None or (new is not None and new < cur):
                by_date[d] = s
    return sorted(by_date.values(), key=lambda s: s["date"])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_scenes = []

    print("=" * 72)
    print(f"Sentinel-2 L2A availability over {TILE}  (Jun-Sep)  for {YEARS}")
    print("No credentials needed - this only searches the public CDSE catalog.")
    print("=" * 72)

    for year in YEARS:
        print(f"\nQuerying {year} ...")
        scenes = query_year(year)
        all_scenes.extend(scenes)

        if not scenes:
            print(f"  No L2A scenes returned for {year}.")
            continue

        print(f"\n  {year}: {len(scenes)} distinct acquisition date(s)")
        print(f"  {'Date':<12}{'Cloud %':>9}   Scene")
        print(f"  {'-'*11:<12}{'-'*8:>9}   {'-'*46}")
        for s in scenes:
            cc = "n/a" if s["cloud_cover"] is None else f"{s['cloud_cover']:.1f}"
            print(f"  {s['date']:<12}{cc:>9}   {s['name'][:46]}")

        print(f"\n  Cloud-free summary for {year}:")
        valid = [s for s in scenes if s["cloud_cover"] is not None]
        for t in THRESHOLDS:
            n = sum(1 for s in valid if s["cloud_cover"] <= t)
            print(f"    <= {t:>2d}% cloud : {n} scene(s)")

    # Save CSV record
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["year", "date", "cloud_cover", "name"])
        w.writeheader()
        for s in all_scenes:
            w.writerow(s)

    print("\n" + "=" * 72)
    print("OVERALL (this is the key part to paste back)")
    for year in YEARS:
        ys = [s for s in all_scenes if s["year"] == year and s["cloud_cover"] is not None]
        n10 = sum(1 for s in ys if s["cloud_cover"] <= 10)
        n20 = sum(1 for s in ys if s["cloud_cover"] <= 20)
        print(f"  {year}: {len(ys)} dated scenes | {n10} at <=10% | {n20} at <=20% cloud")
    print(f"\nFull list saved to:\n  {OUT_CSV}")
    print("=" * 72)
    print("\nRule of thumb: for a vegetation-health study aim for <=10-20% tile")
    print("cloud, then still mask residual cloud per-pixel via the SCL band.")
    print("If each year gives ~4+ scenes at <=20%, extending Task 3 is feasible.")


if __name__ == "__main__":
    main()
