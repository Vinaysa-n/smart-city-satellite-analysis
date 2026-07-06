"""
select_task3_scenes_2022_2025.py
---------------------------------
Builds the SCENE MANIFEST for the four-year Task 3 rebuild.

Decision locked: 12 scenes per year (2022-2025), June-September window,
<=20% tile cloud, aiming for ~3 per month for balanced monthly coverage,
then backfilling to 12 from the next-cleanest scenes if a month is short.

  * No login / token needed - this only SEARCHES the public CDSE catalog.
  * It downloads NOTHING. It just decides which 48 scenes to download next
    and writes their product IDs to a manifest CSV.

Output: task3_scene_manifest_2022_2025.csv
        (year, month, date, cloud %, product_id, product_name)
        -> the download step will read this file.

"""

import os
import time
import csv
import datetime

try:
    import requests
except ImportError:
    raise SystemExit("Missing 'requests'. In the Anaconda prompt run:\n"
                     "    pip install requests")

# ----------------------------------------------------------------------
# Selection rule - already filled in for your thesis.
# ----------------------------------------------------------------------
TILE         = "T32TNQ"     # Parma tile (T32TPQ excluded automatically)
PRODUCT_TYPE = "MSIL2A"     # Level-2A only
YEARS        = [2022, 2023, 2024, 2025]
MONTHS       = [6, 7, 8, 9]    # June .. September
CLOUD_CAP    = 20.0           # %, hard ceiling for any selected scene
PER_MONTH    = 3              # target scenes per month (3 x 4 = 12)
PER_YEAR     = 12             # final count per year

OUT_DIR  = r"C:\Users\vinay\OneDrive\Desktop\Thesis\outputs"
MANIFEST = os.path.join(OUT_DIR, "task3_scene_manifest_2022_2025.csv")

ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"


# ----------------------------------------------------------------------
def month_window(year, month):
    start = datetime.date(year, month, 1)
    if month == 12:
        end = datetime.date(year, 12, 31)
    else:
        end = datetime.date(year, month + 1, 1) - datetime.timedelta(days=1)
    return (f"{start.isoformat()}T00:00:00.000Z",
            f"{end.isoformat()}T23:59:59.999Z")


def build_filter(year, month):
    start, end = month_window(year, month)
    return ("Collection/Name eq 'SENTINEL-2' "
            f"and contains(Name,'{PRODUCT_TYPE}') "
            f"and contains(Name,'{TILE}') "
            f"and ContentDate/Start ge {start} "
            f"and ContentDate/Start le {end}")


def cloud_of(attrs):
    for a in attrs or []:
        if a.get("Name") == "cloudCover":
            try:
                return float(a.get("Value"))
            except (TypeError, ValueError):
                return None
    return None


def query(year, month):
    """All distinct acquisitions for one year+month, with product IDs."""
    params = {"$filter": build_filter(year, month),
              "$expand": "Attributes", "$top": "200"}
    rows, url, first = [], ODATA_URL, True
    while url:
        for attempt in range(3):
            try:
                r = (requests.get(url, params=params, timeout=60) if first
                     else requests.get(url, timeout=60))
                r.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == 2:
                    print(f"  ! network error {year}-{month:02d}: {exc}")
                    return []
                time.sleep(3)
        data = r.json()
        for p in data.get("value", []):
            name = p.get("Name", "")
            if TILE not in name:                 # never let T32TPQ slip in
                continue
            rows.append({
                "year": year, "month": month,
                "date": p.get("ContentDate", {}).get("Start", "")[:10],
                "cloud_cover": cloud_of(p.get("Attributes")),
                "product_id": p.get("Id", ""),
                "product_name": name,
            })
        url = data.get("@odata.nextLink")
        first = False

    # collapse multiple processing baselines of the same acquisition ->
    # keep the one with the lowest reported cloud cover (with its own ID).
    by_date = {}
    for s in rows:
        d = s["date"]
        if d not in by_date:
            by_date[d] = s
        else:
            cur, new = by_date[d]["cloud_cover"], s["cloud_cover"]
            if cur is None or (new is not None and new < cur):
                by_date[d] = s
    return list(by_date.values())


def select_year(year):
    """Pick 12 cleanest scenes, balanced by month, all <= CLOUD_CAP."""
    per_month = {}
    for m in MONTHS:
        scenes = [s for s in query(year, m)
                  if s["cloud_cover"] is not None and s["cloud_cover"] <= CLOUD_CAP]
        scenes.sort(key=lambda s: s["cloud_cover"])
        per_month[m] = scenes

    chosen, leftovers = [], []
    # pass 1: up to PER_MONTH cleanest from each month
    for m in MONTHS:
        chosen.extend(per_month[m][:PER_MONTH])
        leftovers.extend(per_month[m][PER_MONTH:])
    # pass 2: backfill to PER_YEAR from the cleanest leftovers
    leftovers.sort(key=lambda s: s["cloud_cover"])
    i = 0
    while len(chosen) < PER_YEAR and i < len(leftovers):
        chosen.append(leftovers[i])
        i += 1

    if len(chosen) < PER_YEAR:
        print(f"  ! {year}: only {len(chosen)} scenes <= {CLOUD_CAP:.0f}% cloud "
              f"(wanted {PER_YEAR}). Consider raising CLOUD_CAP.")
    chosen.sort(key=lambda s: s["date"])
    return chosen


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_chosen = []

    print("=" * 72)
    print("Task 3 four-year scene manifest  |  12/year, Jun-Sep, <=20% cloud")
    print("No credentials needed - catalog search only, nothing downloaded.")
    print("=" * 72)

    for year in YEARS:
        print(f"\nSelecting {year} ...")
        chosen = select_year(year)
        all_chosen.extend(chosen)
        mc = {m: 0 for m in MONTHS}
        for s in chosen:
            mc[s["month"]] += 1
        print(f"  picked {len(chosen)} scenes  "
              f"(Jun {mc[6]}, Jul {mc[7]}, Aug {mc[8]}, Sep {mc[9]})")
        for s in chosen:
            print(f"    {s['date']}   cloud {s['cloud_cover']:5.1f}%   "
                  f"{s['product_name'][:46]}")

    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["year", "month", "date",
                                           "cloud_cover", "product_id",
                                           "product_name"])
        w.writeheader()
        for s in all_chosen:
            w.writerow(s)

    print("\n" + "=" * 72)
    print(f"TOTAL selected: {len(all_chosen)} scenes")
    for year in YEARS:
        n = sum(1 for s in all_chosen if s["year"] == year)
        print(f"  {year}: {n} scenes")
    print(f"\nManifest written to:\n  {MANIFEST}")
    print("=" * 72)
    print("\nNext: review the dates above. The download step will read this")
    print("manifest, pull each scene, clip to Parma, compute NDRE/LAI, then")
    print("delete the big zip so only ~1 GB sits on disk at a time.")


if __name__ == "__main__":
    main()
