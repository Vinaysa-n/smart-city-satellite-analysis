"""
Download Parma meteorology from Open-Meteo (ERA5 reanalysis, gap-free)
for 2022-2025, April 1 - September 30 each year, aggregated to daily.

Output: parma_meteo_openmeteo_2022_2025_daily.csv  (expected 732 rows = 183 days x 4 years)
"""

import requests
import time
import pandas as pd


# SETTINGS
# Parma city centre. ERA5 grid is ~25 km, so the exact point is not critical.
LATITUDE = 44.80
LONGITUDE = 10.33

YEARS = [2022, 2023, 2024, 2025]
WINDOW_START = "04-01"   # April 1
WINDOW_END = "09-30"     # September 30

# Daily aggregation calendar.
# "Europe/Berlin" = Italian local day (CET/CEST) -- the standard meteo convention.
# so the meteo days line up exactly with the NO2 days.
TIMEZONE = "Europe/Berlin"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "wind_speed_10m",
    "precipitation",
]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OUT_DAILY = "parma_meteo_openmeteo_2022_2025_daily.csv"


# FETCH ONE YEAR (hourly) WITH SIMPLE RETRY
def fetch_year(year):
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": f"{year}-{WINDOW_START}",
        "end_date": f"{year}-{WINDOW_END}",
        "hourly": ",".join(HOURLY_VARS),
        "timezone": TIMEZONE,
        "wind_speed_unit": "ms",
        "temperature_unit": "celsius",
        "precipitation_unit": "mm",
    }
    for attempt in range(3):
        try:
            r = requests.get(ARCHIVE_URL, params=params, timeout=60)
            r.raise_for_status()
            data = r.json()
            if "hourly" not in data:
                raise ValueError(f"No hourly data returned: {data}")
            return data["hourly"]
        except Exception as e:
            print(f"  attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    raise RuntimeError(f"Failed to fetch {year} after 3 attempts")



# MAIN: download -> aggregate to daily -> stitch all years

def main():
    all_daily = []

    for year in YEARS:
        print(f"Downloading {year}  ({year}-{WINDOW_START} -> {year}-{WINDOW_END}) ...")
        hourly = fetch_year(year)

        df = pd.DataFrame(hourly)
        df["time"] = pd.to_datetime(df["time"])
        df["date"] = df["time"].dt.date

        daily = df.groupby("date").agg(
            temp_mean_c=("temperature_2m", "mean"),
            temp_min_c=("temperature_2m", "min"),
            temp_max_c=("temperature_2m", "max"),
            pressure_mean_hpa=("surface_pressure", "mean"),
            humidity_mean_pct=("relative_humidity_2m", "mean"),
            wind_mean_ms=("wind_speed_10m", "mean"),
            rain_sum_mm=("precipitation", "sum"),
        ).reset_index()

        for c in daily.columns:
            if c != "date":
                daily[c] = daily[c].round(2)

        missing = int(daily.isna().sum().sum())
        print(f"  {len(daily)} days, missing values: {missing}")
        all_daily.append(daily)
        time.sleep(1)

    combined = pd.concat(all_daily, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    combined.to_csv(OUT_DAILY, index=False)

    print("\n" + "=" * 55)
    print("DONE")
    print(f"  Total days:              {len(combined)}  (expected 732)")
    print(f"  Date range:              {combined['date'].min()} -> {combined['date'].max()}")
    print(f"  Missing values anywhere: {int(combined.isna().sum().sum())}")
    print(f"  Saved:                   {OUT_DAILY}")
    print("=" * 55)


if __name__ == "__main__":
    main()
