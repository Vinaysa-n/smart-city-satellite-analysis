"""
Cross-check the Open-Meteo meteo file against NASA POWER (an INDEPENDENT
reanalysis: NASA's MERRA-2 model, different from the ERA5 behind Open-Meteo).

Two independent reanalyses agreeing on Parma's daily weather is strong evidence
the data is correct. Free, no login (like Open-Meteo).

It prints:
  * agreement statistics per variable (mean abs difference, bias, correlation)
  * 5 sample dates per calendar month, Open-Meteo vs NASA side by side
"""

import time
import requests
import numpy as np
import pandas as pd

# your Open-Meteo file (the .xls that is actually a CSV)
OPENMETEO_FILE = r"C:\Users\vinay\OneDrive\Desktop\Thesis\files for thesis\parma_meteo_openmeteo_2022_2025_daily.xls"

LAT, LON = 44.80, 10.33
START, END = "20220401", "20250930"

NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
# NASA POWER variables -> meaning (units): T2M mean/min/max C, PS kPa,
# RH2M %, WS10M m/s, PRECTOTCORR mm/day
NASA_VARS = "T2M,T2M_MIN,T2M_MAX,PS,RH2M,WS10M,PRECTOTCORR"


def fetch_nasa():
    params = {
        "parameters": NASA_VARS, "community": "AG",
        "longitude": LON, "latitude": LAT,
        "start": START, "end": END, "format": "JSON",
    }
    for attempt in range(3):
        try:
            r = requests.get(NASA_URL, params=params, timeout=120)
            r.raise_for_status()
            p = r.json()["properties"]["parameter"]
            break
        except Exception as e:
            print(f"  NASA fetch attempt {attempt+1} failed: {e}")
            time.sleep(5)
    else:
        raise RuntimeError("NASA POWER fetch failed")

    df = pd.DataFrame({k: pd.Series(v) for k, v in p.items()})
    df.index = pd.to_datetime(df.index, format="%Y%m%d")
    df = df.replace(-999.0, np.nan)
    out = pd.DataFrame({
        "date": df.index,
        "temp_mean_c": df["T2M"].values,
        "temp_min_c": df["T2M_MIN"].values,
        "temp_max_c": df["T2M_MAX"].values,
        "pressure_mean_hpa": df["PS"].values * 10.0,   # kPa -> hPa
        "humidity_mean_pct": df["RH2M"].values,
        "wind_mean_ms": df["WS10M"].values,
        "rain_sum_mm": df["PRECTOTCORR"].values,
    })
    return out.reset_index(drop=True)


def main():
    om = pd.read_csv(OPENMETEO_FILE)
    om["date"] = pd.to_datetime(om["date"])
    print(f"Open-Meteo file: {len(om)} rows")

    print("Fetching NASA POWER (independent reanalysis) ...")
    nasa = fetch_nasa()
    # keep only Apr-Sep dates that exist in your file
    nasa = nasa[nasa["date"].isin(om["date"])]

    cols = ["temp_mean_c", "temp_min_c", "temp_max_c", "pressure_mean_hpa",
            "humidity_mean_pct", "wind_mean_ms", "rain_sum_mm"]
    merged = om.merge(nasa, on="date", suffixes=("_om", "_nasa"))
    print(f"Compared on {len(merged)} common days.\n")

    print("=== AGREEMENT per variable (Open-Meteo vs NASA POWER) ===")
    print(f"{'variable':18s} {'mean|diff|':>10s} {'bias(OM-NASA)':>14s} {'corr':>7s}")
    for c in cols:
        a, b = merged[f"{c}_om"], merged[f"{c}_nasa"]
        mad = (a - b).abs().mean()
        bias = (a - b).mean()
        corr = a.corr(b)
        print(f"{c:18s} {mad:10.2f} {bias:14.2f} {corr:7.3f}")

    print("\n=== 5 sample dates per month (OM = Open-Meteo, NA = NASA) ===")
    merged["mon"] = merged["date"].dt.month
    for mon in sorted(merged["mon"].unique()):
        sub = merged[merged["mon"] == mon].reset_index(drop=True)
        idx = np.linspace(0, len(sub) - 1, min(5, len(sub))).astype(int)
        name = pd.Timestamp(2000, mon, 1).strftime("%B")
        print(f"\n-- {name} --")
        print(f"  {'date':12s} {'Tmean OM/NA':>14s} {'Press OM/NA':>16s} "
              f"{'RH OM/NA':>12s} {'Wind OM/NA':>13s}")
        for i in idx:
            r = sub.loc[i]
            print(f"  {r['date'].date()!s:12s} "
                  f"{r['temp_mean_c_om']:5.1f}/{r['temp_mean_c_nasa']:<5.1f}   "
                  f"{r['pressure_mean_hpa_om']:6.0f}/{r['pressure_mean_hpa_nasa']:<6.0f}  "
                  f"{r['humidity_mean_pct_om']:4.0f}/{r['humidity_mean_pct_nasa']:<4.0f}  "
                  f"{r['wind_mean_ms_om']:4.1f}/{r['wind_mean_ms_nasa']:<4.1f}")

    print("\nReading it: temp/pressure/humidity should match closely (corr > 0.9,")
    print("small bias). Wind and rain differ more between reanalyses -- that is")
    print("expected and not an error in your data.")


if __name__ == "__main__":
    main()
