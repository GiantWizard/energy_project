# Scans single ERCOT days for price volatility to find a better week to
# analyze than the calm 2026-06-28 to 2026-07-04 week already pulled.
import gridstatus
import pandas as pd

iso = gridstatus.Ercot()

candidate_dates = [
    "2026-05-18", "2026-05-25",
    "2026-06-01", "2026-06-08", "2026-06-15", "2026-06-22",
    "2026-06-25", "2026-07-05",
]

rows = []
for d in candidate_dates:
    try:
        df = iso.get_spp(d, market="REAL_TIME_15_MIN", location_type="Trading Hub", verbose=False)
        df = df[df["Location"] == "HB_HOUSTON"]
        stats = {
            "date": d,
            "min": df["SPP"].min(),
            "max": df["SPP"].max(),
            "mean": df["SPP"].mean(),
            "std": df["SPP"].std(),
            "range": df["SPP"].max() - df["SPP"].min(),
            "n": len(df),
        }
        rows.append(stats)
        print(stats)
    except Exception as e:
        print(f"{d}: FAILED - {e}")

result = pd.DataFrame(rows)
result.to_csv("volatility_scan.csv", index=False)
print("\nSorted by range:")
print(result.sort_values("range", ascending=False))
