# Pulls one week of ERCOT real-time (15-min) settlement point prices for
# HB_HOUSTON and saves to a local CSV so later runs don't re-fetch.
#
# HB_HOUSTON was picked after checking all 7 trading hubs on a sample day
# (2026-06-28): all had full 96/day interval coverage, but HB_WEST and HB_PAN
# showed negative SPPs from renewable curtailment, so HB_HOUSTON (no negative
# prices, ERCOT's major urban load-center hub) was the cleaner first pass.
import gridstatus
import pandas as pd

iso = gridstatus.Ercot()

START = "2026-06-28"
END = "2026-07-05"  # exclusive end, gives us about one week

df = iso.get_spp(
    START,
    end=END,
    market="REAL_TIME_15_MIN",
    location_type="Trading Hub",
    verbose=True,
)

df = df[df["Location"] == "HB_HOUSTON"].copy()
df = df.sort_values("Interval Start").reset_index(drop=True)

out_path = "raw_hb_houston_rtm_spp.csv"
df.to_csv(out_path, index=False)

print(f"Saved {len(df)} rows to {out_path}")
print(f"Date range: {df['Interval Start'].min()} to {df['Interval Start'].max()}")
print(df["SPP"].describe())
