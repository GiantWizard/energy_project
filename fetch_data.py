# Pulls one week of ERCOT RTM 15-min prices for HB_HOUSTON.
# Picked over HB_WEST/HB_PAN, which showed negative SPPs from curtailment.
import gridstatus

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
