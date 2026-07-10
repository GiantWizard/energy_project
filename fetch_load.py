# Pulls ERCOT system-wide load for both weeks (fuel-mix isn't available
# historically). get_load() only covers a rolling 14-day window, so the
# older volatile week needs get_hourly_load_post_settlements() instead.
import gridstatus

iso = gridstatus.Ercot()

df_calm = iso.get_load("2026-06-28", end="2026-07-05", verbose=True)
df_calm.to_csv("raw_load_calm_week.csv", index=False)
print(f"\ncalm_week: saved {len(df_calm)} rows to raw_load_calm_week.csv")
print(f"Date range: {df_calm['Interval Start'].min()} to {df_calm['Interval Start'].max()}")
print(df_calm["Load"].describe())

df_volatile_all = iso.get_hourly_load_post_settlements("2026-01-24", end="2026-01-31", verbose=True)
df_volatile = df_volatile_all[
    (df_volatile_all["Interval Start"] >= "2026-01-24")
    & (df_volatile_all["Interval Start"] < "2026-01-31")
].reset_index(drop=True)
df_volatile.to_csv("raw_load_volatile_week.csv", index=False)
print(f"\nvolatile_week: saved {len(df_volatile)} rows to raw_load_volatile_week.csv")
print(f"Date range: {df_volatile['Interval Start'].min()} to {df_volatile['Interval Start'].max()}")
print(df_volatile["ERCOT"].describe())
