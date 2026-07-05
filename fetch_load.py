"""
Pull ERCOT system-wide actual load for both weeks under analysis, as the
exogenous demand-level feature for Experiment 1.

Note: gridstatus.Ercot.get_fuel_mix() only supports "today"/"yesterday" for
ERCOT (confirmed empirically -- raises NotSupported() for any other date,
and there is no separate historical fuel-mix archive report type the way
there is for load and price). So renewable-penetration / fuel-mix features
are not achievable for historical weeks with this library; this script
pulls load only, and RESULTS.md documents the fuel-mix gap honestly rather
than faking it.

Two different load methods are needed for the two weeks:

- The calm week (2026-06-28) is within `get_load()`'s rolling window
  (`iso.LOAD_HISTORICAL_MAX_DAYS` = 14 days back from today), so `get_load()`
  works directly and returns system-wide totals under a "Load" column.
- The volatile week (2026-01-24) is ~5 months in the past, well outside that
  14-day window -- `get_load()` raises NotSupported() for it (confirmed
  empirically). `get_hourly_load_post_settlements()` pulls from ERCOT's
  separate historical load archive instead, which has no such rolling-window
  limit, but returns a different schema: hourly load broken out by weather
  zone (Coast, East, Far West, North, North Central, South, South Central,
  West) plus an "ERCOT" column for the system-wide total -- that "ERCOT"
  column is the one used downstream as the system-wide demand feature.
"""
import gridstatus
import pandas as pd

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
