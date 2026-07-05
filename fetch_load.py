# Pulls ERCOT system-wide actual load for both weeks, as the exogenous
# demand-level feature for Experiment 1.
#
# get_fuel_mix() only supports "today"/"yesterday" for ERCOT (confirmed
# empirically, raises NotSupported() for any other date, and there's no
# separate historical fuel-mix archive the way there is for load and price).
# So renewable-penetration / fuel-mix features aren't achievable historically
# with this library; this script pulls load only.
#
# Two different load methods are needed for the two weeks. The calm week
# (2026-06-28) is within get_load()'s rolling 14-day window, so get_load()
# works directly and returns system-wide totals under a "Load" column. The
# volatile week (2026-01-24) is about 5 months in the past, outside that
# window, so get_load() raises NotSupported() for it. get_hourly_load_post_
# settlements() pulls from ERCOT's separate historical load archive instead,
# with no such window limit, but returns hourly load broken out by weather
# zone plus an "ERCOT" column for the system-wide total, which is the one
# used downstream as the demand feature.
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
