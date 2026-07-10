# Fetches ERCOT's annual RTM price archive directly instead of using
# gridstatus.Ercot.get_rtm_spp(year), which breaks under pandas 2.3.x
# (astype("timedelta64[h]") is no longer allowed).
import gridstatus
from gridstatus import utils
import pandas as pd

RTID = 13061  # HISTORICAL_RTM_LOAD_ZONE_AND_HUB_PRICES_RTID
YEAR = 2026

iso = gridstatus.Ercot()
doc_info = iso._get_document(
    report_type_id=RTID,
    constructed_name_contains=f"{YEAR}.zip",
    verbose=True,
)
print(f"Archive doc: {doc_info}")

x = utils.get_zip_file(doc_info.url, verbose=True)
all_sheets = pd.read_excel(x, sheet_name=None)
df = pd.concat(all_sheets.values())

count = df[["Delivery Hour", "Delivery Interval"]].isnull().all(axis=1).sum()
if count == 1:
    df = df.dropna(subset=["Delivery Hour", "Delivery Interval"], how="all")
elif count > 1:
    raise ValueError("Parsing error, more than expected null rows found")

df["Delivery Interval"] = df["Delivery Interval"].astype("Int64")

df.rename(
    columns={
        "Delivery Hour": "HourEnding",
        "Delivery Interval": "DeliveryInterval",
        "Delivery Date": "DeliveryDate",
    },
    inplace=True,
)

interval_length = pd.Timedelta(minutes=15)
df["HourBeginning"] = df["HourEnding"] - 1
df["Interval Start"] = (
    pd.to_datetime(df["DeliveryDate"])
    + pd.to_timedelta(df["HourBeginning"], unit="h")  # pandas 2.3 safe replacement
    + ((df["DeliveryInterval"] - 1) * interval_length)
)
df["Interval Start"] = df["Interval Start"].dt.tz_localize(
    iso.default_timezone, ambiguous="infer", nonexistent="shift_forward",
)
df["Interval End"] = df["Interval Start"] + interval_length

print(f"\nTotal rows: {len(df)}")
print(f"Date range: {df['Interval Start'].min()} to {df['Interval Start'].max()}")
print(f"Columns: {df.columns.tolist()}")

# Data is long-format: one row per (Interval Start, Settlement Point Name).
# Keep just HB_HOUSTON, matching the rest of the pipeline.
print(f"Sample Settlement Point Names: {sorted(df['Settlement Point Name'].unique())[:15]}")

out = df[df["Settlement Point Name"] == "HB_HOUSTON"][["Interval Start", "Settlement Point Price"]]
out = out.rename(columns={"Settlement Point Price": "SPP"}).copy()
out = out.sort_values("Interval Start").reset_index(drop=True)
out.to_csv("historical_archive_hb_houston_2026.csv", index=False)
print(f"\nSaved {len(out)} rows to historical_archive_hb_houston_2026.csv")
print(out["SPP"].describe())
