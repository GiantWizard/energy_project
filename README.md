# Energy Markets Bridge

A pipeline that pulls ERCOT real-time electricity prices and tests whether
market-microstructure-style features borrowed conceptually from a Hypixel
Skyblock Bazaar market-analysis project
([GiantWizard/wiz](https://github.com/GiantWizard/wiz)) carry real signal.
It runs several different angles (point forecasting at two horizons,
exogenous demand data, direction classification, and anomaly detection, on
both a calm week and a real high-volatility winter event) instead of
reworking one experiment's framing until it sounds better.

## Why this project

The Skyblock bazaar pipeline treats in-game item prices as a high-frequency
order-book-like series and builds features around short-horizon momentum,
volatility, and spread to understand price formation. Electricity real-time
markets (ERCOT settlement point prices) are a real-world analogue: a
continuously updating price series shaped by supply/demand imbalance,
published every 15 minutes. This project reuses that feature-engineering
thinking, not the code, applied to real grid data, as a bridge toward
Cornell CBE's Energy Economics and Engineering concentration (grid
economics, price formation).

Headline result (full story in `RESULTS.md`): a naive next-interval
forecast (predict price = last observed price) beats a RandomForest using
the Skyblock-style features at 15-minute resolution, on both a calm week
and a genuinely volatile winter scarcity-pricing week. ERCOT RTM prices are
dominated by the most recent observation regardless of how volatile the
period is. Two angles do show the features doing real work: direction
classification (predicting up/down instead of the exact price beats a
naive "same as last move" baseline on both weeks) and anomaly detection
(IsolationForest on momentum/volatility/spread alone flags intervals with
16-23x higher volatility on the volatile week, and those flagged intervals
line up with a real, dated ERCOT Weather Watch event from 2026-01-24 to
01-27).

## What's here

- `fetch_data.py`: pulls one week of ERCOT `HB_HOUSTON` real-time 15-min
  settlement point prices (the calm week), saves to
  `raw_hb_houston_rtm_spp.csv`.
- `fetch_historical_archive.py`: downloads ERCOT's annual historical RTM
  archive (works around a pandas 2.3 compatibility bug in `gridstatus`'s
  `get_rtm_spp`) to scan the full year for the most volatile week, saving
  `raw_hb_houston_volatile_week.csv` and the full-year archive.
- `scan_volatility.py`: the scan script that discovered ERCOT's live
  document list only retains about 9-10 days of history, a real constraint
  documented in `RESULTS.md`.
- `fetch_load.py`: pulls ERCOT system-wide demand (load) for both weeks,
  as the exogenous feature for Experiment 1.
- `analysis.py`: the prior round's script, three experiments (t+1 forecast,
  t+4 forecast, anomaly detection) on the calm week only.
- `experiments.py`: this round's comprehensive script, all four new
  experiments (exogenous load, direction classification, volatile-week
  forecasting, deepened anomaly detection) across both weeks. Saves
  `experiments_plot.png` and `experiments_results.txt`.
- `RESULTS.md`: the full, current results writeup, including the two prior
  rounds' history, all four new experiments, and an honest final verdict.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See `RESULTS.md` for the full writeup.
