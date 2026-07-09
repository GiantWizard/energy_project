# Energy Markets Bridge

A pipeline that pulls real ERCOT electricity prices and tests whether the
features I built for a Hypixel Skyblock Bazaar price project
([GiantWizard/wiz](https://github.com/GiantWizard/wiz)) actually help
predict anything in a real market. It runs through several different
experiments (short-term forecasting at two time horizons, adding in real
demand data, predicting direction instead of exact price, and flagging
unusual price spikes, all tested against both a calm week and an actual
volatile week) instead of running one experiment and reframing the writeup
until it sounds better.

## Why this project

The Skyblock bazaar project treats in-game item prices like a fast-moving
market and builds features around short-term momentum, how choppy the
price has been, and the gap between consecutive prices, to try to
understand how prices move. ERCOT's real-time electricity prices behave in
a similar way: a price that updates every 15 minutes based on how much
electricity people are using versus how much is available. This project
takes that same feature-building approach, not the actual code, and
applies it to real grid data, as a bridge toward Cornell CBE's Energy
Economics and Engineering concentration, which deals with exactly this
kind of grid economics and price formation.

Headline result (the full story is in `RESULTS.md`): a naive forecast that
just guesses the next price will equal the last price beats a
RandomForest model built on the Skyblock-style features, at 15-minute
resolution, on both a calm week and an actual volatile winter week when
prices spiked because supply couldn't keep up with demand. ERCOT's
real-time prices are dominated by whatever the price just did, no matter
how volatile things get. Two experiments do show the features doing real
work. Predicting the direction of the next move (up or down) beats a
naive baseline on both weeks, and flagging anomalies with an
IsolationForest model trained only on momentum, volatility, and spread
catches intervals with 16 to 23 times higher volatility during the
volatile week, and those flagged intervals line up with a real, dated
ERCOT Weather Watch alert from 2026-01-24 to 01-27.

## What's here

- `fetch_data.py`: pulls one week of ERCOT `HB_HOUSTON` real-time 15-min
  settlement point prices (the calm week), saves to
  `raw_hb_houston_rtm_spp.csv`.
- `fetch_historical_archive.py`: downloads ERCOT's annual historical RTM
  archive (works around a pandas 2.3 compatibility bug in `gridstatus`'s
  `get_rtm_spp`) to scan the full year for the most volatile week, saving
  `raw_hb_houston_volatile_week.csv` and the full-year archive.
- `scan_volatility.py`: the scan script that discovered ERCOT's live
  document list only keeps about 9-10 days of history, a real constraint
  documented in `RESULTS.md`.
- `fetch_load.py`: pulls ERCOT system-wide demand (load) for both weeks,
  used as the extra feature for Experiment 1.
- `analysis.py`: the earlier script, three experiments (t+1
  forecast, t+4 forecast, anomaly detection) on the calm week only.
- `experiments.py`: the extended script, running all four additional
  experiments (exogenous load, direction classification, volatile-week
  forecasting, deepened anomaly detection) across both weeks. Saves
  `experiments_plot.png` and `experiments_results.txt`.
- `RESULTS.md`: the full, current results writeup, including the earlier
  history, all four additional experiments, and an honest final verdict.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See `RESULTS.md` for the full writeup.
