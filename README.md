# Energy Markets Bridge

This project is a pipeline that pulls ERCOT electricity prices and tests whether the features I built previously for a Hypixel Skyblock price project ([GiantWizard/wiz](https://github.com/GiantWizard/wiz)) can help predict anything in a real market. It runs through several different experiements, from short-term forcasting to predicting direction or flagging unusual price spikes.

## Why this project

wiz treats in-game prices like a fast-moving market and builds features around short-term momentum, how choppy prices are, the price gap between consecutive orders in the order book etc. to try and understand how prices may move. ERCOT's electricity prices behave in a similar way, as prices update every 15 minutes based on how much electricity people are using and how much is available. The project takes the features (not the code) and tries to apply it to actual data instead of just simulated markets.

## Results

The core result is that a native forcast that literally just guesses the next price will equal the last price beats a RandomForest model built on Skyblock-esque features at a 15 minute resolution on both a calm week and a volatile week, which is a little disheartening. This shows that ERCOT's real time prices are dominated by whatever the price just did, no matter how volatile thigns get. Two experiments do show the features doing work, though. Predicting the direction of the next move beats a naive baseline, and flagging anomolaies with an IsolationForest model trained on momentum, volatility, and spread caches itnervals with 16-23 times higher volatility during the volatille week, and the flagged intervals line up with a dated ERCOT Weather Watch alert from 2026-01-24 to 01-27.

## What's here

- `fetch_data.py`: pulls one week of ERCOT `HB_HOUSTON` real-time 15-min settlement point prices (the calm week), saves to `raw_hb_houston_rtm_spp.csv`.
- `fetch_historical_archive.py`: downloads ERCOT's annual historical RTM archive to scan the full year for the most volatile week, saving `raw_hb_houston_volatile_week.csv` and the full-year archive.
- `scan_volatility.py`: the scan script that discovered ERCOT's live document list only keeps about 9-10 days of history (documented in `RESULTS.md`).
- `fetch_load.py`: pulls ERCOT demand for both weeks, used as the extra feature for Experiment 1.
- `analysis.py`: the earlier script, three experiments (t+1 forecast, t+4 forecast, anomaly detection) on the calm week only.
- `experiments.py`: the extended script, running all four additional experiments (exogenous load, direction classification, volatile-week forecasting, deepened anomaly detection) across both weeks. Saves `experiments_plot.png` and `experiments_results.txt`.
- `RESULTS.md`: the full, current results writeup, including the earlier history, all four additional experiments, and a verdict.

## Setup

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

See `RESULTS.md` for the full writeup.
