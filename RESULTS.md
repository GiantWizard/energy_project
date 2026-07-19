# Results: Energy Markets Bridge

## Revision history

1. Initial approach: built a RandomForest model to predict the price 15 minutes ahead, using a using a calm week of data (2026-06-28 to 07-04). Unfortunately, it lost to a naive baseline that just predicts the last price (MAE 2.232 vs 2.002) and the three features borrowed from wiz barely mattered, making up only about 8% of the model's feature iportance compared to `lag_1` (last observed price) making up 83%.
2. Two new angles were tried: forcasting an hour ahead did a little better, though that was still mostly driven by the `hour` feature. Anomaly detection on the other hand worked fairly cleanly, as the intervals it flagged as unusual showed 4 to 7 times higher spread and volatility than the normal ones.
3. Current results: tried four differnt approaches to see if a better result was possible at all. Added demand data, switched from predicting price to predicting direction, tested against a volatile week instead of just a calm one, checked the anomaly detection results against a named ERCOT grid event. Two of the four helped, to an extent. The other two were useless compared to the baseline.

## Data

- ISO / node: ERCOT, `HB_HOUSTON` (Houston Load Zone Trading Hub), real-time market, 15-minute settlement point prices (SPP).
- Calm week: 2026-06-28 to 2026-07-04, 672 rows. Min $8.89, max $66.13, mean $27.18, std $7.87.
- Volatile week: 2026-01-24 to 2026-01-30, 672 rows, zero gaps. Min -$6.66, max $1170.38, mean $106.03, std $105.57 or about 13x the calm week's volatility. This was a winter cold snap where supply couldn't keep up with demand and prices spiked (see Experiment 4).
- Demand data: ERCOT system-wide hourly load for both weeks. Calm week via `gridstatus.get_load`; volatile week via `gridstatus.get_hourly_load_post_settlements`, since `get_load` only covers a rolling 14-day window and the volatile week is over 5 months in the past. See Setup Friction below.

### How the volatile week was found

Because ERCOT's live feed only keeps a rolling 9-10 day of data, any date requested prior to June 27, 2026 simply returns "no documents found." Thus, the original method using `gridstatus.Ercot.get_spp()` didn't work. 

Switching to the annual historical archive hit a compatibility bug, as gridstatus internally uses `.astype("timedelta64[h]")` which pandas 2.3.x no longer supports.

To bypass this, I wrote `fetch_historical_archive.py` to download the raw archive directly and rebuild the interval logic with the supported `pd.to_timedelta(..., unit="h")` function instead. This pulled the entire 2026 year-to-date dataset (Jan 1 – Jul 4; 408,388 rows), allowing me to scan day by day for the highest volatility at `HB_HOUSTON`. 

## Feature engineering

| Feature | Definition | Skyblock analogue |
|---|---|---|
| `momentum_4` | % change in price over the last 4 intervals | Essentially tracking whether an item's price is trending up or down over a certain window |
| `volatility_8` | Rolling std-dev of price over the last 8 intervals | Same as a rolling measure of how choppy a price has been recently |
| `spread_1` | Absolute difference between consecutive interval prices | Similar to the gap between an item's buy and sell price |
| `load`, `load_change_4`| System-wide demand level and its 1-hour % change | This is a new grid-specific signal, since an in-game item market has no "total demand" number the way the power grid does |

## Experiment 1: exogenous demand data

Adding demand data was the logical next step as power prices are heavily driven by grind demand. I originally intended to include fuel-mix and renewable penetration data but `gridstatus.Ercot.get_fuel_mix()` only supports live data and throws a `NotSupported()` error for historical queries. Since there's no historical archive for fuel mix, I decided to use historic al load data instead, pulling it with `get_hourly_load_post_settlements`.


| Model | Calm week t+1 MAE | Volatile week t+1 MAE |
|---|---|---|
| Naive baseline | 2.002 | 4.443 |
| Price-only model | 2.232 (-11.5%) | 7.755 (-74.6%) |
| Price + load model | 2.155 (-7.7%) | 8.122 (-82.8%) |

Observations:
- Marginal impact: Adding load features narrowed the gap slightly during the calm week but still failed to beat the naive baseline. On the volatile week, adding load decreased performance.
- Lag-1 still remains dominant, as `load` picked up 3.1% feature importance during the calm week, the `lag_1` price feature still completely dominates the moden's predictions.
- Hourly load data is too coarse and slow-moving when resampled onto a 15-minute grid. It fails to capture the rapid shifts of real-time settlement prices, offering little value beyond what the price history already provides.

## Experiment 2: direction classification

A genuine, repeatable win.

Reframing the task from predicting exact prices (regression) to predicting the direction of the next price move yielded a stable performance improvement, which is consistent with my observations from other times I tried to model markets.

To keep the evaluation rigorous, the RandomForest classifier was compared against a non-trivial naive baseline that always guesses the direction of the last observed move instead of a simple majority-class guess.

| Metric | Calm week accuracy | Volatile week accuracy |
|---|---|---|
| Naive (same as last direction) | 0.586 | 0.564 |
| RandomForest classifier | 0.624 | 0.579 |
| Beats naive? | Yes (+3.8 pts) | Yes (+1.5 pts) |

Observations:
- The model successfully outperformed the baseline in both market conditions, proving that direction is a lot more predictable than exact numerical values
- Unlike the point-forcasting models where `lag_1` completely dominated, feature importances here were distributed evenly, with all seven features within a 11.7% to 19.2% band.
- The top contributors for the calm week were `hour`, `momentum_4`, and `lag_4`, confirming that the engineered features (`momentum_4`, `volatility_8`, `spread_1`) carry meaningful predictive power.

## Experiment 3: forecasting on the volatile week

The hypothesis is that a highly volatile period would provide a stronger signal for my model to find, allowing the model to close the performance gap with the baseline. However, the higher volatility made the model's relative performace significantly worse.

| | t+1 MAE | t+4 (1hr) MAE |
|---|---|---|
| Naive | 4.443 | 9.321 |
| Model | 7.755 (-74.6%) | 15.929 (-70.9%) |

| Horizon | Naive MAE | Model MAE | Relative Loss |
| --- | --- | --- | --- |
| t+1 | 4.443 | 7.755 | -74.6% |
| t+4 | 9.321 | 15.929 | -70.9% |

Observations:
- While the model trailed the baseline by just 11.5% during the calm week, it trailed significantly more in the volatile week, plummeting to 74.6%
- During sharp price spikes, the most reliable indicator is the immediate past price. Because a RandomForest operates by averaging historical trees, it can't react fast enough to a spike.This causes its predictions to visually lag behind the acutal spikes by a step (as seen in `experiments_plot.png`).
- Counterintuitively, higher raw volatility did not introduce usable structure for the model. It only expanded the margin for error, giving the baseline a huge inherent advantage. 

## Experiment 4: Anomaly detection via IsolationForest

This stands as the project's strongest result. The IsolationForest model was trained exclusively on engieered features (`momentum`, `volatility`, and `spread`), and intentionally omits raw lag prices.

| Metric | Calm week | Volatile week |
|---|---|---|
| Flagged | 34/665 (5.1%) | 34/665 (5.1%) |
| Median `spread_1` (Normal vs. Flagged) | 0.74 vs. 5.57 (7.5x) | 3.68 vs. 61.07 (16.6x) |
| Median `volatility_8` (Normal vs. Flagged) | 1.74 vs. 7.38 (4.2x) | 8.44 vs. 194.29 (23x) |

Observations:
- The model performed how a robust anomaly detector would, as high volatility yielded much more pronnounced signals. Flagged intervals in the volatile week showed massive multipliers compared to normal baseline operations (16.6x for spread, 23x for volatility).
- The single most extreme anomaly captured the year-to-date price peak on January 28, 2026, at 07:00 CST ($1,170.38/MWh). The top five flagged anomalies cluster around the actual price spikes on January 25 (evening) and January 28 (morning), visible in the second panel of `experiments_plot.png`.
- An official ERCOT press release from January 21, 2026 ("_ERCOT Issues Weather Watch_," ercot.com/news/release/01212026-ercot-issues-weather) explicitly warned of grid stress spanning January 24-27 due to sub-freezing temperatures, which confirms the model's reliability.
- Note: while ERCOT's advisory confirms the grid stress, independent media coverage quantifying the exact $/MWh levels during this window is sparse, so can't be precisely reported here.


## Summary

| Experiment | Result |
|---|---|
| 1. Exogenous load data | Slight improvement for the calm week, butdidn't beat the baseline. Confirmed essentially marginal impact.|
| 2. Direction classification | Beat the baseline across both weeks (+3.8 / +1.5 accuracy points). This was the only task where engineered features showed meaningful importance.|
| 3. Volatile-period forecasting | Made the model's relative underperformance worse instead of better.|
| 4. Deepened anomaly detection | Delivered 16–23x signal separation during the volatile week, and flagged anomalies matched a documented ERCOT grid event.|

## Setup friction

- `get_spp()` only contains a 9-10 day rolling history, making it useless for historical scans.
- Historical fuel-mix data doesn't exist in the archive, only data for today and yesterday exist
- The annual archive call (`get_rtm_spp(year)`) throws a fatal error because `gridstatus` internally relies on `astype("timedelta64[h]")`, which is deprecated in modern pandas. This required bypassing the internal function entirely to reconstruct the interval logic manually via `pd.to_timedelta(..., unit="h")`.
- The standard `get_load()` endpoint caps out at a 14-day rolling window. Reaching back five months to analyze the volatile week required shifting to the `get_hourly_load_post_settlements()` archive.
