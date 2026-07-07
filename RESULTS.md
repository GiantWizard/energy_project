# Results: Energy Markets Bridge

## Revision history

1. Weekend one: built a RandomForest model to forecast the price 15 minutes ahead, using a calm week of data (2026-06-28 to 07-04). It lost to a naive baseline that just predicts the last price again (MAE 2.232 vs 2.002), and the three features borrowed from the Skyblock project barely mattered: together they made up only about 8% of the model's feature importance, while `lag_1` (just the last observed price) made up 83% on its own.
2. First rework: instead of just rewriting the summary to sound better, tried two new angles. Forecasting an hour ahead instead of 15 minutes did a little better, though that was driven mostly by the `hour` feature. Anomaly detection worked cleanly: the intervals it flagged as unusual showed 4 to 7 times higher spread and volatility than the normal ones.
3. This rework: tried four different angles to see if a better result was possible at all. Added real demand data, switched from predicting the exact price to predicting direction, tested against an actual volatile week instead of another calm one, and checked the anomaly detection results against a real, named ERCOT grid event. Two of the four helped, in a modest but real way. The other two did not, and are reported below exactly as they came out.

## Data

- ISO / node: ERCOT, `HB_HOUSTON` (Houston Load Zone Trading Hub), real-time market, 15-minute settlement point prices (SPP).
- Calm week: 2026-06-28 to 2026-07-04, 672 rows, zero gaps. Min $8.89, max $66.13, mean $27.18, std $7.87.
- Volatile week (new this round): 2026-01-24 to 2026-01-30, 672 rows, zero gaps. Min -$6.66, max $1170.38, mean $106.03, std $105.57, about 13x the calm week's volatility. This was an actual winter cold snap where supply couldn't keep up with demand and prices spiked (see Experiment 4).
- Demand data (new this round): ERCOT system-wide hourly load, pulled for both weeks. Calm week via `gridstatus.get_load`; volatile week via `gridstatus.get_hourly_load_post_settlements`, since `get_load` only covers a rolling 14-day window and the volatile week is over 5 months in the past. See Setup Friction below.

### How the volatile week was found

`gridstatus.Ercot.get_spp()`, the method used for the original calm-week pull, draws from ERCOT's live rolling document list, which turned out to only keep about 9-10 days of real-time SPP documents. I confirmed this by testing it directly: every date requested before 2026-06-27 failed with "no documents found," while 2026-06-27 onward worked. So scanning the last month or two for a volatile day wasn't going to work through that method.

Instead, ERCOT publishes an annual historical RTM archive (`gridstatus.Ercot.get_rtm_spp(year)`, report type 13061) that covers the whole year in one file. That call hits a real bug under pandas 2.3.x: gridstatus's internal `parse_doc` function calls `.astype("timedelta64[h]")`, which pandas 2.3 no longer allows (it only permits s/ms/us/ns units). Rather than patch the installed library, `fetch_historical_archive.py` downloads the same underlying file and rebuilds the same interval logic using `pd.to_timedelta(..., unit="h")` instead, which pandas 2.3 accepts. That produced the full 2026-01-01 to 2026-07-04 year-to-date archive (408,388 rows across all settlement points), which I scanned day by day for `HB_HOUSTON` to find the highest-volatility week.

## Feature engineering

| Feature | Definition | Skyblock analogue |
|---|---|---|
| `momentum_4` | % change in price over the last 4 intervals (1 hour) | Same idea as tracking whether an item's price is trending up or down over a short window |
| `volatility_8` | Rolling std-dev of price over the last 8 intervals (2 hours) | Same as a rolling measure of how choppy a price has been recently |
| `spread_1` | Absolute difference between consecutive interval prices | Similar to the gap between an item's buy and sell price |
| `load`, `load_change_4` (new) | System-wide demand level and its 1-hour % change | No Skyblock equivalent. This is a genuinely new, grid-specific signal, since an in-game item market has no "total demand" number the way the power grid does |

## Experiment 1: exogenous demand (load) data

Marginal help, doesn't flip the result.

The reasoning here was different from just re-tuning the price-only features: adding real demand data is a legitimate reason to expect improvement, since price is partly a function of demand.

Fuel-mix and renewable-penetration data, which I originally wanted to add too, turned out not to be accessible historically at all. `gridstatus.Ercot.get_fuel_mix()` only supports `"today"`/`"yesterday"` for ERCOT (I confirmed this by calling it directly, it raises `NotSupported()` for any other date), and there's no separate historical fuel-mix archive the way there is for load and price. That's a real, confirmed limitation, not something I worked around. I used load (demand level) instead, since it's available historically through a separate archive (`get_hourly_load_post_settlements`).

| | Calm week t+1 MAE | Volatile week t+1 MAE |
|---|---|---|
| Naive baseline | 2.002 | 4.443 |
| Price-only model | 2.232 (-11.5%) | 7.755 (-74.6%) |
| Price + load model | 2.155 (-7.7%) | 8.122 (-82.8%) |

Adding load features narrowed the gap slightly on the calm week (model MAE improved from 2.232 to 2.155) but still didn't beat naive, and made things worse on the volatile week. `load` itself picked up modest feature importance (3.1% on the calm week), but it didn't change the underlying picture: `lag_1` still dominates. This doesn't flip the original finding. A likely reason is that hourly load, resampled onto a 15-minute grid, is a coarse and slow-moving signal compared to how fast settlement prices actually move, so it adds only a little on top of what the price series already tells you about itself.

## Experiment 2: direction classification

A genuine, repeatable win.

Instead of predicting the exact next price, I reframed the task as predicting just the direction of the next move, up or down, and compared that against a baseline that always guesses "same direction as the last observed move." That's a fair, non-trivial baseline, not just picking whichever direction happens to be more common.

| | Calm week accuracy | Volatile week accuracy |
|---|---|---|
| Naive (same as last direction) | 0.586 | 0.564 |
| RandomForest classifier | 0.624 | 0.579 |
| Beats naive? | Yes (+3.8 pts) | Yes (+1.5 pts) |

This is a real, if modest, win on both weeks. More interesting is that the feature importances look completely different from the point-forecasting task. On the calm week, `hour` (19.2%), `momentum_4` (17.6%), and `lag_4` (15.2%) lead, with all seven features landing in a narrow 11.7%-19.2% band. No single feature dominates the way `lag_1` did at 83% in the point-forecast task. The Skyblock-style features (`momentum_4`, `volatility_8`, `spread_1`) are actually pulling weight here, not just registering a token contribution. This is the clearest evidence in the whole project that the features are useful for something, just not for pinning down the exact price.

## Experiment 3: forecasting on the volatile week

Makes the model's relative loss worse instead of better.

The hypothesis going in was that a genuinely more volatile period should give the model more real signal to find, which could close the gap with naive.

| | t+1 MAE | t+4 (1hr) MAE |
|---|---|---|
| Naive | 4.443 | 9.321 |
| Model (price-only) | 7.755 (-74.6%) | 15.929 (-70.9%) |

That's not what happened. If anything, the opposite did. On the calm week the model lost to naive by 11.5% (t+1); on the volatile week it lost by 74.6%. During sharp, scarcity-driven price spikes, the single most predictive thing is still "what did the price just do" (naive), and an averaging model like RandomForest actually falls further behind because it can't react fast enough to the spike's speed and size. You can see this directly in `experiments_plot.png`'s third panel, where the model's t+1 predictions lag the real spikes by a step. Higher raw volatility gave the model more error to make, not more structure to actually use, relative to the naive baseline. That's a counterintuitive finding, but a real one worth keeping rather than something to smooth over.

## Experiment 4: anomaly detection, deepened

The strongest result in the project, now with a real-world check.

I ran the same IsolationForest (momentum, volatility, and spread only, no lag prices) on the volatile week.

| | Calm week | Volatile week |
|---|---|---|
| Flagged | 34/665 (5.1%) | 34/665 (5.1%) |
| Median `spread_1`, normal vs. flagged | 0.74 vs. 5.57 (7.5x) | 3.68 vs. 61.07 (16.6x) |
| Median `volatility_8`, normal vs. flagged | 1.74 vs. 7.38 (4.2x) | 8.44 vs. 194.29 (23x) |

The separation is far stronger on the volatile week, which is exactly what you'd want from a working anomaly detector: bigger real anomalies produce a bigger, clearer signal. The single most extreme flagged interval is 2026-01-28 07:00 CST at $1170.38/MWh (the year-to-date price peak), and the top 5 flagged intervals are dominated by the two real spike clusters on 2026-01-25 (evening) and 2026-01-28 (morning). See `experiments_plot.png`, second panel, where the flags sit almost exactly on the two visible price spikes.

I checked this against the real world through a live web search rather than just assuming it lined up. ERCOT's own press release from 2026-01-21, "ERCOT Issues Weather Watch" (ercot.com/news/release/01212026-ercot-issues-weather), announces a Weather Watch for 2026-01-24 through 2026-01-27: below-freezing temperatures with a chance of frozen precipitation, higher electrical demand, and the possibility of lower reserves. That window lines up almost exactly with the volatile week found independently by the volatility scan, and with its price spikes on 01-25 and 01-28. This is a dated, first-party ERCOT document naming a real grid-stress event in the same window the anomaly detector flagged on its own. A second search for outlet coverage quantifying the actual $/MWh levels during this window came up empty beyond ERCOT's own advisory, worth noting so this doesn't get overstated as more independently confirmed than it actually is.

## Summary: what actually worked

| Experiment | Result |
|---|---|
| 1. Exogenous load data | Marginal improvement, doesn't flip the naive-wins result. Fuel-mix data confirmed unavailable historically. |
| 2. Direction classification | Genuine win on both weeks (+3.8 / +1.5 accuracy points), and the only task where the Skyblock features show balanced, substantial importance rather than a token contribution. |
| 3. Volatile-period forecasting | Made the model's relative underperformance worse instead of better. A real, counterintuitive negative finding. |
| 4. Deepened anomaly detection | Strongest result in the project: 16-23x separation on the volatile week vs. 4-7x on the calm week, and the flagged intervals correspond to a real, named, dated ERCOT Weather Watch event. |

Updated recommendation: lead with anomaly detection (Experiment 4) as the primary result, with direction classification (Experiment 2) as a solid secondary result. Point forecasting (Experiments 1 and 3) is reported honestly as not working, including the counterintuitive finding that more volatility makes the point-forecast gap bigger instead of smaller. That's a real, defensible, non-obvious finding in its own right, not a null result to bury.

## Setup friction encountered (this round)

- ERCOT's live/recent document list for real-time SPP (`get_spp`) only keeps a rolling 9-10 days or so, which I confirmed by testing it directly since the library doesn't document this up front.
- `get_fuel_mix()` genuinely has no historical access for ERCOT (today/yesterday only), confirmed both by calling it directly and by reading the library source. There's no workaround through `gridstatus` for this one.
- `get_rtm_spp(year)` (the annual archive) hit a real pandas 2.3 incompatibility bug in gridstatus 0.36.0's internal parsing (`astype("timedelta64[h]")`). Worked around by rebuilding the same interval-construction logic outside the buggy call instead of patching the installed library.
- `get_load()` only covers a rolling 14-day window for ERCOT. The separate `get_hourly_load_post_settlements()` archive was needed to get load data for the volatile week, which is about 5 months back.

## Is this project worth keeping for fall?

Yes, and this round strengthens the case rather than just re-confirming the prior finding. The project now has two genuinely different validated results (anomaly detection, direction classification) instead of one, plus two honestly-reported negative findings (exogenous load doesn't flip the point-forecast result, more volatility makes the point-forecast gap worse) that are informative rather than embarrassing. The anomaly-detection result is now checked against a real, dated, first-party ERCOT grid event, the strongest single piece of evidence in the project that the Skyblock-style feature engineering is doing genuine work on real grid data and not just describing a plausible-sounding analogy. It's still not as strong a lead as `battery_project`'s independently-verified R²=0.82, but it's a more substantial, multi-angle piece of work than either prior round, with an honest account of what didn't work sitting right next to what did.
