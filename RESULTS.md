# Results: Energy Markets Bridge

## Revision history

1. Weekend one: built a next-15-min RandomForest price forecast on a calm week (2026-06-28 to 07-04). It lost to a naive random-walk baseline (MAE 2.232 vs 2.002), and the three Skyblock-analogue features contributed only about 8% combined feature importance vs. `lag_1` alone at 83%.
2. First rework: instead of rewording the summary, tested a longer forecast horizon (a modest win, driven mostly by `hour`) and anomaly detection (a clean win: flagged intervals showed 4-7x higher spread/volatility than normal ones).
3. This rework: tried four genuinely different angles to see if a better result is achievable at all: exogenous demand data, direction/regime classification instead of point forecasting, a real high-volatility period instead of just a different calm week, and a deeper check of the anomaly-detection angle against a real, named ERCOT grid event. Two of the four produce genuine, if modest, improvements. Two do not, and are reported honestly below.

## Data

- ISO / node: ERCOT, `HB_HOUSTON` (Houston Load Zone Trading Hub), real-time market, 15-minute settlement point prices (SPP).
- Calm week: 2026-06-28 to 2026-07-04, 672 rows, zero gaps. Min $8.89, max $66.13, mean $27.18, std $7.87.
- Volatile week (new this round): 2026-01-24 to 2026-01-30, 672 rows, zero gaps. Min -$6.66, max $1170.38, mean $106.03, std $105.57, about 13x the calm week's volatility. This is a real winter cold-snap scarcity-pricing event (see Experiment 4).
- Demand data (new this round): ERCOT system-wide hourly load, pulled for both weeks. Calm week via `gridstatus.get_load`; volatile week via `gridstatus.get_hourly_load_post_settlements`, since the live/recent-only `get_load` only supports a rolling 14-day window and the volatile week is over 5 months in the past. See Setup Friction below.

### How the volatile week was found

`gridstatus.Ercot.get_spp()`, the method used for the original calm-week pull, draws from ERCOT's live rolling document list, which turned out to only retain about 9-10 days of real-time SPP documents. Confirmed empirically: every date requested before 2026-06-27 failed with "no documents found," while 2026-06-27 onward succeeded. This meant scanning the last month or two for a volatile day couldn't work as originally planned through that method.

Found a different path instead: ERCOT publishes an annual historical RTM archive (`gridstatus.Ercot.get_rtm_spp(year)`, report type 13061) covering the full year in one file. This hit a real bug under pandas 2.3.x: gridstatus's internal `parse_doc` calls `.astype("timedelta64[h]")`, which pandas 2.3 no longer permits (only s/ms/us/ns units are allowed). Rather than patch the installed library, `fetch_historical_archive.py` downloads the same underlying file and reimplements the same interval-construction logic with `pd.to_timedelta(..., unit="h")`, which is pandas-2.3-compatible. This produced the full 2026-01-01 to 2026-07-04 year-to-date archive (408,388 rows across all settlement points), scanned day by day for `HB_HOUSTON` to find the highest-volatility week.

## Feature engineering

| Feature | Definition | Skyblock analogue |
|---|---|---|
| `momentum_4` | % change in price over the last 4 intervals (1 hour) | Short-horizon buy/sell price momentum |
| `volatility_8` | Rolling std-dev of price over the last 8 intervals (2 hours) | Rolling volatility / choppiness feature |
| `spread_1` | Absolute difference between consecutive interval prices | Bid/ask spread proxy |
| `load`, `load_change_4` (new) | System-wide demand level and its 1-hour % change | Not a Skyblock analogue. A genuinely new, grid-specific exogenous signal; order books don't have a "system demand" equivalent |

## Experiment 1: exogenous demand (load) data

Marginal help, doesn't flip the result.

Hypothesis: unlike re-tuning price-only features, adding real demand data is a legitimate reason to expect improvement, since price is partly a function of demand.

Fuel-mix / renewable-penetration data, as originally proposed, turned out not to be accessible historically at all. `gridstatus.Ercot.get_fuel_mix()` only supports `"today"`/`"yesterday"` for ERCOT (confirmed by direct call, raises `NotSupported()` for any other date), and there is no separate historical fuel-mix archive report the way there is for load and price. This is a real, confirmed limitation, not worked around. Load (demand level) was used instead, since it's available historically through a separate archive (`get_hourly_load_post_settlements`).

| | Calm week t+1 MAE | Volatile week t+1 MAE |
|---|---|---|
| Naive baseline | 2.002 | 4.443 |
| Price-only model | 2.232 (-11.5%) | 7.755 (-74.6%) |
| Price + load model | 2.155 (-7.7%) | 8.122 (-82.8%) |

Adding load features narrowed the gap slightly on the calm week (model MAE improved from 2.232 to 2.155) but still didn't beat naive, and actively made things worse on the volatile week. `load` itself picked up modest feature importance (3.1% on the calm week) but didn't change the fundamental picture: `lag_1` still dominates. This does not flip the original finding. A likely reason: hourly load resampled onto a 15-min grid is a coarse, slowly-changing signal relative to how fast settlement prices move, so it adds limited marginal information on top of the price series' own recent history.

## Experiment 2: direction classification

A genuine, repeatable win.

Reframed the task: instead of predicting the exact next price, predict just the direction of the next move (up or down), compared against a naive "same direction as the last observed move" baseline. This is a fair, non-trivial baseline, not majority-class guessing.

| | Calm week accuracy | Volatile week accuracy |
|---|---|---|
| Naive (same as last direction) | 0.586 | 0.564 |
| RandomForest classifier | 0.624 | 0.579 |
| Beats naive? | Yes (+3.8 pts) | Yes (+1.5 pts) |

This is a real, if modest, win on both weeks. More importantly, the feature importances look completely different from the point-forecasting task: on the calm week, `hour` (19.2%), `momentum_4` (17.6%), and `lag_4` (15.2%) lead, with all seven features landing in a narrow 11.7%-19.2% band. No single feature dominates the way `lag_1` did at 83% in the point-forecast task. The Skyblock-style features (`momentum_4`, `volatility_8`, `spread_1`) are doing real, substantive work here, not just registering a token contribution. This is the clearest evidence in the whole project that the features are useful for something, just not for point-forecasting the exact price.

## Experiment 3: forecasting on the volatile week

Makes the model's relative loss worse, not better.

Hypothesis, as given in the task: a genuinely more volatile period should give the model more real signal to find, potentially closing the gap with naive.

| | t+1 MAE | t+4 (1hr) MAE |
|---|---|---|
| Naive | 4.443 | 9.321 |
| Model (price-only) | 7.755 (-74.6%) | 15.929 (-70.9%) |

This did not work. If anything, the opposite happened. On the calm week the model lost to naive by 11.5% (t+1); on the volatile week it lost by 74.6%. During sharp, scarcity-driven spikes, the single most predictive thing is still "what did the price just do" (naive), and a smoothing ensemble model actually falls further behind because it under-reacts to the spike's speed and magnitude, visible directly in `experiments_plot.png`'s third panel, where the model's t+1 predictions lag the real spikes by a step. Higher raw volatility gave the model more error to make, not more structure to exploit relative to the naive baseline. This is a legitimate, somewhat counterintuitive finding worth keeping, not something to force into a positive result.

## Experiment 4: anomaly detection, deepened

The strongest result in the project, now with a real-world check.

Ran the same IsolationForest (momentum/volatility/spread only, no lag prices) on the volatile week.

| | Calm week | Volatile week |
|---|---|---|
| Flagged | 34/665 (5.1%) | 34/665 (5.1%) |
| Median `spread_1`, normal vs. flagged | 0.74 vs. 5.57 (7.5x) | 3.68 vs. 61.07 (16.6x) |
| Median `volatility_8`, normal vs. flagged | 1.74 vs. 7.38 (4.2x) | 8.44 vs. 194.29 (23x) |

The separation is far stronger on the volatile week, which is exactly what you'd want from a working anomaly detector: bigger real anomalies produce a bigger, clearer signal. The single most extreme flagged interval is 2026-01-28 07:00 CST at $1170.38/MWh (the year-to-date price peak), and the top 5 flagged intervals are dominated by the two real spike clusters on 2026-01-25 (evening) and 2026-01-28 (morning). See `experiments_plot.png`, second panel, where the flags sit almost exactly on the two visible price spikes.

Real-world correspondence check, done through a live web search rather than assumed: ERCOT's own press release from 2026-01-21, "ERCOT Issues Weather Watch" (ercot.com/news/release/01212026-ercot-issues-weather), announces a Weather Watch for 2026-01-24 through 2026-01-27: below-freezing temperatures with the possibility of frozen precipitation, higher electrical demand, and the potential for lower reserves. That window lines up almost exactly with the independently-discovered volatile week and its price spikes on 01-25 and 01-28. This is a dated, first-party ERCOT document naming a real grid-stress event in the same window the anomaly detector flagged, not a coincidence noticed after the fact. A second search for outlet coverage quantifying the actual $/MWh levels during this window came up empty beyond ERCOT's own advisory, worth noting so this isn't overstated as more independently corroborated than it is.

## Summary: what actually worked

| Experiment | Result |
|---|---|
| 1. Exogenous load data | Marginal improvement, doesn't flip the naive-wins result. Fuel-mix data confirmed unavailable historically. |
| 2. Direction classification | Genuine win on both weeks (+3.8 / +1.5 accuracy points), and the only task where the Skyblock features show balanced, substantial importance rather than a token contribution. |
| 3. Volatile-period forecasting | Made the model's relative underperformance worse, not better. A real, counterintuitive negative finding. |
| 4. Deepened anomaly detection | Strongest result in the project: 16-23x separation on the volatile week vs. 4-7x on the calm week, and the flagged intervals correspond to a real, named, dated ERCOT Weather Watch event. |

Updated recommendation: lead with anomaly detection (Experiment 4) as the primary result, with direction classification (Experiment 2) as a solid secondary result. Point forecasting (Experiments 1 and 3) is reported honestly as not working, including the counterintuitive finding that more volatility makes the point-forecast gap worse, not better. That's a real, defensible, non-obvious empirical finding in its own right, not a null result to bury.

## Setup friction encountered (this round)

- ERCOT's live/recent document list for real-time SPP (`get_spp`) only retains a rolling 9-10 days or so, confirmed empirically and not documented up front by the library.
- `get_fuel_mix()` genuinely has no historical access for ERCOT (today/yesterday only), confirmed by direct call and by reading the library source. No workaround exists via `gridstatus` for this one.
- `get_rtm_spp(year)` (the annual archive) hit a real pandas 2.3 incompatibility bug in gridstatus 0.36.0's internal parsing (`astype("timedelta64[h]")`). Worked around by reimplementing the equivalent interval-construction logic outside the buggy call rather than patching the installed library.
- `get_load()` only supports a rolling 14-day window for ERCOT. The separate `get_hourly_load_post_settlements()` archive was needed to get load data for the volatile week, which is about 5 months back.

## Is this project worth keeping for fall?

Yes, and this round strengthens the case rather than just re-confirming the prior finding. The project now has two genuinely different validated results (anomaly detection, direction classification) rather than one, plus two honestly-reported negative findings (exogenous load doesn't flip the point-forecast result, more volatility makes the point-forecast gap worse) that are informative rather than embarrassing. The anomaly-detection result is now checked against a real, dated, first-party ERCOT grid event, the strongest single piece of evidence in the project that the Skyblock-style feature engineering is doing genuine work on real grid data, not just describing a plausible-sounding analogy. It's still not as strong a lead as `battery_project`'s independently-verified R²=0.82, but it's a more substantial, multi-angle piece of work than either prior round, with an honest account of what didn't work sitting right next to what did.
