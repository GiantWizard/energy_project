"""
Energy-markets bridge project, weekend one (reworked after verification pass).

Original weekend-one result: a next-15-min-interval RandomForest forecast did
NOT beat a naive random-walk baseline (MAE 2.232 vs 2.002), and the three
Skyblock-analogue features contributed ~8% combined feature importance vs.
lag_1 alone at 83%. Rather than reframe that away, this script actually tests
two different angles the verification pass suggested, to see if either one
gives the engineered features real work to do:

  1. Longer-horizon forecast (1-hour-ahead, i.e. 4 intervals out) --
     does the naive lag-1 baseline stay dominant when the target is further
     from the most recent observation?
  2. Anomaly/spike detection -- do volatility_8 / spread_1 carry signal for
     "is this an unusual interval," a different question than point
     forecasting, using an unsupervised IsolationForest on the engineered
     features (not lag prices, so lag_1 can't just dominate by definition)?

All three experiments (original next-interval forecast, 1-hour-ahead forecast,
anomaly detection) are run and reported. Whichever actually shows the
Skyblock-style features doing real work is the one RESULTS.md leads with; if
none do, RESULTS.md says so plainly.

Feature <-> Skyblock-bazaar mapping:
  - momentum_4  : % change over the last 4 intervals (1 hour)
                  ~ Skyblock's short-horizon buy/sell price momentum signal.
  - volatility_8: rolling std-dev of price over the last 8 intervals (2 hours)
                  ~ Skyblock's rolling volatility feature (choppy/risky price).
  - spread_1    : abs difference between consecutive interval prices
                  ~ Skyblock's bid/ask spread proxy (order-to-order gap).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

RAW_PATH = "raw_hb_houston_rtm_spp.csv"

df = pd.read_csv(RAW_PATH, parse_dates=["Interval Start"])
df = df.sort_values("Interval Start").reset_index(drop=True)
df = df[["Interval Start", "SPP"]].rename(columns={"Interval Start": "ts", "SPP": "price"})

# ---- Feature engineering (shared across all three experiments) --------
df["momentum_4"] = df["price"].pct_change(4)
df["volatility_8"] = df["price"].rolling(8).std()
df["spread_1"] = df["price"].diff().abs()
df["lag_1"] = df["price"].shift(1)
df["lag_2"] = df["price"].shift(2)
df["lag_4"] = df["price"].shift(4)
df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60.0

feature_cols = ["momentum_4", "volatility_8", "spread_1", "lag_1", "lag_2", "lag_4", "hour"]
skyblock_features = ["momentum_4", "volatility_8", "spread_1"]

results_lines = []


def run_forecast(horizon_steps, label):
    """Train/test a RandomForest forecasting `horizon_steps` intervals ahead."""
    d = df.copy()
    d["target"] = d["price"].shift(-horizon_steps)
    model_df = d.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)

    split_idx = int(len(model_df) * 0.8)
    train, test = model_df.iloc[:split_idx], model_df.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5

    # naive baseline: predict target = current price (random walk, held flat over the horizon)
    naive_preds = test["price"]
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = mean_squared_error(y_test, naive_preds) ** 0.5

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    skyblock_importance = importances[skyblock_features].sum()

    print(f"\n=== {label} ===")
    print(f"Test set: {test['ts'].min()} to {test['ts'].max()} ({len(test)} intervals)")
    print(f"Model  MAE: {mae:.3f}  RMSE: {rmse:.3f}")
    print(f"Naive  MAE: {naive_mae:.3f}  RMSE: {naive_rmse:.3f}")
    print(f"Model beats naive: {mae < naive_mae}")
    print("Feature importances:")
    print(importances)
    print(f"Skyblock-feature combined importance: {skyblock_importance:.4f}")

    results_lines.append(f"\n[{label}]")
    results_lines.append(f"test_start={test['ts'].min()}")
    results_lines.append(f"test_end={test['ts'].max()}")
    results_lines.append(f"n_test={len(test)}")
    results_lines.append(f"model_mae={mae:.4f}")
    results_lines.append(f"model_rmse={rmse:.4f}")
    results_lines.append(f"naive_mae={naive_mae:.4f}")
    results_lines.append(f"naive_rmse={naive_rmse:.4f}")
    results_lines.append(f"model_beats_naive={mae < naive_mae}")
    results_lines.append(f"skyblock_feature_importance_combined={skyblock_importance:.4f}")
    results_lines.append("feature_importances:")
    for k, v in importances.items():
        results_lines.append(f"  {k}={v:.4f}")

    return test, preds, naive_preds, importances, mae, naive_mae


# Experiment 1: original next-15-min-interval forecast (baseline for comparison)
test_1, preds_1, naive_1, imp_1, mae_1, naive_mae_1 = run_forecast(1, "Forecast: next-15-min interval (t+1)")

# Experiment 2: 1-hour-ahead forecast (4 intervals out)
test_4, preds_4, naive_4, imp_4, mae_4, naive_mae_4 = run_forecast(4, "Forecast: 1-hour-ahead (t+4)")

# ---- Experiment 3: anomaly / spike detection -----------------------------
# Question: do the Skyblock-style features (momentum, volatility, spread) flag
# genuinely unusual intervals, using only those features (not lag prices,
# so lag_1 can't trivially dominate as it does in the forecasting task)?
anomaly_features = ["momentum_4", "volatility_8", "spread_1"]
anomaly_df = df.dropna(subset=anomaly_features).reset_index(drop=True)

iso_forest = IsolationForest(n_estimators=300, contamination=0.05, random_state=0)
anomaly_df["anomaly_score"] = iso_forest.fit_predict(anomaly_df[anomaly_features])
anomaly_df["anomaly_score_raw"] = iso_forest.decision_function(anomaly_df[anomaly_features])

flagged = anomaly_df[anomaly_df["anomaly_score"] == -1].sort_values("anomaly_score_raw")
n_flagged = len(flagged)

print(f"\n=== Anomaly detection (IsolationForest on {anomaly_features}) ===")
print(f"Flagged {n_flagged} / {len(anomaly_df)} intervals ({100*n_flagged/len(anomaly_df):.1f}%) as anomalous")
print("\nMost anomalous intervals:")
print(flagged[["ts", "price", "momentum_4", "volatility_8", "spread_1", "anomaly_score_raw"]].head(10).to_string(index=False))

# sanity check: do flagged intervals actually correspond to unusually large price moves?
normal_spread = anomaly_df.loc[anomaly_df["anomaly_score"] == 1, "spread_1"]
flagged_spread = flagged["spread_1"]
print(f"\nMedian spread_1, normal intervals: {normal_spread.median():.3f}")
print(f"Median spread_1, flagged intervals: {flagged_spread.median():.3f}")
print(f"Median volatility_8, normal intervals: {anomaly_df.loc[anomaly_df['anomaly_score']==1,'volatility_8'].median():.3f}")
print(f"Median volatility_8, flagged intervals: {flagged['volatility_8'].median():.3f}")

results_lines.append("\n[Anomaly detection: IsolationForest on momentum_4, volatility_8, spread_1]")
results_lines.append(f"n_total={len(anomaly_df)}")
results_lines.append(f"n_flagged={n_flagged}")
results_lines.append(f"pct_flagged={100*n_flagged/len(anomaly_df):.2f}")
results_lines.append(f"median_spread_1_normal={normal_spread.median():.4f}")
results_lines.append(f"median_spread_1_flagged={flagged_spread.median():.4f}")
results_lines.append(f"median_volatility_8_normal={anomaly_df.loc[anomaly_df['anomaly_score']==1,'volatility_8'].median():.4f}")
results_lines.append(f"median_volatility_8_flagged={flagged['volatility_8'].median():.4f}")
results_lines.append("top_10_flagged_intervals:")
for _, row in flagged.head(10).iterrows():
    results_lines.append(
        f"  ts={row['ts']} price={row['price']:.2f} momentum_4={row['momentum_4']:.4f} "
        f"volatility_8={row['volatility_8']:.3f} spread_1={row['spread_1']:.3f} "
        f"score={row['anomaly_score_raw']:.4f}"
    )

# ---- Plots ----------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(12, 13))

axes[0].plot(test_1["ts"], test_1["price"].shift(-1).ffill(), label="Actual", linewidth=1.0, alpha=0.4)
axes[0].plot(test_1["ts"], preds_1, label="Predicted (RF, t+1)", linewidth=1.2)
axes[0].plot(test_1["ts"], naive_1, label="Naive (t-1)", linewidth=0.8, linestyle="--", alpha=0.6)
axes[0].set_title(f"t+1 forecast: model MAE {mae_1:.2f} vs naive MAE {naive_mae_1:.2f}")
axes[0].set_ylabel("$/MWh")
axes[0].legend(fontsize=8)

axes[1].plot(test_4["ts"], test_4["price"], label="Actual (current)", linewidth=1.0, alpha=0.4)
axes[1].plot(test_4["ts"], preds_4, label="Predicted (RF, t+4 = 1hr ahead)", linewidth=1.2)
axes[1].plot(test_4["ts"], naive_4, label="Naive (t-1)", linewidth=0.8, linestyle="--", alpha=0.6)
axes[1].set_title(f"t+4 (1hr-ahead) forecast: model MAE {mae_4:.2f} vs naive MAE {naive_mae_4:.2f}")
axes[1].set_ylabel("$/MWh")
axes[1].legend(fontsize=8)

axes[2].plot(anomaly_df["ts"], anomaly_df["price"], label="Price", linewidth=1.0, color="steelblue")
axes[2].scatter(flagged["ts"], flagged["price"], color="red", s=20, zorder=5, label=f"Flagged anomalies (n={n_flagged})")
axes[2].set_title("IsolationForest anomaly flags on momentum/volatility/spread features")
axes[2].set_ylabel("$/MWh")
axes[2].set_xlabel("Time")
axes[2].legend(fontsize=8)

plt.tight_layout()
plt.savefig("forecast_plot.png", dpi=120)
print("\nSaved combined plot to forecast_plot.png")

with open("model_metrics.txt", "w") as f:
    f.write("\n".join(results_lines) + "\n")
print("Saved model_metrics.txt")
