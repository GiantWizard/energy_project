"""
Energy-markets bridge project — round 3 rework.

Prior honest finding: on a calm week (2026-06-28 to 2026-07-04), a next-15-min
RandomForest forecast lost to a naive random-walk baseline, and the three
Skyblock-analogue features (momentum_4, volatility_8, spread_1) barely
registered (~8% combined feature importance vs lag_1 at 83%). The only
result that held up was IsolationForest anomaly detection using those
features alone.

This script tests four genuinely different angles to see if a better result
is actually achievable, not just re-argued:

  1. Exogenous data: add ERCOT system-wide load (demand level + demand
     momentum) as features alongside the price-based ones. (Fuel-mix /
     renewable-penetration was also attempted but is NOT available
     historically via gridstatus for ERCOT -- see fetch_load.py docstring
     and RESULTS.md for the confirmed limitation.)
  2. Reframe the forecasting target as next-interval price DIRECTION
     (up/down) instead of exact price, compared against a naive
     "same direction as last move" baseline.
  3. Pull a genuinely more volatile period. Using ERCOT's annual historical
     RTM archive (see fetch_historical_archive.py), scanned all of 2026
     year-to-date and found 2026-01-24 to 2026-01-30 as the most volatile
     week (max $1170.38/MWh vs the calm week's max of $66.13; std $105.57
     vs $7.87) -- a real winter cold-snap scarcity-pricing event.
  4. Deepen the anomaly-detection angle: check whether IsolationForest's
     flagged intervals on the volatile week correspond to the actual real
     price spike, and separately research what ERCOT event this period
     corresponds to.

Everything below is run for real, on both the calm week and the volatile
week, and reported honestly regardless of outcome.
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, precision_score

SKYBLOCK_FEATURES = ["momentum_4", "volatility_8", "spread_1"]
results_lines = []


def log(msg):
    print(msg)
    results_lines.append(msg)


def load_price(path):
    df = pd.read_csv(path, parse_dates=["Interval Start"])
    df = df.sort_values("Interval Start").reset_index(drop=True)
    df = df[["Interval Start", "SPP"]].rename(columns={"Interval Start": "ts", "SPP": "price"})
    return df


def load_demand(path, total_col):
    df = pd.read_csv(path, parse_dates=["Interval Start"])
    df = df[["Interval Start", total_col]].rename(columns={"Interval Start": "ts", total_col: "load"})
    return df.sort_values("ts").reset_index(drop=True)


def build_features(price_df, demand_df=None):
    df = price_df.copy()
    df["momentum_4"] = df["price"].pct_change(4)
    df["volatility_8"] = df["price"].rolling(8).std()
    df["spread_1"] = df["price"].diff().abs()
    df["lag_1"] = df["price"].shift(1)
    df["lag_2"] = df["price"].shift(2)
    df["lag_4"] = df["price"].shift(4)
    df["hour"] = df["ts"].dt.hour + df["ts"].dt.minute / 60.0

    if demand_df is not None:
        # load is hourly; resample onto the 15-min price grid via forward-fill
        # (each hourly load reading applies to the following interval until
        # the next hourly reading), then add demand level and demand momentum
        d = demand_df.set_index("ts").reindex(
            pd.date_range(demand_df["ts"].min(), df["ts"].max(), freq="15min", tz=df["ts"].dt.tz)
        ).ffill().rename_axis("ts").reset_index()
        df = pd.merge_asof(df.sort_values("ts"), d.sort_values("ts"), on="ts", direction="backward")
        df["load_change_4"] = df["load"].pct_change(4)

    return df


def run_forecast(df, feature_cols, horizon_steps, label):
    d = df.copy()
    d["target"] = d["price"].shift(-horizon_steps)
    model_df = d.dropna(subset=feature_cols + ["target"]).reset_index(drop=True)
    if len(model_df) < 50:
        log(f"[{label}] SKIPPED - insufficient rows ({len(model_df)})")
        return None

    split_idx = int(len(model_df) * 0.8)
    train, test = model_df.iloc[:split_idx], model_df.iloc[split_idx:]
    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    model = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    naive_preds = test["price"]
    naive_mae = mean_absolute_error(y_test, naive_preds)
    naive_rmse = mean_squared_error(y_test, naive_preds) ** 0.5

    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    skyblock_imp = importances.reindex(SKYBLOCK_FEATURES).dropna().sum()

    log(f"\n[{label}]")
    log(f"  n_test={len(test)}  model_mae={mae:.3f}  naive_mae={naive_mae:.3f}  beats_naive={mae < naive_mae}  (delta={100*(naive_mae-mae)/naive_mae:+.1f}%)")
    log(f"  model_rmse={rmse:.3f}  naive_rmse={naive_rmse:.3f}")
    log(f"  feature_importances: {importances.round(4).to_dict()}")
    log(f"  skyblock_combined_importance={skyblock_imp:.4f}")

    return {"mae": mae, "naive_mae": naive_mae, "rmse": rmse, "naive_rmse": naive_rmse,
            "beats_naive": mae < naive_mae, "importances": importances, "test": test, "preds": preds}


def run_direction_classification(df, feature_cols, label):
    d = df.copy()
    d["next_price"] = d["price"].shift(-1)
    d["direction"] = np.sign(d["next_price"] - d["price"])  # -1, 0, +1
    d["last_direction"] = np.sign(d["price"] - d["price"].shift(1))
    model_df = d.dropna(subset=feature_cols + ["direction", "last_direction"]).reset_index(drop=True)
    model_df = model_df[model_df["direction"] != 0]  # drop exact ties for a clean binary task
    if len(model_df) < 50:
        log(f"[{label}] SKIPPED - insufficient rows")
        return None

    split_idx = int(len(model_df) * 0.8)
    train, test = model_df.iloc[:split_idx], model_df.iloc[split_idx:]
    X_train, y_train = train[feature_cols], train["direction"]
    X_test, y_test = test[feature_cols], test["direction"]

    clf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=0)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    # naive baseline: predict the same direction as the last observed move
    naive_preds = test["last_direction"]
    naive_preds = naive_preds.replace(0, 1)  # break ties arbitrarily, same as any classifier would have to
    naive_acc = accuracy_score(y_test, naive_preds)

    importances = pd.Series(clf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    log(f"\n[{label}]")
    log(f"  n_test={len(test)}  model_acc={acc:.3f}  naive_acc={naive_acc:.3f}  beats_naive={acc > naive_acc}")
    log(f"  feature_importances: {importances.round(4).to_dict()}")

    return {"acc": acc, "naive_acc": naive_acc, "beats_naive": acc > naive_acc}


def run_anomaly_detection(df, label, contamination=0.05):
    anomaly_df = df.dropna(subset=SKYBLOCK_FEATURES).reset_index(drop=True)
    iso_forest = IsolationForest(n_estimators=300, contamination=contamination, random_state=0)
    anomaly_df["anomaly_flag"] = iso_forest.fit_predict(anomaly_df[SKYBLOCK_FEATURES])
    anomaly_df["anomaly_score"] = iso_forest.decision_function(anomaly_df[SKYBLOCK_FEATURES])

    flagged = anomaly_df[anomaly_df["anomaly_flag"] == -1].sort_values("anomaly_score")
    normal = anomaly_df[anomaly_df["anomaly_flag"] == 1]

    log(f"\n[Anomaly detection: {label}]")
    log(f"  flagged {len(flagged)} / {len(anomaly_df)} ({100*len(flagged)/len(anomaly_df):.1f}%)")
    log(f"  median spread_1   normal={normal['spread_1'].median():.3f}  flagged={flagged['spread_1'].median():.3f}")
    log(f"  median volatility_8 normal={normal['volatility_8'].median():.3f}  flagged={flagged['volatility_8'].median():.3f}")
    log(f"  price at single most extreme interval overall: {anomaly_df.loc[anomaly_df['price'].idxmax(), ['ts','price']].to_dict()}")
    log(f"  top 5 flagged intervals:")
    for _, row in flagged.head(5).iterrows():
        log(f"    ts={row['ts']} price={row['price']:.2f} spread_1={row['spread_1']:.2f} volatility_8={row['volatility_8']:.2f}")

    return anomaly_df, flagged


# ============================================================================
# Load data
# ============================================================================
calm_price = load_price("raw_hb_houston_rtm_spp.csv")
volatile_price = load_price("raw_hb_houston_volatile_week.csv")
calm_load = load_demand("raw_load_calm_week.csv", "Load")
volatile_load = load_demand("raw_load_volatile_week.csv", "ERCOT")

calm_feat = build_features(calm_price, calm_load)
volatile_feat = build_features(volatile_price, volatile_load)

price_only_cols = ["momentum_4", "volatility_8", "spread_1", "lag_1", "lag_2", "lag_4", "hour"]
price_plus_load_cols = price_only_cols + ["load", "load_change_4"]

log("=" * 70)
log("CALM WEEK STATS (2026-06-28 to 2026-07-04)")
log(f"  price: min={calm_price['price'].min():.2f} max={calm_price['price'].max():.2f} std={calm_price['price'].std():.2f}")
log("VOLATILE WEEK STATS (2026-01-24 to 2026-01-30, ERCOT winter scarcity event)")
log(f"  price: min={volatile_price['price'].min():.2f} max={volatile_price['price'].max():.2f} std={volatile_price['price'].std():.2f}")
log("=" * 70)

# ============================================================================
# Experiment 1: exogenous load data, price-only vs price+load, both weeks
# ============================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 1: exogenous load (demand) features")
log("=" * 70)

r_calm_t1_priceonly = run_forecast(calm_feat, price_only_cols, 1, "Calm week, t+1, price-only")
r_calm_t1_withload = run_forecast(calm_feat, price_plus_load_cols, 1, "Calm week, t+1, price+load")
r_vol_t1_priceonly = run_forecast(volatile_feat, price_only_cols, 1, "Volatile week, t+1, price-only")
r_vol_t1_withload = run_forecast(volatile_feat, price_plus_load_cols, 1, "Volatile week, t+1, price+load")

# ============================================================================
# Experiment 2: direction classification instead of point forecast
# ============================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 2: next-interval direction classification (up/down)")
log("=" * 70)

d_calm = run_direction_classification(calm_feat, price_only_cols, "Calm week, direction classification")
d_vol = run_direction_classification(volatile_feat, price_only_cols, "Volatile week, direction classification")

# ============================================================================
# Experiment 3: volatile week already loaded above; also re-run t+4 for it
# ============================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 3: volatile-week forecast at both horizons")
log("=" * 70)

r_vol_t4_priceonly = run_forecast(volatile_feat, price_only_cols, 4, "Volatile week, t+4 (1hr ahead), price-only")
r_vol_t4_withload = run_forecast(volatile_feat, price_plus_load_cols, 4, "Volatile week, t+4 (1hr ahead), price+load")

# ============================================================================
# Experiment 4: anomaly detection on both weeks, deepened
# ============================================================================
log("\n" + "=" * 70)
log("EXPERIMENT 4: anomaly detection, deepened")
log("=" * 70)

calm_anomaly_df, calm_flagged = run_anomaly_detection(calm_feat, "calm week")
vol_anomaly_df, vol_flagged = run_anomaly_detection(volatile_feat, "volatile week")

# ============================================================================
# Plots
# ============================================================================
fig, axes = plt.subplots(4, 1, figsize=(13, 17))

axes[0].plot(calm_price["ts"], calm_price["price"], label="Calm week price", color="steelblue")
axes[0].scatter(calm_flagged["ts"], calm_flagged["price"], color="red", s=20, zorder=5, label=f"Flagged (n={len(calm_flagged)})")
axes[0].set_title("Calm week (2026-06-28 to 07-04): price + anomaly flags")
axes[0].legend(fontsize=8)
axes[0].set_ylabel("$/MWh")

axes[1].plot(volatile_price["ts"], volatile_price["price"], label="Volatile week price", color="darkorange")
axes[1].scatter(vol_flagged["ts"], vol_flagged["price"], color="red", s=20, zorder=5, label=f"Flagged (n={len(vol_flagged)})")
axes[1].set_title("Volatile week (2026-01-24 to 01-30, winter scarcity event): price + anomaly flags")
axes[1].legend(fontsize=8)
axes[1].set_ylabel("$/MWh")

if r_vol_t1_priceonly is not None:
    t = r_vol_t1_priceonly["test"]
    axes[2].plot(t["ts"], t["target"] if "target" in t.columns else t["price"].shift(-1), label="Actual", alpha=0.5, linewidth=1)
    axes[2].plot(t["ts"], r_vol_t1_priceonly["preds"], label="Predicted (RF, t+1, price-only)", linewidth=1.2)
    axes[2].set_title(f"Volatile week t+1 forecast: model MAE {r_vol_t1_priceonly['mae']:.1f} vs naive {r_vol_t1_priceonly['naive_mae']:.1f}")
    axes[2].legend(fontsize=8)
    axes[2].set_ylabel("$/MWh")

if r_vol_t4_priceonly is not None:
    t = r_vol_t4_priceonly["test"]
    axes[3].plot(t["ts"], t["price"], label="Actual (current)", alpha=0.5, linewidth=1)
    axes[3].plot(t["ts"], r_vol_t4_priceonly["preds"], label="Predicted (RF, t+4=1hr, price-only)", linewidth=1.2)
    axes[3].set_title(f"Volatile week t+4 forecast: model MAE {r_vol_t4_priceonly['mae']:.1f} vs naive {r_vol_t4_priceonly['naive_mae']:.1f}")
    axes[3].legend(fontsize=8)
    axes[3].set_ylabel("$/MWh")
    axes[3].set_xlabel("Time")

plt.tight_layout()
plt.savefig("experiments_plot.png", dpi=120)
log("\nSaved experiments_plot.png")

with open("experiments_results.txt", "w") as f:
    f.write("\n".join(results_lines) + "\n")
log("Saved experiments_results.txt")
