# %% [markdown]
# # 04 - Bayesian Additive Time Series Modeling with Facebook Prophet
# **Grid Stress Forecaster — Predicting Renewable Energy Curtailment**
# 
# This notebook covers:
# 1. Setting up **Facebook Prophet** with custom high-order hourly seasonality.
# 2. Incorporating **US Statutory Holidays** and grid stress period effects.
# 3. Injecting exogenous physical regressors: `net_load_mw`, `solar_elevation_deg`, `temperature_c`, and `wind_speed_mps`.
# 4. Performing cross-validation across rolling historical test cutoffs.
# 5. Visualizing 48-hour point forecasts, uncertainty bounds, and component decompositions.

# %%
import os
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import setup_logger, load_config, ensure_dir, calculate_regression_metrics
from scripts.prophet_model import EnergyProphetForecaster

config = load_config()
fig_dir = project_root / config["paths"]["figures_dir"]
ensure_dir(str(fig_dir))

# Load preprocessed clean features
clean_data_path = str(project_root / config["paths"]["processed_features_file"])
df = pd.read_parquet(clean_data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Train / Test split: Hold out the last 48 hours for test evaluation
test_hours = 48
train_df = df.iloc[:-test_hours].copy()
test_df = df.iloc[-test_hours:].copy()

print(f"Dataset Split -> Training Observations: {len(train_df)} | Holdout Evaluation: {len(test_df)}")

# %% [markdown]
# ## 1. Model Initialization & Training
# Initializing `EnergyProphetForecaster` with configured seasonal priors and exogenous regressors.

# %%
forecaster = EnergyProphetForecaster(config=config)
print("Configured Exogenous Regressors:", forecaster.exogenous_regressors)

print("Fitting Prophet model on training historical data...")
forecaster.train(train_df)
print("Training completed.")

# %% [markdown]
# ## 2. 48-Hour Ahead Out-of-Sample Forecasting
# Generating predictions for the 48-hour test horizon with 90% confidence intervals.

# %%
forecast_df = forecaster.predict(test_df)
print("Sample Forecast Output (First 5 hours):")
forecast_df.head()

# %% [markdown]
# ## 3. Performance Metric Evaluation
# Comparing actual curtailment observations with Prophet predictions.

# %%
y_actual = test_df["curtailment_mw"].values
y_pred = forecast_df["yhat"].values

metrics = calculate_regression_metrics(y_actual, y_pred)
print("\n--- Prophet 48-Hour Horizon Performance ---")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 4. Visualization: Actual vs. Prophet Forecast with Uncertainty Band

# %%
fig, ax = plt.subplots(figsize=(14, 6), dpi=200)

timestamps = test_df["timestamp"]
ax.plot(timestamps, y_actual, color="#2c3e50", marker="o", markersize=4, linestyle=":", label="Observed Curtailment (MW)")
ax.plot(timestamps, y_pred, color="#e67e22", linewidth=2.2, label="Prophet Forecast (MW)")

ax.fill_between(
    timestamps,
    forecast_df["yhat_lower"].values,
    forecast_df["yhat_upper"].values,
    color="#f39c12",
    alpha=0.25,
    label="90% Confidence Interval"
)

ax.axhline(250.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Alert Threshold (250 MW)")

ax.set_title("Prophet 48-Hour Renewable Curtailment Forecast vs. Actual", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Timestamp (UTC)", fontsize=11)
ax.set_ylabel("Renewable Curtailment (MW)", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

save_path = str(fig_dir / "prophet_forecast_eval.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# ## 5. Rolling Cross-Validation Diagnostics
# Evaluating historical error distributions across multiple temporal horizons.

# %%
cv_metrics = forecaster.evaluate_cross_validation(train_df.iloc[-4000:])
print("\nProphet Cross-Validation Summary:")
for k, v in cv_metrics.items():
    print(f"  {k}: {v}")
