# %% [markdown]
# # 03 - Feature Engineering: Multi-Seasonal Harmonics & Astronomical Dynamics
# **Grid Stress Forecaster — Predicting Renewable Energy Curtailment**
# 
# This notebook covers:
# 1. Astronomical calculation of **Solar Elevation Angle** from California geographical coordinates.
# 2. **Cyclical Calendar Transforms** (sin/cos of hour, day of week, day of year).
# 3. **Fourier Series Harmonic Decomposition** across daily (24h), weekly (168h), and annual (8760h) periodicity.
# 4. **Autoregressive Lags & Rolling Moving Averages** without lookahead leakage.
# 5. Feature ranking and correlation with renewable energy curtailment.

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
from sklearn.ensemble import RandomForestRegressor

from scripts.utils import setup_logger, load_config, ensure_dir
from scripts.preprocessing import GridDataPreprocessor
from scripts.feature_engineering import EnergyFeatureEngineer

config = load_config()
fig_dir = project_root / config["paths"]["figures_dir"]
ensure_dir(str(fig_dir))

# 1. Clean data with Preprocessor
raw_data_path = str(project_root / config["paths"]["eia_raw_file"])
clean_data_path = str(project_root / config["paths"]["processed_features_file"])

preprocessor = GridDataPreprocessor(config=config)
df_clean = preprocessor.process(input_path=raw_data_path, output_path=clean_data_path)
print(f"Cleaned dataset: {len(df_clean)} records")

# %% [markdown]
# ## 1. Feature Engineering Transformation
# Applying the feature engineer to create cyclical encodings, solar elevation, Fourier features, and autoregressive lags.

# %%
feature_engineer = EnergyFeatureEngineer(config=config)
df_features = feature_engineer.build_features(df_clean, drop_na=True)
print(f"Engineered Dataset: {df_features.shape[0]} rows x {df_features.shape[1]} columns")
print("\nSample Engineered Columns:")
print(list(df_features.columns[:25]))

# %% [markdown]
# ## 2. Solar Elevation Angle vs. Solar Photovoltaic Dispatch
# Examining how well astronomical calculations mirror actual photovoltaic telemetry.

# %%
sample_slice = df_features.iloc[500:572]  # 72 hours (3 days)

fig, ax1 = plt.subplots(figsize=(14, 5), dpi=200)
ax2 = ax1.twinx()

ax1.plot(sample_slice["timestamp"], sample_slice["solar_elevation_deg"], color="#e67e22", linewidth=2.2, label="Calculated Solar Elevation Angle (°)")
ax2.plot(sample_slice["timestamp"], sample_slice["solar_generation_mw"], color="#f1c40f", linewidth=2.0, linestyle="--", label="Actual Solar Generation (MW)")

ax1.set_ylabel("Solar Elevation Angle (° Above Horizon)", color="#e67e22", fontsize=11)
ax2.set_ylabel("Solar Photovoltaic Generation (MW)", color="#f39c12", fontsize=11)
ax1.set_title("Astronomical Solar Elevation Angle vs. Observed PV Generation (3-Day Sequence)", fontsize=13, fontweight="bold")
ax1.grid(True, linestyle=":", alpha=0.6)

# Align legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

save_path = str(fig_dir / "solar_elevation_validation.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# ## 3. Multi-Seasonal Fourier Harmonic Decomposition
# Visualizing how Fourier sine/cosine pairs reconstruct 24-hour daily and 168-hour weekly cyclical power dynamics.

# %%
fig, (ax_d, ax_w) = plt.subplots(2, 1, figsize=(14, 7), sharex=False, dpi=200)

sample_fourier = df_features.iloc[100:100 + 168]  # 1 week

# Daily harmonics
ax_d.plot(sample_fourier["timestamp"], sample_fourier["fourier_d1_sin"], label="Daily Harmonic 1 (sin)", color="#3498db")
ax_d.plot(sample_fourier["timestamp"], sample_fourier["fourier_d2_sin"], label="Daily Harmonic 2 (sin)", color="#2ecc71", linestyle="--")
ax_d.set_title("Daily Fourier Harmonics (24-Hour Periodicity)", fontsize=12, fontweight="bold")
ax_d.set_ylabel("Harmonic Amplitude")
ax_d.legend(loc="upper right")
ax_d.grid(True, linestyle=":")

# Weekly harmonics
ax_w.plot(sample_fourier["timestamp"], sample_fourier["fourier_w1_sin"], label="Weekly Fundamental (168-Hour)", color="#9b59b6")
ax_w.plot(sample_fourier["timestamp"], sample_fourier["fourier_w2_sin"], label="Weekly Harmonic 2", color="#e74c3c", linestyle="--")
ax_w.set_title("Weekly Fourier Harmonics (168-Hour Periodicity)", fontsize=12, fontweight="bold")
ax_w.set_ylabel("Harmonic Amplitude")
ax_w.legend(loc="upper right")
ax_w.grid(True, linestyle=":")

plt.tight_layout()
save_path = str(fig_dir / "fourier_decomposition.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# ## 4. Feature Importance Analysis for Curtailment Prediction
# Fitting a Random Forest regressor to measure relative Gini feature importances.

# %%
feature_subset = [
    "net_load_mw", "solar_generation_mw", "wind_generation_mw", "system_demand_mw",
    "solar_elevation_deg", "temperature_c", "wind_speed_mps", "ghi_estimated",
    "renewable_penetration_ratio", "hour_sin", "hour_cos", "month_sin",
    "net_load_lag_1h", "net_load_lag_24h", "solar_generation_lag_24h",
    "net_load_roll_24h_mean", "fourier_d1_sin", "fourier_d1_cos"
]

X = df_features[feature_subset]
y = df_features["curtailment_mw"]

rf = RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X, y)

importances = pd.Series(rf.feature_importances_, index=feature_subset).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6), dpi=200)
importances.plot(kind="barh", color="#2c3e50", ax=ax)
ax.set_title("Random Forest Gini Feature Importance: Predicting Curtailment (MW)", fontsize=13, fontweight="bold")
ax.set_xlabel("Relative Feature Importance Score", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
save_path = str(fig_dir / "feature_importance.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# **Conclusion**:
# `net_load_mw`, `renewable_penetration_ratio`, and `solar_elevation_deg` dominate the predictive signal, confirming that physical grid inertia and solar geometry drive renewable curtailment.
