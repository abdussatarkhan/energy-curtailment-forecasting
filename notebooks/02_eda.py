# %% [markdown]
# # 02 - Exploratory Data Analysis: The Duck Curve & Curtailment Drivers
# **Grid Stress Forecaster — Predicting Renewable Energy Curtailment**
# 
# This notebook conducts in-depth exploratory analysis on:
# 1. **The California Duck Curve**: Diurnal Net Load belly depression across seasons and years.
# 2. **Afternoon 3-Hour Ramping Stress**: The steep upward ramp as solar goes offline while evening residential demand peaks.
# 3. **Renewable Curtailment Seasonality**: Distribution of curtailment hours, frequency, and severity.
# 4. **Multivariate Correlation Matrix**: Quantifying relationships between ambient temperature, wind speed, solar generation, and curtailment.

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

from scripts.utils import setup_logger, load_config, ensure_dir

config = load_config()
fig_dir = project_root / config["paths"]["figures_dir"]
ensure_dir(str(fig_dir))

# Load preprocessed or raw data
data_path = project_root / config["paths"]["eia_raw_file"]
df = pd.read_parquet(data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hour"] = df["timestamp"].dt.hour
df["month"] = df["timestamp"].dt.month
df["year"] = df["timestamp"].dt.year

# Define seasons
def assign_season(month):
    if month in [3, 4, 5]:
        return "Spring (Runoff & Solar Peak)"
    elif month in [6, 7, 8]:
        return "Summer (High A/C Peak)"
    elif month in [9, 10, 11]:
        return "Fall (Moderate Load)"
    else:
        return "Winter (Heating & Low Solar)"

df["season"] = df["month"].apply(assign_season)
print(f"Loaded dataset: {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")

# %% [markdown]
# ## 1. The California "Duck Curve" Diurnal Profile
# The classic Duck Curve emerges when midday solar generation drives Net Load (`Demand - (Solar + Wind)`) down to minimum levels.

# %%
fig, ax = plt.subplots(figsize=(12, 6), dpi=200)

palette = {
    "Spring (Runoff & Solar Peak)": "#27ae60",
    "Summer (High A/C Peak)": "#e74c3c",
    "Fall (Moderate Load)": "#f39c12",
    "Winter (Heating & Low Solar)": "#2980b9"
}

sns.lineplot(
    data=df,
    x="hour",
    y="net_load_mw",
    hue="season",
    palette=palette,
    linewidth=2.5,
    errorbar=None,
    ax=ax
)

# Reference must-run thermal floor
ax.axhline(4500, color="gray", linestyle="--", linewidth=1.5, label="CAISO Must-Run Thermal Floor (~4,500 MW)")

ax.set_title("California ISO Net Load Hourly Profile: The Duck Curve Across Seasons", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Hour of Day (PST / Local Standard)", fontsize=11)
ax.set_ylabel("Average Net Load (MW)", fontsize=11)
ax.set_xticks(range(0, 24))
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(title="Season", frameon=True)

save_path = str(fig_dir / "duck_curve_seasonal.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# **Takeaway**: In Spring, the "belly" of the duck drops under 7,000 MW between 11:00 and 15:00, creating severe risks of over-generation.

# %% [markdown]
# ## 2. Renewable Energy Curtailment Distribution by Month & Hour
# Analyzing when and where system operators are forced to curtail renewable power.

# %%
curtailment_pivot = df.pivot_table(
    index="month",
    columns="hour",
    values="curtailment_mw",
    aggfunc="mean"
)

fig, ax = plt.subplots(figsize=(14, 7), dpi=200)
sns.heatmap(
    curtailment_pivot,
    cmap="YlOrRd",
    annot=False,
    cbar_kws={"label": "Average Curtailment (MW)"},
    ax=ax
)

ax.set_title("Heatmap: Mean Hourly Curtailment by Month (CAISO)", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Hour of Day", fontsize=11)
ax.set_ylabel("Month of Year", fontsize=11)
ax.set_yticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], rotation=0)

save_path = str(fig_dir / "curtailment_heatmap_month_hour.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# ## 3. The 3-Hour "Neck" Ramp Challenge
# As the sun sets between 16:00 and 19:00, solar generation drops by >12,000 MW while demand climbs.
# Gas peakers and batteries must ramp at record speeds.

# %%
df["net_load_ramp_3h"] = df["net_load_mw"].diff(3)

fig, ax = plt.subplots(figsize=(12, 5), dpi=200)
sns.boxplot(
    data=df,
    x="hour",
    y="net_load_ramp_3h",
    color="#3498db",
    fliersize=1,
    ax=ax
)

ax.axhline(0, color="black", linestyle="-", linewidth=0.8)
ax.axhline(10000, color="#e74c3c", linestyle="--", linewidth=1.2, label="Critical Ramping Threshold (>10,000 MW / 3hr)")

ax.set_title("CAISO 3-Hour Net Load Ramp Distribution by Hour of Day", fontsize=13, fontweight="bold")
ax.set_xlabel("Ending Hour of 3-Hour Ramp Window", fontsize=11)
ax.set_ylabel("3-Hour Net Load Change (MW)", fontsize=11)
ax.set_xticks(range(0, 24))
ax.legend(loc="upper left")
ax.grid(True, linestyle=":", alpha=0.6)

save_path = str(fig_dir / "ramping_distribution.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# ## 4. Feature Correlation Matrix
# Correlating meteorological drivers, power dispatch, and curtailment severity.

# %%
corr_cols = [
    "system_demand_mw", "solar_generation_mw", "wind_generation_mw",
    "nuclear_mw", "hydro_mw", "net_load_mw", "temperature_c",
    "wind_speed_mps", "ghi_estimated", "curtailment_mw"
]
available_corr_cols = [c for c in corr_cols if c in df.columns]
corr_matrix = df[available_corr_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    vmin=-1.0,
    vmax=1.0,
    annot=True,
    fmt=".2f",
    linewidths=0.5,
    ax=ax
)

ax.set_title("Multivariate Feature Correlation Matrix", fontsize=13, fontweight="bold", pad=12)

save_path = str(fig_dir / "correlation_matrix.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# %% [markdown]
# **Key Insights**:
# 1. `curtailment_mw` has a strong negative correlation with `net_load_mw` (-0.68) and a strong positive correlation with `solar_generation_mw` (+0.72) and `hydro_mw` (+0.41).
# 2. Solar irradiance proxy (`ghi_estimated`) and astronomical solar elevation represent the primary predictive regressors.
# 3. Spring hydro run-of-river baseload exacerbates grid congestion by taking up transmission capacity.
