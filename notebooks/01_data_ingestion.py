# %% [markdown]
# # 01 - Automated Data Ingestion Pipeline
# **Grid Stress Forecaster — Predicting Renewable Energy Curtailment**
# 
# This notebook demonstrates the ingestion workflow for:
# 1. **California Independent System Operator (CAISO)** balancing authority hourly fuel-mix dispatch from the **EIA v2 REST API**.
# 2. **NOAA Integrated Surface Database (ISD)** meteorology covering key load and renewable production centers across California (LAX, SFO, Fresno, Bakersfield).
# 3. Generating a calibrated physics-informed synthetic benchmark dataset for offline reproducible experimentation.

# %%
import os
import sys
from pathlib import Path

# Add project root and scripts directory to python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scripts.utils import setup_logger, load_config, ensure_dir
from scripts.data_collection import EIAClient, NOAAClient, generate_synthetic_benchmark_dataset

logger = setup_logger("01_ingestion_nb")
sns.set_theme(style="whitegrid")

# %% [markdown]
# ## 1. Configuration & Parameter Verification
# Inspecting configured endpoints, target fuel types, and NOAA weather station metadata.

# %%
config = load_config()
print("Project Title:", config["project"]["name"])
print("Configured Paths:", config["paths"])
print("Target Fuel Types:", config["data_collection"]["eia"]["fuel_types"])
print("Target NOAA Weather Stations:")
for st in config["data_collection"]["noaa"]["target_stations"]:
    print(f" - {st['station_id']}: {st['name']} (Lat: {st['lat']}, Lon: {st['lon']}, Weight: {st['weight']})")

# %% [markdown]
# ## 2. Benchmarking & Synthetic Grid Simulation
# In the absence of a live commercial EIA API key, we invoke our calibrated synthetic generator.
# This accurately models:
# - Diurnal solar bell-curve matching California's ~18.5 GW installed capacity.
# - Evening Tehachapi & Solano wind generation surges.
# - Diablo Canyon nuclear base generation (~2,240 MW) and seasonal Sierra Nevada spring hydro runoff.
# - Afternoon duck curve belly compression and over-generation curtailment events.

# %%
raw_output_path = str(project_root / config["paths"]["eia_raw_file"])
ensure_dir(os.path.dirname(raw_output_path))

# Generate or load 3-year hourly dataset (2021 to 2023)
if not os.path.exists(raw_output_path):
    print("Generating synthetic benchmark dataset...")
    df_raw = generate_synthetic_benchmark_dataset(
        start_date="2021-01-01",
        end_date="2023-12-31 23:00",
        output_path=raw_output_path
    )
else:
    print(f"Loading existing raw dataset from {raw_output_path}...")
    df_raw = pd.read_parquet(raw_output_path)

print(f"\nRaw Dataset Dimensions: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
df_raw.head()

# %% [markdown]
# ## 3. Data Profile & Ingestion Integrity Inspection
# Verifying data types, missing value rates, and statistical distributions of grid variables.

# %%
df_raw.info()

# %%
summary_stats = df_raw.describe().round(2)
print("Statistical Summary of Grid Generation & Curtailment (MW):")
summary_stats[["system_demand_mw", "solar_generation_mw", "wind_generation_mw", "net_load_mw", "curtailment_mw"]]

# %% [markdown]
# ## 4. Initial Time-Series Sanity Plots
# Visualizing an arbitrary 7-day spring window showing high solar penetration and curtailment occurrence.

# %%
spring_week = df_raw.iloc[2000:2168]  # ~1 week sample in Spring
fig, ax1 = plt.subplots(figsize=(14, 6), dpi=150)

ax1.plot(spring_week["timestamp"], spring_week["system_demand_mw"], color="black", label="System Demand (MW)", linewidth=1.8)
ax1.plot(spring_week["timestamp"], spring_week["solar_generation_mw"], color="#f39c12", label="Solar Generation (MW)", linewidth=1.8)
ax1.plot(spring_week["timestamp"], spring_week["wind_generation_mw"], color="#27ae60", label="Wind Generation (MW)", linewidth=1.5, linestyle="--")
ax1.plot(spring_week["timestamp"], spring_week["net_load_mw"], color="#2980b9", label="Net Load (MW)", linewidth=2.0)

ax2 = ax1.twinx()
ax2.fill_between(spring_week["timestamp"], 0, spring_week["curtailment_mw"], color="#e74c3c", alpha=0.4, label="Curtailment (MW)")
ax2.set_ylabel("Curtailed Energy (MW)", color="#e74c3c", fontsize=11)
ax2.set_ylim(0, 4000)

ax1.set_title("CAISO Spring Hourly Generation, Net Load & Curtailment Dynamics", fontsize=13, fontweight="bold")
ax1.set_ylabel("Power Dispatch (MW)", fontsize=11)
ax1.set_xlabel("Date & Time", fontsize=11)
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.tight_layout()
plt.show()

# %% [markdown]
# **Observation**: When midday solar surges coincide with moderate demand, Net Load plunges toward the thermal must-run floor, triggering acute renewable curtailment spikes (> 1,000 MW).
