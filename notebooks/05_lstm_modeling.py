# %% [markdown]
# # 05 - Deep Multivariate LSTM Sequence Modeling
# **Grid Stress Forecaster — Predicting Renewable Energy Curtailment**
# 
# This notebook implements:
# 1. **3D Multivariate Sequence Tensor Construction**: Sliding past 48 hours of grid telemetry into (N, 48, F) inputs and (N, 48) multi-step forecast targets.
# 2. **Feature & Target Scaling**: RobustScaler on exogenous features and MinMaxScaler on target Megawatts.
# 3. **Stacked LSTM Architecture**: Recurrent neural network with dropout regularization, batch normalization, and dense projection.
# 4. **Training Dynamics & Early Stopping**: Tracking train/validation loss curves.
# 5. **Holdout Evaluation**: Comparing 48-hour forward pass with observed curtailment.

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
from scripts.lstm_model import EnergyLSTMForecaster

config = load_config()
fig_dir = project_root / config["paths"]["figures_dir"]
ensure_dir(str(fig_dir))

# Load data
clean_data_path = str(project_root / config["paths"]["processed_features_file"])
df = pd.read_parquet(clean_data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Split into Train, Validation (last 168h of train), and Test (last 48h)
test_hours = 48
val_hours = 168
train_df = df.iloc[:-(test_hours + val_hours)].copy()
val_df = df.iloc[-(test_hours + val_hours):-test_hours].copy()
test_df = df.iloc[-test_hours:].copy()

print(f"Dataset Partitions -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %% [markdown]
# ## 1. Model Architecture & Data Windowing
# Initializing `EnergyLSTMForecaster`. Input sequence length = 48 hours, Output forecast horizon = 48 hours.

# %%
lstm_forecaster = EnergyLSTMForecaster(config=config)
print("Multivariate Feature Columns:", lstm_forecaster.feature_cols)

# Prepare sequence tensors
X_train, y_train = lstm_forecaster.prepare_data(train_df, is_training=True)
X_val, y_val = lstm_forecaster.prepare_data(val_df, is_training=False)

print(f"\nSequence Tensor Dimensions:")
print(f"X_train: {X_train.shape} (Windows, Lookback Hours, Features)")
print(f"y_train: {y_train.shape} (Windows, Forecast Horizon Hours)")
print(f"X_val:   {X_val.shape}")

# %% [markdown]
# ## 2. Model Training with Regularization & Early Stopping
# Training the neural network while monitoring Huber loss on validation sequences.

# %%
model_save_path = str(project_root / "models/lstm_best_weights.keras")
history = lstm_forecaster.train(train_df, val_df=val_df, model_save_path=model_save_path)
print("LSTM Training Phase Concluded.")

# %% [markdown]
# ## 3. Training & Validation Loss Curves
# Inspecting learning convergence and verifying absence of catastrophic overfitting.

# %%
if history and hasattr(history, "history"):
    fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
    ax.plot(history.history["loss"], label="Training Huber Loss", color="#2980b9", linewidth=2.0)
    ax.plot(history.history["val_loss"], label="Validation Huber Loss", color="#e74c3c", linewidth=2.0, linestyle="--")
    ax.set_title("LSTM Convergence: Training vs. Validation Loss", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Huber Loss", fontsize=11)
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    save_path = str(fig_dir / "lstm_training_loss.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved loss curves to: {save_path}")
    plt.show()

# %% [markdown]
# ## 4. 48-Hour Ahead Out-of-Sample Forecasting
# Passing the last 48 hours of validation history into the trained LSTM network to forecast the test window.

# %%
input_window = pd.concat([val_df, train_df]).iloc[-48:]
predicted_curtailment_mw = lstm_forecaster.predict_horizon(input_window)

y_actual = test_df["curtailment_mw"].values[:48]
timestamps = test_df["timestamp"].values[:48]

metrics = calculate_regression_metrics(y_actual, predicted_curtailment_mw)
print("\n--- LSTM 48-Hour Test Horizon Metrics ---")
for k, v in metrics.items():
    print(f"  {k}: {v}")

# %% [markdown]
# ## 5. Visualization: Actual vs. LSTM Forecast

# %%
fig, ax = plt.subplots(figsize=(14, 6), dpi=200)

ax.plot(timestamps, y_actual, color="#2c3e50", marker="o", markersize=4, linestyle=":", label="Observed Curtailment (MW)")
ax.plot(timestamps, predicted_curtailment_mw, color="#27ae60", linewidth=2.2, label="LSTM 48-Step Forecast (MW)")
ax.axhline(250.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Curtailment Alert Threshold (250 MW)")

ax.set_title("Multivariate LSTM 48-Hour Curtailment Forecast vs. Actual Grid Dispatch", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Timestamp (UTC)", fontsize=11)
ax.set_ylabel("Renewable Curtailment (MW)", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left")

save_path = str(fig_dir / "lstm_forecast_eval.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved figure to: {save_path}")
plt.show()

# Save final model
lstm_forecaster.save(str(project_root / "models/lstm_model"))
