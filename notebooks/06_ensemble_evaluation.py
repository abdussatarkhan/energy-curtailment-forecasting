# %% [markdown]
# # 06 - Stacked Ensemble Generalization & Comparative Evaluation
# **Grid Stress Forecaster — Predicting Renewable Energy Curtailment**
# 
# This notebook covers:
# 1. **Stacking Meta-Learner**: Blending Prophet and LSTM via constrained convex quadratic optimization.
# 2. **Comparative Benchmark Evaluation**: Persistence (t-168h) vs. Prophet vs. LSTM vs. Stacked Ensemble.
# 3. **Curtailment Event Classification**: Precision, Recall, F1-Score, and PR-AUC for critical alert thresholds (> 250 MW).
# 4. **Exporting Production Predictions**: Serializing 48-hour forecasts for ingestion by the Plotly Dash Operations Room.

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
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from scripts.utils import (
    setup_logger,
    load_config,
    ensure_dir,
    calculate_regression_metrics,
    calculate_classification_metrics,
    plot_curtailment_forecast
)
from scripts.prophet_model import EnergyProphetForecaster
from scripts.lstm_model import EnergyLSTMForecaster
from scripts.ensemble import StackedCurtailmentEnsemble

config = load_config()
fig_dir = project_root / config["paths"]["figures_dir"]
ensure_dir(str(fig_dir))

# Load clean features
clean_data_path = str(project_root / config["paths"]["processed_features_file"])
df = pd.read_parquet(clean_data_path)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Time splits: Train, Val (168h), Test (48h)
test_hours = 48
val_hours = 168
train_df = df.iloc[:-(test_hours + val_hours)].copy()
val_df = df.iloc[-(test_hours + val_hours):-test_hours].copy()
test_df = df.iloc[-test_hours:].copy()

y_actual = test_df["curtailment_mw"].values
timestamps = test_df["timestamp"].values

print(f"Data Partitions -> Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# %% [markdown]
# ## 1. Training Base Models & Meta-Learner Stacking
# Initializing `StackedCurtailmentEnsemble` and optimizing blending weights.

# %%
ensemble = StackedCurtailmentEnsemble(config=config)
ensemble.train_base_models(train_df, val_df=val_df)

print(f"\nFinal Optimized Stacking Weights:")
print(f"  - Prophet Weight: {ensemble.weights[0]:.4f}")
print(f"  - LSTM Weight:    {ensemble.weights[1]:.4f}")
print(f"  - Residual Standard Deviation: {ensemble.residual_std:.2f} MW")

# %% [markdown]
# ## 2. Generating Comparative Predictions
# Benchmarking against 1-week persistence (`t-168h`), Prophet standalone, LSTM standalone, and Ensemble.

# %%
# 1. Baseline Persistence (same hour last week)
y_persist = df["curtailment_mw"].shift(168).iloc[-test_hours:].values
y_persist = np.nan_to_num(y_persist, nan=0.0)

# 2. Standalone Prophet
prophet_res = ensemble.prophet_forecaster.predict(test_df)
y_prophet = prophet_res["yhat"].values

# 3. Standalone LSTM
input_window = pd.concat([val_df, train_df]).iloc[-48:]
y_lstm = ensemble.lstm_forecaster.predict_horizon(input_window)

# 4. Stacked Ensemble
forecast_results = ensemble.predict_48h(input_window, test_df)
y_ensemble = forecast_results["ensemble_mw"].values

# %% [markdown]
# ## 3. Performance Scorecard: Regression & Event Classification Metrics

# %%
models = {
    "Persistence Baseline (t-168h)": y_persist,
    "Facebook Prophet": y_prophet,
    "Multivariate LSTM": y_lstm,
    "Stacked Ensemble (Prophet + LSTM)": y_ensemble
}

scorecard_rows = []
for name, preds in models.items():
    reg = calculate_regression_metrics(y_actual, preds)
    cls = calculate_classification_metrics(y_actual, preds, threshold=250.0)
    scorecard_rows.append({
        "Model Architecture": name,
        "MAE (MW)": reg["MAE"],
        "RMSE (MW)": reg["RMSE"],
        "MAPE (%)": reg["MAPE"],
        "R2 Score": reg["R2"],
        "Precision": cls["Precision"],
        "Recall": cls["Recall"],
        "F1-Score": cls["F1_Score"],
        "ROC-AUC": cls["ROC_AUC"]
    })

scorecard_df = pd.DataFrame(scorecard_rows).set_index("Model Architecture")
print("\n=== Model Benchmark Scorecard ===")
scorecard_df.round(3)

# %% [markdown]
# ## 4. Comparative Forecast Plot with Alert Tiers

# %%
fig, ax = plt.subplots(figsize=(15, 6), dpi=200)

ax.plot(timestamps, y_actual, color="#2c3e50", marker="o", markersize=4, linestyle=":", label="Observed Curtailment (MW)")
ax.plot(timestamps, y_persist, color="#95a5a6", linestyle="--", linewidth=1.2, label="Persistence Baseline (t-168h)")
ax.plot(timestamps, y_prophet, color="#e67e22", linestyle="-.", linewidth=1.5, label="Prophet Standalone")
ax.plot(timestamps, y_lstm, color="#27ae60", linestyle="--", linewidth=1.5, label="LSTM Standalone")
ax.plot(timestamps, y_ensemble, color="#2980b9", linewidth=2.5, label="Stacked Ensemble (Optimal)")

ax.fill_between(
    timestamps,
    forecast_results["lower_bound_mw"].values,
    forecast_results["upper_bound_mw"].values,
    color="#3498db",
    alpha=0.2,
    label="Ensemble 90% Confidence Band"
)

# Curtailment Alert Band
ax.axhline(250.0, color="#e74c3c", linestyle="--", linewidth=1.5, label="Curtailment Alert Level (>250 MW)")
ax.axhline(1000.0, color="#c0392b", linestyle=":", linewidth=1.5, label="Severe Curtailment (>1000 MW)")

ax.set_title("Comprehensive Model Comparison: 48-Hour Curtailment Forecast", fontsize=14, fontweight="bold", pad=12)
ax.set_xlabel("Forecast Timestamp (UTC)", fontsize=11)
ax.set_ylabel("Renewable Curtailment (MW)", fontsize=11)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="upper left", frameon=True, ncol=2)

save_path = str(fig_dir / "ensemble_comparison_forecast.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved comparative figure to: {save_path}")
plt.show()

# %% [markdown]
# ## 5. Precision-Recall & ROC Event Classification Analysis
# Measuring classification performance when identifying hours with curtailment > 250 MW.

# %%
fig, (ax_pr, ax_roc) = plt.subplots(1, 2, figsize=(14, 5), dpi=200)
y_binary = (y_actual >= 250.0).astype(int)

for name, preds in models.items():
    max_p = max(np.max(preds), 500.0)
    probs = np.clip(preds / max_p, 0.0, 1.0)
    
    if len(np.unique(y_binary)) > 1:
        # PR Curve
        prec, rec, _ = precision_recall_curve(y_binary, probs)
        pr_score = auc(rec, prec)
        ax_pr.plot(rec, prec, label=f"{name} (AUC = {pr_score:.2f})")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_binary, probs)
        roc_score = auc(fpr, tpr)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {roc_score:.2f})")

ax_pr.set_title("Precision-Recall Curve (Curtailment > 250 MW)", fontsize=12, fontweight="bold")
ax_pr.set_xlabel("Recall")
ax_pr.set_ylabel("Precision")
ax_pr.legend(loc="lower left")
ax_pr.grid(True, linestyle=":")

ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=12, fontweight="bold")
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.legend(loc="lower right")
ax_roc.grid(True, linestyle=":")

plt.tight_layout()
save_path = str(fig_dir / "pr_roc_evaluation.png")
plt.savefig(save_path, dpi=300, bbox_inches="tight")
print(f"Saved PR/ROC curves to: {save_path}")
plt.show()

# %% [markdown]
# ## 6. Exporting Forecasts for Operations Dashboard
# Serializing the forecast output to Parquet for real-time visualization in Plotly Dash.

# %%
predictions_out_path = str(project_root / config["paths"]["predictions_file"])
ensure_dir(os.path.dirname(predictions_out_path))

# Combine forecast results with metadata
export_df = forecast_results.copy()
export_df["actual_curtailment_mw"] = y_actual
export_df["net_load_mw"] = test_df["net_load_mw"].values
export_df["solar_generation_mw"] = test_df["solar_generation_mw"].values
export_df["wind_generation_mw"] = test_df["wind_generation_mw"].values
export_df["temperature_c"] = test_df["temperature_c"].values

export_df.to_parquet(predictions_out_path, index=False)
print(f"\nProduction forecast predictions serialized to: {predictions_out_path} ({len(export_df)} rows)")
export_df.head()
