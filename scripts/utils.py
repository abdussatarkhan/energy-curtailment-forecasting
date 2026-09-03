"""
scripts/utils.py - General utilities, metrics, configuration helpers, and visualization tools.
Grid Stress Forecaster — Predicting Renewable Energy Curtailment
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path

import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


def setup_logger(name: str = "GridStressForecaster", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configures and returns a thread-safe, standardized logger with console and file handlers.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger already initialized
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger("utils")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Loads YAML configuration settings safely.
    """
    path = Path(config_path)
    if not path.is_file():
        # Try finding relative to project root
        project_root = Path(__file__).resolve().parent.parent
        alt_path = project_root / config_path
        if alt_path.is_file():
            path = alt_path
        else:
            raise FileNotFoundError(f"Configuration file not found at '{config_path}' or '{alt_path}'.")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.debug("Loaded configuration from %s", path)
    return config


def ensure_dir(dir_path: str) -> str:
    """
    Ensures a directory exists, creating all necessary parents.
    """
    os.makedirs(dir_path, exist_ok=True)
    return dir_path


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard regression performance metrics: MAE, RMSE, MAPE, and R-squared.
    Handles zeros safely when calculating Mean Absolute Percentage Error (MAPE).
    """
    y_true = np.asarray(y_true, dtype=np.float64).flatten()
    y_pred = np.asarray(y_pred, dtype=np.float64).flatten()

    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))

    # Epsilon-stabilized MAPE calculation (ignoring near-zero baseline)
    mask = np.abs(y_true) > 1e-2
    if np.any(mask):
        mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)
    else:
        mape = 0.0

    # Normalized RMSE (normalized by range or mean)
    denom = (np.max(y_true) - np.min(y_true))
    nrmse = float(rmse / denom) if denom > 1e-5 else 0.0

    return {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "NRMSE": round(nrmse, 4),
        "MAPE": round(mape, 4),
        "R2": round(r2, 4)
    }


def calculate_classification_metrics(
    y_true_continuous: np.ndarray,
    y_pred_continuous: np.ndarray,
    threshold: float = 250.0
) -> Dict[str, float]:
    """
    Converts continuous curtailment forecasts into binary curtailment incident events
    (True if curtailment > threshold MW) and computes operational classification metrics.
    """
    y_true_binary = (np.asarray(y_true_continuous) >= threshold).astype(int).flatten()
    y_pred_binary = (np.asarray(y_pred_continuous) >= threshold).astype(int).flatten()

    precision = float(precision_score(y_true_binary, y_pred_binary, zero_division=0))
    recall = float(recall_score(y_true_binary, y_pred_binary, zero_division=0))
    f1 = float(f1_score(y_true_binary, y_pred_binary, zero_division=0))

    try:
        # Normalize predictions to [0, 1] range as pseudo-probability proxy for ROC/PR-AUC
        max_val = max(np.max(y_pred_continuous), threshold * 2)
        probs = np.clip(np.asarray(y_pred_continuous) / max_val, 0.0, 1.0)
        roc_auc = float(roc_auc_score(y_true_binary, probs)) if len(np.unique(y_true_binary)) > 1 else 0.5
        pr_auc = float(average_precision_score(y_true_binary, probs)) if len(np.unique(y_true_binary)) > 1 else 0.0
    except Exception as e:
        logger.warning(f"Failed to compute AUC metrics: {e}")
        roc_auc, pr_auc = 0.5, 0.0

    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()

    return {
        "Threshold_MW": float(threshold),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1_Score": round(f1, 4),
        "ROC_AUC": round(roc_auc, 4),
        "PR_AUC": round(pr_auc, 4),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn)
    }


def create_time_splits(
    df: pd.DataFrame,
    test_hours: int = 48,
    val_hours: int = 168
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates strictly temporal train/val/test splits preventing future data leakage.
    Default: Last 48 hours for test, prior 168 hours (7 days) for validation, remainder for train.
    """
    total_len = len(df)
    if total_len < (test_hours + val_hours + 100):
        raise ValueError(f"Dataframe length {total_len} is too short for split requirements.")

    train_end_idx = total_len - (test_hours + val_hours)
    val_end_idx = total_len - test_hours

    train_df = df.iloc[:train_end_idx].copy()
    val_df = df.iloc[train_end_idx:val_end_idx].copy()
    test_df = df.iloc[val_end_idx:].copy()

    logger.info(
        f"Time splits created - Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows"
    )
    return train_df, val_df, test_df


def save_json(data: Dict[str, Any], file_path: str) -> None:
    """Saves dictionary data to a formatted JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)
    logger.info(f"Saved JSON metrics to {file_path}")


def load_json(file_path: str) -> Dict[str, Any]:
    """Loads JSON data from file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_curtailment_forecast(
    timestamps: pd.DatetimeIndex,
    actual: np.ndarray,
    predicted: np.ndarray,
    lower_bound: Optional[np.ndarray] = None,
    upper_bound: Optional[np.ndarray] = None,
    threshold: float = 250.0,
    title: str = "48-Hour Ahead Renewable Curtailment Forecast",
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Generates a publication-grade forecast plot comparing Actual vs Predicted Curtailment (MW),
    highlighting alert zones and confidence intervals.
    """
    sns.set_theme(style="whitegrid", palette="deep")
    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)

    # Threshold line
    ax.axhline(threshold, color="#e74c3c", linestyle="--", linewidth=1.5, label=f"Curtailment Alert Threshold ({threshold:.0f} MW)", alpha=0.85)

    # Predicted line and uncertainty band
    ax.plot(timestamps, predicted, color="#2980b9", linewidth=2.2, label="Ensemble Forecast (MW)")
    if lower_bound is not None and upper_bound is not None:
        ax.fill_between(
            timestamps,
            np.maximum(0, lower_bound),
            upper_bound,
            color="#3498db",
            alpha=0.25,
            label="90% Prediction Interval"
        )

    # Actual observations
    ax.plot(timestamps, actual, color="#2c3e50", linewidth=2.0, linestyle=":", marker="o", markersize=4, label="Observed Curtailment (MW)")

    # Formatting
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Forecast Timestamp (UTC)", fontsize=11, labelpad=10)
    ax.set_ylabel("Renewable Curtailment (MW)", fontsize=11, labelpad=10)
    ax.set_ylim(bottom=-10)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Forecast plot saved to {save_path}")

    return fig
