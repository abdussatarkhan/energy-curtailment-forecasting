"""
scripts/ensemble.py - Stacked Generalization combining Prophet and LSTM Forecasters.
Implements meta-learner stacking, optimal convex weight blending, uncertainty bounds,
and operational curtailment risk classification.
"""

import os
import sys
import argparse
import joblib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LinearRegression
from scipy.optimize import minimize

from utils import (
    setup_logger,
    load_config,
    ensure_dir,
    calculate_regression_metrics,
    calculate_classification_metrics
)
from prophet_model import EnergyProphetForecaster
from lstm_model import EnergyLSTMForecaster

logger = setup_logger("ensemble")


class StackedCurtailmentEnsemble:
    """
    Combines generalized additive seasonal modeling (Facebook Prophet) with deep sequence
    learning (LSTM) through a constrained stacking meta-learner.

    1. Base Model 1: Prophet captures macro seasonal trends, statutory holidays, and temperature effects.
    2. Base Model 2: LSTM captures non-linear hourly inertia, multi-hour sequence ramp dynamics, and weather fronts.
    3. Meta-Learner: Constrained non-negative least squares or Ridge regression prevents negative weights
       and optimizes minimum variance forecasting error.
    4. Uncertainty Bounds: Synthesizes Prophet's posterior interval with empirical residual dispersion.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.ens_cfg = self.config.get("ensemble", {})
        self.curtailment_threshold = self.config.get("preprocessing", {}).get("curtailment_threshold_mw", 250.0)
        self.meta_learner_type = self.ens_cfg.get("meta_learner", "Ridge")
        self.meta_model: Optional[Any] = None
        self.weights = np.array([0.40, 0.60])  # Default prior: 40% Prophet, 60% LSTM
        self.residual_std: float = 45.0
        self.prophet_forecaster = EnergyProphetForecaster(config=self.config)
        self.lstm_forecaster = EnergyLSTMForecaster(config=self.config)

    def fit_meta_weights(
        self,
        prophet_val_preds: np.ndarray,
        lstm_val_preds: np.ndarray,
        y_val_actual: np.ndarray
    ) -> np.ndarray:
        """
        Solves constrained convex optimization for optimal ensemble blending weights:
        min || y - (w1 * p1 + w2 * p2) ||^2  s.t.  w1 + w2 = 1, w1, w2 >= 0
        """
        logger.info("Optimizing ensemble blending weights on validation predictions...")
        P = np.column_stack([prophet_val_preds.flatten(), lstm_val_preds.flatten()])
        y = y_val_actual.flatten()

        def loss_func(w):
            pred = P @ w
            return np.mean((y - pred) ** 2)

        cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
        bnds = [(0.0, 1.0), (0.0, 1.0)]
        init_w = np.array([0.5, 0.5])

        res = minimize(loss_func, init_w, method="SLSQP", bounds=bnds, constraints=cons)
        if res.success:
            self.weights = res.x
            logger.info(f"Optimal Ensemble Weights: Prophet = {self.weights[0]:.4f}, LSTM = {self.weights[1]:.4f}")
        else:
            logger.warning("Convex optimization did not converge. Falling back to Ridge regression.")
            ridge = Ridge(alpha=self.ens_cfg.get("ridge_alpha", 1.0), positive=True)
            ridge.fit(P, y)
            raw_w = ridge.coef_
            s = np.sum(raw_w)
            self.weights = raw_w / s if s > 0 else np.array([0.5, 0.5])
            logger.info(f"Ridge-derived weights: Prophet = {self.weights[0]:.4f}, LSTM = {self.weights[1]:.4f}")

        # Compute empirical residual standard deviation for prediction intervals
        stacked_preds = P @ self.weights
        self.residual_std = float(np.std(y - stacked_preds))
        logger.info(f"Ensemble validation residual std: {self.residual_std:.2f} MW")
        return self.weights

    def train_base_models(
        self,
        train_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None
    ) -> None:
        """
        Fits both Prophet and LSTM on historical data, then optimizes blending weights on validation set.
        """
        logger.info("=== Starting Ensemble Base Model Training ===")
        # 1. Fit Prophet
        logger.info("Fitting Base Model 1: Facebook Prophet...")
        self.prophet_forecaster.train(train_df)

        # 2. Fit LSTM
        logger.info("Fitting Base Model 2: Multivariate Deep LSTM...")
        self.lstm_forecaster.train(train_df, val_df=val_df)

        # 3. Stack meta-weights if validation data is provided
        if val_df is not None and len(val_df) >= 48:
            logger.info("Generating validation forecasts for meta-stacking...")
            prophet_val = self.prophet_forecaster.predict(val_df)["yhat"].values
            
            # For LSTM, predict the horizon
            lstm_preds = []
            for start_idx in range(0, len(val_df) - 48 + 1, 24):
                input_window = pd.concat([train_df, val_df.iloc[:start_idx]]).iloc[-48:]
                chunk_pred = self.lstm_forecaster.predict_horizon(input_window)
                lstm_preds.extend(chunk_pred[:min(24, len(val_df) - start_idx)])
            
            # Pad or slice to match length
            min_len = min(len(prophet_val), len(lstm_preds), len(val_df))
            if min_len > 0:
                y_val = val_df["curtailment_mw"].values[:min_len]
                p_val = prophet_val[:min_len]
                l_val = np.array(lstm_preds)[:min_len]
                self.fit_meta_weights(p_val, l_val, y_val)

        logger.info("Ensemble training process successfully completed.")

    def predict_48h(
        self,
        recent_history_df: pd.DataFrame,
        future_exog_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Executes full 48-hour forward pass across Prophet and LSTM, then stacks predictions.
        recent_history_df: at least past 48 hours with all telemetry
        future_exog_df: future 48 hours containing weather forecast & calendar features
        """
        if len(future_exog_df) != 48:
            logger.warning(f"future_exog_df has {len(future_exog_df)} rows; expected 48 for standard horizon.")

        # 1. Base Forecast 1: Prophet
        prophet_df = self.prophet_forecaster.predict(future_exog_df)
        p_pred = prophet_df["yhat"].values

        # 2. Base Forecast 2: LSTM
        l_pred = self.lstm_forecaster.predict_horizon(recent_history_df)
        if len(l_pred) < len(future_exog_df):
            l_pred = np.pad(l_pred, (0, len(future_exog_df) - len(l_pred)), mode="edge")
        elif len(l_pred) > len(future_exog_df):
            l_pred = l_pred[:len(future_exog_df)]

        # 3. Stack predictions using convex weights
        stacked = self.weights[0] * p_pred + self.weights[1] * l_pred
        stacked = np.maximum(0.0, stacked)

        # 4. Uncertainty intervals: 90% confidence (+/- 1.645 * sigma)
        sigma = max(self.residual_std, 25.0)
        lower_band = np.maximum(0.0, stacked - 1.645 * sigma)
        upper_band = stacked + 1.645 * sigma

        # 5. Operational Alert Classifications:
        # Green (< 250 MW), Amber (250 - 1000 MW), Red (> 1000 MW severe event)
        alert_levels = []
        for val in stacked:
            if val < self.curtailment_threshold:
                alert_levels.append("LOW (Green)")
            elif val < 1000.0:
                alert_levels.append("MODERATE (Amber)")
            else:
                alert_levels.append("CRITICAL (Red)")

        results_df = pd.DataFrame({
            "timestamp": future_exog_df["timestamp"].values,
            "prophet_mw": np.round(p_pred, 1),
            "lstm_mw": np.round(l_pred, 1),
            "ensemble_mw": np.round(stacked, 1),
            "lower_bound_mw": np.round(lower_band, 1),
            "upper_bound_mw": np.round(upper_band, 1),
            "curtailment_alert": alert_levels,
            "curtailment_event_prob": np.round(np.clip(stacked / (self.curtailment_threshold * 2.5), 0.0, 1.0), 3)
        })

        return results_df

    def evaluate_test_set(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates ensemble output against true test set values across regression and classification metrics.
        """
        reg_metrics = calculate_regression_metrics(y_true, y_pred)
        cls_metrics = calculate_classification_metrics(y_true, y_pred, threshold=self.curtailment_threshold)
        summary = {
            "regression_metrics": reg_metrics,
            "classification_metrics": cls_metrics,
            "weights": {"prophet": float(self.weights[0]), "lstm": float(self.weights[1])}
        }
        return summary

    def save_ensemble(self, output_dir: str = "models/ensemble") -> None:
        """Saves weights and sub-models."""
        ensure_dir(output_dir)
        meta_state = {
            "weights": self.weights,
            "residual_std": self.residual_std,
            "curtailment_threshold": self.curtailment_threshold
        }
        joblib.dump(meta_state, os.path.join(output_dir, "meta_weights.pkl"))
        self.prophet_forecaster.save_model(os.path.join(output_dir, "prophet.pkl"))
        self.lstm_forecaster.save(os.path.join(output_dir, "lstm"))
        logger.info(f"Ensemble successfully saved to {output_dir}")

    def load_ensemble(self, input_dir: str = "models/ensemble") -> None:
        """Loads weights and sub-models."""
        meta_state = joblib.load(os.path.join(input_dir, "meta_weights.pkl"))
        self.weights = meta_state["weights"]
        self.residual_std = meta_state["residual_std"]
        self.curtailment_threshold = meta_state["curtailment_threshold"]
        self.prophet_forecaster.load_model(os.path.join(input_dir, "prophet.pkl"))
        self.lstm_forecaster.load(os.path.join(input_dir, "lstm"))
        logger.info(f"Ensemble successfully loaded from {input_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stacked Generalization Forecaster CLI")
    parser.add_argument("--data", type=str, default="data/processed/energy_features_clean.parquet")
    parser.add_argument("--output-dir", type=str, default="models/ensemble")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = pd.read_parquet(args.data)

    test_hours = 48
    val_hours = 168
    train_df = df.iloc[:-(test_hours + val_hours)]
    val_df = df.iloc[-(test_hours + val_hours):-test_hours]
    test_df = df.iloc[-test_hours:]

    ensemble = StackedCurtailmentEnsemble(config=cfg)
    ensemble.train_base_models(train_df, val_df=val_df)
    ensemble.save_ensemble(args.output_dir)

    # Predict 48 hours
    recent_history = pd.concat([val_df, train_df]).iloc[-48:]
    forecast_results = ensemble.predict_48h(recent_history, test_df)

    eval_results = ensemble.evaluate_test_set(
        test_df["curtailment_mw"].values,
        forecast_results["ensemble_mw"].values
    )
    logger.info(f"Ensemble Test Results: {eval_results}")


if __name__ == "__main__":
    main()
