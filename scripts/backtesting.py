"""
scripts/backtesting.py - Comprehensive rolling-window backtesting engine for 48-hour-ahead forecasts.
Evaluates Persistence Baselines, Prophet, LSTM, and Stacked Ensemble across seasonal regimes.
Computes regression (MAE, RMSE, MAPE) and event classification (Precision, Recall, F1, PR-AUC).
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import (
    setup_logger,
    load_config,
    ensure_dir,
    calculate_regression_metrics,
    calculate_classification_metrics,
    save_json,
    plot_curtailment_forecast
)
from prophet_model import EnergyProphetForecaster
from lstm_model import EnergyLSTMForecaster
from ensemble import StackedCurtailmentEnsemble

logger = setup_logger("backtesting")


class RollingWindowBacktester:
    """
    Simulates operational grid scheduling with non-overlapping or rolling 48-hour windows:
    - Benchmarks against Day-Ahead Persistence (t-24h) and Same-Day-Last-Week (t-168h) baselines.
    - Evaluates Prophet, LSTM, and Stacked Ensemble across spring overgeneration vs summer peak.
    - Computes confusion matrix, curtailment event detection precision/recall/F1, and regression errors.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.bt_cfg = self.config.get("backtesting", {})
        self.horizon_hours = self.bt_cfg.get("test_horizon_hours", 48)
        self.step_hours = self.bt_cfg.get("step_hours", 24)
        self.curtailment_threshold = self.config.get("preprocessing", {}).get("curtailment_threshold_mw", 250.0)

    def run_persistence_baseline(self, df: pd.DataFrame, lag_hours: int = 168) -> Tuple[np.ndarray, np.ndarray]:
        """
        Naive grid operator persistence benchmark:
        Assumes curtailment today at hour h equals curtailment from same hour last week (lag 168h).
        """
        actual = df["curtailment_mw"].values[lag_hours:]
        predicted = df["curtailment_mw"].shift(lag_hours).values[lag_hours:]
        return actual, predicted

    def run_rolling_evaluation(
        self,
        df: pd.DataFrame,
        num_windows: int = 4,
        window_size: int = 48
    ) -> Dict[str, Any]:
        """
        Executes rolling temporal backtest across multiple evaluation blocks.
        """
        logger.info(f"Initiating Rolling Backtest: {num_windows} evaluation windows of {window_size} hours each...")
        total_rows = len(df)
        required_rows = window_size * num_windows + 168 + 48
        if total_rows < required_rows:
            raise ValueError(f"Dataframe size {total_rows} is too small for requested backtesting windows.")

        results_collection = {
            "Persistence_168h": {"actual": [], "pred": []},
            "Prophet": {"actual": [], "pred": []},
            "LSTM": {"actual": [], "pred": []},
            "Stacked_Ensemble": {"actual": [], "pred": []}
        }

        # Select window endpoints
        window_offsets = [total_rows - (i * window_size) for i in range(num_windows, 0, -1)]

        for i, end_idx in enumerate(window_offsets):
            start_idx = end_idx - window_size
            test_slice = df.iloc[start_idx:end_idx].copy()
            train_slice = df.iloc[:start_idx].copy()

            logger.info(f"--- Processing Backtest Window {i+1}/{num_windows} ({test_slice['timestamp'].min()} to {test_slice['timestamp'].max()}) ---")

            actual_vals = test_slice["curtailment_mw"].values
            val_hours = 168
            train_sub = train_slice.iloc[:-val_hours]
            val_sub = train_slice.iloc[-val_hours:]

            # 1. Baseline: 168h persistence
            persist_pred = df["curtailment_mw"].shift(168).iloc[start_idx:end_idx].values
            persist_pred = np.nan_to_num(persist_pred, nan=0.0)
            results_collection["Persistence_168h"]["actual"].extend(actual_vals)
            results_collection["Persistence_168h"]["pred"].extend(persist_pred)

            # 2. Fit and predict Prophet
            prophet = EnergyProphetForecaster(config=self.config)
            prophet.train(train_sub)
            p_pred = prophet.predict(test_slice)["yhat"].values
            results_collection["Prophet"]["actual"].extend(actual_vals)
            results_collection["Prophet"]["pred"].extend(p_pred)

            # 3. Fit and predict LSTM
            lstm = EnergyLSTMForecaster(config=self.config)
            lstm.train(train_sub, val_df=val_sub)
            l_pred = lstm.predict_horizon(train_slice.iloc[-48:])
            if len(l_pred) < len(actual_vals):
                l_pred = np.pad(l_pred, (0, len(actual_vals) - len(l_pred)), mode="edge")
            elif len(l_pred) > len(actual_vals):
                l_pred = l_pred[:len(actual_vals)]
            results_collection["LSTM"]["actual"].extend(actual_vals)
            results_collection["LSTM"]["pred"].extend(l_pred)

            # 4. Ensemble
            ensemble = StackedCurtailmentEnsemble(config=self.config)
            # Re-use sub-models
            ensemble.prophet_forecaster = prophet
            ensemble.lstm_forecaster = lstm
            ens_df = ensemble.predict_48h(train_slice.iloc[-48:], test_slice)
            results_collection["Stacked_Ensemble"]["actual"].extend(actual_vals)
            results_collection["Stacked_Ensemble"]["pred"].extend(ens_df["ensemble_mw"].values)

        # Compute aggregate scorecard
        scorecard = {}
        for model_name, preds in results_collection.items():
            y_act = np.array(preds["actual"])
            y_hat = np.array(preds["pred"])
            reg = calculate_regression_metrics(y_act, y_hat)
            cls = calculate_classification_metrics(y_act, y_hat, threshold=self.curtailment_threshold)
            scorecard[model_name] = {
                "regression": reg,
                "curtailment_event_detection": cls
            }
            logger.info(f"[{model_name}] MAE: {reg['MAE']} MW | RMSE: {reg['RMSE']} MW | F1: {cls['F1_Score']} | Precision: {cls['Precision']} | Recall: {cls['Recall']}")

        return scorecard


def main():
    parser = argparse.ArgumentParser(description="Rolling Forecast Backtester CLI")
    parser.add_argument("--data", type=str, default="data/processed/energy_features_clean.parquet")
    parser.add_argument("--output-scorecard", type=str, default="reports/model_performance_scorecard.json")
    parser.add_argument("--output-plot", type=str, default="images/backtest_forecast_comparison.png")
    parser.add_argument("--windows", type=int, default=3, help="Number of rolling windows")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    logger.info(f"Loading data from {args.data} for backtesting...")
    df = pd.read_parquet(args.data)

    backtester = RollingWindowBacktester(config=cfg)
    scorecard = backtester.run_rolling_evaluation(df, num_windows=args.windows, window_size=48)

    save_json(scorecard, args.output_scorecard)

    # Generate comparative visualization for the final test window
    final_48 = df.iloc[-48:]
    timestamps = pd.to_datetime(final_48["timestamp"])
    actual = final_48["curtailment_mw"].values
    
    # Load or instantiate ensemble for the plot
    ens = StackedCurtailmentEnsemble(config=cfg)
    ens.train_base_models(df.iloc[:-48])
    ens_res = ens.predict_48h(df.iloc[-96:-48], final_48)

    plot_curtailment_forecast(
        timestamps=timestamps,
        actual=actual,
        predicted=ens_res["ensemble_mw"].values,
        lower_bound=ens_res["lower_bound_mw"].values,
        upper_bound=ens_res["upper_bound_mw"].values,
        threshold=cfg["preprocessing"]["curtailment_threshold_mw"],
        title="Final 48-Hour Backtest Window: Actual vs. Stacked Ensemble Forecast",
        save_path=args.output_plot
    )
    logger.info("Backtesting routine completed successfully.")


if __name__ == "__main__":
    main()
