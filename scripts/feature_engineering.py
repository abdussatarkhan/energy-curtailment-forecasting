"""
scripts/feature_engineering.py - Advanced feature creation for solar elevation,
Fourier multi-seasonal terms, autoregressive lags (t-1 to t-48), rolling windows, and ramp rates.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from utils import setup_logger, load_config, ensure_dir

logger = setup_logger("feature_engineering")


class EnergyFeatureEngineer:
    """
    Transforms clean hourly grid data into predictive features for Prophet and LSTM forecasters:
    1. Cyclical trigonometric encodings (hour of day, day of week, month).
    2. Astronomical solar elevation angle and zenith calculations based on California coordinates.
    3. Multi-horizon Fourier decomposition terms (daily, weekly, annual harmonic frequencies).
    4. Autoregressive lags from t-1 up to t-48 for solar, wind, net load, and curtailment.
    5. Rolling multi-scale aggregate statistics (6h, 24h, 168h rolling mean, std, min, max).
    6. System ramping dynamics (1h and 3h net load gradient MW/hr).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        feat_cfg = self.config.get("feature_engineering", {})
        self.lat = feat_cfg.get("solar_position", {}).get("latitude", 36.7783)
        self.lon = feat_cfg.get("solar_position", {}).get("longitude", -119.4179)
        self.lag_hours = feat_cfg.get("lag_hours", [1, 2, 3, 6, 12, 24, 48])
        self.rolling_windows = feat_cfg.get("rolling_windows_hours", [6, 24, 168])
        self.fourier_cfg = feat_cfg.get("fourier_terms", {"daily_order": 5, "weekly_order": 3, "annual_order": 4})
        self.curtailment_threshold = self.config.get("preprocessing", {}).get("curtailment_threshold_mw", 250.0)

    def add_cyclical_calendar_features(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """
        Extracts temporal attributes and computes sine/cosine continuous circular representations.
        """
        logger.info("Computing calendar and cyclical trigonometric features...")
        df = df.copy()
        ts = pd.to_datetime(df[time_col])

        # Raw calendar components
        df["hour"] = ts.dt.hour
        df["day_of_week"] = ts.dt.dayofweek
        df["day_of_year"] = ts.dt.dayofyear
        df["month"] = ts.dt.month
        df["quarter"] = ts.dt.quarter
        df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

        # Cyclical transformations
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24.0)

        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7.0)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7.0)

        df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12.0)

        df["day_of_year_sin"] = np.sin(2 * np.pi * (df["day_of_year"] - 1) / 365.25)
        df["day_of_year_cos"] = np.cos(2 * np.pi * (df["day_of_year"] - 1) / 365.25)

        return df

    def add_solar_elevation_angle(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """
        Computes the solar elevation angle (degrees above horizon) based on astronomical
        declination, equation of time, and hour angle for California coordinates.
        Elevation angle correlates strongly with solar photovoltaic capacity factor.
        """
        logger.info("Computing astronomical solar elevation angle...")
        df = df.copy()
        ts = pd.to_datetime(df[time_col])

        day_of_year = ts.dt.dayofyear.values
        hour = ts.dt.hour.values + ts.dt.minute.values / 60.0

        # Fractional year in radians
        gamma = 2 * np.pi / 365.0 * (day_of_year - 1 + (hour - 12) / 24.0)

        # Equation of time (minutes)
        eqtime = 229.18 * (
            0.000075 + 0.001868 * np.cos(gamma) - 0.032077 * np.sin(gamma)
            - 0.014615 * np.cos(2 * gamma) - 0.040849 * np.sin(2 * gamma)
        )

        # Solar declination angle (radians)
        decl = (
            0.006918 - 0.399912 * np.cos(gamma) + 0.070257 * np.sin(gamma)
            - 0.006758 * np.cos(2 * gamma) + 0.000907 * np.sin(2 * gamma)
            - 0.002697 * np.cos(3 * gamma) + 0.00148 * np.sin(3 * gamma)
        )

        # Time offset and solar time
        time_offset = eqtime + 4 * self.lon
        true_solar_time = (hour * 60 + time_offset) % 1440
        solar_hour_angle = (true_solar_time / 4.0 - 180) * (np.pi / 180.0)

        # Solar zenith angle calculation
        lat_rad = np.radians(self.lat)
        cos_zenith = np.sin(lat_rad) * np.sin(decl) + np.cos(lat_rad) * np.cos(decl) * np.cos(solar_hour_angle)
        cos_zenith = np.clip(cos_zenith, -1.0, 1.0)
        zenith = np.arccos(cos_zenith)

        solar_elevation_deg = 90.0 - np.degrees(zenith)
        # Bounded at 0 (sun below horizon)
        df["solar_elevation_deg"] = np.maximum(0.0, solar_elevation_deg)
        df["is_daylight"] = (df["solar_elevation_deg"] > 2.0).astype(int)

        return df

    def add_fourier_seasonal_terms(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """
        Generates Fourier harmonic terms for multi-frequency seasonal modeling:
        - Daily cycle: 24 hours
        - Weekly cycle: 168 hours
        - Annual cycle: 8766 hours (365.25 * 24)
        """
        logger.info("Computing multi-seasonal Fourier features...")
        df = df.copy()
        ts = pd.to_datetime(df[time_col])

        # Reference elapsed hours from start
        min_ts = ts.min()
        elapsed_hours = (ts - min_ts).dt.total_seconds() / 3600.0

        # Daily Fourier components
        daily_order = self.fourier_cfg.get("daily_order", 5)
        for k in range(1, daily_order + 1):
            df[f"fourier_d{k}_sin"] = np.sin(2 * np.pi * k * elapsed_hours / 24.0)
            df[f"fourier_d{k}_cos"] = np.cos(2 * np.pi * k * elapsed_hours / 24.0)

        # Weekly Fourier components
        weekly_order = self.fourier_cfg.get("weekly_order", 3)
        for k in range(1, weekly_order + 1):
            df[f"fourier_w{k}_sin"] = np.sin(2 * np.pi * k * elapsed_hours / 168.0)
            df[f"fourier_w{k}_cos"] = np.cos(2 * np.pi * k * elapsed_hours / 168.0)

        # Annual Fourier components
        annual_order = self.fourier_cfg.get("annual_order", 4)
        for k in range(1, annual_order + 1):
            df[f"fourier_a{k}_sin"] = np.sin(2 * np.pi * k * elapsed_hours / 8766.0)
            df[f"fourier_a{k}_cos"] = np.cos(2 * np.pi * k * elapsed_hours / 8766.0)

        return df

    def add_autoregressive_lags_and_ramps(
        self,
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generates historical lagged values and system ramping metrics (delta MW/hour).
        """
        logger.info(f"Generating autoregressive lags {self.lag_hours} and system ramp dynamics...")
        df = df.copy()

        targets = target_columns or ["solar_generation_mw", "wind_generation_mw", "net_load_mw", "curtailment_mw"]

        for col in targets:
            if col in df.columns:
                # Add individual lags
                for lag in self.lag_hours:
                    df[f"{col}_lag_{lag}h"] = df[col].shift(lag)

                # Ramping rates: 1-hour and 3-hour gradients
                df[f"{col}_ramp_1h"] = df[col].diff(1)
                df[f"{col}_ramp_3h"] = df[col].diff(3)

        # Net load specific ratios
        if "net_load_mw" in df.columns and "system_demand_mw" in df.columns:
            # Renewable penetration ratio: (Solar + Wind) / Demand
            renewables = df.get("solar_generation_mw", 0) + df.get("wind_generation_mw", 0)
            df["renewable_penetration_ratio"] = np.clip(renewables / (df["system_demand_mw"] + 1e-5), 0.0, 2.5)

        return df

    def add_rolling_window_aggregations(
        self,
        df: pd.DataFrame,
        target_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Computes rolling statistics (mean, std, min, max) over multiple horizons (e.g. 6h, 24h, 168h).
        Shifted by 1 step to ensure no lookahead leakage.
        """
        logger.info(f"Computing rolling window statistics for windows {self.rolling_windows}h...")
        df = df.copy()

        targets = target_columns or ["solar_generation_mw", "wind_generation_mw", "net_load_mw"]

        for col in targets:
            if col in df.columns:
                for w in self.rolling_windows:
                    # Shift by 1 to prevent lookahead bias
                    shifted = df[col].shift(1)
                    df[f"{col}_roll_{w}h_mean"] = shifted.rolling(window=w, min_periods=max(1, w // 4)).mean()
                    df[f"{col}_roll_{w}h_std"] = shifted.rolling(window=w, min_periods=max(1, w // 4)).std().fillna(0)
                    df[f"{col}_roll_{w}h_min"] = shifted.rolling(window=w, min_periods=max(1, w // 4)).min()
                    df[f"{col}_roll_{w}h_max"] = shifted.rolling(window=w, min_periods=max(1, w // 4)).max()

        # Binary curtailment event label
        if "curtailment_mw" in df.columns:
            df["curtailment_event"] = (df["curtailment_mw"] >= self.curtailment_threshold).astype(int)

        return df

    def build_features(self, df: pd.DataFrame, drop_na: bool = True) -> pd.DataFrame:
        """
        Executes full feature engineering workflow in optimal dependency order.
        """
        logger.info("Executing comprehensive feature engineering pipeline...")
        df = self.add_cyclical_calendar_features(df)
        df = self.add_solar_elevation_angle(df)
        df = self.add_fourier_seasonal_terms(df)
        df = self.add_autoregressive_lags_and_ramps(df)
        df = self.add_rolling_window_aggregations(df)

        if drop_na:
            initial_len = len(df)
            df = df.dropna().reset_index(drop=True)
            logger.info(f"Dropped {initial_len - len(df)} initial rows due to lag initialization. Remaining rows: {len(df)}")

        return df


def main():
    parser = argparse.ArgumentParser(description="Feature Engineering Pipeline CLI")
    parser.add_argument("--input", type=str, default="data/processed/energy_features_clean.parquet")
    parser.add_argument("--output", type=str, default="data/processed/energy_features_engineered.parquet")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    engineer = EnergyFeatureEngineer(config=cfg)

    logger.info(f"Loading preprocessed data from {args.input}...")
    df = pd.read_parquet(args.input)
    featured_df = engineer.build_features(df)

    ensure_dir(os.path.dirname(args.output))
    featured_df.to_parquet(args.output, index=False)
    logger.info(f"Engineered features dataset successfully saved to {args.output} ({featured_df.shape[0]} rows, {featured_df.shape[1]} cols)")


if __name__ == "__main__":
    main()
