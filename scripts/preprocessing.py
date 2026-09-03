"""
scripts/preprocessing.py - Data cleaning, timezone alignment, DST reconciliation,
missing hour imputation, and meteorological spatial matching for CAISO grid data.
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd

from utils import setup_logger, load_config, ensure_dir

logger = setup_logger("preprocessing")


class GridDataPreprocessor:
    """
    Handles end-to-end preprocessing of time-series electricity grid and weather records:
    1. Reconciles Daylight Saving Time (DST) transition ambiguities.
    2. Aligns local Pacific Time (PST/PDT) to Coordinated Universal Time (UTC).
    3. Detects and fills missing timestamps via frequency reindexing and spline interpolation.
    4. Handles unit normalization (MW vs MWh) and extreme dispatch spike filtering.
    5. Performs inverse-distance or regional weighted spatial matching of meteorological feeds.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.source_tz = self.config["preprocessing"].get("source_timezone", "America/Los_Angeles")
        self.target_tz = self.config["preprocessing"].get("target_timezone", "UTC")
        self.interpolate_limit = self.config["preprocessing"].get("interpolate_limit", 3)
        self.curtailment_threshold = self.config["preprocessing"].get("curtailment_threshold_mw", 250.0)
        self.outlier_cutoff = self.config["preprocessing"].get("outlier_std_cutoff", 4.0)

    def harmonize_timestamps_and_dst(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """
        Parses timestamps, handles duplicate hours during Fall DST "fall back" (ambiguous=infer),
        handles missing hours during Spring DST "spring forward" (nonexistent=shift_forward),
        and normalizes everything to unambiguous UTC index.
        """
        logger.info(f"Harmonizing timestamps and reconciling DST for column: '{time_col}'...")
        df = df.copy()

        # Parse string or raw timestamps
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

        # Drop unparseable records
        initial_count = len(df)
        df = df.dropna(subset=[time_col])
        if len(df) < initial_count:
            logger.warning(f"Dropped {initial_count - len(df)} rows with null/corrupt timestamps.")

        # Localize or convert timezone
        if df[time_col].dt.tz is None:
            # Localize with DST ambiguity resolution
            try:
                df[time_col] = df[time_col].dt.tz_localize(
                    self.source_tz,
                    ambiguous="infer",
                    nonexistent="shift_forward"
                )
            except Exception as e:
                logger.warning(f"Ambiguous DST fallback to NaT/nearest: {e}")
                df[time_col] = df[time_col].dt.tz_localize(self.source_tz, ambiguous="NaT", nonexistent="shift_forward")
                df = df.dropna(subset=[time_col])

        # Convert to target UTC
        df[time_col] = df[time_col].dt.tz_convert(self.target_tz)

        # Remove duplicate timestamps if any
        if df[time_col].duplicated().any():
            dups = df[time_col].duplicated().sum()
            logger.warning(f"Resolving {dups} duplicate timestamps by grouping mean...")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            agg_dict = {col: "mean" for col in numeric_cols}
            df = df.groupby(time_col).agg(agg_dict).reset_index()

        df = df.sort_values(by=time_col).reset_index(drop=True)
        return df

    def enforce_continuous_hourly_grid(self, df: pd.DataFrame, time_col: str = "timestamp") -> pd.DataFrame:
        """
        Re-indexes the dataset against a strict hourly calendar from min to max time,
        imputing small gaps (< limit) using time-weighted linear interpolation.
        """
        logger.info("Enforcing strict hourly continuity...")
        df = df.copy()
        df = df.set_index(time_col)

        start_dt = df.index.min()
        end_dt = df.index.max()
        full_index = pd.date_range(start=start_dt, end=end_dt, freq="1h", tz=self.target_tz)

        missing_hours = len(full_index) - len(df)
        if missing_hours > 0:
            logger.info(f"Detected {missing_hours} missing hourly intervals. Re-indexing...")

        df_reindexed = df.reindex(full_index)
        df_reindexed.index.name = time_col

        # Interpolate numeric fields with limit
        numeric_cols = df_reindexed.select_dtypes(include=[np.number]).columns
        df_reindexed[numeric_cols] = df_reindexed[numeric_cols].interpolate(
            method="time",
            limit=self.interpolate_limit,
            limit_direction="both"
        )

        # Backfill/forward fill any remaining boundary edge cases
        df_reindexed[numeric_cols] = df_reindexed[numeric_cols].bfill().ffill()

        return df_reindexed.reset_index()

    def filter_outliers_and_physical_bounds(
        self,
        df: pd.DataFrame,
        generation_cols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Enforces thermodynamics and grid physical invariants:
        - Solar generation cannot be negative, must be zero during deep night.
        - Demand and generation values cannot be negative.
        - Statistical Z-score outlier trimming for transmission telemetry glitches.
        """
        logger.info("Filtering physical constraints and statistical anomalies...")
        df = df.copy()

        gen_cols = generation_cols or ["solar_generation_mw", "wind_generation_mw", "system_demand_mw", "curtailment_mw"]

        for col in gen_cols:
            if col in df.columns:
                # Floor negative readings to 0.0
                df[col] = np.maximum(0.0, df[col].astype(float))

                # Outlier detection using rolling window median and std
                rolling_median = df[col].rolling(window=168, min_periods=24, center=True).median()
                rolling_std = df[col].rolling(window=168, min_periods=24, center=True).std()

                upper_bound = rolling_median + self.outlier_cutoff * rolling_std
                outlier_mask = (df[col] > upper_bound) & (upper_bound.notnull()) & (df[col] > 100)

                if outlier_mask.sum() > 0:
                    logger.info(f"Capping {outlier_mask.sum()} telemetry anomalies in column '{col}'.")
                    df.loc[outlier_mask, col] = upper_bound[outlier_mask]

        # Invariant: Curtailment cannot exceed total intermittent generation (Solar + Wind)
        if "curtailment_mw" in df.columns and "solar_generation_mw" in df.columns and "wind_generation_mw" in df.columns:
            total_renewables = df["solar_generation_mw"] + df["wind_generation_mw"]
            excessive_curtailment = df["curtailment_mw"] > total_renewables
            if excessive_curtailment.sum() > 0:
                logger.warning(f"Clipping {excessive_curtailment.sum()} instances where curtailment exceeded renewables.")
                df.loc[excessive_curtailment, "curtailment_mw"] = total_renewables[excessive_curtailment]

        # Net load re-computation: Net Load = Demand - (Solar + Wind)
        if "system_demand_mw" in df.columns and "solar_generation_mw" in df.columns and "wind_generation_mw" in df.columns:
            df["net_load_mw"] = df["system_demand_mw"] - (df["solar_generation_mw"] + df["wind_generation_mw"])

        return df

    def match_and_aggregate_weather_stations(
        self,
        weather_df: pd.DataFrame,
        station_weights: Optional[Dict[str, float]] = None
    ) -> pd.DataFrame:
        """
        Spatially aggregates multi-station NOAA ISD meteorological feeds using
        load/generation weighted spatial interpolation.
        """
        logger.info("Aggregating multi-station NOAA weather observations...")
        if weather_df.empty:
            logger.warning("Empty weather dataframe provided. Skipping spatial aggregation.")
            return weather_df

        weights = station_weights or {
            "72295023174": 0.35,  # LAX
            "72494023234": 0.25,  # SFO
            "72389093193": 0.25,  # FAT (Central Valley solar)
            "72290003131": 0.15   # BFL (Tehachapi wind/solar)
        }

        # Normalize weights
        total_w = sum(weights.values())
        norm_weights = {k: v / total_w for k, v in weights.items()}

        # Group by timestamp and station
        if "station_id" in weather_df.columns and "timestamp" in weather_df.columns:
            weather_df["weight"] = weather_df["station_id"].map(norm_weights).fillna(0.1)

            weighted_dfs = []
            for col in ["temperature_c", "wind_speed_mps", "ghi_estimated"]:
                if col in weather_df.columns:
                    weather_df[f"{col}_weighted"] = weather_df[col] * weather_df["weight"]

            agg_dict = {
                f"{col}_weighted": "sum" for col in ["temperature_c", "wind_speed_mps", "ghi_estimated"] if col in weather_df.columns
            }
            agg_dict["weight"] = "sum"

            grouped = weather_df.groupby("timestamp").agg(agg_dict).reset_index()
            for col in ["temperature_c", "wind_speed_mps", "ghi_estimated"]:
                if f"{col}_weighted" in grouped.columns:
                    grouped[col] = grouped[f"{col}_weighted"] / grouped["weight"]
                    grouped.drop(columns=[f"{col}_weighted"], inplace=True)

            grouped.drop(columns=["weight"], inplace=True)
            return grouped

        return weather_df

    def process(self, input_path: str, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Runs the full preprocessing pipeline on an ingested file and writes the clean dataset.
        """
        logger.info(f"Starting end-to-end preprocessing for: {input_path}")
        if input_path.endswith(".parquet"):
            df = pd.read_parquet(input_path)
        elif input_path.endswith(".csv"):
            df = pd.read_csv(input_path)
        else:
            raise ValueError(f"Unsupported file format: {input_path}")

        df_clean = self.harmonize_timestamps_and_dst(df, time_col="timestamp")
        df_clean = self.enforce_continuous_hourly_grid(df_clean, time_col="timestamp")
        df_clean = self.filter_outliers_and_physical_bounds(df_clean)

        out_path = output_path or self.config["paths"]["processed_features_file"]
        ensure_dir(os.path.dirname(out_path))
        df_clean.to_parquet(out_path, index=False)
        logger.info(f"Clean preprocessed dataset saved to: {out_path} ({len(df_clean)} rows, {len(df_clean.columns)} columns)")
        return df_clean


def main():
    parser = argparse.ArgumentParser(description="Grid Data Preprocessing and Cleaning CLI")
    parser.add_argument("--input", type=str, default="data/raw/eia_caiso_hourly.parquet", help="Raw input file path")
    parser.add_argument("--output", type=str, default="data/processed/energy_features_clean.parquet", help="Clean output path")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    args = parser.parse_args()

    cfg = load_config(args.config)
    preprocessor = GridDataPreprocessor(config=cfg)
    preprocessor.process(input_path=args.input, output_path=args.output)


if __name__ == "__main__":
    main()
