"""
scripts/data_collection.py - Automated ingestion for EIA API v2 and NOAA ISD Weather Data.
Includes a physics-informed synthetic generator replicating CAISO 2019-2023 grid dispatch,
duck curve troughs, and renewable curtailment dynamics for reproducibility.
"""

import os
import sys
import argparse
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

import requests
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import setup_logger, load_config, ensure_dir

logger = setup_logger("data_collection")


class EIAClient:
    """
    Client for extracting hourly electric grid generation and demand data
    from the U.S. Energy Information Administration (EIA) v2 REST API.
    """

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = api_key or os.environ.get("EIA_API_KEY", "")
        self.base_url = base_url or "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
        if not self.api_key:
            logger.warning("No EIA API key provided. Set EIA_API_KEY environment variable or pass --api-key.")

    def fetch_hourly_fuel_generation(
        self,
        respondent: str = "CISO",
        start_date: str = "2021-01-01T00",
        end_date: str = "2023-12-31T23",
        fuel_types: Optional[List[str]] = None,
        max_retries: int = 3,
        page_size: int = 5000
    ) -> pd.DataFrame:
        """
        Paginates through EIA v2 API to extract hourly generation records by fuel type.
        """
        if not self.api_key:
            raise ValueError("EIA API key is required to query live EIA endpoints.")

        all_records: List[Dict] = []
        offset = 0

        logger.info(f"Querying EIA API for respondent={respondent} from {start_date} to {end_date}...")

        while True:
            params = {
                "api_key": self.api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": respondent,
                "start": start_date,
                "end": end_date,
                "offset": offset,
                "length": page_size,
                "sort[0][column]": "period",
                "sort[0][direction]": "asc"
            }

            if fuel_types:
                for fuel in fuel_types:
                    params[f"facets[fueltype][]"] = fuel

            response = None
            for attempt in range(1, max_retries + 1):
                try:
                    res = requests.get(self.base_url, params=params, timeout=30)
                    if res.status_code == 200:
                        response = res.json()
                        break
                    elif res.status_code == 429:
                        wait = attempt * 5
                        logger.warning(f"Rate limited (429). Retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        logger.error(f"EIA API error HTTP {res.status_code}: {res.text[:200]}")
                        time.sleep(2)
                except requests.RequestException as exc:
                    logger.error(f"Network error on attempt {attempt}: {exc}")
                    time.sleep(attempt * 2)

            if not response or "response" not in response or "data" not in response["response"]:
                logger.error("Failed to retrieve valid data chunk from EIA API.")
                break

            data_chunk = response["response"]["data"]
            if not data_chunk:
                break

            all_records.extend(data_chunk)
            offset += len(data_chunk)
            total_available = response["response"].get("total", offset)
            logger.info(f"Fetched {offset} / {total_available} records...")

            if offset >= total_available:
                break

        if not all_records:
            logger.warning("Zero records returned from EIA API.")
            return pd.DataFrame()

        df = pd.DataFrame(all_records)
        logger.info(f"EIA extraction complete. Total records: {len(df)}")
        return df


class NOAAClient:
    """
    Client for downloading Integrated Surface Database (ISD) global hourly meteorological data.
    """

    def __init__(self, base_url: str = "https://www.ncei.noaa.gov/data/global-hourly/access"):
        self.base_url = base_url

    def fetch_station_data(self, station_id: str, year: int) -> pd.DataFrame:
        """
        Downloads raw CSV for a given station and year directly from NOAA NCEI archive.
        """
        url = f"{self.base_url}/{year}/{station_id}.csv"
        logger.info(f"Downloading NOAA station {station_id} for year {year} from {url}")

        try:
            res = requests.get(url, timeout=45)
            if res.status_code == 200:
                from io import StringIO
                df = pd.read_csv(StringIO(res.text), low_memory=False)
                logger.info(f"Retrieved {len(df)} records for station {station_id} ({year})")
                return df
            else:
                logger.warning(f"NOAA station {station_id} ({year}) returned HTTP {res.status_code}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed downloading NOAA data for station {station_id}: {e}")
            return pd.DataFrame()


def generate_synthetic_benchmark_dataset(
    start_date: str = "2021-01-01",
    end_date: str = "2023-12-31 23:00",
    output_path: str = "data/raw/eia_caiso_hourly.parquet",
    seed: int = 42
) -> pd.DataFrame:
    """
    Generates a realistic, physics-grounded synthetic dataset modeling CAISO hourly power grid
    dynamics, duck curve evolution, seasonal weather patterns, and solar/wind curtailment events.
    Used for immediate offline reproducibility, training validation, and dashboard demonstrations.
    """
    np.random.seed(seed)
    date_range = pd.date_range(start=start_date, end=end_date, freq="h", tz="America/Los_Angeles")
    n_hours = len(date_range)
    logger.info(f"Generating synthetic CAISO grid dataset ({n_hours} hours)...")

    # Time indicators
    hours = date_range.hour.values
    day_of_year = date_range.dayofyear.values
    day_of_week = date_range.dayofweek.values

    # 1. Base Load Demand (MW) - High in summer afternoon (A/C), lower in spring
    # Base seasonal cycle + diurnal cycle + weekend reduction
    seasonal_demand = 26000 + 7000 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)
    diurnal_demand = 5000 * np.sin(2 * np.pi * (hours - 8) / 24) + 3000 * np.sin(4 * np.pi * (hours - 6) / 24)
    weekend_effect = np.where(day_of_week >= 5, -2500, 0)
    noise_demand = np.random.normal(0, 800, n_hours)
    system_demand_mw = np.clip(seasonal_demand + diurnal_demand + weekend_effect + noise_demand, 16000, 52000)

    # 2. Solar Generation (MW) - Peaks midday, higher in summer/spring, zero at night
    solar_angle = np.maximum(0, np.sin(np.pi * (hours - 6) / 14))
    solar_angle = np.where((hours >= 6) & (hours <= 20), solar_angle, 0.0)
    seasonal_solar_factor = 0.65 + 0.35 * np.sin(2 * np.pi * (day_of_year - 80) / 365.25)

    # Cloud cover attenuation factor (random beta distribution clusters)
    cloud_noise = np.random.beta(5, 1.5, n_hours)
    max_solar_capacity_mw = 18500  # CAISO solar nameplate ~18.5 GW
    solar_generation_mw = max_solar_capacity_mw * (solar_angle ** 1.3) * seasonal_solar_factor * cloud_noise
    solar_generation_mw = np.clip(solar_generation_mw, 0, max_solar_capacity_mw)

    # 3. Wind Generation (MW) - Typically peaks late evening and night (Tehachapi/Solano passes)
    wind_diurnal = 0.4 + 0.3 * np.sin(2 * np.pi * (hours - 18) / 24)
    wind_seasonal = 0.5 + 0.3 * np.sin(2 * np.pi * (day_of_year - 120) / 365.25)
    wind_noise = np.random.weibull(2.0, n_hours) * 0.4
    max_wind_capacity_mw = 8000
    wind_generation_mw = np.clip(max_wind_capacity_mw * wind_diurnal * wind_seasonal * wind_noise, 300, max_wind_capacity_mw)

    # 4. Inflexible Baserush Generation: Hydro (run-of-river spring runoff) + Nuclear (Diablo Canyon 2.2 GW)
    nuclear_mw = np.random.normal(2240, 30, n_hours)
    # Hydro snowmelt peak in April-June
    hydro_peak = np.exp(-((day_of_year - 140) / 40) ** 2)
    hydro_mw = 2500 + 4500 * hydro_peak + np.random.normal(0, 150, n_hours)

    # 5. Net Load = Demand - (Solar + Wind)
    net_load_mw = system_demand_mw - (solar_generation_mw + wind_generation_mw)

    # 6. Curtailment Mechanism: Occurs when (Solar + Wind + Must-Run Hydro + Nuclear) exceeds Demand
    # or Net Load falls below thermal must-run floor (~4,500 MW in CAISO)
    thermal_must_run_floor_mw = 4500.0
    over_generation_potential = thermal_must_run_floor_mw - net_load_mw
    
    # Transmission congestion factor and negative pricing probability
    congestion_factor = np.where(solar_generation_mw > 11000, 1.25, 1.0)
    curtailment_raw = np.maximum(0.0, over_generation_potential * congestion_factor)
    curtailment_event_mask = (curtailment_raw > 150.0) & (hours >= 9) & (hours <= 17)
    
    curtailment_mw = np.where(curtailment_event_mask, curtailment_raw * np.random.uniform(0.75, 1.05, n_hours), 0.0)
    curtailment_mw = np.clip(curtailment_mw, 0, 4800)

    # 7. Ambient Weather Profiles (LAX / Central Valley composite)
    temp_seasonal = 18.0 + 8.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25)
    temp_diurnal = 5.5 * np.sin(2 * np.pi * (hours - 9) / 24)
    temperature_c = temp_seasonal + temp_diurnal + np.random.normal(0, 1.8, n_hours)

    wind_speed_mps = 3.5 + 2.5 * np.sin(2 * np.pi * (hours - 14) / 24) + np.random.normal(0, 1.0, n_hours)
    wind_speed_mps = np.clip(wind_speed_mps, 0.2, 28.0)

    ghi_estimated = np.where(solar_angle > 0, solar_angle * 1050 * cloud_noise, 0.0)

    # Compile DataFrame
    df = pd.DataFrame({
        "timestamp": date_range,
        "system_demand_mw": np.round(system_demand_mw, 1),
        "solar_generation_mw": np.round(solar_generation_mw, 1),
        "wind_generation_mw": np.round(wind_generation_mw, 1),
        "nuclear_mw": np.round(nuclear_mw, 1),
        "hydro_mw": np.round(hydro_mw, 1),
        "net_load_mw": np.round(net_load_mw, 1),
        "curtailment_mw": np.round(curtailment_mw, 1),
        "temperature_c": np.round(temperature_c, 2),
        "wind_speed_mps": np.round(wind_speed_mps, 2),
        "ghi_estimated": np.round(ghi_estimated, 1)
    })

    ensure_dir(os.path.dirname(output_path))
    df.to_parquet(output_path, index=False)
    logger.info(f"Synthetic benchmark dataset successfully saved to {output_path} ({len(df)} rows)")
    return df


def main():
    parser = argparse.ArgumentParser(description="CAISO Energy and NOAA Weather Data Ingestion CLI")
    parser.add_argument("--source", type=str, choices=["eia", "noaa", "benchmark", "all"], default="benchmark",
                        help="Data source to collect from")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--api-key", type=str, default=None, help="EIA API Key")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start Date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default="2023-12-31", help="End Date YYYY-MM-DD")
    parser.add_argument("--output", type=str, default=None, help="Output destination file path")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.source in ["benchmark", "all"]:
        out = args.output or cfg["paths"]["eia_raw_file"]
        generate_synthetic_benchmark_dataset(
            start_date=args.start,
            end_date=args.end + " 23:00",
            output_path=out
        )

    if args.source == "eia":
        client = EIAClient(api_key=args.api_key)
        out = args.output or cfg["paths"]["eia_raw_file"]
        df_eia = client.fetch_hourly_fuel_generation(
            respondent=cfg["data_collection"]["eia"]["default_respondent"],
            start_date=f"{args.start}T00",
            end_date=f"{args.end}T23",
            fuel_types=cfg["data_collection"]["eia"]["fuel_types"]
        )
        if not df_eia.empty:
            ensure_dir(os.path.dirname(out))
            df_eia.to_parquet(out, index=False)
            logger.info(f"Saved EIA raw data to {out}")

    if args.source == "noaa":
        client = NOAAClient()
        stations = cfg["data_collection"]["noaa"]["target_stations"]
        frames = []
        for st in stations:
            df_st = client.fetch_station_data(st["station_id"], 2023)
            if not df_st.empty:
                df_st["station_id"] = st["station_id"]
                frames.append(df_st)
        if frames:
            merged = pd.concat(frames, ignore_index=True)
            out = args.output or cfg["paths"]["noaa_raw_file"]
            ensure_dir(os.path.dirname(out))
            merged.to_parquet(out, index=False)
            logger.info(f"Saved NOAA raw data to {out}")


if __name__ == "__main__":
    main()
