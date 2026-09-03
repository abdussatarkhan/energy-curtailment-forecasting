-- ============================================================================
-- Grid Stress Forecaster — Predicting Renewable Energy Curtailment
-- Database: PostgreSQL 14+ / TimescaleDB
-- Schema: DDL Definitions and Advanced Operational Analytics Queries
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. TABLE DEFINITIONS (DDL)
-- ----------------------------------------------------------------------------

CREATE TABLE IF NOT EXISTS caiso_hourly_dispatch (
    timestamp_utc TIMESTAMPTZ PRIMARY KEY,
    system_demand_mw NUMERIC(10, 2) NOT NULL,
    solar_generation_mw NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    wind_generation_mw NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    nuclear_mw NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    hydro_mw NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    thermal_gas_mw NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    net_load_mw NUMERIC(10, 2) GENERATED ALWAYS AS (
        system_demand_mw - (solar_generation_mw + wind_generation_mw)
    ) STORED,
    curtailment_mw NUMERIC(10, 2) NOT NULL DEFAULT 0.0,
    locational_marginal_price_usd NUMERIC(8, 2) DEFAULT 35.00
);

CREATE INDEX IF NOT EXISTS idx_caiso_dispatch_time ON caiso_hourly_dispatch (timestamp_utc DESC);
CREATE INDEX IF NOT EXISTS idx_caiso_curtailment ON caiso_hourly_dispatch (curtailment_mw) WHERE curtailment_mw > 250.0;

CREATE TABLE IF NOT EXISTS noaa_weather_hourly (
    station_id VARCHAR(20) NOT NULL,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    temperature_c NUMERIC(5, 2),
    dew_point_c NUMERIC(5, 2),
    wind_speed_mps NUMERIC(5, 2),
    ghi_estimated_wm2 NUMERIC(7, 2),
    cloud_ceiling_m INTEGER,
    PRIMARY KEY (station_id, timestamp_utc)
);

CREATE TABLE IF NOT EXISTS model_forecast_log (
    forecast_run_id UUID NOT NULL,
    forecast_generated_at TIMESTAMPTZ NOT NULL,
    target_timestamp_utc TIMESTAMPTZ NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    predicted_curtailment_mw NUMERIC(10, 2) NOT NULL,
    lower_bound_mw NUMERIC(10, 2),
    upper_bound_mw NUMERIC(10, 2),
    curtailment_alert_tier VARCHAR(20),
    PRIMARY KEY (forecast_run_id, target_timestamp_utc, model_name)
);

-- ----------------------------------------------------------------------------
-- 2. DUCK CURVE ANALYSIS: Average Net Load & Solar Dispatch by Hour and Season
-- ----------------------------------------------------------------------------
-- Calculates the diurnal depression in net load across California's four seasons.

WITH seasonal_hourly AS (
    SELECT 
        EXTRACT(HOUR FROM timestamp_utc AT TIME ZONE 'America/Los_Angeles') AS hour_pst,
        CASE 
            WHEN EXTRACT(MONTH FROM timestamp_utc) IN (3, 4, 5) THEN '1_Spring'
            WHEN EXTRACT(MONTH FROM timestamp_utc) IN (6, 7, 8) THEN '2_Summer'
            WHEN EXTRACT(MONTH FROM timestamp_utc) IN (9, 10, 11) THEN '3_Fall'
            ELSE '4_Winter'
        END AS season,
        AVG(system_demand_mw) AS avg_demand_mw,
        AVG(solar_generation_mw) AS avg_solar_mw,
        AVG(wind_generation_mw) AS avg_wind_mw,
        AVG(net_load_mw) AS avg_net_load_mw,
        AVG(curtailment_mw) AS avg_curtailment_mw,
        COUNT(*) AS total_sample_hours
    FROM caiso_hourly_dispatch
    GROUP BY 1, 2
)
SELECT 
    hour_pst,
    season,
    ROUND(avg_demand_mw, 1) AS avg_demand_mw,
    ROUND(avg_solar_mw, 1) AS avg_solar_mw,
    ROUND(avg_net_load_mw, 1) AS avg_net_load_mw,
    ROUND(avg_curtailment_mw, 1) AS avg_curtailment_mw,
    -- Belly of the duck indicator: Drop from morning peak (07:00) to midday trough (13:00)
    ROUND(avg_demand_mw - avg_net_load_mw, 1) AS solar_penetration_mw
FROM seasonal_hourly
ORDER BY season, hour_pst;

-- ----------------------------------------------------------------------------
-- 3. THE 3-HOUR EVENING RAMPING CHALLENGE
-- ----------------------------------------------------------------------------
-- Quantifies the steep net load ramp between 16:00 and 20:00 PST when solar PV
-- disappears and residential evening demand surges.

WITH hourly_ramps AS (
    SELECT 
        timestamp_utc,
        timestamp_utc AT TIME ZONE 'America/Los_Angeles' AS timestamp_pst,
        net_load_mw,
        solar_generation_mw,
        system_demand_mw,
        -- Window function: net load 3 hours prior
        LAG(net_load_mw, 3) OVER (ORDER BY timestamp_utc) AS net_load_t_minus_3h,
        net_load_mw - LAG(net_load_mw, 3) OVER (ORDER BY timestamp_utc) AS ramp_3h_mw
    FROM caiso_hourly_dispatch
)
SELECT 
    DATE(timestamp_pst) AS dispatch_date,
    EXTRACT(HOUR FROM timestamp_pst) AS ramp_end_hour_pst,
    ROUND(net_load_t_minus_3h, 1) AS start_net_load_mw,
    ROUND(net_load_mw, 1) AS end_net_load_mw,
    ROUND(ramp_3h_mw, 1) AS upward_ramp_3h_mw,
    CASE 
        WHEN ramp_3h_mw >= 12000 THEN 'CRITICAL RAMP (>12 GW)'
        WHEN ramp_3h_mw >= 8000 THEN 'SEVERE RAMP (8-12 GW)'
        ELSE 'MODERATE RAMP'
    END AS ramp_severity_tier
FROM hourly_ramps
WHERE EXTRACT(HOUR FROM timestamp_pst) BETWEEN 17 AND 21
  AND ramp_3h_mw >= 8000
ORDER BY upward_ramp_3h_mw DESC
LIMIT 50;

-- ----------------------------------------------------------------------------
-- 4. HYDRO SNOWMELT & SOLAR OVER-GENERATION SYNTHESIS
-- ----------------------------------------------------------------------------
-- In Spring (April-June), Sierra Nevada snowmelt causes run-of-river hydro
-- to run baseload, competing with solar and creating negative pricing events.

SELECT 
    DATE_TRUNC('week', timestamp_utc) AS dispatch_week,
    ROUND(AVG(hydro_mw), 1) AS weekly_avg_hydro_mw,
    ROUND(AVG(solar_generation_mw), 1) AS weekly_avg_solar_mw,
    ROUND(MIN(net_load_mw), 1) AS weekly_min_net_load_mw,
    ROUND(SUM(curtailment_mw), 1) AS total_weekly_curtailment_mwh,
    COUNT(CASE WHEN locational_marginal_price_usd < 0 THEN 1 END) AS negative_price_hours,
    ROUND(AVG(locational_marginal_price_usd), 2) AS avg_wholesale_lmp_usd
FROM caiso_hourly_dispatch
GROUP BY 1
HAVING SUM(curtailment_mw) > 500.0
ORDER BY total_weekly_curtailment_mwh DESC
LIMIT 25;

-- ----------------------------------------------------------------------------
-- 5. FORECAST ACCURACY & ECONOMIC IMPACT SCORECARD
-- ----------------------------------------------------------------------------
-- Compares Ensemble model forecasts against actual dispatch, calculating
-- Mean Absolute Error and financial value of avoided green energy waste.

WITH prediction_evaluation AS (
    SELECT 
        f.target_timestamp_utc,
        f.model_name,
        f.predicted_curtailment_mw,
        d.curtailment_mw AS actual_curtailment_mw,
        ABS(f.predicted_curtailment_mw - d.curtailment_mw) AS absolute_error_mw,
        CASE 
            WHEN d.curtailment_mw >= 250.0 AND f.predicted_curtailment_mw >= 250.0 THEN 'TRUE_POSITIVE'
            WHEN d.curtailment_mw < 250.0 AND f.predicted_curtailment_mw >= 250.0 THEN 'FALSE_POSITIVE'
            WHEN d.curtailment_mw >= 250.0 AND f.predicted_curtailment_mw < 250.0 THEN 'FALSE_NEGATIVE'
            ELSE 'TRUE_NEGATIVE'
        END AS event_detection_class
    FROM model_forecast_log f
    JOIN caiso_hourly_dispatch d ON f.target_timestamp_utc = d.timestamp_utc
    WHERE f.target_timestamp_utc >= NOW() - INTERVAL '30 days'
)
SELECT 
    model_name,
    COUNT(*) AS total_forecast_hours,
    ROUND(AVG(absolute_error_mw), 2) AS mae_mw,
    ROUND(SQRT(AVG(POWER(absolute_error_mw, 2))), 2) AS rmse_mw,
    COUNT(CASE WHEN event_detection_class = 'TRUE_POSITIVE' THEN 1 END) AS tp,
    COUNT(CASE WHEN event_detection_class = 'FALSE_POSITIVE' THEN 1 END) AS fp,
    COUNT(CASE WHEN event_detection_class = 'FALSE_NEGATIVE' THEN 1 END) AS fn,
    -- Operational Precision & Recall
    ROUND(
        COUNT(CASE WHEN event_detection_class = 'TRUE_POSITIVE' THEN 1 END)::NUMERIC / 
        NULLIF(COUNT(CASE WHEN event_detection_class IN ('TRUE_POSITIVE', 'FALSE_POSITIVE') THEN 1 END), 0),
        4
    ) AS precision_score,
    ROUND(
        COUNT(CASE WHEN event_detection_class = 'TRUE_POSITIVE' THEN 1 END)::NUMERIC / 
        NULLIF(COUNT(CASE WHEN event_detection_class IN ('TRUE_POSITIVE', 'FALSE_NEGATIVE') THEN 1 END), 0),
        4
    ) AS recall_score
FROM prediction_evaluation
GROUP BY model_name
ORDER BY mae_mw ASC;
