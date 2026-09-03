# Operational Dashboards

This directory contains specifications and configuration guides for the interactive grid forecasting dashboards.

## Dashboards Overview

### 1. Plotly Dash Operations Room (`app/dash_app.py`)
- **Real-Time 48-Hour Curtailment Forecast**: Dual-axis visualization tracking Net Load (MW) vs. Solar/Wind Over-Generation.
- **Dynamic Alert Engine**: Automated tier triggers (GREEN: Normal, AMBER: Elevated Curtailment Risk > 500 MW, RED: Severe System Over-Supply > 1,500 MW).
- **Duck Curve Real-Time Tracker**: Visualizes belly-of-the-duck deepening across afternoon hours (11:00 to 16:00 PST) with ramping constraints.
- **Economic Dispatch Loss Estimator**: Translates curtailed MWh into lost revenue and wholesale price depression estimates (locational marginal pricing near -$25/MWh).

### How to Run:
```bash
python app/dash_app.py
```
Open your browser at `http://127.0.0.1:8050`.

### 2. Exported HTML Visualizations
Static interactive plots generated from notebooks and backtesting scripts can be viewed directly in standard web browsers.
