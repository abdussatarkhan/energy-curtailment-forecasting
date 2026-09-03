"""
app/dash_app.py - Real-Time Grid Stress Forecaster & Curtailment Operations Room.
Interactive Plotly Dash Application displaying 48-Hour Forecasts, Uncertainty Bounds,
Duck Curve Tracking, Operational Risk Tiers, and Economic Loss Estimations.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, dash_table

# Setup paths
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "scripts"))

# Import fallback generator if data not present
try:
    from scripts.utils import load_config
    config = load_config()
except Exception:
    config = {
        "dashboard": {"host": "127.0.0.1", "port": 8050, "debug": False, "title": "Grid Stress Forecaster"},
        "preprocessing": {"curtailment_threshold_mw": 250.0}
    }


def load_or_generate_forecast_data() -> pd.DataFrame:
    """
    Loads latest forecast predictions Parquet file or synthesizes a 48-hour
    operational sequence with solar, wind, net load, and curtailment alerts.
    """
    pred_path = project_root / "data/processed/forecast_predictions.parquet"
    if os.path.exists(pred_path):
        df = pd.read_parquet(pred_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df

    # Generate synthetic 48-hour forward pass for standalone dashboard demonstration
    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    timestamps = [now + timedelta(hours=i) for i in range(48)]
    hours = np.array([ts.hour for ts in timestamps])

    solar_profile = np.maximum(0, np.sin(np.pi * (hours - 6) / 14))
    solar_profile = np.where((hours >= 6) & (hours <= 20), solar_profile, 0.0)
    solar_mw = solar_profile * 16500.0 + np.random.normal(0, 300, 48)
    solar_mw = np.clip(solar_mw, 0, 17500)

    wind_mw = 3200 + 1500 * np.sin(2 * np.pi * (hours - 18) / 24) + np.random.normal(0, 200, 48)
    wind_mw = np.clip(wind_mw, 400, 7500)

    demand_mw = 25000 + 5000 * np.sin(2 * np.pi * (hours - 8) / 24) + np.random.normal(0, 400, 48)
    net_load_mw = demand_mw - (solar_mw + wind_mw)

    # Curtailment simulation
    over_gen = np.maximum(0, 5200.0 - net_load_mw)
    curt_raw = np.where((hours >= 10) & (hours <= 16), over_gen * 1.3, 0.0)
    ensemble_mw = np.clip(curt_raw, 0, 4200)

    prophet_mw = np.clip(ensemble_mw * np.random.uniform(0.9, 1.1, 48), 0, 4500)
    lstm_mw = np.clip(ensemble_mw * np.random.uniform(0.92, 1.08, 48), 0, 4500)
    lower_mw = np.maximum(0, ensemble_mw - 80)
    upper_mw = ensemble_mw + 110

    alert_tiers = []
    for val in ensemble_mw:
        if val < 250.0:
            alert_tiers.append("LOW (Green)")
        elif val < 1000.0:
            alert_tiers.append("MODERATE (Amber)")
        else:
            alert_tiers.append("CRITICAL (Red)")

    df = pd.DataFrame({
        "timestamp": timestamps,
        "actual_curtailment_mw": ensemble_mw * np.random.uniform(0.95, 1.05, 48),
        "prophet_mw": np.round(prophet_mw, 1),
        "lstm_mw": np.round(lstm_mw, 1),
        "ensemble_mw": np.round(ensemble_mw, 1),
        "lower_bound_mw": np.round(lower_mw, 1),
        "upper_bound_mw": np.round(upper_mw, 1),
        "net_load_mw": np.round(net_load_mw, 1),
        "solar_generation_mw": np.round(solar_mw, 1),
        "wind_generation_mw": np.round(wind_mw, 1),
        "temperature_c": np.round(21.0 + 6.0 * np.sin(2 * np.pi * (hours - 9) / 24), 1),
        "curtailment_alert": alert_tiers,
        "curtailment_event_prob": np.round(np.clip(ensemble_mw / 750.0, 0.0, 1.0), 3)
    })
    return df


# Initialize Dash App
app = dash.Dash(
    __name__,
    title=config.get("dashboard", {}).get("title", "Grid Stress Forecaster"),
    update_title=None,
    suppress_callback_exceptions=True
)
server = app.server

# Data store
df_init = load_or_generate_forecast_data()

# App Layout
app.layout = html.Div(className="dashboard-container", children=[

    # 1. Header Banner
    html.Div(className="dashboard-header", children=[
        html.Div(className="dashboard-title", children=[
            html.H1("⚡ Grid Stress Forecaster — CAISO Renewable Curtailment"),
            html.P(f"California ISO Balancing Authority | Automated 48-Hour Operational Horizon | System Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC")
        ]),
        html.Div(id="system-status-badge", children=[
            html.Span("SYSTEM NOMINAL", className="status-badge alert-green")
        ])
    ]),

    # 2. Key Performance Metric Cards
    html.Div(className="kpi-grid", children=[
        html.Div(className="kpi-card", children=[
            html.Div("Peak Curtailment Risk", className="kpi-label"),
            html.Div(id="kpi-peak-curtailment", className="kpi-value", children="0 MW"),
            html.Div(id="kpi-peak-time", className="kpi-sub", children="Next 48 Hours")
        ]),
        html.Div(className="kpi-card", children=[
            html.Div("Total Curtailed Volume", className="kpi-label"),
            html.Div(id="kpi-total-volume", className="kpi-value", children="0 MWh"),
            html.Div("Lost Renewable Clean Energy", className="kpi-sub")
        ]),
        html.Div(className="kpi-card", children=[
            html.Div("Economic Value at Risk", className="kpi-label"),
            html.Div(id="kpi-economic-loss", className="kpi-value", children="$0"),
            html.Div("Wholesale Price Depressed (-$25/MWh)", className="kpi-sub")
        ]),
        html.Div(className="kpi-card", children=[
            html.Div("Minimum Net Load Belly", className="kpi-label"),
            html.Div(id="kpi-min-netload", className="kpi-value", children="0 MW"),
            html.Div("Critical Inflexible Generation Floor", className="kpi-sub")
        ])
    ]),

    # 3. Interactive Controls Panel
    html.Div(className="control-panel", children=[
        html.Div(className="control-item", children=[
            html.Label("Forecasting Model Architecture:"),
            dcc.Dropdown(
                id="model-selector",
                options=[
                    {"label": "Stacked Ensemble (Prophet + LSTM)", "value": "ensemble_mw"},
                    {"label": "Facebook Prophet (Bayesian GAM)", "value": "prophet_mw"},
                    {"label": "Multivariate Deep LSTM", "value": "lstm_mw"}
                ],
                value="ensemble_mw",
                clearable=False,
                style={"backgroundColor": "#0f172a", "color": "#000"}
            )
        ]),
        html.Div(className="control-item", children=[
            html.Label("Curtailment Alert Threshold (MW):"),
            dcc.Slider(
                id="threshold-slider",
                min=100,
                max=1500,
                step=50,
                value=250,
                marks={100: "100", 250: "250 (Default)", 500: "500", 1000: "1000", 1500: "1500"},
                tooltip={"placement": "bottom", "always_visible": False}
            )
        ]),
        html.Div(className="control-item", children=[
            html.Label("Operational Forecast Window:"),
            dcc.RadioItems(
                id="horizon-filter",
                options=[
                    {"label": " Next 24 Hours  ", "value": 24},
                    {"label": " Full 48 Hours  ", "value": 48}
                ],
                value=48,
                inline=True,
                style={"color": "#f8fafc", "marginTop": "6px"}
            )
        ])
    ]),

    # 4. Primary Forecast Chart (48-Hour Ahead with Uncertainty Bounds)
    html.Div(className="graph-card", children=[
        html.Div("📈 48-Hour Ahead Renewable Curtailment Forecast & Confidence Interval", className="graph-title"),
        dcc.Graph(id="curtailment-forecast-chart", config={"displayModeBar": False})
    ]),

    # 5. Row with Two Visualizations: The Duck Curve and Fuel Mix Ramping
    html.Div(className="graphs-row", children=[
        html.Div(className="graph-card", children=[
            html.Div("🦆 CAISO Duck Curve: Real-Time Net Load vs. Solar Production", className="graph-title"),
            dcc.Graph(id="duck-curve-chart", config={"displayModeBar": False})
        ]),
        html.Div(className="graph-card", children=[
            html.Div("🔋 Renewable Generation Dispatch & Ramping Dynamics", className="graph-title"),
            dcc.Graph(id="generation-mix-chart", config={"displayModeBar": False})
        ])
    ]),

    # 6. High-Risk Curtailment Event Log Table
    html.Div(className="graph-card", children=[
        html.Div("🚨 Scheduled High-Risk Curtailment Incidents (Actions Required)", className="graph-title"),
        html.Div(id="alert-table-container")
    ]),

    # Auto-refresh Interval Component (5 minutes)
    dcc.Interval(id="interval-component", interval=300 * 1000, n_intervals=0)
])


# Callback: Update Dashboard Graphics and KPIs
@app.callback(
    [
        Output("system-status-badge", "children"),
        Output("kpi-peak-curtailment", "children"),
        Output("kpi-peak-time", "children"),
        Output("kpi-total-volume", "children"),
        Output("kpi-economic-loss", "children"),
        Output("kpi-min-netload", "children"),
        Output("curtailment-forecast-chart", "figure"),
        Output("duck-curve-chart", "figure"),
        Output("generation-mix-chart", "figure"),
        Output("alert-table-container", "children")
    ],
    [
        Input("model-selector", "value"),
        Input("threshold-slider", "value"),
        Input("horizon-filter", "value"),
        Input("interval-component", "n_intervals")
    ]
)
def update_dashboard(selected_model, threshold_val, horizon_hours, n_intervals):
    df = load_or_generate_forecast_data().iloc[:horizon_hours].copy()

    # 1. KPIs
    pred_series = df[selected_model]
    peak_mw = float(pred_series.max())
    peak_idx = pred_series.idxmax()
    peak_time_str = df.loc[peak_idx, "timestamp"].strftime("%b %d, %H:%M UTC")
    total_mwh = float(pred_series.sum())
    economic_loss = total_mwh * 25.0  # Assumes $25/MWh average value of lost green energy
    min_netload = float(df["net_load_mw"].min()) if "net_load_mw" in df.columns else 4200.0

    # Status Badge Logic
    if peak_mw >= 1000.0:
        badge = html.Span("🔴 CRITICAL OVER-SUPPLY ALERT", className="status-badge alert-red")
    elif peak_mw >= threshold_val:
        badge = html.Span("🟡 ELEVATED CURTAILMENT RISK", className="status-badge alert-amber")
    else:
        badge = html.Span("🟢 SYSTEM BALANCED", className="status-badge alert-green")

    # 2. Main Forecast Chart
    fig_forecast = go.Figure()

    # Confidence Interval Fill
    if "lower_bound_mw" in df.columns and "upper_bound_mw" in df.columns and selected_model == "ensemble_mw":
        fig_forecast.add_trace(go.Scatter(
            x=list(df["timestamp"]) + list(df["timestamp"])[::-1],
            y=list(df["upper_bound_mw"]) + list(df["lower_bound_mw"])[::-1],
            fill="toself",
            fillcolor="rgba(56, 189, 248, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            showlegend=True,
            name="90% Prediction Interval"
        ))

    # Forecast Trace
    fig_forecast.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df[selected_model],
        mode="lines+markers",
        line=dict(color="#38bdf8", width=3),
        marker=dict(size=5, color="#38bdf8"),
        name="Model Curtailment Forecast (MW)"
    ))

    # Observed Actuals (if present)
    if "actual_curtailment_mw" in df.columns:
        fig_forecast.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["actual_curtailment_mw"],
            mode="lines",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            name="Observed Baseline (MW)"
        ))

    # Alert Threshold Line
    fig_forecast.add_hline(
        y=threshold_val,
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Alert Threshold ({threshold_val} MW)",
        annotation_position="bottom right",
        annotation_font_color="#ef4444"
    )

    fig_forecast.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(gridcolor="#334155", showgrid=True),
        yaxis=dict(gridcolor="#334155", title="Megawatts (MW)", showgrid=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 3. Duck Curve Chart
    fig_duck = go.Figure()
    fig_duck.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["net_load_mw"],
        mode="lines",
        line=dict(color="#818cf8", width=3),
        name="Net Load (MW)"
    ))
    fig_duck.add_hline(
        y=4500,
        line_dash="dash",
        line_color="#64748b",
        annotation_text="Thermal Must-Run Floor (4,500 MW)",
        annotation_font_color="#94a3b8"
    )
    fig_duck.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="Net Load (MW)")
    )

    # 4. Generation Mix Chart
    fig_gen = go.Figure()
    fig_gen.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["solar_generation_mw"],
        mode="lines",
        stackgroup="one",
        fillcolor="rgba(245, 158, 11, 0.6)",
        line=dict(color="#f59e0b", width=1.5),
        name="Solar PV (MW)"
    ))
    fig_gen.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["wind_generation_mw"],
        mode="lines",
        stackgroup="one",
        fillcolor="rgba(16, 185, 129, 0.6)",
        line=dict(color="#10b981", width=1.5),
        name="Wind (MW)"
    ))
    fig_gen.update_layout(
        template="plotly_dark",
        paper_bgcolor="#1e293b",
        plot_bgcolor="#1e293b",
        font=dict(family="Inter, sans-serif", color="#f8fafc"),
        margin=dict(l=40, r=40, t=30, b=40),
        xaxis=dict(gridcolor="#334155"),
        yaxis=dict(gridcolor="#334155", title="Renewable MW")
    )

    # 5. Alert Incidents Table
    alert_df = df[df[selected_model] >= threshold_val][["timestamp", selected_model, "net_load_mw", "curtailment_alert", "curtailment_event_prob"]].copy()
    if not alert_df.empty:
        alert_df["timestamp"] = alert_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M UTC")
        alert_df.rename(columns={
            "timestamp": "Event Timestamp",
            selected_model: "Curtailed Volume (MW)",
            "net_load_mw": "Net Load (MW)",
            "curtailment_alert": "Risk Tier",
            "curtailment_event_prob": "Event Probability"
        }, inplace=True)
        table = dash_table.DataTable(
            data=alert_df.to_dict("records"),
            columns=[{"name": col, "id": col} for col in alert_df.columns],
            style_header={
                "backgroundColor": "#0f172a",
                "color": "#38bdf8",
                "fontWeight": "600",
                "border": "1px solid #334155"
            },
            style_cell={
                "backgroundColor": "#1e293b",
                "color": "#f8fafc",
                "border": "1px solid #334155",
                "fontFamily": "Inter, sans-serif",
                "padding": "10px"
            },
            style_data_conditional=[
                {
                    "if": {"filter_query": "{Risk Tier} contains 'CRITICAL'"},
                    "color": "#ef4444",
                    "fontWeight": "bold"
                },
                {
                    "if": {"filter_query": "{Risk Tier} contains 'MODERATE'"},
                    "color": "#f59e0b"
                }
            ],
            page_size=6
        )
    else:
        table = html.Div(
            f"No hours exceed the current alert threshold ({threshold_val} MW) within the selected horizon.",
            style={"color": "#10b981", "padding": "20px", "textAlign": "center"}
        )

    return (
        badge,
        f"{peak_mw:,.0f} MW",
        f"At {peak_time_str}",
        f"{total_mwh:,.0f} MWh",
        f"${economic_loss:,.0f}",
        f"{min_netload:,.0f} MW",
        fig_forecast,
        fig_duck,
        fig_gen,
        table
    )


if __name__ == "__main__":
    host = config.get("dashboard", {}).get("host", "127.0.0.1")
    port = config.get("dashboard", {}).get("port", 8050)
    debug = config.get("dashboard", {}).get("debug", False)
    print(f"Starting Grid Stress Forecaster Dashboard at http://{host}:{port}")
    app.run_server(host=host, port=port, debug=debug)
