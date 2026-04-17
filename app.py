"""
Module 7: Dashboard & Reporting
=================================
Interactive Dash/Plotly dashboard for strategy monitoring.
Run: python dashboard/app.py
Opens at http://localhost:8050
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os, sys, json

# ── Synthetic demo data (used when live backtest hasn't been run) ─────────────

def generate_demo_data(n_days: int = 756, seed: int = 42) -> dict:
    """Generate realistic synthetic backtest results for dashboard demo."""
    np.random.seed(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    # Momentum strategy: moderate Sharpe with realistic drawdowns
    mu_m, sig_m = 0.0006, 0.0090
    raw_m = np.random.randn(n_days) * sig_m + mu_m
    # Add regime changes
    raw_m[100:150] -= 0.003  # mini drawdown
    raw_m[350:400] -= 0.005  # larger drawdown (2022-style)
    raw_m[500:560] += 0.002  # recovery

    # Risk parity strategy: lower vol, lower return
    raw_rp = np.random.randn(n_days) * 0.0055 + 0.0003
    raw_rp[350:400] -= 0.004

    # Benchmark (S&P 500 proxy)
    raw_bm = np.random.randn(n_days) * 0.010 + 0.0004
    raw_bm[350:400] -= 0.006

    mom_curve = 1_000_000 * np.cumprod(1 + raw_m)
    rp_curve  = 1_000_000 * np.cumprod(1 + raw_rp)
    bm_curve  = 1_000_000 * np.cumprod(1 + raw_bm)

    def max_dd_series(curve):
        roll_max = np.maximum.accumulate(curve)
        return (curve - roll_max) / roll_max

    # Monthly returns
    monthly_dates = pd.bdate_range("2022-01-03", periods=n_days)[::21]
    monthly_rets  = pd.Series(raw_m).rolling(21).sum()[::21].values[:len(monthly_dates)]

    # Positions heatmap (20 tickers, weekly)
    tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","JNJ","V",
               "PG","XOM","UNH","MA","HD","CVX","MRK","ABBV","PFE","BAC"]
    n_weeks = 52
    week_dates = pd.bdate_range("2022-01-03", periods=n_weeks*5, freq="W-FRI")[:n_weeks]
    positions = np.random.randn(n_weeks, 20) * 0.05
    positions[positions < -0.06] = 0
    positions[positions > 0.08] = 0

    return {
        "dates":       dates,
        "mom_curve":   mom_curve,
        "rp_curve":    rp_curve,
        "bm_curve":    bm_curve,
        "mom_returns": pd.Series(raw_m, index=dates),
        "rp_returns":  pd.Series(raw_rp, index=dates),
        "mom_dd":      max_dd_series(mom_curve),
        "rp_dd":       max_dd_series(rp_curve),
        "bm_dd":       max_dd_series(bm_curve),
        "monthly_dates": monthly_dates[:len(monthly_rets)],
        "monthly_rets":  monthly_rets,
        "tickers":     tickers,
        "week_dates":  week_dates,
        "positions":   positions,
    }


D = generate_demo_data()

COLORS = {
    "momentum":   "#1a73e8",
    "risk_parity":"#34a853",
    "benchmark":  "#9aa0a6",
    "loss":       "#ea4335",
    "gain":       "#34a853",
    "bg":         "#0f1117",
    "card":       "#1a1d27",
    "border":     "#2d3142",
    "text":       "#e8eaed",
    "muted":      "#9aa0a6",
}

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'IBM Plex Mono', monospace", size=11, color=COLORS["text"]),
    margin=dict(l=50, r=20, t=40, b=40),
    hovermode="x unified",
    legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
    xaxis=dict(gridcolor=COLORS["border"], showgrid=True, zeroline=False),
    yaxis=dict(gridcolor=COLORS["border"], showgrid=True, zeroline=False),
)


# ── Chart Builders ────────────────────────────────────────────────────────────

def build_equity_curve():
    fig = go.Figure()
    norm = lambda x: x / x[0]

    fig.add_trace(go.Scatter(
        x=D["dates"], y=norm(D["mom_curve"]),
        name="Momentum L/S", line=dict(color=COLORS["momentum"], width=2),
        fill="tozeroy", fillcolor="rgba(26,115,232,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=D["dates"], y=norm(D["rp_curve"]),
        name="Risk Parity", line=dict(color=COLORS["risk_parity"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=D["dates"], y=norm(D["bm_curve"]),
        name="S&P 500 (BM)", line=dict(color=COLORS["benchmark"], width=1.5, dash="dot"),
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Equity Curves (Normalised)", font=dict(size=13)),
        yaxis=dict(**LAYOUT["yaxis"], tickformat=".2f"),
        height=300,
    )
    return fig


def build_drawdown_chart():
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=D["dates"], y=D["mom_dd"] * 100,
        name="Momentum", fill="tozeroy",
        line=dict(color=COLORS["loss"], width=1.5),
        fillcolor="rgba(234,67,53,0.15)",
    ))
    fig.add_trace(go.Scatter(
        x=D["dates"], y=D["rp_dd"] * 100,
        name="Risk Parity", line=dict(color=COLORS["risk_parity"], width=1.5),
        fillcolor="rgba(52,168,83,0.08)", fill="tozeroy",
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Drawdown (%)", font=dict(size=13)),
        yaxis=dict(**LAYOUT["yaxis"], ticksuffix="%"),
        height=220,
    )
    return fig


def build_monthly_returns():
    colors = [COLORS["gain"] if r > 0 else COLORS["loss"] for r in D["monthly_rets"]]
    fig = go.Figure(go.Bar(
        x=D["monthly_dates"],
        y=D["monthly_rets"] * 100,
        marker_color=colors,
        name="Monthly Return",
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Monthly Returns — Momentum Strategy", font=dict(size=13)),
        yaxis=dict(**LAYOUT["yaxis"], ticksuffix="%"),
        height=220,
        showlegend=False,
    )
    return fig


def build_position_heatmap():
    fig = go.Figure(go.Heatmap(
        z=D["positions"].T * 100,
        x=[str(d.date()) for d in D["week_dates"]],
        y=D["tickers"],
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="Weight %", ticksuffix="%",
                      tickfont=dict(size=10, color=COLORS["text"])),
        hoverongaps=False,
        xgap=1, ygap=1,
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Position Weights Heatmap (Weekly)", font=dict(size=13)),
        height=400,
        xaxis=dict(tickangle=45, nticks=12, gridcolor="transparent"),
        yaxis=dict(autorange="reversed", gridcolor="transparent"),
    )
    return fig


def build_return_distribution():
    r = D["mom_returns"].values * 100
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=r, nbinsx=60,
        marker_color=COLORS["momentum"],
        opacity=0.8, name="Daily Returns",
    ))
    var95 = np.percentile(r, 5)
    fig.add_vline(x=var95, line_color=COLORS["loss"], line_dash="dash",
                  annotation_text=f"VaR 95%: {var95:.2f}%",
                  annotation_font_color=COLORS["loss"])
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Return Distribution", font=dict(size=13)),
        xaxis=dict(**LAYOUT["xaxis"], ticksuffix="%"),
        height=250,
        showlegend=False,
    )
    return fig


def build_rolling_sharpe():
    r = D["mom_returns"]
    rolling_sr = (r.rolling(63).mean() / r.rolling(63).std() * np.sqrt(252)).fillna(0)
    fig = go.Figure()
    fig.add_hline(y=0, line_color=COLORS["border"])
    fig.add_hline(y=1, line_color=COLORS["gain"], line_dash="dot", opacity=0.5)
    fig.add_trace(go.Scatter(
        x=D["dates"], y=rolling_sr,
        name="Rolling Sharpe (63d)",
        line=dict(color=COLORS["momentum"], width=2),
        fill="tozeroy",
        fillcolor="rgba(26,115,232,0.08)",
    ))
    fig.update_layout(
        **LAYOUT,
        title=dict(text="Rolling Sharpe Ratio (63-day)", font=dict(size=13)),
        height=220,
    )
    return fig


def build_risk_dashboard():
    """4-panel risk summary chart."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("VaR 95%", "Realised Volatility", "Gross Exposure", "Net Exposure"),
        vertical_spacing=0.15,
    )
    r = D["mom_returns"]
    var_series  = r.rolling(21).apply(lambda x: -np.percentile(x, 5)) * 100
    vol_series  = r.rolling(21).std() * np.sqrt(252) * 100
    gross_exp   = pd.Series(np.random.uniform(0.8, 1.5, len(D["dates"])), index=D["dates"])
    net_exp     = pd.Series(np.random.uniform(-0.3, 0.3, len(D["dates"])), index=D["dates"])

    kw = dict(line_width=1.5)
    fig.add_trace(go.Scatter(x=D["dates"], y=var_series,
                  line=dict(color=COLORS["loss"], **kw), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=D["dates"], y=vol_series,
                  line=dict(color=COLORS["momentum"], **kw), showlegend=False), row=1, col=2)
    fig.add_trace(go.Scatter(x=D["dates"], y=gross_exp,
                  line=dict(color=COLORS["risk_parity"], **kw), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=D["dates"], y=net_exp,
                  line=dict(color=COLORS["benchmark"], **kw), showlegend=False), row=2, col=2)

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="'IBM Plex Mono', monospace", size=10, color=COLORS["text"]),
        margin=dict(l=50, r=20, t=50, b=40),
        height=380,
        hovermode="x unified",
    )
    for ax in fig.layout:
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(gridcolor=COLORS["border"], showgrid=True, zeroline=False)
    return fig


# ── Metrics Cards Data ────────────────────────────────────────────────────────

def compute_summary_metrics() -> dict:
    r = D["mom_returns"]
    ann_ret = ((1 + r).prod() ** (252 / len(r)) - 1)
    ann_vol = r.std() * np.sqrt(252)
    sharpe  = (r.mean() - 0.05/252) / r.std() * np.sqrt(252)
    mdd     = min((np.cumprod(1+r) / np.maximum.accumulate(np.cumprod(1+r)) - 1))

    return {
        "Annual Return":   f"{ann_ret:+.2%}",
        "Annual Vol":      f"{ann_vol:.2%}",
        "Sharpe Ratio":    f"{sharpe:.3f}",
        "Max Drawdown":    f"{mdd:.2%}",
        "VaR 95%":         f"{-np.percentile(r, 5):.4f}",
        "Win Rate":        f"{(r > 0).mean():.2%}",
    }


# ── Dash App ──────────────────────────────────────────────────────────────────

def create_app():
    try:
        import dash
        from dash import dcc, html
        import dash_bootstrap_components as dbc
    except ImportError:
        print("[ERROR] Install dash and dash-bootstrap-components: pip install dash dash-bootstrap-components")
        return None

    app = dash.Dash(
        __name__,
        external_stylesheets=[
            dbc.themes.DARKLY,
            "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap",
        ],
        title="QuantSys Dashboard",
    )

    metrics = compute_summary_metrics()

    CARD_STYLE = {
        "background": COLORS["card"],
        "border": f"1px solid {COLORS['border']}",
        "borderRadius": "8px",
        "padding": "16px 20px",
        "marginBottom": "12px",
    }

    metric_cards = dbc.Row([
        dbc.Col(html.Div([
            html.P(k, style={"fontSize": "11px", "color": COLORS["muted"],
                             "margin": "0 0 4px 0", "letterSpacing": "0.05em"}),
            html.H4(v, style={"margin": 0, "fontFamily": "'IBM Plex Mono'",
                               "color": COLORS["gain"] if "+" in v else
                               (COLORS["loss"] if v.startswith("-") else COLORS["text"])}),
        ], style=CARD_STYLE), width=2)
        for k, v in metrics.items()
    ], className="g-2")

    app.layout = html.Div([
        # Header
        html.Div([
            html.Div([
                html.H2("QUANTSYS", style={
                    "fontFamily": "'IBM Plex Mono'", "fontWeight": "600",
                    "color": COLORS["text"], "letterSpacing": "0.15em",
                    "margin": 0, "fontSize": "20px",
                }),
                html.Span("LIVE DASHBOARD", style={
                    "fontSize": "10px", "color": COLORS["momentum"],
                    "letterSpacing": "0.2em", "marginLeft": "12px",
                    "border": f"1px solid {COLORS['momentum']}",
                    "padding": "2px 8px", "borderRadius": "3px",
                }),
            ], style={"display": "flex", "alignItems": "center"}),
            html.P("Quantitative Trading System  ·  IIT Roorkee × Zetheta 2026",
                   style={"color": COLORS["muted"], "fontSize": "11px",
                          "margin": "4px 0 0 0", "letterSpacing": "0.05em"}),
        ], style={
            "background": COLORS["card"],
            "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "16px 24px",
            "marginBottom": "20px",
        }),

        # Main content
        html.Div([
            # Metric cards
            metric_cards,

            # Equity curve
            dbc.Row([
                dbc.Col(dcc.Graph(figure=build_equity_curve(), config={"displayModeBar": False}), width=12)
            ], style={"marginTop": "8px"}),

            # Drawdown + Rolling Sharpe
            dbc.Row([
                dbc.Col(dcc.Graph(figure=build_drawdown_chart(), config={"displayModeBar": False}), width=6),
                dbc.Col(dcc.Graph(figure=build_rolling_sharpe(), config={"displayModeBar": False}), width=6),
            ], style={"marginTop": "8px"}),

            # Monthly returns + Distribution
            dbc.Row([
                dbc.Col(dcc.Graph(figure=build_monthly_returns(), config={"displayModeBar": False}), width=7),
                dbc.Col(dcc.Graph(figure=build_return_distribution(), config={"displayModeBar": False}), width=5),
            ], style={"marginTop": "8px"}),

            # Position heatmap
            dbc.Row([
                dbc.Col(dcc.Graph(figure=build_position_heatmap(), config={"displayModeBar": False}), width=12)
            ], style={"marginTop": "8px"}),

            # Risk panel
            dbc.Row([
                dbc.Col(dcc.Graph(figure=build_risk_dashboard(), config={"displayModeBar": False}), width=12)
            ], style={"marginTop": "8px"}),

        ], style={"padding": "0 24px 40px"}),

    ], style={
        "background": COLORS["bg"],
        "minHeight": "100vh",
        "fontFamily": "'IBM Plex Mono', monospace",
    })

    return app


if __name__ == "__main__":
    app = create_app()
    if app:
        print("\n[DASHBOARD] Starting at http://localhost:8050")
        app.run(debug=True, host="0.0.0.0", port=8050)
    else:
        print("Install dependencies: pip install dash dash-bootstrap-components")
