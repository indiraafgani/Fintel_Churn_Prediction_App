"""
FINTel — Visualization Module
Plotly chart builders. Segment colours: Low=green, Mid=yellow, High=red.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict

C_NAVY      = "#0F1D3D"
C_PRIMARY   = "#1A3462"
C_SECONDARY = "#476996"
C_SOFT      = "#9AADC2"
C_BG        = "#EBECEF"
C_WHITE     = "#FFFFFF"
C_DANGER    = "#E74C3C"
C_WARNING   = "#F39C12"
C_SUCCESS   = "#27AE60"

SEGMENT_COLORS = {"High": C_DANGER, "Mid": C_WARNING, "Low": C_SUCCESS}

_BASE = dict(
    paper_bgcolor=C_WHITE,
    plot_bgcolor=C_WHITE,
    font=dict(family="Inter, sans-serif", color=C_NAVY),
)


def churn_score_gauge(score: int, segment: str) -> go.Figure:
    color = SEGMENT_COLORS.get(segment, C_SECONDARY)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 44, "color": color, "family": "Inter"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickvals": [0, 33, 67, 100],
                "ticktext": ["0", "33", "67", "100"],
                "tickfont": {"size": 10, "color": C_SOFT},
                "tickwidth": 1, "tickcolor": C_SOFT,
            },
            "bar": {"color": color, "thickness": 0.28},
            "bgcolor": C_WHITE, "borderwidth": 0,
            "steps": [
                {"range": [0, 33],   "color": "#ECFDF5"},
                {"range": [33, 67],  "color": "#FEF9C3"},
                {"range": [67, 100], "color": "#FEF2F2"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "thickness": 0.85, "value": score},
        },
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=30, b=10), **_BASE)
    return fig


def shap_individual_bar(shap_display: List[Dict], title: str = "Individual SHAP (Top 8)") -> go.Figure:
    names   = [d["name"]  for d in shap_display]
    values  = [d["value"] for d in shap_display]
    dirs    = [d["direction"] for d in shap_display]
    colors  = [C_DANGER if d == "up" else C_SUCCESS for d in dirs]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1], orientation="h",
        marker_color=colors[::-1], marker_line_width=0,
        text=[f"{v:+.4f}" for v in values[::-1]], textposition="outside",
        textfont={"size": 10},
        hovertemplate="%{y}: %{x:+.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=12, color=C_PRIMARY), x=0),
        height=320, margin=dict(l=10, r=80, t=40, b=16), **_BASE,
        xaxis=dict(showgrid=True, gridcolor=C_BG, zeroline=True,
                   zerolinecolor=C_SOFT, zerolinewidth=1,
                   tickfont=dict(size=10, color=C_SOFT)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=C_NAVY)),
    )
    return fig


def shap_global_bar(global_importance: List[tuple], top_n: int = 10) -> go.Figure:
    items  = global_importance[:top_n]
    names  = [i[0] for i in items]
    values = [i[1] for i in items]

    fig = go.Figure(go.Bar(
        x=values[::-1], y=names[::-1], orientation="h",
        marker_color=C_SECONDARY, marker_line_width=0,
        text=[f"{v:.4f}" for v in values[::-1]], textposition="outside",
        textfont={"size": 10},
        hovertemplate="%{y}: %{x:.4f}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Global Feature Importance (mean |SHAP|)", font=dict(size=12, color=C_PRIMARY), x=0),
        height=340, margin=dict(l=10, r=80, t=40, b=16), **_BASE,
        xaxis=dict(showgrid=True, gridcolor=C_BG, zeroline=False,
                   tickfont=dict(size=10, color=C_SOFT)),
        yaxis=dict(showgrid=False, tickfont=dict(size=11, color=C_NAVY)),
    )
    return fig


def churn_distribution_donut(n_churn: int, n_no_churn: int) -> go.Figure:
    total = n_churn + n_no_churn
    pct   = int(n_churn / total * 100) if total > 0 else 0

    fig = go.Figure(go.Pie(
        labels=["Churn", "No Churn"], values=[n_churn, n_no_churn],
        hole=0.56, marker_colors=[C_DANGER, C_SUCCESS],
        textfont_size=12,
        hovertemplate="%{label}: %{value} (%{percent})<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Churn vs No Churn", font=dict(size=13, color=C_PRIMARY), x=0.5),
        height=280, showlegend=True,
        margin=dict(l=16, r=16, t=40, b=16),
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font=dict(size=11)),
        **_BASE,
    )
    fig.add_annotation(
        text=f"{pct}%", x=0.5, y=0.5, showarrow=False,
        font=dict(size=24, color=C_DANGER, family="Inter"),
    )
    return fig


def segment_bar(df: pd.DataFrame) -> go.Figure:
    if "Churn_Segment" not in df.columns:
        return go.Figure()
    counts = df["Churn_Segment"].value_counts().reindex(["High", "Mid", "Low"], fill_value=0)
    colors = [SEGMENT_COLORS[s] for s in counts.index]

    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=colors, marker_line_width=0,
        text=counts.values, textposition="outside",
        hovertemplate="%{x}: %{y}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text="Churn Segment Distribution", font=dict(size=13, color=C_PRIMARY), x=0.5),
        height=280, margin=dict(l=16, r=16, t=40, b=16), **_BASE,
        xaxis=dict(showgrid=False, tickfont=dict(size=12, color=C_NAVY)),
        yaxis=dict(showgrid=True, gridcolor=C_BG, tickfont=dict(size=10, color=C_SOFT)),
    )
    return fig


def probability_histogram(df: pd.DataFrame) -> go.Figure:
    if "Churn_Probability" not in df.columns:
        return go.Figure()

    churn    = df[df["Predicted_Label"] == "Churn"]["Churn_Probability"]
    no_churn = df[df["Predicted_Label"] == "No Churn"]["Churn_Probability"]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=no_churn, name="No Churn", nbinsx=20,
        marker_color=C_SUCCESS, opacity=0.75,
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra>No Churn</extra>",
    ))
    fig.add_trace(go.Histogram(
        x=churn, name="Churn", nbinsx=20,
        marker_color=C_DANGER, opacity=0.75,
        hovertemplate="Prob: %{x:.2f}<br>Count: %{y}<extra>Churn</extra>",
    ))
    fig.update_layout(
        barmode="overlay",
        title=dict(text="Churn Probability Distribution", font=dict(size=13, color=C_PRIMARY), x=0.5),
        height=280, margin=dict(l=16, r=16, t=40, b=16), **_BASE,
        xaxis=dict(title="Probability", showgrid=True, gridcolor=C_BG,
                   tickfont=dict(size=10, color=C_SOFT)),
        yaxis=dict(title="Count", showgrid=True, gridcolor=C_BG,
                   tickfont=dict(size=10, color=C_SOFT)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def churn_score_histogram(df: pd.DataFrame) -> go.Figure:
    if "Churn_Score" not in df.columns:
        return go.Figure()

    fig = go.Figure(go.Histogram(
        x=df["Churn_Score"], nbinsx=20, marker_color=C_SECONDARY, opacity=0.85,
        hovertemplate="Score: %{x}<br>Count: %{y}<extra></extra>",
    ))
    for x, color, label in [(33, C_SUCCESS, "Low|Mid"), (67, C_DANGER, "Mid|High")]:
        fig.add_vline(x=x, line_dash="dash", line_color=color, line_width=1.5,
                      annotation_text=label, annotation_font_size=10,
                      annotation_font_color=color)
    fig.update_layout(
        title=dict(text="Churn Score Distribution", font=dict(size=13, color=C_PRIMARY), x=0.5),
        height=280, margin=dict(l=16, r=16, t=40, b=16), **_BASE,
        xaxis=dict(title="Churn Score", showgrid=True, gridcolor=C_BG,
                   tickfont=dict(size=10, color=C_SOFT)),
        yaxis=dict(title="Count", showgrid=True, gridcolor=C_BG,
                   tickfont=dict(size=10, color=C_SOFT)),
    )
    return fig
