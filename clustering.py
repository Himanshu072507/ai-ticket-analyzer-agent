"""Pure-pandas helpers: match tickets to AI-supplied clusters and build per-cluster trends."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


def match_tickets_to_clusters(
    df: pd.DataFrame, desc_col: str, clusters: list[dict]
) -> dict[str, pd.DataFrame]:
    """Return {cluster_name: filtered_df} via case-insensitive substring OR-match on keywords."""
    descriptions = df[desc_col].astype(str).str.lower()
    out: dict[str, pd.DataFrame] = {}
    for cluster in clusters:
        keywords = [k.lower() for k in cluster.get("keywords", []) if k]
        if not keywords:
            out[cluster["name"]] = df.iloc[0:0]
            continue
        mask = pd.Series(False, index=df.index)
        for kw in keywords:
            mask |= descriptions.str.contains(kw, regex=False, na=False)
        out[cluster["name"]] = df[mask]
    return out


def cluster_trend(
    cluster_df: pd.DataFrame, date_col: str | None, granularity: str = "D"
) -> go.Figure:
    """Small line chart of cluster volume over time. Empty figure if no usable dates."""
    fig = go.Figure()
    fig.update_layout(
        template="simple_white", height=200, margin=dict(l=10, r=10, t=30, b=10)
    )
    if not date_col or cluster_df.empty or date_col not in cluster_df.columns:
        fig.add_annotation(text="No trend available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig
    dates = pd.to_datetime(cluster_df[date_col], errors="coerce").dropna()
    if dates.empty:
        fig.add_annotation(text="No trend available", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig
    series = (
        dates.to_frame(name="date").set_index("date").assign(n=1)["n"].resample(granularity).sum()
    )
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Tickets")
    return fig
