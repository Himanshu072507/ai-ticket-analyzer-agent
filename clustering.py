"""Pure-pandas helpers: match tickets to AI-supplied clusters and build per-cluster trends."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from charts import mini_trend


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
    if not date_col or cluster_df.empty or date_col not in cluster_df.columns:
        return mini_trend(pd.Series(dtype=int))
    dates = pd.to_datetime(cluster_df[date_col], errors="coerce").dropna()
    if dates.empty:
        return mini_trend(pd.Series(dtype=int))
    series = (
        dates.to_frame(name="date").set_index("date").assign(n=1)["n"].resample(granularity).sum()
    )
    return mini_trend(series)
