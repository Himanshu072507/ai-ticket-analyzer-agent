"""Plotly chart builders for the ticket analyzer Streamlit app."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

_LAYOUT = dict(template="simple_white", height=400, margin=dict(l=10, r=10, t=40, b=10))


def _empty_fig(title: str) -> go.Figure:
    """Return a blank figure with a 'No data available' annotation."""
    fig = go.Figure()
    fig.update_layout(title=title, **_LAYOUT)
    fig.add_annotation(text="No data available", xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False, font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def category_bar(df: pd.DataFrame, category_col: str) -> go.Figure:
    """Horizontal bar of top 15 categories by ticket count."""
    title = "Tickets by Category"
    if df.empty or category_col not in df.columns or df[category_col].dropna().empty:
        return _empty_fig(title)
    counts = df[category_col].dropna().value_counts().head(15).sort_values(ascending=True)
    fig = px.bar(x=counts.values, y=counts.index, orientation="h",
                 labels={"x": "Tickets", "y": "Category"}, title=title)
    fig.update_layout(**_LAYOUT)
    return fig


def department_bar(df: pd.DataFrame, department_col: str) -> go.Figure:
    """Vertical bar of top 10 departments by ticket count."""
    title = "Tickets by Department"
    if df.empty or department_col not in df.columns or df[department_col].dropna().empty:
        return _empty_fig(title)
    counts = df[department_col].dropna().value_counts().head(10)
    fig = px.bar(x=counts.index, y=counts.values,
                 labels={"x": "Department", "y": "Tickets"}, title=title)
    fig.update_layout(**_LAYOUT)
    return fig


def volume_over_time(df: pd.DataFrame, date_col: str, granularity: str = "D") -> go.Figure:
    """Line chart of ticket volume resampled by granularity (D/W/M)."""
    title = "Ticket Volume Over Time"
    if df.empty or date_col not in df.columns:
        return _empty_fig(title)
    dates = pd.to_datetime(df[date_col], errors="coerce").dropna()
    if dates.empty:
        return _empty_fig(title)
    series = dates.to_frame(name="date").set_index("date").assign(n=1)["n"].resample(granularity).sum()
    fig = px.line(x=series.index, y=series.values,
                  labels={"x": "Date", "y": "Tickets"}, title=title)
    fig.update_layout(**_LAYOUT)
    return fig


def status_donut(df: pd.DataFrame, status_col: str) -> go.Figure:
    """Donut chart of ticket status breakdown."""
    title = "Status Breakdown"
    if df.empty or status_col not in df.columns or df[status_col].dropna().empty:
        return _empty_fig(title)
    counts = df[status_col].dropna().value_counts()
    fig = go.Figure(data=[go.Pie(labels=counts.index, values=counts.values, hole=0.4)])
    fig.update_layout(title=title, **_LAYOUT)
    return fig


def category_dept_heatmap(df: pd.DataFrame, category_col: str, department_col: str) -> go.Figure:
    """Heatmap of top 10 categories x top 10 departments."""
    title = "Category x Department"
    if (df.empty or category_col not in df.columns or department_col not in df.columns
            or df[[category_col, department_col]].dropna().empty):
        return _empty_fig(title)
    sub = df[[category_col, department_col]].dropna()
    top_cats = sub[category_col].value_counts().head(10).index
    top_depts = sub[department_col].value_counts().head(10).index
    sub = sub[sub[category_col].isin(top_cats) & sub[department_col].isin(top_depts)]
    if sub.empty:
        return _empty_fig(title)
    matrix = pd.crosstab(sub[category_col], sub[department_col]).reindex(
        index=top_cats, columns=top_depts, fill_value=0)
    fig = px.imshow(matrix, color_continuous_scale="Blues", text_auto=True,
                    labels=dict(x="Department", y="Category", color="Tickets"), title=title)
    fig.update_layout(**_LAYOUT)
    return fig
