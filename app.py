"""Streamlit UI for the AI Ticket Analyzer Agent."""
from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, analyze, list_ollama_models
from charts import (
    category_bar,
    category_dept_heatmap,
    department_bar,
    status_donut,
    volume_over_time,
)

load_dotenv()

st.set_page_config(page_title="AI Ticket Analyzer Agent", page_icon=":bar_chart:", layout="wide")

FIELD_SYNONYMS = {
    "category": ["category", "issue type", "type", "issue category", "ticket type"],
    "description": ["description", "details", "explanation", "summary", "comment", "issue"],
    "department": ["department", "team", "dept", "business unit", "sbu"],
    "date": ["date", "created", "created at", "created on", "opened", "reported on", "timestamp"],
    "priority": ["priority", "severity"],
    "status": ["status", "state"],
}


def auto_map(columns: list[str]) -> dict[str, str | None]:
    """Pick the best column for each logical field by matching against synonyms."""
    lowered = {c: c.strip().lower() for c in columns}
    mapping: dict[str, str | None] = {}
    for field, synonyms in FIELD_SYNONYMS.items():
        match = next(
            (orig for orig, low in lowered.items() if any(s == low or s in low for s in synonyms)),
            None,
        )
        mapping[field] = match
    return mapping


def read_upload(uploaded) -> tuple[pd.DataFrame | None, list[str]]:
    """Return (df, sheet_names). sheet_names is empty for CSV, the list of sheets for xlsx."""
    name = uploaded.name.lower()
    data = uploaded.getvalue()
    if name.endswith(".csv"):
        return pd.read_csv(io.BytesIO(data)), []
    xls = pd.ExcelFile(io.BytesIO(data))
    return None, xls.sheet_names


def read_sheet(uploaded, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(uploaded.getvalue()), sheet_name=sheet_name)


def kpi_strip(df: pd.DataFrame, m: dict) -> None:
    cols = st.columns(4)
    cols[0].metric("Total tickets", len(df))

    if m.get("status"):
        open_like = df[m["status"]].astype(str).str.lower().isin(["open", "in progress", "new"])
        cols[1].metric("Open / In Progress", f"{open_like.mean() * 100:.0f}%")
    else:
        cols[1].metric("Open %", "-")

    if m.get("date"):
        dates = pd.to_datetime(df[m["date"]], errors="coerce").dropna()
        if not dates.empty:
            age_days = (pd.Timestamp.now().normalize() - dates).dt.days
            cols[2].metric("Avg age (days)", f"{age_days.mean():.0f}")
        else:
            cols[2].metric("Avg age (days)", "-")
    else:
        cols[2].metric("Avg age (days)", "-")

    if m.get("category"):
        top = df[m["category"]].value_counts()
        cols[3].metric("Top category", top.index[0] if not top.empty else "-")
    else:
        cols[3].metric("Top category", "-")


def render_insights(insights: dict) -> None:
    st.subheader("AI Summary")

    st.markdown("**Top themes**")
    for theme in insights.get("top_themes", []):
        st.markdown(f"- **{theme['title']}** — {theme['detail']}")

    st.markdown("**Priority ranking**")
    for i, item in enumerate(insights.get("priority_ranking", []), 1):
        with st.expander(f"{i}. {item['issue']} — {item['impact']}", expanded=i == 1):
            st.markdown(f"**Suggested owner:** {item['suggested_owner']}")
            st.markdown(f"**Rationale:** {item['rationale']}")

    spot = insights.get("department_spotlight") or {}
    if spot:
        st.markdown(f"**Department spotlight — {spot.get('department', '')}**")
        st.info(spot.get("finding", ""))

    st.markdown("**Recommendations**")
    for rec in insights.get("recommendations", []):
        st.markdown(f"- **{rec['action']}** → {rec['expected_outcome']}")


def insights_markdown(insights: dict) -> str:
    out = ["# AI Ticket Analyzer Agent — Report", f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    out.append("## Top themes")
    for t in insights.get("top_themes", []):
        out.append(f"- **{t['title']}** — {t['detail']}")
    out.append("\n## Priority ranking")
    for i, item in enumerate(insights.get("priority_ranking", []), 1):
        out.append(f"{i}. **{item['issue']}** ({item['impact']})")
        out.append(f"   - Owner: {item['suggested_owner']}")
        out.append(f"   - Rationale: {item['rationale']}")
    spot = insights.get("department_spotlight") or {}
    if spot:
        out.append(f"\n## Department spotlight — {spot.get('department', '')}")
        out.append(spot.get("finding", ""))
    out.append("\n## Recommendations")
    for rec in insights.get("recommendations", []):
        out.append(f"- **{rec['action']}** — {rec['expected_outcome']}")
    return "\n".join(out)


st.title("AI Ticket Analyzer Agent")
st.caption("Upload an Excel or CSV of support tickets to get charts, KPIs, and prioritized AI recommendations.")

with st.sidebar:
    st.subheader("Settings")
    provider = st.radio(
        "AI provider",
        ["Gemini (cloud)", "Ollama (local)"],
        index=0 if st.session_state.get("provider", "gemini") == "gemini" else 1,
        help="Gemini needs an API key. Ollama runs fully offline on your laptop.",
    )
    provider_id = "gemini" if provider.startswith("Gemini") else "ollama"
    st.session_state["provider"] = provider_id

    api_key = ""
    ollama_model = DEFAULT_OLLAMA_MODEL
    ollama_host = DEFAULT_OLLAMA_HOST

    if provider_id == "gemini":
        env_key = os.getenv("GEMINI_API_KEY", "")
        api_key = st.text_input(
            "Google Gemini API key",
            value=st.session_state.get("api_key", env_key),
            type="password",
            help="Paste your Gemini API key. Stored only in this session. Get one free at aistudio.google.com/app/apikey.",
        )
        st.session_state["api_key"] = api_key
        if env_key and api_key == env_key:
            st.caption("Loaded from `.env`.")
        elif not api_key:
            st.caption("AI insights disabled until a key is provided.")
    else:
        ollama_host = st.text_input("Ollama host", value=DEFAULT_OLLAMA_HOST)
        models = list_ollama_models(ollama_host)
        if models:
            default_idx = models.index(DEFAULT_OLLAMA_MODEL) if DEFAULT_OLLAMA_MODEL in models else 0
            ollama_model = st.selectbox("Model", models, index=default_idx)
        else:
            st.warning(
                "No Ollama models found. Pull one first, e.g. `ollama pull llama3.1:8b`."
            )
            ollama_model = st.text_input("Model name", value=DEFAULT_OLLAMA_MODEL)

uploaded = st.file_uploader("Upload tickets", type=["xlsx", "csv"])
if not uploaded:
    st.info("Tip: try `sample_tickets.xlsx` bundled with the project.")
    st.stop()

df, sheets = read_upload(uploaded)
if df is None:
    sheet = st.selectbox("Pick a sheet", sheets) if len(sheets) > 1 else sheets[0]
    df = read_sheet(uploaded, sheet)

if df.empty:
    st.error("File is empty.")
    st.stop()

st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns.")

st.subheader("Column mapping")
st.caption("Auto-detected below. Override any field if the guess is wrong.")
detected = auto_map(list(df.columns))
options = [None] + list(df.columns)
mapping_cols = st.columns(3)
mapping: dict[str, str | None] = {}
for i, field in enumerate(FIELD_SYNONYMS):
    with mapping_cols[i % 3]:
        default = detected.get(field)
        mapping[field] = st.selectbox(
            field.capitalize(), options, index=options.index(default) if default in options else 0,
            format_func=lambda x: "— none —" if x is None else x,
        )

if not mapping["category"] or not mapping["description"]:
    st.error("Category and Description are required. Pick them above to continue.")
    st.stop()

st.divider()
kpi_strip(df, mapping)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(category_bar(df, mapping["category"]), use_container_width=True)
with c2:
    if mapping["department"]:
        st.plotly_chart(department_bar(df, mapping["department"]), use_container_width=True)

if mapping["date"]:
    gran_label = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=0, horizontal=True)
    gran = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[gran_label]
    st.plotly_chart(volume_over_time(df, mapping["date"], gran), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    if mapping["status"]:
        st.plotly_chart(status_donut(df, mapping["status"]), use_container_width=True)
with c4:
    if mapping["department"]:
        st.plotly_chart(
            category_dept_heatmap(df, mapping["category"], mapping["department"]),
            use_container_width=True,
        )

st.divider()

cache_key = (uploaded.name, tuple(sorted(mapping.items())), provider_id, ollama_model if provider_id == "ollama" else None)
if st.session_state.get("insights_key") != cache_key:
    st.session_state.pop("insights", None)

provider_ready = (provider_id == "gemini" and bool(api_key)) or (
    provider_id == "ollama" and bool(ollama_model)
)

if "insights" not in st.session_state:
    if not provider_ready:
        if provider_id == "gemini":
            st.info("Paste a Gemini API key in the sidebar (or set `GEMINI_API_KEY` in `.env`) to generate AI insights.")
        else:
            st.info("Select an Ollama model in the sidebar to generate AI insights.")
    elif st.button("Generate AI insights", type="primary"):
        spinner_label = (
            f"Ollama ({ollama_model}) is analyzing your tickets..."
            if provider_id == "ollama"
            else "Gemini is analyzing your tickets..."
        )
        with st.spinner(spinner_label):
            try:
                st.session_state["insights"] = analyze(
                    df, mapping,
                    provider=provider_id,
                    api_key=api_key,
                    ollama_model=ollama_model,
                    ollama_host=ollama_host,
                )
                st.session_state["insights_key"] = cache_key
            except Exception as e:
                st.error(str(e))

if "insights" in st.session_state:
    render_insights(st.session_state["insights"])
    md = insights_markdown(st.session_state["insights"])
    st.download_button(
        "Download summary as Markdown", md,
        file_name=f"ticket-insights-{datetime.now():%Y%m%d-%H%M}.md", mime="text/markdown",
    )
