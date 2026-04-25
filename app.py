"""Streamlit UI for the AI Ticket Analyzer Agent."""
from __future__ import annotations

import io
import os
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analyzer import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    analyze_category,
    analyze_overview,
    list_ollama_models,
)
from clustering import cluster_trend, match_tickets_to_clusters
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
    "ticket_id": ["ticket id", "ticket number", "ticket no", "ticket #", "incident id",
                  "incident number", "incident", "ref", "reference", "id", "case id",
                  "case number"],
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
        # First, try exact matches
        match = next(
            (orig for orig, low in lowered.items() if any(s == low for s in synonyms)),
            None,
        )
        # If no exact match, try substring matches
        if match is None:
            match = next(
                (orig for orig, low in lowered.items() if any(s in low for s in synonyms)),
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


def _trim_words(text: str, max_words: int) -> str:
    """Truncate text to max_words, appending an ellipsis if cut."""
    words = (text or "").split()
    if len(words) <= max_words:
        return text or ""
    return " ".join(words[:max_words]) + "…"


def drilldown_markdown(overview: dict, clusters_by_category: dict[str, dict]) -> str:
    """Serialize the headline + any explored category clusters into markdown."""
    out = ["# AI Ticket Analyzer Agent — Drill-Down Report",
           f"_Generated {datetime.now():%Y-%m-%d %H:%M}_", ""]
    out.append("## Headline")
    out.append(overview.get("headline", ""))
    out.append("\n## Categories (ranked)")
    for c in overview.get("ranked_categories", []):
        out.append(f"- **{c['name']}** — {c['count']} tickets ({c['percent']:.1f}%)")
    if clusters_by_category:
        out.append("\n## Explored categories")
        for cat, payload in clusters_by_category.items():
            out.append(f"\n### {cat}")
            for cl in payload.get("clusters", []):
                out.append(f"- **{cl['name']}** — {cl['summary']}")
                out.append(f"  - Root cause: {cl['root_cause']}")
                out.append(f"  - Owner: {cl['suggested_owner']}")
                out.append(f"  - Suggested fix: {cl['suggested_fix']}")
    return "\n".join(out)


def render_drilldown(
    df: pd.DataFrame,
    mapping: dict,
    provider_id: str,
    api_key: str,
    ollama_model: str,
    ollama_host: str,
) -> None:
    overview = st.session_state.get("overview")
    if not overview:
        return

    st.subheader("AI Drill-Down")
    st.markdown(f"**Headline:** {_trim_words(overview['headline'], 25)}")

    cat_col = mapping["category"]
    counts = df[cat_col].dropna().astype(str).value_counts()
    total = int(counts.sum())
    categories = [
        {"name": str(name), "count": int(n), "percent": (n / total) * 100}
        for name, n in counts.items()
    ]
    if not categories:
        st.info("No categories in the data.")
        return

    if "selected_category" not in st.session_state:
        st.session_state["selected_category"] = categories[0]["name"]
    st.session_state.setdefault("category_clusters", {})

    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown("**Categories**")
        for c in categories:
            label = f"{c['name']} — {c['count']} ({c['percent']:.0f}%)"
            if st.button(label, key=f"cat-btn-{c['name']}", use_container_width=True):
                st.session_state["selected_category"] = c["name"]

    selected = st.session_state["selected_category"]

    with right:
        st.markdown(f"**Top issues — {selected}**")
        cat_col, desc_col = mapping["category"], mapping["description"]
        cat_df = df[df[cat_col].astype(str) == str(selected)]

        id_col = mapping.get("ticket_id")

        def _fmt(row) -> str:
            desc = str(row[desc_col])[:200]
            if id_col and id_col in row.index and pd.notna(row[id_col]):
                return f"- **{row[id_col]}** — {desc}"
            return f"- {desc}"

        if len(cat_df) < 5:
            st.info("Too few tickets to cluster — showing raw descriptions.")
            for _, row in cat_df.dropna(subset=[desc_col]).head(10).iterrows():
                st.markdown(_fmt(row))
            return

        cached = st.session_state["category_clusters"].get(selected)
        if not cached:
            with st.spinner(f"Clustering '{selected}'..."):
                try:
                    cached = analyze_category(
                        df, mapping, selected,
                        provider=provider_id, api_key=api_key,
                        ollama_model=ollama_model, ollama_host=ollama_host,
                    )
                    st.session_state["category_clusters"][selected] = cached
                except Exception as e:
                    st.error(f"Could not cluster '{selected}': {e}")
                    if st.button("Retry", key=f"retry-{selected}"):
                        st.rerun()
                    return

        clusters = cached.get("clusters", [])
        if not clusters:
            st.warning("No clusters returned — try a different category.")
            return
        matches = match_tickets_to_clusters(cat_df, desc_col, clusters)

        for cl in clusters:
            sub = matches.get(cl["name"], cat_df.iloc[0:0])
            count_label = f"{len(sub)} tickets" if len(sub) else "— no keyword match"
            summary = _trim_words(cl["summary"], 12)
            with st.expander(f"{cl['name']} · {count_label} — {summary}"):
                if mapping.get("date") and not sub.empty:
                    st.plotly_chart(
                        cluster_trend(sub, mapping["date"], "D"),
                        use_container_width=True,
                    )
                st.markdown(f"**Root cause:** {_trim_words(cl['root_cause'], 20)}")
                fix_cols = st.columns([1, 2])
                with fix_cols[0]:
                    st.markdown(f"**Owner:** {_trim_words(cl['suggested_owner'], 4)}")
                with fix_cols[1]:
                    st.markdown(f"**Fix:** {_trim_words(cl['suggested_fix'], 15)}")
                if not sub.empty:
                    st.markdown("**Example tickets:**")
                    for _, row in sub.dropna(subset=[desc_col]).head(10).iterrows():
                        st.markdown(_fmt(row))
                else:
                    st.caption("No tickets matched the AI-supplied keywords.")


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

if st.session_state.get("loaded_file") != uploaded.name:
    st.session_state.pop("dept_ms", None)
    st.session_state["loaded_file"] = uploaded.name

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

st.subheader("Filters")
filter_cols = st.columns(2)

selected_departments: list[str] = []
if mapping["department"]:
    with filter_cols[0]:
        all_depts = sorted(df[mapping["department"]].dropna().astype(str).unique().tolist())
        ALL_LABEL = "All departments"
        options = [ALL_LABEL] + all_depts
        if "dept_ms" not in st.session_state:
            st.session_state["dept_ms"] = [ALL_LABEL]
        selection = st.multiselect(
            "Departments", options, key="dept_ms",
            placeholder="Choose departments",
            help="Default is All departments. Remove the chip to pick specific ones.",
        )
        selected_departments = (
            all_depts if ALL_LABEL in selection else [s for s in selection if s != ALL_LABEL]
        )
else:
    with filter_cols[0]:
        st.caption("Department filter unavailable — no department column mapped.")

date_range: tuple = ()
parsed_dates = None
if mapping["date"]:
    parsed_dates = pd.to_datetime(df[mapping["date"]], errors="coerce")
    with filter_cols[1]:
        valid_dates = parsed_dates.dropna()
        if not valid_dates.empty:
            min_d, max_d = valid_dates.min().date(), valid_dates.max().date()
            date_range = st.date_input(
                "Date range", value=(min_d, max_d), min_value=min_d, max_value=max_d,
            )
        else:
            st.caption("Date filter unavailable — date column has no parseable dates.")
else:
    with filter_cols[1]:
        st.caption("Date filter unavailable — no date column mapped.")

filtered_df = df
if mapping["department"] and selected_departments:
    filtered_df = filtered_df[filtered_df[mapping["department"]].astype(str).isin(selected_departments)]
if mapping["date"] and isinstance(date_range, tuple) and len(date_range) == 2 and parsed_dates is not None:
    start, end = date_range
    filtered_df = filtered_df[
        (parsed_dates.loc[filtered_df.index].dt.date >= start)
        & (parsed_dates.loc[filtered_df.index].dt.date <= end)
    ]

if len(filtered_df) != len(df):
    st.caption(f"Showing {len(filtered_df):,} of {len(df):,} rows after filters.")
if filtered_df.empty:
    st.warning("No tickets match the current filters.")
    st.stop()

st.divider()
kpi_strip(filtered_df, mapping)

c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(category_bar(filtered_df, mapping["category"]), use_container_width=True)
with c2:
    if mapping["department"]:
        st.plotly_chart(department_bar(filtered_df, mapping["department"]), use_container_width=True)

if mapping["date"]:
    gran_label = st.radio("Granularity", ["Daily", "Weekly", "Monthly"], index=0, horizontal=True)
    gran = {"Daily": "D", "Weekly": "W", "Monthly": "ME"}[gran_label]
    st.plotly_chart(volume_over_time(filtered_df, mapping["date"], gran), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    if mapping["status"]:
        st.plotly_chart(status_donut(filtered_df, mapping["status"]), use_container_width=True)
with c4:
    if mapping["department"]:
        st.plotly_chart(
            category_dept_heatmap(filtered_df, mapping["category"], mapping["department"]),
            use_container_width=True,
        )

st.divider()

filter_signature = (
    tuple(selected_departments),
    tuple(str(d) for d in date_range) if isinstance(date_range, tuple) else (),
)
cache_key = (
    uploaded.name,
    tuple(sorted(mapping.items())),
    provider_id,
    ollama_model if provider_id == "ollama" else None,
    filter_signature,
)
if st.session_state.get("overview_key") != cache_key:
    st.session_state.pop("overview", None)
    st.session_state.pop("category_clusters", None)
    st.session_state.pop("selected_category", None)

provider_ready = (provider_id == "gemini" and bool(api_key)) or (
    provider_id == "ollama" and bool(ollama_model)
)

if "overview" not in st.session_state:
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
                st.session_state["overview"] = analyze_overview(
                    filtered_df, mapping,
                    provider=provider_id, api_key=api_key,
                    ollama_model=ollama_model, ollama_host=ollama_host,
                )
                st.session_state["overview_key"] = cache_key
            except Exception as e:
                st.error(str(e))

if "overview" in st.session_state:
    render_drilldown(filtered_df, mapping, provider_id, api_key, ollama_model, ollama_host)
    md = drilldown_markdown(
        st.session_state["overview"],
        st.session_state.get("category_clusters", {}),
    )
    st.download_button(
        "Download report as Markdown", md,
        file_name=f"ticket-drilldown-{datetime.now():%Y%m%d-%H%M}.md",
        mime="text/markdown",
    )
