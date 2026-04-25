# AI Drill-Down Dashboard — Design

**Date:** 2026-04-25
**Project:** AI Ticket Analyzer Agent (`~/ticket-analyzer`)
**Status:** Approved, pending implementation plan

## Goal

Replace the current flat "AI Summary" section (top_themes / priority_ranking / recommendations) with a three-level drill-down that lets a support PM move from a one-paragraph headline → category breakdown → top sub-issues per category → individual issue detail. Existing charts and KPIs are unchanged.

## Scope

**In scope:**
- New "AI Drill-Down" section that replaces the current `render_insights()` block.
- Two new analyzer entry points (overview + per-category) sharing existing provider plumbing.
- New `clustering.py` module for keyword-based ticket-to-cluster matching.
- One new chart helper (`mini_trend`) for the per-issue trend.
- Full pytest scaffold with fixtures, mocked provider calls, and a smoke test.

**Out of scope:**
- Changes to file upload, column mapping, KPI strip, or the 5 existing charts.
- Persistence (SQLite/Supabase) — clusters live in `st.session_state` only.
- Multi-file batch upload.
- Streamlit Cloud deploy.

## Page Layout

Top of page (unchanged): sidebar (provider/keys), file upload, column mapping, KPI strip, 5 existing charts (category bar, dept bar, time series, status donut, heatmap).

Below the existing divider, the **AI Drill-Down section** has three nested levels:

### Level 1 — Headline narrative (full width)
2–3 sentence AI summary anchoring what to look at first. Example:
> "Across 60 tickets, Login (24%) and Payments (18%) dominate. Volume up 30% in the last 14 days. Onboarding is small (8%) but every ticket is High priority — worth a deeper look."

### Level 2 — Master-detail (categories → top issues)
Two columns:
- **Left pane (≈1/3 width):** ranked list of top categories with ticket count + % bar. Click selects a category. Selected category is highlighted; selection state stored in `st.session_state["selected_category"]`.
- **Right pane (≈2/3 width):** for the selected category, shows 3–5 AI-clustered "top issues" as cards. Each card shows: cluster name, count, % of category, 1-line summary. First click on a category triggers a lazy AI call to cluster that category's descriptions; result cached in session.

### Level 3 — Issue detail
Click a top-issue card → expands inline (right pane) to show:
- **Mini trend line** — this sub-issue's volume over time (skipped if no date column).
- **Example tickets** — 5–10 raw descriptions matching the cluster's keywords (capped at `SNIPPET_CHARS`).
- **AI root-cause summary** — 2–3 sentences from the cluster payload.
- **Suggested owner** + **suggested fix** — both from the cluster payload.

## Data Flow & AI Calls

**First "Generate AI insights" click → one AI call (`analyze_overview`)**

Returns:
```json
{
  "headline": "string (2-3 sentences)",
  "ranked_categories": [
    { "name": "string", "count": int, "percent": float }
  ]
}
```
Top 10 categories only. No clusters yet — just the ranking that drives the left pane.

**First click on a category → lazy AI call (`analyze_category(category_name)`)**

Returns:
```json
{
  "clusters": [
    {
      "name": "string",
      "summary": "string (1 line)",
      "keywords": ["string", "..."],
      "root_cause": "string (2-3 sentences)",
      "suggested_owner": "string",
      "suggested_fix": "string"
    }
  ]
}
```
3–5 clusters. Built from descriptions in that one category only (smaller payload than full overview).

**Local cluster matching (no AI call)**

After receiving clusters, `clustering.match_tickets_to_clusters()` filters the category's tickets by case-insensitive substring OR-match against each cluster's `keywords`. This produces the per-cluster count, examples, and trend without another AI call.

**Click an issue card → no AI call.** Pure local: render trend, examples, and the cluster's pre-fetched root_cause/owner/fix.

## Caching

All AI results stored in `st.session_state`:
- Overview: keyed by `(file_name, mapping, provider, model)`.
- Per-category clusters: nested dict keyed by category name under the same overview key.

Cache invalidates on: file re-upload with different name, mapping change, provider switch, model switch.

## Module Structure

### `analyzer.py` (modified)
Replace single `analyze()` with two functions sharing `_call_gemini`, `_call_ollama`, retry/fallback, and `_is_transient`:

- `analyze_overview(df, mapping, provider, api_key, ollama_model, ollama_host) -> dict`
  - Returns `{headline, ranked_categories}`.
  - New `OVERVIEW_SCHEMA`.
  - Payload: existing `_build_stats(df, mapping)`.

- `analyze_category(df, mapping, category_name, provider, api_key, ollama_model, ollama_host) -> dict`
  - Returns `{clusters: [...]}`.
  - New `CATEGORY_SCHEMA`.
  - Payload: descriptions filtered to `category_name` only, sampled the same way as `_sample_descriptions` but without the cross-category quota logic (one-category context → can use a simpler sampling cap).

The old `analyze()` and the old `RESPONSE_SCHEMA` are removed.

### `clustering.py` (new, ~40 lines, no AI)
- `match_tickets_to_clusters(df, desc_col, clusters) -> dict[str, pd.DataFrame]`
  - For each cluster, returns the subset of `df` whose description (case-insensitive) contains any of the cluster's keywords.
- `cluster_trend(cluster_df, date_col, granularity) -> go.Figure`
  - Small line chart of cluster volume over time. Returns an empty figure if `date_col` is None or has no parseable dates.

### `charts.py` (modified)
Add one helper:
- `mini_trend(series, title) -> go.Figure` — small line chart, height 200, used by `cluster_trend`.

### `app.py` (modified)
- Remove `render_insights()` and `insights_markdown()`.
- Add `render_drilldown(df, mapping, provider_id, api_key, ollama_model, ollama_host)`:
  - Renders headline (Level 1).
  - Renders master-detail (Level 2) using `st.columns([1, 2])`.
  - Renders issue-detail expanders (Level 3) inside the right pane.
- Add lazy state management for `st.session_state["selected_category"]` and `st.session_state["category_clusters"]`.
- The Markdown download button is replaced by a new `drilldown_markdown(overview, clusters_by_category)` that serializes whatever has been generated so far (overview always; categories the user explored).

## Edge Cases

- **No date column:** mini trend chart is hidden in the issue detail; everything else still works.
- **Category with <5 tickets:** skip AI clustering; right pane shows "Too few tickets to cluster — showing raw descriptions" and lists them inline.
- **AI cluster keywords match zero tickets:** cluster card renders with count "—" and a small warning; no trend or examples shown; root_cause/owner/fix from the AI payload still display.
- **AI call fails on a category:** error shown inline in the right pane with a "Retry" button; other categories remain usable; left pane stays interactive.
- **Provider switched mid-session:** all cached overview + clusters cleared (cache key includes provider/model).
- **Re-upload same file:** cache key includes file name + mapping; survives re-upload, invalidates on any mapping change.
- **Long descriptions:** existing 200-char `SNIPPET_CHARS` cap applied to examples in the issue detail.
- **AI returns malformed JSON:** existing `json.JSONDecodeError` handling in analyzer raises a clear error surfaced in the UI.

## Verification

### Setup
- `tests/` directory.
- `pytest` + `pytest-mock` added to `requirements.txt` under a dev section.
- `conftest.py` with shared fixtures: `sample_df` (loads `sample_tickets.xlsx`), `default_mapping`, `mock_clusters_response`, `mock_overview_response`.

### `tests/test_clustering.py` (~6 tests)
- Keyword match: case-insensitive.
- Keyword match: partial substring.
- Keyword match: multiple keywords (OR).
- Keyword match: no-match returns empty df.
- `cluster_trend` returns a Plotly figure with the right number of points.
- `cluster_trend` handles empty df without raising.

### `tests/test_analyzer.py` (~8 tests, all mocked — no real API calls)
- `_build_stats`: required keys present, `top_categories` truncated to 15, samples respect `MAX_SAMPLES`.
- `analyze_overview` returns `{headline, ranked_categories}` (Gemini path mocked).
- `analyze_overview` returns `{headline, ranked_categories}` (Ollama path mocked).
- `analyze_category` returns `{clusters: [...]}` (Gemini path mocked).
- `analyze_category` returns `{clusters: [...]}` (Ollama path mocked).
- Gemini retry: transient 503 retries up to `MAX_RETRIES` then falls back to next model.
- Ollama unreachable: raises clear error message containing the host.
- Schema validation: response missing required keys raises clearly.

### `tests/test_app_helpers.py` (~4 tests)
- `auto_map`: matches synonyms across casing/whitespace.
- `auto_map`: returns None for missing fields.
- `read_upload`: csv branch returns df + empty sheet list.
- `read_upload`: xlsx branch returns None + sheet name list.

### `tests/test_smoke.py` (~1 test)
- Streamlit `AppTest`: launch app, upload `sample_tickets.xlsx`, verify expected widgets render. Marked as `@pytest.mark.smoke` so it can be skipped in CI if flaky.

### CI / Docs
- Add a `pytest -q` target (Make or just README instructions).
- README gets a "Running tests" section.

## Trade-offs Considered

- **Replacing the existing AI Summary entirely (vs keeping both)** — chosen for simplicity per user feedback. Drill-down replaces it; charts and KPIs above remain as the "broad summary."
- **Lazy per-category cluster calls (vs one fat call)** — chosen because most users will only drill into 2–3 categories, not all 10. Cuts LLM cost and latency on first generate.
- **Local keyword matching for cluster → tickets (vs another AI call to assign each ticket)** — chosen because keyword OR-match on the AI-supplied keywords is fast, transparent, and good enough for example surfacing. Avoids ticket-by-ticket LLM cost.
- **Streamlit master-detail via two `st.columns` (vs Streamlit native components)** — Streamlit has no first-class master-detail widget; columns + session_state is the standard pattern.
