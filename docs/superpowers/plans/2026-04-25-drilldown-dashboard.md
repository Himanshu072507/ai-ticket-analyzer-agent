# AI Drill-Down Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current flat `render_insights()` block with a three-level drill-down (headline narrative → category master-detail → per-issue details) backed by lazy per-category AI cluster calls and local keyword matching.

**Architecture:** Two new analyzer entry points (`analyze_overview`, `analyze_category`) reuse the existing Gemini/Ollama provider plumbing. A new `clustering.py` does no AI — it matches tickets to AI-supplied keywords. `app.py` gets a new `render_drilldown()` driven by `st.session_state` so per-category clusters are fetched lazily and cached.

**Tech Stack:** Streamlit, pandas, Plotly, `google-genai` SDK (Gemini), Ollama HTTP API, pytest + pytest-mock.

**Spec:** `docs/superpowers/specs/2026-04-25-drilldown-dashboard-design.md`

---

## File Structure

**New files:**
- `clustering.py` — pure-pandas helpers for matching tickets to clusters and building per-cluster trends. ~50 lines, no AI.
- `tests/__init__.py` — empty package marker.
- `tests/conftest.py` — shared fixtures (`sample_df`, `default_mapping`, mock responses).
- `tests/test_clustering.py` — keyword matching + trend chart tests.
- `tests/test_analyzer.py` — `_build_stats`, `analyze_overview`, `analyze_category`, retry, error tests (all mocked, no real API).
- `tests/test_app_helpers.py` — `auto_map`, `read_upload` tests.
- `tests/test_smoke.py` — Streamlit `AppTest` end-to-end smoke (marker `smoke`).

**Modified files:**
- `analyzer.py` — replace `RESPONSE_SCHEMA` and `analyze()` with `OVERVIEW_SCHEMA`, `CATEGORY_SCHEMA`, `analyze_overview()`, `analyze_category()`. Reuse `_call_gemini`, `_call_ollama`, `_is_transient`, retry logic. Remove the old single-call API.
- `charts.py` — add `mini_trend(series, title)` helper.
- `app.py` — remove `render_insights()` and `insights_markdown()`. Add `render_drilldown()` with master-detail layout and lazy per-category cluster fetch. Update download button to serialize the drill-down state.
- `requirements.txt` — add `pytest`, `pytest-mock` under a dev section.
- `README.md` — add a "Running tests" section.
- `pytest.ini` — register the `smoke` marker.

---

### Task 1: Test scaffold + dev dependencies

**Files:**
- Modify: `requirements.txt`
- Create: `pytest.ini`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Add dev deps to `requirements.txt`**

Append to `requirements.txt`:

```
# dev
pytest>=8.0
pytest-mock>=3.12
```

- [ ] **Step 2: Install dev deps**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pip install pytest pytest-mock`
Expected: both packages install without conflict.

- [ ] **Step 3: Create `pytest.ini`**

```ini
[pytest]
testpaths = tests
markers =
    smoke: end-to-end Streamlit AppTest smoke test (slow, optional)
```

- [ ] **Step 4: Create `tests/__init__.py`**

Empty file:

```python
```

- [ ] **Step 5: Create `tests/conftest.py`**

```python
"""Shared fixtures for ticket-analyzer tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "sample_tickets.xlsx"


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """The bundled 60-row sample workbook as a DataFrame."""
    return pd.read_excel(SAMPLE_PATH)


@pytest.fixture
def default_mapping(sample_df: pd.DataFrame) -> dict[str, str | None]:
    """Best-effort mapping for the bundled sample file."""
    cols = {c.lower(): c for c in sample_df.columns}
    return {
        "category": cols.get("category"),
        "description": cols.get("description"),
        "department": cols.get("department"),
        "date": cols.get("date"),
        "priority": cols.get("priority"),
        "status": cols.get("status"),
    }


@pytest.fixture
def mock_overview_response() -> dict:
    return {
        "headline": "Login (24%) and Payments (18%) dominate. Volume up 30% in the last 14 days.",
        "ranked_categories": [
            {"name": "Login", "count": 14, "percent": 23.3},
            {"name": "Payments", "count": 11, "percent": 18.3},
            {"name": "Onboarding", "count": 5, "percent": 8.3},
        ],
    }


@pytest.fixture
def mock_clusters_response() -> dict:
    return {
        "clusters": [
            {
                "name": "OTP not received",
                "summary": "Users not getting login OTP via SMS.",
                "keywords": ["otp", "sms", "code"],
                "root_cause": "SMS gateway throttling during peak hours.",
                "suggested_owner": "Platform / Notifications",
                "suggested_fix": "Add fallback email OTP and burst capacity.",
            },
            {
                "name": "Password reset broken",
                "summary": "Reset link expires before user clicks.",
                "keywords": ["password", "reset", "link"],
                "root_cause": "Token TTL set too low.",
                "suggested_owner": "Identity",
                "suggested_fix": "Raise TTL to 30 minutes.",
            },
        ]
    }
```

- [ ] **Step 6: Verify pytest discovers the suite**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest --collect-only -q`
Expected: `0 tests collected` (no test files yet) and no import errors.

- [ ] **Step 7: Commit**

```bash
cd ~/ticket-analyzer
git add requirements.txt pytest.ini tests/__init__.py tests/conftest.py
git commit -m "test: bootstrap pytest scaffold with shared fixtures"
```

---

### Task 2: `clustering.py` — keyword matching (TDD)

**Files:**
- Create: `clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Write the failing test for `match_tickets_to_clusters`**

Create `tests/test_clustering.py`:

```python
"""Tests for clustering.py — pure pandas, no AI."""
from __future__ import annotations

import pandas as pd
import pytest

from clustering import cluster_trend, match_tickets_to_clusters


@pytest.fixture
def desc_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "description": [
                "OTP not received on phone",
                "SMS code never arrived",
                "Password reset link expired immediately",
                "Cannot reset password",
                "App keeps crashing on launch",
            ]
        }
    )


def test_match_case_insensitive(desc_df: pd.DataFrame) -> None:
    clusters = [{"name": "otp", "keywords": ["OTP", "SMS"]}]
    out = match_tickets_to_clusters(desc_df, "description", clusters)
    assert list(out.keys()) == ["otp"]
    assert len(out["otp"]) == 2


def test_match_partial_substring(desc_df: pd.DataFrame) -> None:
    clusters = [{"name": "reset", "keywords": ["reset"]}]
    out = match_tickets_to_clusters(desc_df, "description", clusters)
    assert len(out["reset"]) == 2


def test_match_multiple_keywords_or(desc_df: pd.DataFrame) -> None:
    clusters = [{"name": "auth", "keywords": ["otp", "password"]}]
    out = match_tickets_to_clusters(desc_df, "description", clusters)
    assert len(out["auth"]) == 3


def test_match_no_match_returns_empty(desc_df: pd.DataFrame) -> None:
    clusters = [{"name": "billing", "keywords": ["invoice", "refund"]}]
    out = match_tickets_to_clusters(desc_df, "description", clusters)
    assert out["billing"].empty
```

- [ ] **Step 2: Run tests, expect failure**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_clustering.py -v`
Expected: `ModuleNotFoundError: No module named 'clustering'`.

- [ ] **Step 3: Implement minimal `clustering.py`**

Create `clustering.py`:

```python
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
```

- [ ] **Step 4: Run tests, expect pass**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_clustering.py -v`
Expected: 4 passed.

- [ ] **Step 5: Add the failing trend tests**

Append to `tests/test_clustering.py`:

```python
def test_cluster_trend_returns_points_for_dated_df() -> None:
    df = pd.DataFrame({
        "date": pd.to_datetime(["2026-04-20", "2026-04-20", "2026-04-21"]),
    })
    fig = cluster_trend(df, "date", "D")
    assert len(fig.data) == 1
    assert list(fig.data[0].y) == [2, 1]


def test_cluster_trend_handles_empty_df() -> None:
    fig = cluster_trend(pd.DataFrame(), "date", "D")
    assert len(fig.data) == 0
    assert any("No trend" in (a.text or "") for a in fig.layout.annotations)


def test_cluster_trend_handles_no_date_col() -> None:
    df = pd.DataFrame({"description": ["x"]})
    fig = cluster_trend(df, None, "D")
    assert len(fig.data) == 0
```

- [ ] **Step 6: Run tests, expect pass**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_clustering.py -v`
Expected: 7 passed.

- [ ] **Step 7: Commit**

```bash
cd ~/ticket-analyzer
git add clustering.py tests/test_clustering.py
git commit -m "feat: add clustering module with keyword matching and per-cluster trend"
```

---

### Task 3: `analyzer.py` — overview + category schemas and helpers (TDD, mocked)

**Files:**
- Modify: `analyzer.py` (replace `RESPONSE_SCHEMA` and `analyze()`)
- Create: `tests/test_analyzer.py`

- [ ] **Step 1: Write the failing test for `_build_stats` shape**

Create `tests/test_analyzer.py`:

```python
"""Tests for analyzer.py — provider calls are mocked. No real API hits."""
from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

import analyzer
from analyzer import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    MAX_RETRIES,
    _build_stats,
    analyze_category,
    analyze_overview,
)


def test_build_stats_shape(sample_df, default_mapping):
    stats = _build_stats(sample_df, default_mapping)
    assert stats["total_tickets"] == len(sample_df)
    assert "top_categories" in stats and len(stats["top_categories"]) <= 15
    assert "description_samples" in stats
```

- [ ] **Step 2: Run tests, expect pass for `_build_stats` (still exists), failure for missing imports**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_analyzer.py -v`
Expected: `ImportError: cannot import name 'analyze_category' from 'analyzer'`.

- [ ] **Step 3: Replace `RESPONSE_SCHEMA` block in `analyzer.py`**

In `analyzer.py`, find the block:

```python
_STR = {"type": "string"}
RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "top_themes": _arr(3, 5, {"title": _STR, "detail": _STR}),
        "priority_ranking": _arr(3, 7, {
            "issue": _STR, "impact": _STR, "suggested_owner": _STR, "rationale": _STR,
        }),
        "department_spotlight": {
            "type": "object",
            "properties": {"department": _STR, "finding": _STR},
            "required": ["department", "finding"],
        },
        "recommendations": _arr(3, 6, {"action": _STR, "expected_outcome": _STR}),
    },
    "required": ["top_themes", "priority_ranking", "department_spotlight", "recommendations"],
}
```

Replace it with:

```python
_STR = {"type": "string"}
_NUM = {"type": "number"}
_INT = {"type": "integer"}

OVERVIEW_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "headline": _STR,
        "ranked_categories": {
            "type": "array",
            "minItems": 1,
            "maxItems": 10,
            "items": {
                "type": "object",
                "properties": {"name": _STR, "count": _INT, "percent": _NUM},
                "required": ["name", "count", "percent"],
            },
        },
    },
    "required": ["headline", "ranked_categories"],
}

CATEGORY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "minItems": 1,
            "maxItems": 5,
            "items": {
                "type": "object",
                "properties": {
                    "name": _STR,
                    "summary": _STR,
                    "keywords": {"type": "array", "items": _STR, "minItems": 1, "maxItems": 8},
                    "root_cause": _STR,
                    "suggested_owner": _STR,
                    "suggested_fix": _STR,
                },
                "required": ["name", "summary", "keywords", "root_cause",
                             "suggested_owner", "suggested_fix"],
            },
        }
    },
    "required": ["clusters"],
}
```

- [ ] **Step 4: Update `_call_gemini` to take a schema parameter**

In `analyzer.py`, replace:

```python
def _call_gemini(client: genai.Client, model: str, payload: str) -> str:
    response = client.models.generate_content(
        model=model,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            max_output_tokens=4096,
        ),
    )
    if not response.text:
        raise RuntimeError("Empty response.")
    return response.text
```

with:

```python
def _call_gemini(client: genai.Client, model: str, payload: str, schema: dict) -> str:
    response = client.models.generate_content(
        model=model,
        contents=payload,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=4096,
        ),
    )
    if not response.text:
        raise RuntimeError("Empty response.")
    return response.text
```

- [ ] **Step 5: Update `_call_ollama` to take a schema parameter**

In `analyzer.py`, replace:

```python
def _call_ollama(host: str, model: str, payload: str) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        "format": RESPONSE_SCHEMA,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }).encode()
```

with:

```python
def _call_ollama(host: str, model: str, payload: str, schema: dict) -> str:
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        "format": schema,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 4096},
    }).encode()
```

- [ ] **Step 6: Replace `_analyze_gemini` and `_analyze_ollama` to take a schema**

Replace `_analyze_gemini`:

```python
def _analyze_gemini(payload: str, api_key: str, schema: dict) -> dict:
    if not api_key:
        raise RuntimeError("Gemini API key is required.")
    client = genai.Client(api_key=api_key)
    last_err: Exception | None = None
    for model in [MODEL, *FALLBACK_MODELS]:
        for attempt in range(MAX_RETRIES):
            try:
                return json.loads(_call_gemini(client, model, payload, schema))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Gemini response was not valid JSON: {e}") from e
            except Exception as e:
                last_err = e
                if _is_transient(e) and attempt < MAX_RETRIES - 1:
                    time.sleep(2 ** attempt)
                    continue
                break
    raise RuntimeError(
        f"Gemini API call failed after retrying {MODEL} and fallbacks {FALLBACK_MODELS}. "
        f"Last error: {last_err}"
    )
```

Replace `_analyze_ollama`:

```python
def _analyze_ollama(payload: str, model: str, host: str, schema: dict) -> dict:
    try:
        text = _call_ollama(host, model, payload, schema)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Could not reach Ollama at {host}. Is it running? (`ollama serve`). Details: {e}"
        ) from e
    except Exception as e:
        raise RuntimeError(f"Ollama call failed: {e}") from e
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Ollama response was not valid JSON: {e}") from e
```

- [ ] **Step 7: Add `_build_category_payload` helper**

Add near `_build_stats`:

```python
def _build_category_payload(df: pd.DataFrame, mapping: dict, category_name: str) -> str:
    """Slice df to one category and serialize description samples for the cluster call."""
    cat_col, desc_col = mapping["category"], mapping["description"]
    sub = df[df[cat_col].astype(str) == str(category_name)]
    if sub.empty:
        raise RuntimeError(f"No tickets found for category {category_name!r}.")
    samples = (
        sub[desc_col].dropna().astype(str).head(MAX_SAMPLES).map(lambda s: s[:SNIPPET_CHARS]).tolist()
    )
    body = {
        "category": str(category_name),
        "ticket_count": int(len(sub)),
        "description_samples": samples,
    }
    return f"Single-category cluster payload:\n\n{json.dumps(body, indent=2, default=str)}"
```

- [ ] **Step 8: Replace `analyze()` with `analyze_overview()` and `analyze_category()`**

Remove the existing `analyze()` function entirely and replace with:

```python
def analyze_overview(
    df: pd.DataFrame,
    mapping: dict,
    provider: str = "gemini",
    api_key: str = "",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    """Generate the headline narrative and ranked category list."""
    if df.empty or not mapping.get("category") or not mapping.get("description"):
        raise RuntimeError("DataFrame must be non-empty with 'category' and 'description' mapped.")
    payload = f"Ticket analytics payload:\n\n{json.dumps(_build_stats(df, mapping), indent=2, default=str)}"
    if provider == "ollama":
        return _analyze_ollama(payload, ollama_model, ollama_host, OVERVIEW_SCHEMA)
    if provider == "gemini":
        return _analyze_gemini(payload, api_key, OVERVIEW_SCHEMA)
    raise RuntimeError(f"Unknown provider: {provider!r}. Use 'gemini' or 'ollama'.")


def analyze_category(
    df: pd.DataFrame,
    mapping: dict,
    category_name: str,
    provider: str = "gemini",
    api_key: str = "",
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    ollama_host: str = DEFAULT_OLLAMA_HOST,
) -> dict:
    """Cluster a single category's tickets into 1-5 named sub-issues."""
    if not mapping.get("category") or not mapping.get("description"):
        raise RuntimeError("Mapping must include 'category' and 'description'.")
    payload = _build_category_payload(df, mapping, category_name)
    if provider == "ollama":
        return _analyze_ollama(payload, ollama_model, ollama_host, CATEGORY_SCHEMA)
    if provider == "gemini":
        return _analyze_gemini(payload, api_key, CATEGORY_SCHEMA)
    raise RuntimeError(f"Unknown provider: {provider!r}. Use 'gemini' or 'ollama'.")
```

- [ ] **Step 9: Run the `_build_stats` test, expect pass**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_analyzer.py -v`
Expected: `test_build_stats_shape PASSED`.

- [ ] **Step 10: Add overview + category provider tests (mocked)**

Append to `tests/test_analyzer.py`:

```python
def test_analyze_overview_gemini_path(sample_df, default_mapping, mock_overview_response):
    with patch.object(analyzer, "_call_gemini", return_value=json.dumps(mock_overview_response)):
        out = analyze_overview(sample_df, default_mapping, provider="gemini", api_key="dummy")
    assert "headline" in out
    assert isinstance(out["ranked_categories"], list)


def test_analyze_overview_ollama_path(sample_df, default_mapping, mock_overview_response):
    with patch.object(analyzer, "_call_ollama", return_value=json.dumps(mock_overview_response)):
        out = analyze_overview(sample_df, default_mapping, provider="ollama")
    assert "headline" in out


def test_analyze_category_gemini_path(sample_df, default_mapping, mock_clusters_response):
    cat = sample_df[default_mapping["category"]].value_counts().index[0]
    with patch.object(analyzer, "_call_gemini", return_value=json.dumps(mock_clusters_response)):
        out = analyze_category(sample_df, default_mapping, str(cat),
                               provider="gemini", api_key="dummy")
    assert "clusters" in out and len(out["clusters"]) >= 1


def test_analyze_category_ollama_path(sample_df, default_mapping, mock_clusters_response):
    cat = sample_df[default_mapping["category"]].value_counts().index[0]
    with patch.object(analyzer, "_call_ollama", return_value=json.dumps(mock_clusters_response)):
        out = analyze_category(sample_df, default_mapping, str(cat), provider="ollama")
    assert "clusters" in out
```

- [ ] **Step 11: Run tests, expect pass**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_analyzer.py -v`
Expected: 5 passed.

- [ ] **Step 12: Add retry + error tests**

Append to `tests/test_analyzer.py`:

```python
def test_gemini_retries_on_transient_then_succeeds(sample_df, default_mapping, mock_overview_response):
    calls = {"n": 0}

    def flaky(client, model, payload, schema):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("503 UNAVAILABLE")
        return json.dumps(mock_overview_response)

    with patch.object(analyzer, "_call_gemini", side_effect=flaky), \
         patch.object(analyzer.time, "sleep", return_value=None):
        out = analyze_overview(sample_df, default_mapping, provider="gemini", api_key="dummy")
    assert calls["n"] == 2
    assert "headline" in out


def test_ollama_unreachable_raises_clear_error(sample_df, default_mapping):
    import urllib.error

    with patch.object(
        analyzer, "_call_ollama",
        side_effect=urllib.error.URLError("Connection refused"),
    ):
        with pytest.raises(RuntimeError, match="Could not reach Ollama"):
            analyze_overview(sample_df, default_mapping, provider="ollama",
                             ollama_host="http://localhost:11434")


def test_overview_invalid_json_raises(sample_df, default_mapping):
    with patch.object(analyzer, "_call_gemini", return_value="not json {{{"):
        with pytest.raises(RuntimeError, match="not valid JSON"):
            analyze_overview(sample_df, default_mapping, provider="gemini", api_key="dummy")
```

- [ ] **Step 13: Run tests, expect pass**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_analyzer.py -v`
Expected: 8 passed.

- [ ] **Step 14: Commit**

```bash
cd ~/ticket-analyzer
git add analyzer.py tests/test_analyzer.py
git commit -m "feat: split analyzer into overview + per-category cluster calls"
```

---

### Task 4: `charts.py` — `mini_trend` helper

**Files:**
- Modify: `charts.py`

- [ ] **Step 1: Append `mini_trend` to `charts.py`**

Add at the end of `charts.py`:

```python
def mini_trend(series: pd.Series, title: str = "") -> go.Figure:
    """Compact line chart for inline issue-detail trends. Empty figure if series is empty."""
    fig = go.Figure()
    fig.update_layout(
        template="simple_white", height=200, margin=dict(l=10, r=10, t=30, b=10),
        title=title or None,
    )
    if series is None or series.empty:
        fig.add_annotation(text="No data", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False)
        return fig
    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines"))
    fig.update_xaxes(title=None)
    fig.update_yaxes(title="Tickets")
    return fig
```

- [ ] **Step 2: Sanity-import to confirm no syntax errors**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && python -c "from charts import mini_trend; print(mini_trend.__doc__)"`
Expected: docstring prints.

- [ ] **Step 3: Commit**

```bash
cd ~/ticket-analyzer
git add charts.py
git commit -m "feat: add mini_trend chart helper"
```

---

### Task 5: `app_helpers` tests for `auto_map` and `read_upload`

**Files:**
- Create: `tests/test_app_helpers.py`

- [ ] **Step 1: Write tests for `auto_map`**

Create `tests/test_app_helpers.py`:

```python
"""Tests for the small pure helpers in app.py."""
from __future__ import annotations

import io

import pandas as pd
import pytest

from app import auto_map, read_upload


class _FakeUpload:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


def test_auto_map_matches_synonyms_with_casing():
    cols = ["Issue Type", "Details", "Team", "Created At", "Severity", "State"]
    out = auto_map(cols)
    assert out["category"] == "Issue Type"
    assert out["description"] == "Details"
    assert out["department"] == "Team"
    assert out["date"] == "Created At"
    assert out["priority"] == "Severity"
    assert out["status"] == "State"


def test_auto_map_returns_none_for_missing_field():
    out = auto_map(["Foo", "Bar"])
    assert out["category"] is None
    assert out["description"] is None


def test_read_upload_csv_branch():
    csv = b"a,b\n1,2\n3,4\n"
    df, sheets = read_upload(_FakeUpload("data.csv", csv))
    assert sheets == []
    assert df is not None
    assert list(df.columns) == ["a", "b"]
    assert len(df) == 2


def test_read_upload_xlsx_branch(tmp_path):
    path = tmp_path / "x.xlsx"
    pd.DataFrame({"a": [1]}).to_excel(path, sheet_name="Sheet1", index=False)
    df, sheets = read_upload(_FakeUpload("x.xlsx", path.read_bytes()))
    assert df is None
    assert sheets == ["Sheet1"]
```

- [ ] **Step 2: Run tests, expect pass**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_app_helpers.py -v`
Expected: 4 passed.

- [ ] **Step 3: Commit**

```bash
cd ~/ticket-analyzer
git add tests/test_app_helpers.py
git commit -m "test: cover auto_map and read_upload helpers"
```

---

### Task 6: `app.py` — replace `render_insights` with `render_drilldown`

**Files:**
- Modify: `app.py`

- [ ] **Step 1: Update `analyzer` import**

In `app.py`, replace:

```python
from analyzer import DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL, analyze, list_ollama_models
```

with:

```python
from analyzer import (
    DEFAULT_OLLAMA_HOST,
    DEFAULT_OLLAMA_MODEL,
    analyze_category,
    analyze_overview,
    list_ollama_models,
)
from clustering import cluster_trend, match_tickets_to_clusters
```

- [ ] **Step 2: Remove `render_insights` and `insights_markdown`**

Delete the entire `render_insights(insights)` function and the entire `insights_markdown(insights)` function from `app.py`.

- [ ] **Step 3: Add `drilldown_markdown` and `render_drilldown`**

Add (in place of the deleted functions):

```python
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
    st.markdown(f"**Headline:** {overview['headline']}")

    categories = overview.get("ranked_categories", [])
    if not categories:
        st.info("No categories returned.")
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

        if len(cat_df) < 5:
            st.info("Too few tickets to cluster — showing raw descriptions.")
            for d in cat_df[desc_col].dropna().astype(str).head(10):
                st.markdown(f"- {d[:200]}")
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
        matches = match_tickets_to_clusters(cat_df, desc_col, clusters)

        for cl in clusters:
            sub = matches.get(cl["name"], cat_df.iloc[0:0])
            count_label = f"{len(sub)} tickets" if len(sub) else "— no keyword match"
            with st.expander(f"{cl['name']} · {count_label} — {cl['summary']}"):
                if mapping.get("date") and not sub.empty:
                    st.plotly_chart(
                        cluster_trend(sub, mapping["date"], "D"),
                        use_container_width=True,
                    )
                st.markdown(f"**Root cause:** {cl['root_cause']}")
                st.markdown(f"**Suggested owner:** {cl['suggested_owner']}")
                st.markdown(f"**Suggested fix:** {cl['suggested_fix']}")
                if not sub.empty:
                    st.markdown("**Example tickets:**")
                    for d in sub[desc_col].dropna().astype(str).head(10):
                        st.markdown(f"- {d[:200]}")
                else:
                    st.caption("No tickets matched the AI-supplied keywords.")
```

- [ ] **Step 4: Replace the bottom of `app.py` (the cache + button + render block)**

Find this block in `app.py`:

```python
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
```

Replace with:

```python
cache_key = (
    uploaded.name,
    tuple(sorted(mapping.items())),
    provider_id,
    ollama_model if provider_id == "ollama" else None,
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
                    df, mapping,
                    provider=provider_id, api_key=api_key,
                    ollama_model=ollama_model, ollama_host=ollama_host,
                )
                st.session_state["overview_key"] = cache_key
            except Exception as e:
                st.error(str(e))

if "overview" in st.session_state:
    render_drilldown(df, mapping, provider_id, api_key, ollama_model, ollama_host)
    md = drilldown_markdown(
        st.session_state["overview"],
        st.session_state.get("category_clusters", {}),
    )
    st.download_button(
        "Download report as Markdown", md,
        file_name=f"ticket-drilldown-{datetime.now():%Y%m%d-%H%M}.md",
        mime="text/markdown",
    )
```

- [ ] **Step 5: Re-run helper tests to make sure imports still work**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_app_helpers.py -v`
Expected: 4 passed.

- [ ] **Step 6: Manual smoke — launch the app and verify the new flow renders**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && streamlit run app.py`
Expected:
- App loads at http://localhost:8501.
- Upload `sample_tickets.xlsx`.
- Existing KPIs + 5 charts still render.
- Click "Generate AI insights" → headline + ranked categories appear in the master pane.
- Click a category in the left pane → top issues load in the right pane (first click slow, subsequent instant).
- Expand a top issue → trend chart, root cause, owner, fix, examples render.
- Stop the server with Ctrl-C.

- [ ] **Step 7: Commit**

```bash
cd ~/ticket-analyzer
git add app.py
git commit -m "feat: replace flat AI summary with three-level drill-down dashboard"
```

---

### Task 7: Streamlit `AppTest` smoke test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the smoke test**

Create `tests/test_smoke.py`:

```python
"""End-to-end Streamlit smoke test. Marked 'smoke'; can be skipped if flaky in CI."""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.smoke


def test_app_renders_until_upload_prompt():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    titles = [el.value for el in at.title]
    assert any("Ticket Analyzer" in t for t in titles)
    infos = [el.value for el in at.info]
    assert any("sample_tickets.xlsx" in (i or "") for i in infos)
```

- [ ] **Step 2: Run the smoke test**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest tests/test_smoke.py -v -m smoke`
Expected: 1 passed.

- [ ] **Step 3: Run the full suite**

Run: `cd ~/ticket-analyzer && source .venv/bin/activate && pytest -q`
Expected: all tests pass (clustering 7, analyzer 8, app_helpers 4, smoke 1 = 20 passed).

- [ ] **Step 4: Commit**

```bash
cd ~/ticket-analyzer
git add tests/test_smoke.py
git commit -m "test: add Streamlit AppTest smoke test"
```

---

### Task 8: README updates

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Read current README**

Run: `cd ~/ticket-analyzer && cat README.md | head -60`

- [ ] **Step 2: Replace the "AI summary" / features section to describe the drill-down**

In `README.md`, find the bullet list of features (the section that mentions "AI-generated priority ranking" / "top themes" — exact text varies). Replace whatever describes the AI section with:

```
- **AI Drill-Down**: a one-paragraph headline narrative, a ranked category list, and lazy per-category clustering. Click a category to see 3–5 AI-named sub-issues; expand a sub-issue for trend, examples, root cause, owner, and suggested fix.
```

- [ ] **Step 3: Add a "Running tests" section near the bottom**

Append to `README.md`:

````
## Running tests

```bash
source .venv/bin/activate
pip install pytest pytest-mock      # one-time
pytest -q                            # core suite (no real API calls)
pytest -q -m smoke                   # also run the Streamlit AppTest smoke test
```

Provider calls are mocked — no Gemini key or Ollama daemon required to run tests.
````

- [ ] **Step 4: Commit**

```bash
cd ~/ticket-analyzer
git add README.md
git commit -m "docs: describe drill-down dashboard and add Running tests section"
```

---

## Self-Review Notes

- **Spec coverage:** headline (Task 6), master-detail (Task 6), issue detail with trend (Tasks 2 + 6), lazy AI per category (Task 3 + 6), local keyword matching (Task 2 + 6), edge cases for <5 tickets / no date col / zero matches / provider switch / re-upload (Task 6 cache logic + render flow), full pytest scaffold including `clustering`, `analyzer`, `app_helpers`, smoke (Tasks 1, 2, 3, 5, 7), README (Task 8). All present.
- **Placeholder scan:** no TBDs; every step has the exact code or command.
- **Type consistency:** `analyze_overview` / `analyze_category` / `OVERVIEW_SCHEMA` / `CATEGORY_SCHEMA` / `match_tickets_to_clusters` / `cluster_trend` / `mini_trend` / `render_drilldown` / `drilldown_markdown` names match across tasks. Cache key field name is `overview_key` everywhere.
