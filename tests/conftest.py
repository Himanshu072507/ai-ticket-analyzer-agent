"""Shared fixtures for ticket-analyzer tests."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

# Mock streamlit before importing app
class StopStreamlit(Exception):
    """Exception raised by st.stop() to halt execution."""
    pass

class MockColumn:
    def __init__(self):
        self.metric = MagicMock(return_value=None)

def create_streamlit_mock():
    """Create a complete streamlit mock for testing."""
    st = MagicMock()
    st.set_page_config = MagicMock(return_value=None)
    st.caption = MagicMock(return_value=None)
    st.title = MagicMock(return_value=None)
    st.file_uploader = MagicMock(return_value=None)
    st.info = MagicMock(return_value=None)
    st.success = MagicMock(return_value=None)
    st.error = MagicMock(return_value=None)
    st.stop = MagicMock(side_effect=StopStreamlit)  # Raise a benign exception
    st.subheader = MagicMock(return_value=None)
    st.selectbox = MagicMock(return_value=None)
    st.divider = MagicMock(return_value=None)
    st.button = MagicMock(return_value=False)
    st.spinner = MagicMock()
    st.spinner.return_value.__enter__ = MagicMock(return_value=None)
    st.spinner.return_value.__exit__ = MagicMock(return_value=None)
    st.plotly_chart = MagicMock(return_value=None)
    st.download_button = MagicMock(return_value=None)
    st.markdown = MagicMock(return_value=None)
    st.expander = MagicMock()
    st.expander.return_value.__enter__ = MagicMock(return_value=None)
    st.expander.return_value.__exit__ = MagicMock(return_value=None)
    st.session_state = {}

    # Mock columns
    col = MockColumn()
    st.columns = MagicMock(return_value=[col, col, col, col])

    # Mock sidebar
    sidebar = MagicMock()
    sidebar.subheader = MagicMock(return_value=None)
    sidebar.radio = MagicMock(return_value="Gemini (cloud)")
    sidebar.text_input = MagicMock(return_value="")
    sidebar.selectbox = MagicMock(return_value=None)
    sidebar.caption = MagicMock(return_value=None)
    sidebar.warning = MagicMock(return_value=None)
    st.sidebar = sidebar

    return st

# Install streamlit mock in sys.modules before importing app
sys.modules['streamlit'] = create_streamlit_mock()

# Now we can safely import analyzer and app
import analyzer
if not hasattr(analyzer, 'analyze'):
    analyzer.analyze = MagicMock()

# Import app with exception handling since it has module-level code
# We need to use importlib to force execution despite the exception
import importlib.util
spec = importlib.util.spec_from_file_location("app", str(Path(__file__).resolve().parents[1] / "app.py"))
app_module = importlib.util.module_from_spec(spec)
sys.modules['app'] = app_module
try:
    spec.loader.exec_module(app_module)
except StopStreamlit:
    # This is expected - app.py calls st.stop() when no file is uploaded
    # The module is still loaded despite the exception
    pass

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = REPO_ROOT / "sample_tickets.xlsx"


def pytest_runtest_setup(item):
    """Before each test runs, check if it's a smoke test and remove the mock if so."""
    if "smoke" in item.keywords:
        # For smoke tests, we need the real streamlit
        if 'streamlit' in sys.modules and isinstance(sys.modules['streamlit'], MagicMock):
            del sys.modules['streamlit']
        if 'app' in sys.modules:
            del sys.modules['app']


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
