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
