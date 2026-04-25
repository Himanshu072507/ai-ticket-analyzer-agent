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
    assert any("No data" in (a.text or "") for a in fig.layout.annotations)


def test_cluster_trend_handles_no_date_col() -> None:
    df = pd.DataFrame({"description": ["x"]})
    fig = cluster_trend(df, None, "D")
    assert len(fig.data) == 0
