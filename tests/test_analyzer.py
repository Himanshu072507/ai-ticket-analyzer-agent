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


def test_analyze_overview_gemini_path(sample_df, default_mapping, mock_overview_response):
    with patch.object(analyzer, "_call_gemini",
                      return_value=json.dumps(mock_overview_response)) as mock_call:
        out = analyze_overview(sample_df, default_mapping, provider="gemini", api_key="dummy")
    assert "headline" in out
    assert isinstance(out["ranked_categories"], list)
    assert mock_call.call_args.args[-1] is analyzer.OVERVIEW_SCHEMA


def test_analyze_overview_ollama_path(sample_df, default_mapping, mock_overview_response):
    with patch.object(analyzer, "_call_ollama",
                      return_value=json.dumps(mock_overview_response)) as mock_call:
        out = analyze_overview(sample_df, default_mapping, provider="ollama")
    assert "headline" in out
    assert mock_call.call_args.args[-1] is analyzer.OVERVIEW_SCHEMA


def test_analyze_category_gemini_path(sample_df, default_mapping, mock_clusters_response):
    cat = sample_df[default_mapping["category"]].value_counts().index[0]
    with patch.object(analyzer, "_call_gemini",
                      return_value=json.dumps(mock_clusters_response)) as mock_call:
        out = analyze_category(sample_df, default_mapping, str(cat),
                               provider="gemini", api_key="dummy")
    assert "clusters" in out and len(out["clusters"]) >= 1
    assert mock_call.call_args.args[-1] is analyzer.CATEGORY_SCHEMA


def test_analyze_category_ollama_path(sample_df, default_mapping, mock_clusters_response):
    cat = sample_df[default_mapping["category"]].value_counts().index[0]
    with patch.object(analyzer, "_call_ollama",
                      return_value=json.dumps(mock_clusters_response)) as mock_call:
        out = analyze_category(sample_df, default_mapping, str(cat), provider="ollama")
    assert "clusters" in out
    assert mock_call.call_args.args[-1] is analyzer.CATEGORY_SCHEMA


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


def test_analyze_overview_groq_path(sample_df, default_mapping, mock_overview_response):
    with patch.object(analyzer, "_call_groq",
                      return_value=json.dumps(mock_overview_response)) as mock_call:
        out = analyze_overview(sample_df, default_mapping, provider="groq", api_key="dummy")
    assert "headline" in out
    assert isinstance(out["ranked_categories"], list)
    assert mock_call.call_args.args[-1] is analyzer.OVERVIEW_SCHEMA


def test_analyze_category_groq_path(sample_df, default_mapping, mock_clusters_response):
    cat = sample_df[default_mapping["category"]].value_counts().index[0]
    with patch.object(analyzer, "_call_groq",
                      return_value=json.dumps(mock_clusters_response)) as mock_call:
        out = analyze_category(sample_df, default_mapping, str(cat),
                               provider="groq", api_key="dummy")
    assert "clusters" in out and len(out["clusters"]) >= 1
    assert mock_call.call_args.args[-1] is analyzer.CATEGORY_SCHEMA


def test_groq_requires_api_key(sample_df, default_mapping):
    with pytest.raises(RuntimeError, match="Groq API key is required"):
        analyze_overview(sample_df, default_mapping, provider="groq", api_key="")


def test_groq_invalid_json_raises(sample_df, default_mapping):
    with patch.object(analyzer, "_call_groq", return_value="not json {{{"):
        with pytest.raises(RuntimeError, match="not valid JSON"):
            analyze_overview(sample_df, default_mapping, provider="groq", api_key="dummy")


def test_unknown_provider_rejected(sample_df, default_mapping):
    with pytest.raises(RuntimeError, match="Unknown provider"):
        analyze_overview(sample_df, default_mapping, provider="bogus", api_key="dummy")
