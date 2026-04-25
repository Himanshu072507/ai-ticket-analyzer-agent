"""Tests for the small pure helpers in app.py."""
from __future__ import annotations

import io
import sys

import pandas as pd
import pytest

# The conftest should have already loaded app, catching StopStreamlit
# Try to import from the already-loaded module in sys.modules
if 'app' in sys.modules:
    auto_map = sys.modules['app'].auto_map
    read_upload = sys.modules['app'].read_upload
else:
    # Fallback: import app, handling the st.stop() that occurs when no file is uploaded
    from tests.conftest import StopStreamlit
    try:
        from app import auto_map, read_upload
    except StopStreamlit:
        import app
        auto_map = app.auto_map
        read_upload = app.read_upload


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
