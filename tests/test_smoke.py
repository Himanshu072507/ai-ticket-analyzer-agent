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
