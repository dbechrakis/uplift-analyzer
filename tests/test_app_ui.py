from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app import main as app_main
from tests.llm_mock import MockLlmServer, content_chunks, sse

APP = str(Path(__file__).resolve().parent.parent / "app" / "main.py")


@pytest.fixture
def app_test(monkeypatch):
    monkeypatch.setenv("UPLIFT_APP_SOURCE", "synthetic")
    monkeypatch.setenv("LLAMA_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(app_main, "SAMPLE_SIZES", [50_000, 100_000, 500_000])
    return AppTest.from_file(APP, default_timeout=90)


def test_app_renders_synthetic_with_banner_and_all_tabs(app_test):
    app_test.run()
    assert not app_test.exception
    assert [t.label for t in app_test.tabs] == app_main.TAB_LABELS
    assert any("SYNTHETIC" in w.value for w in app_test.warning)
    assert len(app_test.metric) >= 5


def test_app_is_fully_usable_with_llm_server_off(app_test):
    app_test.run()
    assert any("no-LLM mode" in w.value for w in app_test.warning)
    assert any("LLM mode is off" in i.value for i in app_test.info)
    app_test.button(key="fallback_run").click().run()
    assert not app_test.exception
    assert any("treatment" in m.value.lower() for m in app_test.markdown)


def test_missing_real_dataset_shows_download_instructions(monkeypatch, tmp_path):
    monkeypatch.setenv("UPLIFT_APP_SOURCE", "criteo")
    monkeypatch.setenv("UPLIFT_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("not found" in e.value for e in at.error)
    assert any("Hugging Face" in c.value for c in at.code)


def test_llm_online_check_uses_the_mock_server(monkeypatch):
    server = MockLlmServer(lambda p, n: sse(*content_chunks("ok")))
    monkeypatch.setattr(app_main, "LlamaClient", lambda settings: server.client())
    app_main.llm_online.clear()
    assert app_main.llm_online("http://mock/v1") is True
