"""End-to-end UI test: runs the real Streamlit script with a fake LLM client."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.fixture
def app(monkeypatch, tmp_path, fake_client):
    client, completions = fake_client("**Answer** with `vector<int>`")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("STUDY_SQUAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("study_squad.agents.make_groq_client", lambda key: client)
    return SimpleNamespace(
        at=AppTest.from_file(APP, default_timeout=120), calls=completions.calls, dir=tmp_path
    )


def test_question_gets_three_answers_and_is_logged(app):
    at = app.at.run()
    assert not at.exception
    at.chat_input[0].set_value("I don't get gradient descent").run()
    assert not at.exception
    turn = at.session_state.turns[-1]
    assert turn["route"]["lead"] == "Simplifier"
    assert set(turn["replies"]) == {"Nerd", "Simplifier", "Challenger"}
    assert len(app.calls) == 3
    assert (app.dir / "study_logs.jsonl").read_text().count("\n") == 1


def test_follow_up_sends_history(app):
    at = app.at.run()
    at.chat_input[0].set_value("what is k-means?").run()
    at.chat_input[0].set_value("how does it work?").run()
    last_call = app.calls[-1]["messages"]
    assert any(m["content"] == "what is k-means?" for m in last_call)


def test_analytics_tab_renders_sample_data(app):
    at = app.at.run()
    assert not at.exception
    labels = {m.label: m.value for m in at.metric}
    assert labels["Questions"] == "58"
    assert labels["Topics found"] != "—"


def test_missing_api_key_shows_help(monkeypatch, tmp_path):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("STUDY_SQUAD_DATA_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("hello").run()
    assert any("API key" in e.value for e in at.error)


def test_follow_up_button_asks_a_routed_question(app):
    at = app.at.run()
    at.chat_input[0].set_value("what is entropy?").run()
    simpler = next(b for b in at.button if b.label == "Explain simpler")
    simpler.click().run()
    assert not at.exception
    turn = at.session_state.turns[-1]
    assert turn["route"]["intent"] == "confused"
    assert turn["route"]["lead"] == "Simplifier"
    assert len(at.session_state.turns) == 2


def test_progress_callback_and_errors_render(monkeypatch, tmp_path, fake_client):
    class RateLimitError(Exception):
        pass

    client, _ = fake_client(error=RateLimitError("429"))
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("STUDY_SQUAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr("study_squad.agents.make_groq_client", lambda key: client)
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.chat_input[0].set_value("hello").run()
    assert not at.exception
    assert any("rate limit" in w.value.lower() for w in at.warning)
