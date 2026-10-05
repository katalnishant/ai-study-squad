from study_squad.agents import AGENTS, StudySquad, build_messages, friendly_error


def test_build_messages_includes_history_and_context():
    msgs = build_messages(
        AGENTS["Nerd"],
        "how does it work?",
        context="[p. 3] K-means assigns points to the nearest centroid.",
        history=[("what is k-means?", "K-means is a clustering algorithm.")],
    )
    roles = [m["role"] for m in msgs]
    assert roles == ["system", "user", "assistant", "user"]
    assert "You are The Nerd" in msgs[0]["content"]
    assert msgs[1]["content"] == "what is k-means?"
    assert "[p. 3]" in msgs[-1]["content"]
    assert msgs[-1]["content"].endswith("Question: how does it work?")


def test_intent_hint_only_for_matching_agent():
    simple = build_messages(AGENTS["Simplifier"], "huh", intent="confused")[0]["content"]
    challenger = build_messages(AGENTS["Challenger"], "huh", intent="deep")[0]["content"]
    assert "lead tutor" in simple
    assert "lead tutor" not in challenger


def test_ask_all_returns_every_agent(fake_client):
    client, completions = fake_client("Here is the answer")
    replies = StudySquad(client, "openai/gpt-oss-120b").ask_all("What is entropy?")
    assert set(replies) == set(AGENTS)
    assert all(r.ok and r.text == "Here is the answer" for r in replies.values())
    assert len(completions.calls) == 3
    assert completions.calls[0]["reasoning_effort"] == "low"


def test_reasoning_effort_not_sent_to_other_models(fake_client):
    client, completions = fake_client()
    StudySquad(client, "qwen/qwen3.8-27b").ask_one("Nerd", [{"role": "user", "content": "hi"}])
    assert "reasoning_effort" not in completions.calls[0]


def test_api_errors_become_friendly_messages(fake_client):
    class RateLimitError(Exception):
        pass

    client, _ = fake_client(error=RateLimitError("429"))
    replies = StudySquad(client, "m").ask_all("q")
    assert all(not r.ok for r in replies.values())
    assert "rate limit" in replies["Nerd"].error.lower()


def test_empty_answer_is_an_error(fake_client):
    client, _ = fake_client(reply="   ")
    reply = StudySquad(client, "m").ask_one("Nerd", [])
    assert not reply.ok


def test_friendly_error_for_retired_model():
    msg = friendly_error(Exception("The model `llama-3.3-70b-versatile` has been decommissioned"))
    assert "not available" in msg


def test_on_reply_called_once_per_agent(fake_client):
    client, _ = fake_client("ok")
    seen = []
    replies = StudySquad(client, "m").ask_all("q", on_reply=lambda r: seen.append(r.agent))
    assert sorted(seen) == sorted(replies)
    assert list(replies) == list(AGENTS)  # result order is stable


def test_prompts_forbid_big_headings():
    msgs = build_messages(AGENTS["Nerd"], "q")
    assert "never use level-1 or level-2 headings" in msgs[0]["content"]
