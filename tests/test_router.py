import pytest

from study_squad.router import detect_intent, order_agents


@pytest.mark.parametrize(
    "text",
    [
        "I don't get it",
        "i dont understand backpropagation",
        "I’m totally lost",  # curly apostrophe
        "Can you explain again?",
        "explain leetcode 1539 in laymans language",
        "This is confusing",
        "still not clear",
    ],
)
def test_confusion_routes_to_simplifier(text):
    route = detect_intent(text)
    assert route.intent == "confused"
    assert route.lead == "Simplifier"
    assert route.matched


@pytest.mark.parametrize("text", ["Quiz me on trees", "am I right about entropy?", "check my answer"])
def test_challenge_routes_to_challenger(text):
    assert detect_intent(text).lead == "Challenger"


@pytest.mark.parametrize(
    "text", ["Derive the normal equation", "time complexity of merge sort", "explain in detail"]
)
def test_depth_routes_to_nerd(text):
    assert detect_intent(text).lead == "Nerd"


def test_general_question_has_no_lead():
    route = detect_intent("What is k-means clustering?")
    assert route.intent == "general"
    assert route.lead is None


def test_confusion_beats_depth():
    # "formula" asks for depth, but confusion has priority
    assert detect_intent("I don't get this formula").lead == "Simplifier"


def test_no_false_positive_on_loss_function():
    assert detect_intent("what is a loss function").intent == "general"


def test_order_agents_moves_lead_first():
    keys = ["Nerd", "Simplifier", "Challenger"]
    assert order_agents(keys, "Challenger") == ["Challenger", "Nerd", "Simplifier"]
    assert order_agents(keys, None) == keys
    assert order_agents(keys, "Unknown") == keys
