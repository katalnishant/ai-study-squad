from study_squad.mining import clean_queries, cluster_topics, feedback_summary, top_keywords, usage_stats

TOPICS = {
    "clustering": [
        "what is k-means clustering",
        "how to choose k in k-means clustering",
        "hierarchical clustering vs k-means clustering",
        "dbscan clustering explained",
    ],
    "deadlock": [
        "what is a deadlock in operating systems",
        "conditions for deadlock",
        "deadlock prevention vs deadlock avoidance",
        "banker's algorithm for deadlock",
    ],
    "regression": [
        "what is linear regression",
        "cost function of linear regression",
        "gradient descent for linear regression",
        "linear regression assumptions",
    ],
}
QUESTIONS = [q for qs in TOPICS.values() for q in qs]


def test_clean_queries_drops_noise():
    raw = ["exit ", "", "python -c 'import os'", 'echo "GROQ_API_KEY=x" > .env', "what is entropy"]
    assert clean_queries(raw) == ["what is entropy"]


def test_top_keywords_finds_main_topics():
    terms = [t for t, _ in top_keywords(QUESTIONS, 5)]
    for topic in ("clustering", "deadlock"):
        assert any(topic in t for t in terms)
    assert "explain" not in terms and "what" not in terms


def test_top_keywords_empty_input():
    assert top_keywords([]) == []
    assert top_keywords(["what is it", "explain me"]) == []


def test_clustering_recovers_the_three_topics():
    result = cluster_topics(QUESTIONS, k=3)
    assert result.k == 3
    groups = {}
    for question, label in zip(result.questions, result.labels, strict=True):
        groups.setdefault(label, set()).add(question)
    expected = [set(qs) for qs in TOPICS.values()]
    assert sorted(map(sorted, groups.values())) == sorted(map(sorted, expected))
    assert result.coords.shape == (len(QUESTIONS), 2)


def test_k_is_chosen_by_silhouette():
    result = cluster_topics(QUESTIONS)
    assert result.silhouette_by_k
    assert result.k == max(result.silhouette_by_k, key=result.silhouette_by_k.get)


def test_clustering_is_deterministic():
    a, b = cluster_topics(QUESTIONS), cluster_topics(QUESTIONS)
    assert a.labels == b.labels and a.k == b.k


def test_too_little_data_returns_none():
    assert cluster_topics(["what is clustering"] * 10) is None
    assert cluster_topics(["a b", "c d"]) is None


def test_usage_stats():
    records = [
        {
            "query": "q1",
            "intent": "confused",
            "lead_agent": "Simplifier",
            "used_pdf": True,
            "latency_ms": {"Nerd": 1000, "Simplifier": 500},
            "responses": {"Nerd": "", "Simplifier": ""},
            "errors": {},
            "timestamp": "2026-10-01T10:00:00",
        },
        {
            "query": "q2",
            "intent": "general",
            "lead_agent": None,
            "used_pdf": False,
            "latency_ms": {"Nerd": 3000},
            "responses": {"Nerd": ""},
            "errors": {"Nerd": "rate limit"},
            "timestamp": "2026-10-02T10:00:00",
        },
    ]
    stats = usage_stats(records)
    assert stats["total"] == 2
    assert stats["pdf_rate"] == 0.5
    assert stats["intent_counts"] == {"confused": 1, "general": 1}
    assert stats["lead_counts"] == {"Simplifier": 1}
    assert stats["avg_latency_ms"]["Nerd"] == 2000
    assert stats["error_rate"] == round(1 / 3, 3)
    assert stats["per_day"] == {"2026-10-01": 1, "2026-10-02": 1}
    assert usage_stats([]) == {"total": 0}


def test_pipeline_quality_on_labelled_sample():
    """Guards the documented result: LSA clustering recovers subjects with ARI >= 0.5."""
    import sys
    from pathlib import Path

    from sklearn.metrics import adjusted_rand_score

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from make_sample_data import SUBJECT_OF, SUBJECTS

    questions = [q for qs in SUBJECTS.values() for q in qs]
    result = cluster_topics(questions)
    assert adjusted_rand_score([SUBJECT_OF[q] for q in questions], result.labels) >= 0.5


def test_feedback_summary():
    votes = [
        {"agent": "Nerd", "rating": 1},
        {"agent": "Nerd", "rating": 0},
        {"agent": "Simplifier", "rating": 1},
    ]
    summary = feedback_summary(votes)
    assert summary["Nerd"] == {"up": 1, "down": 1, "total": 2, "helpful_rate": 0.5}
    assert summary["Simplifier"]["helpful_rate"] == 1.0
    assert feedback_summary([]) == {}
