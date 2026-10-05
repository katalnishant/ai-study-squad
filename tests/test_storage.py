import json

from study_squad.storage import FeedbackRecord, LogRecord, LogStore, normalize, read_feedback, read_records


def test_append_and_load_roundtrip(tmp_path):
    store = LogStore(tmp_path / "logs.jsonl")
    store.append(LogRecord(query="what is apriori?", responses={"Nerd": "..."}, intent="general"))
    store.append(LogRecord(query="quiz me", responses={}, intent="challenge", lead_agent="Challenger"))
    records = store.load()
    assert [r["query"] for r in records] == ["what is apriori?", "quiz me"]
    assert records[1]["lead_agent"] == "Challenger"


def test_reads_both_legacy_formats(tmp_path):
    legacy = [
        {"query": "what is clustering", "responses": {"Nerd": "x"}},
        {"timestamp": "2026-04-22 12:45:49", "topic": "what is boosting", "agents": {"Nerd": "Error: quota"}},
        {"user_query": "old key", "responses": {}},
        {"query": None, "responses": {}},
    ]
    path = tmp_path / "study_logs.json"
    path.write_text(json.dumps(legacy))
    records = read_records(path)
    assert [r["query"] for r in records] == ["what is clustering", "what is boosting", "old key"]
    assert records[1]["timestamp"] == "2026-04-22 12:45:49"
    assert records[1]["errors"] == {"Nerd": "Error: quota"}


def test_corrupt_line_is_skipped(tmp_path):
    path = tmp_path / "logs.jsonl"
    path.write_text('{"query": "ok", "responses": {}}\n{"query": "half-writ\n')
    assert [r["query"] for r in read_records(path)] == ["ok"]


def test_missing_file_is_empty(tmp_path):
    assert read_records(tmp_path / "nope.jsonl") == []


def test_normalize_rejects_garbage():
    assert normalize("not a dict") is None
    assert normalize({"responses": {}}) is None


def test_feedback_latest_rating_wins(tmp_path):
    store = LogStore(tmp_path / "feedback.jsonl")
    store.append(FeedbackRecord(turn_id="t1", agent="Nerd", rating=0))
    store.append(FeedbackRecord(turn_id="t1", agent="Nerd", rating=1))  # changed their mind
    store.append(FeedbackRecord(turn_id="t1", agent="Simplifier", rating=1))
    (tmp_path / "feedback.jsonl").open("a").write("not json\n")
    items = read_feedback(tmp_path / "feedback.jsonl")
    assert sorted((i["agent"], i["rating"]) for i in items) == [("Nerd", 1), ("Simplifier", 1)]
