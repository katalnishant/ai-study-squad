"""Generate data/sample_logs.jsonl, a labelled demo dataset for the Analytics tab.

The questions are typical study questions written for this demo (not real user
data). Every record has ``"sample": true`` and no answers or latencies, so the
dashboard never presents demo numbers as real measurements.

Run:  python scripts/make_sample_data.py
"""

from __future__ import annotations

import json
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from study_squad.router import detect_intent  # noqa: E402
from study_squad.storage import LogRecord  # noqa: E402

SUBJECTS: dict[str, list[str]] = {
    "Clustering": [
        "What is k-means clustering?",
        "How do I choose k in k-means clustering?",
        "Explain the elbow method for clustering",
        "Difference between k-means and hierarchical clustering",
        "What is DBSCAN clustering and when is it better than k-means?",
        "I don't get how k-means centroids are updated",
        "What is the silhouette score in clustering?",
        "Quiz me on clustering algorithms",
    ],
    "Decision trees & ensembles": [
        "How does a decision tree choose where to split?",
        "Derive the formula for entropy and information gain",
        "What is the Gini index in decision trees?",
        "Explain decision tree pruning",
        "What is random forest and why is it better than a single decision tree?",
        "How does gradient boosting work?",
        "Explain XGBoost in simple words",
        "Difference between bagging and boosting",
    ],
    "Regression": [
        "What is linear regression?",
        "Derive the normal equation for linear regression",
        "What is the cost function in linear regression?",
        "Explain gradient descent for linear regression",
        "I'm confused about the learning rate in gradient descent",
        "What is logistic regression and how is it different from linear regression?",
        "What is overfitting in regression models?",
        "Am I right that R-squared always increases when you add features?",
    ],
    "Association rules & data mining": [
        "What is the Apriori algorithm?",
        "Explain support, confidence and lift in association rules",
        "How does FP-growth improve on Apriori?",
        "What is market basket analysis?",
        "Explain association rule mining with an example",
        "What is the KDD process in data mining?",
        "Explain data preprocessing steps in data mining",
        "What is data normalization and why do we need it?",
    ],
    "Neural networks": [
        "How does a neural network learn?",
        "Explain backpropagation step by step",
        "What is an activation function in a neural network?",
        "Explain ReLU vs sigmoid activation functions",
        "What is a convolutional neural network?",
        "Explain dropout in neural networks in simple terms",
        "Test me on neural network basics",
    ],
    "Operating systems": [
        "What is a deadlock in operating systems?",
        "What are the four necessary conditions for deadlock?",
        "Explain the banker's algorithm for deadlock avoidance",
        "Difference between deadlock prevention and deadlock avoidance",
        "What is process scheduling in an operating system?",
        "Explain round robin CPU scheduling",
    ],
    "Computer networks": [
        "Explain the TCP three-way handshake",
        "Difference between TCP and UDP",
        "What is the OSI model?",
        "Explain subnetting with an example",
        "How does DNS resolution work?",
        "What is congestion control in TCP?",
    ],
    "Data structures & algorithms": [
        "Explain Floyd's tortoise and hare algorithm",
        "What is the time complexity of binary search?",
        "Explain binary search on answer with an example",
        "How do I remove linked list elements with a given value?",
        "What is dynamic programming?",
        "Explain the optimal approach for the split array largest sum problem",
        "What is the difference between BFS and DFS?",
    ],
}
QUESTIONS = [q for qs in SUBJECTS.values() for q in qs]
SUBJECT_OF = {q: subject for subject, qs in SUBJECTS.items() for q in qs}

AGENTS = ("Nerd", "Simplifier", "Challenger")


def main() -> None:
    rng = random.Random(7)
    start = datetime(2026, 9, 14, 9, 0)
    out = ROOT / "data" / "sample_logs.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for i, q in enumerate(QUESTIONS):
            route = detect_intent(q)
            ts = start + timedelta(days=i * 18 // len(QUESTIONS), minutes=rng.randint(0, 600))
            record = LogRecord(
                query=q,
                responses={a: "" for a in AGENTS},
                timestamp=ts.isoformat(timespec="seconds"),
                session_id="sample",
                intent=route.intent,
                lead_agent=route.lead,
                used_pdf=rng.random() < 0.35,
                sample=True,
            ).to_dict()
            record["subject"] = SUBJECT_OF[q]  # ground-truth label used by evaluate_clustering.py
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(QUESTIONS)} sample records to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
