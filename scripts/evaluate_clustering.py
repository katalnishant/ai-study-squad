"""Evaluate the topic-clustering pipeline against known subject labels.

The sample dataset has 58 questions from 8 subjects. We hide the labels, run
each clustering variant, and compare its clusters to the true subjects with the
Adjusted Rand Index (ARI: 1.0 = perfect agreement, 0.0 = random).

Run:  python scripts/evaluate_clustering.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from make_sample_data import SUBJECT_OF, SUBJECTS  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

from study_squad.mining import _vectorizer, cluster_topics  # noqa: E402


def kmeans_on_raw_tfidf(questions: list[str], k: int) -> list[int]:
    X = _vectorizer().fit_transform(questions)
    return KMeans(n_clusters=k, n_init=10, random_state=42).fit(X).labels_.tolist()


def best_k_on_raw_tfidf(questions: list[str], k_max: int = 12) -> list[int]:
    X = _vectorizer().fit_transform(questions)
    fits = [KMeans(n_clusters=k, n_init=10, random_state=42).fit(X) for k in range(2, k_max + 1)]
    best = max(fits, key=lambda m: silhouette_score(X, m.labels_))
    return best.labels_.tolist()


def main() -> None:
    questions = [q for qs in SUBJECTS.values() for q in qs]
    truth = [SUBJECT_OF[q] for q in questions]
    n_subjects = len(SUBJECTS)

    rows = []
    labels = best_k_on_raw_tfidf(questions)
    rows.append(("TF-IDF + K-Means (k by silhouette)", len(set(labels)), adjusted_rand_score(truth, labels)))
    labels = kmeans_on_raw_tfidf(questions, n_subjects)
    rows.append(
        (
            f"TF-IDF + K-Means (k = {n_subjects}, told the answer)",
            n_subjects,
            adjusted_rand_score(truth, labels),
        )
    )

    result = cluster_topics(questions)
    assert result is not None and result.questions == questions
    rows.append(
        (
            "TF-IDF + LSA + K-Means (k by silhouette) [used in app]",
            result.k,
            adjusted_rand_score(truth, result.labels),
        )
    )
    fixed = cluster_topics(questions, k=n_subjects)
    rows.append(
        (
            f"TF-IDF + LSA + K-Means (k = {n_subjects}, told the answer)",
            n_subjects,
            adjusted_rand_score(truth, fixed.labels),
        )
    )

    print(f"{len(questions)} questions, {n_subjects} true subjects\n")
    print(f"{'Method':<58}{'k':>4}{'ARI':>8}")
    for name, k, ari in rows:
        print(f"{name:<58}{k:>4}{ari:>8.3f}")


if __name__ == "__main__":
    main()
