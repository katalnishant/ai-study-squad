"""Data mining over the study logs.

* **Cleaning:** drop empty, shell-command and exit messages; filter stop words.
* **TF-IDF keyword mining:** rank the terms (unigrams + bigrams) that best
  characterise what the student studies.
* **Topic clustering:** TF-IDF -> LSA (Truncated SVD) -> K-Means. ``k`` is
  chosen automatically by the highest silhouette score; each cluster is named
  by the top TF-IDF terms of its member questions.
* **2-D topic map:** t-SNE projects the LSA vectors to 2-D so clusters can be
  plotted.
* **Usage statistics:** intent mix, lead-agent counts, latency, PDF usage, activity.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer

# Words that appear in almost every study question and say nothing about the topic.
DOMAIN_STOP_WORDS = {
    "explain",
    "explained",
    "explaining",
    "expalain",
    "explan",
    "tell",
    "teach",
    "please",
    "pls",
    "plz",
    "want",
    "know",
    "understand",
    "detail",
    "detailed",
    "details",
    "simple",
    "simply",
    "language",
    "laymans",
    "layman",
    "basic",
    "basics",
    "advance",
    "advanced",
    "scratch",
    "help",
    "need",
    "like",
    "using",
    "use",
    "does",
    "work",
    "works",
    "working",
    "example",
    "examples",
    "difference",
    "different",
    "between",
    "best",
    "approach",
    "solution",
    "solutions",
    "question",
    "questions",
    "topic",
    "okay",
    "ok",
    "hey",
    "hi",
    "yes",
    "chapter",
    "chap",
    "ch",
    "make",
    "way",
    "good",
    "better",
    "thing",
    "things",
    "mean",
    "means",
    "meaning",
    "really",
    "actually",
    "just",
    "dont",
    "don",
    "get",
    "words",
    "word",
    "terms",
    "term",
    "step",
    "steps",
    "ways",
    "important",
}
STOP_WORDS = sorted(ENGLISH_STOP_WORDS | DOMAIN_STOP_WORDS)

_SHELL_PREFIX = re.compile(r"^\s*(?:python3?|pip3?|echo|source|cd|export|streamlit|git|ls|sudo)\s")
_TOKEN = r"(?u)\b[a-zA-Z][a-zA-Z0-9+#]+\b"


def clean_queries(queries: Sequence[str]) -> list[str]:
    """Remove empty strings, 'exit' commands and pasted terminal commands."""
    cleaned = []
    for q in queries:
        q = (q or "").strip()
        if len(q) < 3 or q.lower().rstrip(" .!") in {"exit", "quit", "bye"}:
            continue
        if _SHELL_PREFIX.match(q) or "/users/" in q.lower() or "api_key" in q.lower():
            continue
        cleaned.append(q)
    return cleaned


def _vectorizer(min_df: int = 1) -> TfidfVectorizer:
    return TfidfVectorizer(
        stop_words=STOP_WORDS,
        token_pattern=_TOKEN,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=min_df,
    )


def top_keywords(queries: Sequence[str], n: int = 10) -> list[tuple[str, float]]:
    """Rank terms by their summed TF-IDF weight across all questions."""
    docs = clean_queries(queries)
    if not docs:
        return []
    vectorizer = _vectorizer()
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:  # every word was a stop word
        return []
    vocab = vectorizer.get_feature_names_out()
    scores = np.asarray(matrix.sum(axis=0)).ravel()
    order = scores.argsort()[::-1]
    picked: list[tuple[str, float]] = []
    for i in order:
        term = str(vocab[i])
        words = set(term.split())
        # skip a unigram already covered by a higher-ranked bigram, and vice versa
        if any(words <= set(p.split()) or set(p.split()) <= words for p, _ in picked):
            continue
        picked.append((term, round(float(scores[i]), 3)))
        if len(picked) == n:
            break
    return picked


@dataclass
class Cluster:
    id: int
    label: str
    terms: list[str]
    size: int
    examples: list[str]


@dataclass
class ClusterResult:
    k: int
    silhouette: float
    clusters: list[Cluster]
    labels: list[int]
    questions: list[str]
    coords: np.ndarray = field(repr=False)
    silhouette_by_k: dict[int, float] = field(default_factory=dict)
    lsa_dims: int = 0


def _project_2d(Z: np.ndarray, random_state: int) -> np.ndarray:
    """2-D map for plotting: t-SNE when there are enough points, otherwise SVD."""
    n = Z.shape[0]
    if n >= 10:
        tsne = TSNE(
            n_components=2,
            perplexity=min(15, (n - 1) / 3),
            metric="cosine",
            init="pca",
            random_state=random_state,
        )
        return tsne.fit_transform(Z)
    if Z.shape[1] >= 2:
        return Z[:, :2]
    return np.hstack([Z, np.zeros_like(Z)])


def cluster_topics(
    queries: Sequence[str],
    k: int | None = None,
    k_max: int = 12,
    random_state: int = 42,
) -> ClusterResult | None:
    """Group questions into topics.

    Pipeline: TF-IDF -> LSA (Truncated SVD + L2 normalisation) -> K-Means.
    LSA turns very sparse short-text vectors into dense "concept" vectors, which
    gives K-Means far cleaner clusters. ``k`` is picked by the best silhouette
    score. Returns None when there is too little data (fewer than 4 distinct
    usable questions).
    """
    docs = clean_queries(queries)
    if len({d.lower() for d in docs}) < 4:
        return None
    vectorizer = _vectorizer()
    try:
        X = vectorizer.fit_transform(docs)
    except ValueError:
        return None

    # Drop questions made only of stop words; they carry no topic signal.
    keep = np.asarray(X.getnnz(axis=1)).ravel() > 0
    docs = [d for d, kept in zip(docs, keep, strict=True) if kept]
    X = X[keep]
    n_docs, n_terms = X.shape
    n_distinct = len({d.lower() for d in docs})
    if n_distinct < 4 or n_terms < 3:
        return None

    dims = int(min(max(2, round(np.sqrt(n_docs)) + 2), n_terms - 1, n_docs - 1))
    lsa = make_pipeline(TruncatedSVD(n_components=dims, random_state=random_state), Normalizer())
    Z = lsa.fit_transform(X)
    # Identical questions collapse to one point; K-Means needs k <= distinct points.
    n_points = len(np.unique(Z.round(6), axis=0))
    if n_points < 3:
        return None

    def fit(n_clusters: int) -> KMeans:
        return KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state).fit(Z)

    scores: dict[int, float] = {}
    if k is None:
        for candidate in range(2, min(k_max, n_points - 1) + 1):
            labels_ = fit(candidate).labels_
            if len(set(labels_)) > 1:
                scores[candidate] = round(float(silhouette_score(Z, labels_)), 4)
        k = max(scores, key=scores.get) if scores else 2
    k = max(2, min(k, n_points - 1))

    model = fit(k)
    labels = model.labels_.tolist()
    silhouette = scores.get(k)
    if silhouette is None:
        silhouette = round(float(silhouette_score(Z, model.labels_)), 4)

    # Name each cluster by the highest mean TF-IDF terms of its member questions.
    terms = vectorizer.get_feature_names_out()
    clusters = []
    for cid in range(k):
        idx = np.flatnonzero(model.labels_ == cid)
        mean_weights = np.asarray(X[idx].mean(axis=0)).ravel()
        # A label term must appear in at least ~30% of the cluster's questions.
        doc_freq = np.asarray((X[idx] > 0).sum(axis=0)).ravel()
        min_df = max(1, int(np.ceil(0.3 * len(idx))))
        top: list[str] = []
        for i in mean_weights.argsort()[::-1]:
            if doc_freq[i] < min_df or mean_weights[i] == 0:
                continue
            term = str(terms[i])
            words = set(term.split())
            if any(words <= set(t.split()) for t in top):
                continue  # already covered, e.g. "network" after "neural network"
            narrower = [t for t in top if set(t.split()) < words]
            if narrower:  # prefer the phrase: "neural" -> "neural network"
                top[top.index(narrower[0])] = term
                top = [t for t in top if t not in narrower[1:]]
            else:
                top.append(term)
            if len(top) == 3:
                break
        label = " · ".join(top[:2])
        if not top:  # no shared vocabulary: a catch-all cluster
            top = [str(terms[i]) for i in mean_weights.argsort()[::-1][:2]]
            label = "mixed: " + " / ".join(top)
        examples = list(dict.fromkeys(docs[i] for i in idx))[:3]
        clusters.append(Cluster(cid, label, top, len(idx), examples))
    clusters.sort(key=lambda c: c.size, reverse=True)

    coords = _project_2d(Z, random_state)
    return ClusterResult(k, silhouette, clusters, labels, docs, coords, scores, dims)


def usage_stats(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts and averages for the analytics dashboard."""
    total = len(records)
    if total == 0:
        return {"total": 0}

    intents = Counter(r.get("intent") or "general" for r in records)
    leads = Counter(r.get("lead_agent") for r in records if r.get("lead_agent"))

    latencies: dict[str, list[int]] = {}
    failures = attempts = 0
    for r in records:
        for agent, ms in (r.get("latency_ms") or {}).items():
            latencies.setdefault(agent, []).append(int(ms))
        attempts += len(r.get("responses") or {})
        failures += len(r.get("errors") or {})

    per_day = Counter(str(r.get("timestamp", ""))[:10] for r in records if r.get("timestamp"))

    return {
        "total": total,
        "pdf_rate": round(sum(bool(r.get("used_pdf")) for r in records) / total, 3),
        "intent_counts": dict(intents),
        "lead_counts": dict(leads),
        "avg_latency_ms": {a: int(np.mean(v)) for a, v in latencies.items() if v},
        "error_rate": round(failures / attempts, 3) if attempts else 0.0,
        "per_day": dict(sorted(per_day.items())),
    }


def feedback_summary(feedback: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per-agent helpfulness from thumbs ratings: up/down counts and helpful rate."""
    summary: dict[str, dict[str, Any]] = {}
    for item in feedback:
        agent = item.get("agent")
        if not agent:
            continue
        row = summary.setdefault(agent, {"up": 0, "down": 0})
        row["up" if int(item.get("rating", 0)) == 1 else "down"] += 1
    for row in summary.values():
        total = row["up"] + row["down"]
        row["total"] = total
        row["helpful_rate"] = round(row["up"] / total, 3) if total else 0.0
    return summary
