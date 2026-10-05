"""Altair charts for the Analytics tab (light theme, validated reference palette)."""

from __future__ import annotations

import altair as alt
import pandas as pd

from study_squad.mining import ClusterResult

BLUE = "#2a78d6"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]  # first three categorical slots: CVD-safe as a set
MUTED = "#c9c8c2"
INK_SECONDARY = "#52514e"
HIGHLIGHT = "#eda100"


def bar_chart(df: pd.DataFrame, x: str, y: str, x_title: str, fmt: str = "d") -> alt.Chart:
    """Horizontal bars sorted by value, with a tooltip on every bar."""
    x_axis = alt.Axis(tickMinStep=1, format="d") if fmt == "d" else alt.Axis()
    return (
        alt.Chart(df)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title, axis=x_axis),
            y=alt.Y(
                f"{y}:N",
                sort="-x",
                title=None,
                scale=alt.Scale(paddingInner=0.35),
                axis=alt.Axis(labelOverlap=False, labelLimit=220),
            ),
            tooltip=[
                alt.Tooltip(f"{y}:N", title=y.title()),
                alt.Tooltip(f"{x}:Q", title=x_title, format=fmt),
            ],
        )
        .properties(height=alt.Step(36))
    )


def topic_map(result: ClusterResult) -> alt.LayerChart:
    """Scatter of questions in t-SNE space; the 3 largest topics coloured, the rest muted."""
    by_id = {c.id: c for c in result.clusters}
    top3 = [c.id for c in result.clusters[:3]]
    df = pd.DataFrame(
        {
            "x": result.coords[:, 0],
            "y": result.coords[:, 1],
            "question": result.questions,
            "topic": [by_id[label].label for label in result.labels],
            "group": [by_id[label].label if label in top3 else "Other topics" for label in result.labels],
        }
    )
    domain = [by_id[i].label for i in top3] + ["Other topics"]
    points = (
        alt.Chart(df)
        .mark_circle(size=110, opacity=0.95, stroke="#ffffff", strokeWidth=2)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                "group:N",
                scale=alt.Scale(domain=domain, range=SERIES[: len(top3)] + [MUTED]),
                legend=alt.Legend(title=None, orient="bottom", labelLimit=320),
            ),
            tooltip=[alt.Tooltip("question:N", title="Question"), alt.Tooltip("topic:N", title="Topic")],
        )
    )
    centers = df.groupby("topic", as_index=False)[["x", "y"]].mean()
    labels = (
        alt.Chart(centers)
        .mark_text(fontSize=11, fontWeight=600, dy=-15, color=INK_SECONDARY)
        .encode(x="x:Q", y="y:Q", text="topic:N")
    )
    return (points + labels).properties(height=440)


def silhouette_chart(result: ClusterResult) -> alt.LayerChart:
    df = pd.DataFrame(
        {"k": list(result.silhouette_by_k), "silhouette": list(result.silhouette_by_k.values())}
    )
    line = (
        alt.Chart(df)
        .mark_line(color=BLUE, strokeWidth=2, point=alt.OverlayMarkDef(size=64, color=BLUE, filled=True))
        .encode(
            x=alt.X("k:O", title="Number of topics (k)", axis=alt.Axis(labelAngle=0)),
            y=alt.Y("silhouette:Q", title="Silhouette score"),
            tooltip=[
                alt.Tooltip("k:O", title="k"),
                alt.Tooltip("silhouette:Q", title="Silhouette", format=".3f"),
            ],
        )
    )
    chosen = (
        alt.Chart(pd.DataFrame({"k": [result.k]}))
        .mark_rule(color=HIGHLIGHT, strokeDash=[4, 4], strokeWidth=2)
        .encode(x="k:O")
    )
    return (line + chosen).properties(height=220)


def activity_chart(per_day: dict[str, int]) -> alt.Chart:
    days = pd.DataFrame(list(per_day.items()), columns=["day", "questions"])
    days["day"] = pd.to_datetime(days["day"]).dt.strftime("%d %b")
    return (
        alt.Chart(days)
        .mark_bar(color=BLUE, cornerRadiusEnd=4)
        .encode(
            x=alt.X("day:O", title=None, sort=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("questions:Q", title="Questions", axis=alt.Axis(tickMinStep=1, format="d")),
            tooltip=[alt.Tooltip("day:O", title="Day"), alt.Tooltip("questions:Q", title="Questions")],
        )
        .properties(height=200)
    )


def helpfulness_chart(summary: dict[str, dict], names: dict[str, str]) -> alt.Chart:
    """Stacked helpful / not helpful votes per agent."""
    rows = []
    for agent, row in summary.items():
        rows.append({"agent": names.get(agent, agent), "vote": "Helpful", "count": row["up"]})
        rows.append({"agent": names.get(agent, agent), "vote": "Not helpful", "count": row["down"]})
    df = pd.DataFrame(rows)
    return (
        alt.Chart(df)
        .mark_bar(cornerRadiusEnd=4, stroke="#ffffff", strokeWidth=2)
        .encode(
            x=alt.X("count:Q", title="Votes", stack=True, axis=alt.Axis(tickMinStep=1, format="d")),
            y=alt.Y("agent:N", title=None, scale=alt.Scale(paddingInner=0.35)),
            color=alt.Color(
                "vote:N",
                scale=alt.Scale(domain=["Helpful", "Not helpful"], range=[SERIES[2], MUTED]),
                legend=alt.Legend(title=None, orient="bottom"),
            ),
            order=alt.Order("vote:N"),
            tooltip=[
                alt.Tooltip("agent:N", title="Agent"),
                alt.Tooltip("vote:N", title="Vote"),
                alt.Tooltip("count:Q", title="Count"),
            ],
        )
        .properties(height=alt.Step(40))
    )
