"""Rule-based intent detection that decides which agent leads the answer.

Each message is normalised and matched against keyword patterns for three
intents. The first intent that matches (in priority order) wins:

* ``confused``  -> The Simplifier leads ("I don't get it", "explain again", ...)
* ``challenge`` -> The Challenger leads ("quiz me", "am I right", ...)
* ``deep``      -> The Nerd leads ("derive", "time complexity", "proof", ...)

Anything else is ``general`` and keeps the default order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "confused": (
        r"\b(?:don'?t|do not|didn'?t|did not|still don'?t|can'?t|cannot)\s+(?:get|understand|follow)\b",
        r"\bconfus\w*",
        r"\b(?:i'?m|i am|totally|completely|so)\s+lost\b",
        r"\bexplain (?:it )?again\b",
        r"\b(?:simpler|simplify|simple words|simple terms|simple language)\b",
        r"\bin simple\b",
        r"\beli5\b",
        r"\blay ?m[ae]n'?s?\b",
        r"\b(?:too|very) (?:hard|complex|complicated|difficult)\b",
        r"\b(?:not|isn'?t|still not) clear\b",
        r"\bmakes? no sense\b",
        r"\bwhat does (?:that|this|it) mean\b",
    ),
    "challenge": (
        r"\b(?:quiz|test|challenge|grill) me\b",
        r"\bam i (?:right|correct|wrong)\b",
        r"\bis (?:this|that|my \w+) (?:right|correct)\b",
        r"\bcheck my (?:answer|understanding|solution|logic)\b",
        r"\binterview questions?\b",
        r"\bcross[- ]question\b",
        r"\bask me\b",
    ),
    "deep": (
        r"\bderiv(?:e|ation)\b",
        r"\bproofs?\b|\bprove\b",
        r"\bformulas?\b|\bequations?\b",
        r"\bmathematical(?:ly)?\b",
        r"\b(?:time|space) complexity\b|\bbig[- ]o\b",
        r"\bin (?:depth|detail)\b|\bdetailed\b",
        r"\btechnical(?:ly)?\b",
        r"\boptimal (?:approach|solution)\b",
    ),
}

INTENT_PRIORITY: tuple[str, ...] = ("confused", "challenge", "deep")

INTENT_LEAD: dict[str, str] = {
    "confused": "Simplifier",
    "challenge": "Challenger",
    "deep": "Nerd",
}

INTENT_LABEL: dict[str, str] = {
    "confused": "Confusion detected",
    "challenge": "Wants to be tested",
    "deep": "Wants depth",
    "general": "General question",
}

_COMPILED = {intent: tuple(re.compile(p) for p in patterns) for intent, patterns in INTENT_PATTERNS.items()}


@dataclass(frozen=True)
class Route:
    intent: str
    lead: str | None
    matched: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        return INTENT_LABEL[self.intent]


def normalize(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text).strip()


def detect_intent(text: str) -> Route:
    """Classify a student message and pick the agent that should lead."""
    clean = normalize(text)
    for intent in INTENT_PRIORITY:
        hits = tuple(m.group(0) for p in _COMPILED[intent] if (m := p.search(clean)))
        if hits:
            return Route(intent=intent, lead=INTENT_LEAD[intent], matched=hits)
    return Route(intent="general", lead=None)


def order_agents(agent_keys: list[str], lead: str | None) -> list[str]:
    """Move the lead agent to the front, keeping the others in their original order."""
    if lead is None or lead not in agent_keys:
        return list(agent_keys)
    return [lead] + [k for k in agent_keys if k != lead]
