"""The three study agents and the orchestration that runs them in parallel."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class Agent:
    key: str
    name: str
    emoji: str
    role: str
    system_prompt: str
    icon: str = ":material/smart_toy:"  # Streamlit Material icon
    color: str = "gray"  # Streamlit markdown/badge colour name
    tagline: str = ""


SHARED_RULES = (
    "You are part of AI Study Squad, a team of three tutors who each answer the same "
    "student question from a different angle. Format answers in Markdown, but never use "
    "level-1 or level-2 headings (# or ##); use ### or **bold** for section titles. Use LaTeX "
    "($...$) only for real formulas. If textbook excerpts are provided, prefer them, "
    "cite pages like (p. 12), and say so plainly when the excerpts do not cover the question. "
    "Never invent page numbers."
)

AGENTS: dict[str, Agent] = {
    "Nerd": Agent(
        key="Nerd",
        name="The Nerd",
        emoji="🎓",
        role="Technical expert",
        icon=":material/school:",
        color="violet",
        tagline="Precise definitions, formulas, complexity and a worked example.",
        system_prompt=(
            "You are The Nerd. Give a rigorous, well-structured technical answer: a precise "
            "definition, how it works step by step, the key formula or complexity where "
            "relevant, and one short worked example. Use headings and bullet points. "
            "Aim for 200-350 words."
        ),
    ),
    "Simplifier": Agent(
        key="Simplifier",
        name="The Simplifier",
        emoji="😊",
        role="Plain-language explainer",
        icon=":material/lightbulb:",
        color="green",
        tagline="One everyday analogy and plain words, so it finally clicks.",
        system_prompt=(
            "You are The Simplifier. Explain the idea like a friendly senior student: one "
            "everyday analogy, plain words, no jargon unless you immediately explain it, and a "
            "few emojis. End with a one-line 'In short:' summary. Aim for 120-200 words."
        ),
    ),
    "Challenger": Agent(
        key="Challenger",
        name="The Challenger",
        emoji="🤔",
        role="Critical thinker",
        icon=":material/psychology_alt:",
        color="orange",
        tagline="Probing questions that test whether you really understand.",
        system_prompt=(
            "You are The Challenger. State the core idea in one sentence, then ask 2-3 "
            "probing questions that expose common misconceptions or edge cases (start them "
            "with 'Wait, are you sure...' or 'But what happens when...'). Do not answer your "
            "own questions. Finish with a one-line hint. Aim for 80-150 words."
        ),
    ),
}

AGENT_ORDER: list[str] = list(AGENTS)

INTENT_HINTS: dict[str, dict[str, str]] = {
    "confused": {
        "Simplifier": "The student says they are confused. You are the lead tutor for this "
        "question: start from zero, go slowly, and use the simplest possible analogy.",
        "Nerd": "The student is confused, so keep the technical answer short and gentle.",
        "Challenger": "The student is confused, so ask easier, confidence-building questions.",
    },
    "challenge": {
        "Challenger": "The student wants to be tested. You are the lead tutor: make your "
        "questions exam-style and specific.",
    },
    "deep": {
        "Nerd": "The student wants depth. You are the lead tutor: include derivations, "
        "formulas or complexity analysis as relevant, up to 500 words.",
    },
}


class ChatClient(Protocol):
    """The subset of the Groq client used here (lets tests pass a fake)."""

    chat: Any


@dataclass
class AgentReply:
    agent: str
    text: str
    latency_ms: int
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def build_messages(
    agent: Agent,
    question: str,
    *,
    context: str | None = None,
    history: Sequence[tuple[str, str]] = (),
    intent: str = "general",
) -> list[dict[str, str]]:
    """Build the chat messages for one agent.

    ``history`` holds earlier (question, this agent's answer) pairs so follow-ups
    like "how does it work?" make sense. Textbook ``context`` is attached only to
    the current question.
    """
    system = f"{SHARED_RULES}\n\n{agent.system_prompt}"
    hint = INTENT_HINTS.get(intent, {}).get(agent.key)
    if hint:
        system += f"\n\n{hint}"

    messages = [{"role": "system", "content": system}]
    for past_question, past_answer in history:
        messages.append({"role": "user", "content": past_question})
        messages.append({"role": "assistant", "content": past_answer})

    if context:
        user = (
            "Textbook excerpts (from the student's uploaded PDF):\n"
            f"<excerpts>\n{context}\n</excerpts>\n\n"
            f"Question: {question}"
        )
    else:
        user = question
    messages.append({"role": "user", "content": user})
    return messages


def friendly_error(exc: Exception) -> str:
    """Turn API exceptions into messages a student can act on."""
    name = type(exc).__name__
    text = str(exc)
    if name == "AuthenticationError":
        return "Invalid or missing Groq API key. Add a valid key in the sidebar or .env file."
    if name == "RateLimitError":
        return "Groq rate limit reached. Wait a few seconds and ask again."
    if "decommissioned" in text or name == "NotFoundError":
        return "This model is not available on Groq any more. Pick another model in the sidebar."
    if name in {"APIConnectionError", "APITimeoutError"}:
        return "Could not reach Groq. Check your internet connection and try again."
    return f"Something went wrong ({name}). Please try again."


class StudySquad:
    """Runs every agent on the same question concurrently."""

    def __init__(
        self,
        client: ChatClient,
        model: str,
        *,
        temperature: float = 0.6,
        reasoning_effort: str | None = "low",
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def _request_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"model": self.model, "temperature": self.temperature}
        # reasoning_effort is only accepted by Groq's reasoning models (gpt-oss).
        if self.reasoning_effort and self.model.startswith("openai/gpt-oss"):
            kwargs["reasoning_effort"] = self.reasoning_effort
        return kwargs

    def ask_one(self, agent_key: str, messages: list[dict[str, str]]) -> AgentReply:
        start = time.perf_counter()
        try:
            response = self.client.chat.completions.create(messages=messages, **self._request_kwargs())
            text = (response.choices[0].message.content or "").strip()
            error = None if text else "The model returned an empty answer. Please try again."
        except Exception as exc:  # noqa: BLE001 - every failure becomes a readable message
            text, error = "", friendly_error(exc)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return AgentReply(agent=agent_key, text=text, latency_ms=latency_ms, error=error)

    def ask_all(
        self,
        question: str,
        *,
        context: str | None = None,
        history: dict[str, Sequence[tuple[str, str]]] | None = None,
        intent: str = "general",
        agent_keys: Sequence[str] = AGENT_ORDER,
        on_reply: Callable[[AgentReply], None] | None = None,
    ) -> dict[str, AgentReply]:
        """Ask every agent concurrently.

        ``on_reply`` is called in the caller's thread as each agent finishes, so a UI
        can show progress without touching UI objects from worker threads.
        """
        history = history or {}
        jobs = {
            key: build_messages(
                AGENTS[key],
                question,
                context=context,
                history=history.get(key, ()),
                intent=intent,
            )
            for key in agent_keys
        }
        results: dict[str, AgentReply] = {}
        with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
            futures = [pool.submit(self.ask_one, key, msgs) for key, msgs in jobs.items()]
            for future in as_completed(futures):
                reply = future.result()
                results[reply.agent] = reply
                if on_reply is not None:
                    on_reply(reply)
        return {key: results[key] for key in jobs}


def make_groq_client(api_key: str) -> ChatClient:
    from groq import Groq

    return Groq(api_key=api_key, max_retries=2, timeout=60)
