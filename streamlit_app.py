"""AI Study Squad: Streamlit front end.

Run locally with:  streamlit run streamlit_app.py
"""

from __future__ import annotations

import uuid
from dataclasses import asdict

import pandas as pd
import streamlit as st

from study_squad import __version__
from study_squad.agents import AGENT_ORDER, AGENTS, AgentReply, StudySquad, make_groq_client
from study_squad.charts import activity_chart, bar_chart, helpfulness_chart, silhouette_chart, topic_map
from study_squad.config import AVAILABLE_MODELS, load_settings, model_label
from study_squad.mining import cluster_topics, feedback_summary, top_keywords, usage_stats
from study_squad.rag import TextbookIndex, format_context
from study_squad.router import INTENT_LABEL, detect_intent, order_agents
from study_squad.storage import FeedbackRecord, LogRecord, LogStore, read_feedback, read_records

st.set_page_config(page_title="AI Study Squad", page_icon=":material/school:", layout="wide")
settings = load_settings()

MIN_SIMILARITY = 0.25  # ignore textbook chunks that barely match the question
COMPARE = "compare"
REPO_URL = "https://github.com/katalnishant/ai-study-squad"

ACCENT = {"Nerd": "#7c3aed", "Simplifier": "#059669", "Challenger": "#ea580c"}
INTENT_STYLE = {  # badge colour and icon for each detected intent
    "confused": ("green", ":material/help:"),
    "challenge": ("orange", ":material/quiz:"),
    "deep": ("violet", ":material/science:"),
    "general": ("gray", ":material/chat_bubble:"),
}
FOLLOW_UPS = [  # (button label, icon, question sent); each phrase triggers the router
    (
        "Explain simpler",
        ":material/sentiment_satisfied:",
        "I don't get it. Explain that again in simpler terms.",
    ),
    ("Go deeper", ":material/science:", "Explain that in detail, with the math or complexity."),
    ("Quiz me", ":material/quiz:", "Quiz me on this."),
]
EXAMPLES = [
    "What is k-means clustering?",
    "I don't get how gradient descent works",
    "Quiz me on decision trees",
    "Derive the formula for information gain",
    "TCP vs UDP: when should I use each?",
]

# --------------------------------------------------------------------------- styles
accent_css = "\n".join(
    f'[class*="st-key-answer-{k}"] {{ border-top: 3px solid {c} !important; }}' for k, c in ACCENT.items()
)
st.markdown(
    f"""
<style>
.block-container {{ max-width: 1120px; padding-top: 3.2rem; padding-bottom: 7rem; }}
.brand {{ display: flex; align-items: center; gap: .8rem; }}
.brand h1 {{ font-size: 1.45rem; font-weight: 700; margin: 0; padding: 0; letter-spacing: -0.01em; }}
.brand p {{ margin: 0; color: #6b7280; font-size: .9rem; }}
.hero {{ text-align: center; padding: 1.6rem 0 .6rem; }}
.hero h2 {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: .3rem; }}
.hero p {{ color: #6b7280; font-size: 1.02rem; max-width: 620px; margin: 0 auto; }}
[class*="st-key-answer-"] {{ background: #ffffff; box-shadow: 0 1px 2px rgba(16,24,40,.04); }}
[class*="st-key-answer-"] p, [class*="st-key-answer-"] li {{ line-height: 1.65; }}
[class*="st-key-agentcard-"] {{ background: #fafafb; }}
{accent_css}
[data-testid="stChatMessage"] {{ background: #f5f6f8; border-radius: .8rem; padding: .7rem 1rem; }}
[data-testid="stMetric"] {{ background: #ffffff; }}
[class*="st-key-answer-"] [data-testid="stHeaderActionElements"] {{ display: none; }}
</style>
""",
    unsafe_allow_html=True,
)

LOGO = """
<svg width="38" height="38" viewBox="0 0 38 38" aria-hidden="true">
  <circle cx="14" cy="15" r="10" fill="#7c3aed" fill-opacity=".9"/>
  <circle cx="24" cy="15" r="10" fill="#059669" fill-opacity=".85"/>
  <circle cx="19" cy="24" r="10" fill="#ea580c" fill-opacity=".85"/>
</svg>
"""


# --------------------------------------------------------------------------- state
def init_state() -> None:
    defaults = {
        "session_id": uuid.uuid4().hex[:12],
        "turns": [],
        "session_logs": [],
        "session_feedback": {},
        "pdf": None,
        "pdf_file_id": None,
        "user_api_key": "",
        "pending_question": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


init_state()


@st.cache_resource(show_spinner=False)
def get_index() -> TextbookIndex:
    return TextbookIndex(persist_dir=settings.chroma_dir)


def get_client(api_key: str):
    return make_groq_client(api_key)  # cheap to build; not cached so keys never linger in memory


def api_key() -> str | None:
    return st.session_state.user_api_key.strip() or settings.groq_api_key or None


def ask(question: str) -> None:
    """Queue a question (from a button or pill) to be answered on the next run."""
    st.session_state.pending_question = question


def on_example_pick() -> None:
    picked = st.session_state.get("example_pick")
    if picked:
        ask(picked)
    st.session_state.example_pick = None


def on_feedback(turn: dict, agent_key: str) -> None:
    rating = st.session_state.get(f"fb-{turn['id']}-{agent_key}")
    if rating is None:
        return
    record = FeedbackRecord(
        turn_id=turn["id"],
        agent=agent_key,
        rating=int(rating),
        query=turn["question"],
        intent=turn["route"]["intent"],
    ).to_dict()
    st.session_state.session_feedback[(turn["id"], agent_key)] = record
    if settings.persist_logs:
        LogStore(settings.feedback_path).append(record)


# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("#### :material/tune: Settings")
    key_input = {
        "type": "password",
        "placeholder": "gsk_…",
        "help": "Free at console.groq.com/keys. Kept only for this browser session.",
    }
    if settings.groq_key_problem:
        st.warning(settings.groq_key_problem, icon=":material/key_off:")
    if settings.groq_api_key and not settings.groq_key_problem:
        st.badge("Groq API key loaded", icon=":material/key:", color="green")
        with st.expander("Use a different key"):
            st.session_state.user_api_key = st.text_input(
                "Groq API key", value=st.session_state.user_api_key, **key_input
            )
    else:
        st.session_state.user_api_key = st.text_input(
            "Groq API key", value=st.session_state.user_api_key, **key_input
        )
    models = list(dict.fromkeys([settings.model, *AVAILABLE_MODELS]))
    model = st.selectbox(
        "Model (runs on Groq)",
        models,
        index=0,
        format_func=model_label,
        help=(
            "Every option is an open-weight model hosted by Groq and called with your Groq key. "
            "GPT-OSS is OpenAI's free open model, but nothing is sent to OpenAI. "
            "Groq retired Llama 3.3 70B for free accounts in Aug 2026; GPT-OSS 120B is its official replacement."
        ),
    )
    st.caption(f"`{model}` · served by Groq")

    st.markdown("#### :material/menu_book: Textbook")
    pdf = st.session_state.pdf
    if pdf:
        with st.container(border=True):
            st.markdown(f":material/picture_as_pdf: **{pdf['filename']}**")
            cached = " · cached" if pdf["cached"] else ""
            st.caption(f"{pdf['pages']} pages · {pdf['chunks']} chunks{cached}")
            if st.button("Remove textbook", icon=":material/close:", type="tertiary"):
                st.session_state.pdf = None
                st.rerun()
    else:
        upload = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            help="Answers will quote your notes and cite page numbers.",
        )
        if upload is not None and upload.file_id != st.session_state.pdf_file_id:
            st.session_state.pdf_file_id = upload.file_id
            try:
                with st.spinner("Reading and indexing your PDF…"):
                    result = get_index().index_pdf(upload.getvalue(), upload.name)
                st.session_state.pdf = asdict(result)
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not index this PDF: {exc}")

    st.markdown("#### :material/history: Session")
    turns_so_far = len(st.session_state.turns)
    st.caption(f"{turns_so_far} question{'s' if turns_so_far != 1 else ''} asked")
    if st.button("New chat", icon=":material/add_comment:", width="stretch", disabled=not turns_so_far):
        st.session_state.turns = []
        st.rerun()

    st.space("medium")
    st.caption(f"v{__version__} · [Source on GitHub]({REPO_URL})")


# --------------------------------------------------------------------------- header
head_left, head_right = st.columns([3, 2], vertical_alignment="center")
with head_left:
    st.markdown(
        f'<div class="brand">{LOGO}<div><h1>AI Study Squad</h1>'
        "<p>Three AI tutors. One question. Every angle covered.</p></div></div>",
        unsafe_allow_html=True,
    )
with head_right, st.container(horizontal=True, horizontal_alignment="right"):
    st.badge(f"{model_label(model).split(' · ')[0]} on Groq", icon=":material/bolt:", color="gray")
    if st.session_state.pdf:
        st.badge("Textbook on", icon=":material/menu_book:", color="blue")
    else:
        st.badge("General knowledge", icon=":material/public:", color="gray")

question = st.chat_input("Ask anything: data mining, ML, DSA, OS, networks…")
if st.session_state.pending_question and not question:
    question = st.session_state.pending_question
st.session_state.pending_question = None

study_tab, analytics_tab, about_tab = st.tabs(
    [":material/forum: Study", ":material/insights: Analytics", ":material/account_tree: How it works"]
)


# --------------------------------------------------------------------------- study tab
def agent_label(key: str, lead: str | None) -> str:
    if key == COMPARE:
        return ":material/view_column: Compare all"
    agent = AGENTS[key]
    star = " :material/star:" if key == lead else ""
    return f"{agent.icon} {agent.name}{star}"


def render_welcome() -> None:
    st.markdown(
        '<div class="hero"><h2>What do you want to learn today?</h2>'
        "<p>Ask once and get three answers: a rigorous one, a simple one, and one that "
        "challenges you. Upload your notes in the sidebar to get answers with page numbers.</p></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(3)
    triggers = {"Nerd": "“derive”, “in detail”", "Simplifier": "“I don't get it”", "Challenger": "“quiz me”"}
    for col, key in zip(cols, AGENT_ORDER, strict=True):
        agent = AGENTS[key]
        with col, st.container(border=True, key=f"agentcard-{key}", height="stretch"):
            st.markdown(f"##### :{agent.color}[{agent.icon}] {agent.name}")
            st.markdown(agent.tagline)
            st.caption(f"Leads when you say {triggers[key]}")
    st.space("small")
    st.pills(
        "Try one of these",
        EXAMPLES,
        key="example_pick",
        on_change=on_example_pick,
    )


def render_answer(turn: dict, key: str, *, compact: bool, latest: bool) -> None:
    agent, reply = AGENTS[key], turn["replies"][key]
    height = 560 if compact else "content"
    with st.container(border=True, key=f"answer-{key}-{turn['id']}-{int(compact)}", height=height):
        st.markdown(f"**:{agent.color}[{agent.icon}] {agent.name}**")
        st.caption(f"{agent.role} · :material/timer: {reply['latency_ms'] / 1000:.1f}s")
        if reply["error"]:
            st.warning(reply["error"], icon=":material/error:")
            return
        st.markdown(reply["text"])  # Markdown only, never raw HTML
        st.space("small")
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.feedback("thumbs", key=f"fb-{turn['id']}-{key}", on_change=on_feedback, args=(turn, key))
            if latest and not compact:
                for label, icon, follow_up in FOLLOW_UPS:
                    st.button(
                        label,
                        icon=icon,
                        type="tertiary",
                        key=f"fu-{turn['id']}-{key}-{label}",
                        on_click=ask,
                        args=(follow_up,),
                    )


def render_turn(turn: dict, *, latest: bool) -> None:
    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(turn["question"])

    route = turn["route"]
    color, icon = INTENT_STYLE[route["intent"]]
    with st.container(horizontal=True, vertical_alignment="center", gap="small"):
        st.badge(INTENT_LABEL[route["intent"]], icon=icon, color=color)
        if route["lead"]:
            st.badge(f"{AGENTS[route['lead']].name} leads", icon=":material/star:", color="gray")
        if turn["pages"]:
            pages = ", ".join(map(str, turn["pages"]))
            st.badge(f"Textbook p. {pages}", icon=":material/menu_book:", color="blue")

    keys = order_agents(AGENT_ORDER, route["lead"])
    view = st.segmented_control(
        "Show answer from",
        [*keys, COMPARE],
        default=keys[0],
        required=True,
        format_func=lambda k: agent_label(k, route["lead"]),
        key=f"view-{turn['id']}",
        label_visibility="collapsed",
    )
    if view == COMPARE:
        for col, key in zip(st.columns(len(keys)), keys, strict=True):
            with col:
                render_answer(turn, key, compact=True, latest=latest)
    else:
        render_answer(turn, view or keys[0], compact=False, latest=latest)


def answer(question: str) -> None:
    key = api_key()
    if not key:
        st.error("Add your Groq API key in the sidebar first (free at console.groq.com/keys).")
        return

    with st.chat_message("user", avatar=":material/person:"):
        st.markdown(question)

    route = detect_intent(question)
    hits, context = [], None
    pdf = st.session_state.pdf
    if pdf:
        try:
            found = get_index().search(pdf["collection"], question)
        except Exception:  # noqa: BLE001 - answer without the book rather than failing
            found = []
            st.warning("Textbook search failed for this question, so the squad answered without it.")
        hits = [h for h in found if h.score >= MIN_SIMILARITY]
        context = format_context(hits) or None

    recent = st.session_state.turns[-settings.history_turns :] if settings.history_turns else []
    history = {
        k: [(t["question"], t["replies"][k]["text"]) for t in recent if not t["replies"][k]["error"]]
        for k in AGENT_ORDER
    }
    squad = StudySquad(
        get_client(key), model, temperature=settings.temperature, reasoning_effort=settings.reasoning_effort
    )

    with st.status("The squad is thinking…", expanded=True) as status:
        lines = {}
        for k in order_agents(AGENT_ORDER, route.lead):
            agent = AGENTS[k]
            lines[k] = st.empty()
            lines[k].markdown(f":{agent.color}[{agent.icon}] **{agent.name}** · :gray[thinking…]")

        def show_progress(reply: AgentReply) -> None:
            agent = AGENTS[reply.agent]
            result = (
                f":red[:material/error: {reply.error}]"
                if reply.error
                else f":green[:material/check_circle: ready in {reply.latency_ms / 1000:.1f}s]"
            )
            lines[reply.agent].markdown(f":{agent.color}[{agent.icon}] **{agent.name}** · {result}")

        replies = squad.ask_all(
            question, context=context, history=history, intent=route.intent, on_reply=show_progress
        )
        failed = sum(1 for r in replies.values() if r.error)
        status.update(
            label="Done" if not failed else f"Done, {failed} agent(s) had a problem",
            state="complete" if not failed else "error",
        )

    turn_id = uuid.uuid4().hex[:10]
    turn = {
        "id": turn_id,
        "question": question,
        "route": {"intent": route.intent, "lead": route.lead, "matched": list(route.matched)},
        "pages": sorted({h.page for h in hits}),
        "replies": {k: asdict(r) for k, r in replies.items()},
    }
    st.session_state.turns.append(turn)

    record = LogRecord(
        query=question,
        responses={k: r.text for k, r in replies.items()},
        turn_id=turn_id,
        session_id=st.session_state.session_id,
        intent=route.intent,
        lead_agent=route.lead,
        used_pdf=bool(hits),
        pages_cited=turn["pages"],
        latency_ms={k: r.latency_ms for k, r in replies.items()},
        errors={k: r.error for k, r in replies.items() if r.error},
        model=model,
    ).to_dict()
    st.session_state.session_logs.append(record)
    if settings.persist_logs:
        LogStore(settings.log_path).append(record)
    st.rerun()  # redraw everything through render_turn


with study_tab:
    turns = st.session_state.turns
    welcome_slot = st.empty()
    if not turns and not question:
        with welcome_slot.container():
            render_welcome()
    else:
        welcome_slot.empty()  # clear it now, not after the (slow) answer finishes
    for i, past in enumerate(turns):
        render_turn(past, latest=(i == len(turns) - 1 and not question))
        if i < len(turns) - 1 or question:
            st.divider()
    if question:
        answer(question)


# --------------------------------------------------------------------------- analytics tab
with analytics_tab:
    saved = read_records(settings.log_path) if settings.persist_logs else []
    sources = (["All saved logs"] if saved else []) + ["This session", "Sample dataset"]
    default = sources[0] if len(saved) >= 4 else "Sample dataset"
    source = st.segmented_control("Data source", sources, default=default, required=True, key="source")

    if source == "All saved logs":
        records = saved
        feedback = read_feedback(settings.feedback_path)
    elif source == "This session":
        records = st.session_state.session_logs
        feedback = list(st.session_state.session_feedback.values())
    else:
        records = read_records(settings.sample_log_path)
        feedback = []
        st.info(
            f"Showing a **sample dataset** of {len(records)} typical study questions so you can see the "
            "mining pipeline work. Ask questions, then switch the data source to see your own.",
            icon=":material/science:",
        )

    queries = [r["query"] for r in records]
    stats = usage_stats(records)
    if stats["total"] == 0:
        st.info("No questions yet. Ask something in the Study tab, or pick the sample dataset.")
    else:
        clusters = cluster_topics(queries)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Questions", stats["total"], border=True)
        m2.metric("Topics found", clusters.k if clusters else "—", border=True)
        m3.metric(
            "Silhouette",
            f"{clusters.silhouette:.2f}" if clusters else "—",
            border=True,
            help="How well separated the topic clusters are (-1 to 1, higher is better).",
        )
        m4.metric("Answered from PDF", f"{stats['pdf_rate']:.0%}", border=True)

        left, right = st.columns(2, gap="large")
        with left:
            st.markdown("##### Top keywords (TF-IDF)")
            kw = pd.DataFrame(top_keywords(queries, 10), columns=["term", "weight"])
            if kw.empty:
                st.caption("Not enough text yet.")
            else:
                st.altair_chart(
                    bar_chart(kw, "weight", "term", "Summed TF-IDF weight", ".2f"), width="stretch"
                )
        with right:
            st.markdown("##### What kind of help was needed")
            intents = pd.DataFrame(
                [(INTENT_LABEL[k], v) for k, v in stats["intent_counts"].items()],
                columns=["intent", "questions"],
            )
            st.altair_chart(bar_chart(intents, "questions", "intent", "Questions"), width="stretch")
            if stats["avg_latency_ms"]:
                st.markdown("##### Average response time")
                lat = pd.DataFrame(
                    [(AGENTS[a].name, ms / 1000) for a, ms in stats["avg_latency_ms"].items() if a in AGENTS],
                    columns=["agent", "seconds"],
                )
                st.altair_chart(bar_chart(lat, "seconds", "agent", "Seconds", ".1f"), width="stretch")

        helpful = feedback_summary(feedback)
        if helpful:
            st.markdown("##### Which tutor helped most")
            st.altair_chart(
                helpfulness_chart(helpful, {k: a.name for k, a in AGENTS.items()}), width="stretch"
            )
            st.caption("From the thumbs up / down under each answer.")

        st.markdown("##### Topic map")
        st.caption("TF-IDF → LSA → K-Means, drawn with t-SNE. Each dot is a question; hover to read it.")
        if clusters is None:
            st.caption("Ask at least 4 different questions to unlock topic clustering.")
        else:
            st.altair_chart(topic_map(clusters), width="stretch")
            table = pd.DataFrame(
                [
                    {"Topic": c.label, "Questions": c.size, "Example questions": " · ".join(c.examples)}
                    for c in clusters.clusters
                ]
            )
            st.dataframe(table, hide_index=True, width="stretch")
            if clusters.silhouette_by_k:
                with st.expander("How the number of topics was chosen", icon=":material/insights:"):
                    st.altair_chart(silhouette_chart(clusters), width="stretch")
                    st.caption(
                        f"K-Means was run for every k and the highest silhouette score was kept "
                        f"(k = {clusters.k}, dashed line). LSA used {clusters.lsa_dims} dimensions."
                    )

        if stats.get("per_day") and len(stats["per_day"]) > 1:
            st.markdown("##### Study activity")
            st.altair_chart(activity_chart(stats["per_day"]), width="stretch")

        export = pd.DataFrame(
            [
                {k: r.get(k) for k in ("timestamp", "query", "intent", "lead_agent", "used_pdf")}
                for r in records
            ]
        )
        st.download_button(
            "Download these logs (CSV)",
            export.to_csv(index=False),
            "study_logs.csv",
            "text/csv",
            icon=":material/download:",
        )


# --------------------------------------------------------------------------- about tab
STEPS = [
    (
        ":material/alt_route:",
        "1 · Route",
        "router.py",
        "Regex rules spot confusion, a request to be tested, or a "
        "request for depth, and pick the lead tutor.",
    ),
    (
        ":material/manage_search:",
        "2 · Retrieve",
        "rag.py",
        "If a PDF is loaded, the closest chunks are found in "
        "ChromaDB by cosine similarity, keeping page numbers.",
    ),
    (
        ":material/groups:",
        "3 · Answer",
        "agents.py",
        "One LLM, three system prompts, called in parallel with the last few turns as memory.",
    ),
    (
        ":material/save:",
        "4 · Log",
        "storage.py",
        "Every question, answer time and rating is appended to a JSON Lines file.",
    ),
    (
        ":material/hub:",
        "5 · Mine",
        "mining.py",
        "TF-IDF keywords, LSA + K-Means topics (k by silhouette) and a t-SNE topic map.",
    ),
]

with about_tab:
    st.markdown("##### How a question flows through the system")
    for col, (icon, title, module, text) in zip(st.columns(len(STEPS)), STEPS, strict=True):
        with col, st.container(border=True, height="stretch"):
            st.markdown(f"**:blue[{icon}] {title}**")
            st.caption(f"`{module}`")
            st.markdown(text)
    st.space("small")
    st.markdown(
        "On a labelled set of 58 questions from 8 subjects, adding LSA before K-Means raised the "
        "Adjusted Rand Index from **0.33 to 0.58**. "
        f"Code, tests and the evaluation script are on [GitHub]({REPO_URL})."
    )
