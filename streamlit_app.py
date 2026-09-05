import os
from pathlib import Path
from uuid import uuid4

import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from main import build_repository_graph, extract_text
from src.ingestion.loader import load_repository
from src.prompts.repository_agent_prompt import REPOSITORY_AGENT_PROMPT
from src.repository.source import prepare_repository
from src.ui.uploads import save_zip_upload


st.set_page_config(
    page_title="Issue2Impact",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 3rem;}
        .hero {
            padding: 1.35rem 1.5rem;
            border: 1px solid rgba(124, 58, 237, 0.28);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(124,58,237,.12), rgba(37,99,235,.06));
            margin-bottom: 1.2rem;
        }
        .hero h1 {margin: 0 0 .35rem 0; font-size: 2.25rem;}
        .hero p {margin: 0; opacity: .82; font-size: 1.02rem;}
        .repo-card, .result-card {
            border: 1px solid rgba(148, 163, 184, .18);
            border-radius: 14px;
            padding: 1rem 1.1rem;
            background: rgba(148, 163, 184, .045);
        }
        .step-row {
            display: flex;
            flex-wrap: wrap;
            gap: .55rem;
            margin: .5rem 0 1rem 0;
        }
        .step {
            border-radius: 999px;
            padding: .42rem .72rem;
            font-size: .84rem;
            border: 1px solid rgba(148, 163, 184, .22);
        }
        .step-done {background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.35);}
        .step-active {background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.38);}
        .step-idle {opacity: .6;}
        .small-muted {opacity: .68; font-size: .86rem;}
        div[data-testid="stMetric"] {
            border: 1px solid rgba(148, 163, 184, .16);
            padding: .75rem 1rem;
            border-radius: 12px;
        }
        div[data-testid="stSidebar"] div[data-testid="stButton"] button,
        div[data-testid="stMain"] div[data-testid="stButton"] button {
            border-radius: 10px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


SESSION_DEFAULTS = {
    "repository": None,
    "repository_file_count": 0,
    "graph": None,
    "checkpointer": None,
    "config": None,
    "result": None,
    "active_query": "",
    "load_error": None,
}

for key, value in SESSION_DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_workflow_state():
    st.session_state.graph = None
    st.session_state.checkpointer = None
    st.session_state.config = None
    st.session_state.result = None
    st.session_state.active_query = ""


def load_selected_repository(source: str, source_type: str):
    reset_workflow_state()
    repository = prepare_repository(source, source_type=source_type)
    documents = load_repository(str(repository.path))
    if not documents:
        raise ValueError("No supported source files were found in this repository.")

    checkpointer = InMemorySaver()
    graph = build_repository_graph(
        repo_path=str(repository.path),
        repository_id=repository.repository_id,
        checkpointer=checkpointer,
    )

    st.session_state.repository = repository
    st.session_state.repository_file_count = len(documents)
    st.session_state.checkpointer = checkpointer
    st.session_state.graph = graph
    st.session_state.result = None
    st.session_state.config = None
    st.session_state.load_error = None


def start_investigation(query: str):
    graph = st.session_state.graph
    if graph is None:
        raise ValueError("Load a repository before starting an investigation.")

    config = {
        "configurable": {"thread_id": f"issue2impact-ui-{uuid4()}"},
        "recursion_limit": 30,
    }
    result = graph.invoke(
        {
            "user_query": query,
            "retry_count": 0,
            "messages": [
                SystemMessage(content=REPOSITORY_AGENT_PROMPT),
                HumanMessage(content=query),
            ],
        },
        config=config,
    )

    st.session_state.config = config
    st.session_state.result = result
    st.session_state.active_query = query


def resume_human_review(approved: bool, feedback: str):
    if st.session_state.graph is None or st.session_state.config is None:
        raise ValueError("There is no paused workflow to resume.")

    st.session_state.result = st.session_state.graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback,
            }
        ),
        config=st.session_state.config,
    )


def get_interrupt_payload(result):
    interrupts = result.get("__interrupt__", []) if result else []
    if not interrupts:
        return None
    return interrupts[0].value


def workflow_steps(result):
    if not result:
        return [
            ("Router", "idle"),
            ("Investigator", "idle"),
            ("Planner", "idle"),
            ("Critic", "idle"),
            ("Human", "idle"),
        ]

    route = result.get("route")
    if route != "repository":
        return [
            ("Router", "done"),
            ("Investigator", "idle"),
            ("Planner", "idle"),
            ("Critic", "idle"),
            ("Human", "idle"),
        ]

    interrupted = bool(result.get("__interrupt__"))
    human_done = "human_approved" in result and not interrupted
    return [
        ("Router", "done" if route else "active"),
        ("Investigator", "done" if result.get("investigation") else "active"),
        ("Planner", "done" if result.get("plan") else "idle"),
        ("Critic", "done" if "plan_approved" in result else "idle"),
        ("Human", "active" if interrupted else ("done" if human_done else "idle")),
    ]


def render_workflow(result):
    pieces = []
    for label, status in workflow_steps(result):
        icon = {"done": "✓", "active": "●", "idle": "○"}[status]
        pieces.append(
            f'<span class="step step-{status}">{icon}&nbsp; {label}</span>'
        )
    st.markdown(f'<div class="step-row">{"".join(pieces)}</div>', unsafe_allow_html=True)


def render_repository_summary():
    repository = st.session_state.repository
    if repository is None:
        st.info("Choose a repository source in the sidebar and load it to begin.")
        return

    st.markdown('<div class="repo-card">', unsafe_allow_html=True)
    left, middle, right = st.columns([1.3, 1, 1])
    with left:
        st.markdown(f"**{repository.name}**")
        st.caption(str(repository.path))
    with middle:
        st.metric("Source", repository.source_type.upper())
    with right:
        st.metric("Source files", st.session_state.repository_file_count)
    st.caption(f"Repository index: `{repository.repository_id}`")
    st.markdown('</div>', unsafe_allow_html=True)


def render_results(result):
    if not result:
        return

    route = result.get("route")
    if route == "general":
        st.subheader("Answer")
        st.markdown(extract_text(result["messages"][-1].content))
        return
    if route == "unsupported":
        st.warning(extract_text(result["messages"][-1].content))
        return

    tabs = st.tabs(["Investigation", "Implementation plan", "Critic review", "Final output"])

    with tabs[0]:
        st.markdown(result.get("investigation", "Investigation is not available yet."))

    with tabs[1]:
        st.markdown(result.get("plan", "Plan is not available yet."))

    with tabs[2]:
        approved = result.get("plan_approved")
        if approved is True:
            st.success("Critic approved the current plan.")
        elif approved is False:
            st.warning("Critic requested another revision or more evidence.")
        else:
            st.info("Critic review has not completed yet.")

        col1, col2 = st.columns(2)
        col1.metric("Retries", result.get("retry_count", 0))
        col2.metric(
            "More evidence",
            "Yes" if result.get("needs_more_evidence") else "No",
        )
        st.markdown("**Feedback**")
        st.markdown(result.get("critic_feedback", "No critic feedback yet."))

    with tabs[3]:
        if result.get("__interrupt__"):
            st.info("The workflow is paused for human approval.")
        elif result.get("messages"):
            st.markdown(extract_text(result["messages"][-1].content))
        else:
            st.info("No final output yet.")


def render_human_approval(result):
    payload = get_interrupt_payload(result)
    if not payload:
        return

    st.divider()
    st.subheader("Human approval")
    st.info(payload.get("message", "Review the proposed implementation plan."))

    with st.expander("Review plan before deciding", expanded=True):
        st.markdown(payload.get("plan", "Plan unavailable."))
        st.markdown("**Critic feedback**")
        st.markdown(payload.get("critic_feedback", "No critic feedback."))

    approve_col, reject_col = st.columns(2)
    with approve_col:
        if st.button("✓ Approve plan", type="primary", use_container_width=True):
            try:
                with st.spinner("Resuming workflow..."):
                    resume_human_review(True, "Approved by human reviewer.")
                st.rerun()
            except Exception as error:
                st.error(f"Unable to resume the workflow: {error}")

    with reject_col:
        with st.popover("Request changes", use_container_width=True):
            feedback = st.text_area(
                "What should the planner change?",
                placeholder="Example: Add explicit malformed-token test coverage and avoid changing the token format.",
                key="human_rejection_feedback",
            )
            if st.button("Send feedback", use_container_width=True):
                if not feedback.strip():
                    st.warning("Add a short reason so the Planner knows what to revise.")
                else:
                    try:
                        with st.spinner("Sending feedback back to the Planner..."):
                            resume_human_review(False, feedback.strip())
                        st.rerun()
                    except Exception as error:
                        st.error(f"Unable to resume the workflow: {error}")


with st.sidebar:
    st.markdown("## Repository")
    source_label = st.radio(
        "Source",
        ["GitHub URL", "Local folder", "ZIP upload"],
        label_visibility="collapsed",
    )

    source_value = None
    source_type = None

    if source_label == "GitHub URL":
        source_type = "github"
        source_value = st.text_input(
            "Public GitHub repository",
            placeholder="https://github.com/owner/repository",
        )
        st.caption("Public repositories only. Git must be installed on the host.")
    elif source_label == "Local folder":
        source_type = "local"
        source_value = st.text_input(
            "Repository folder",
            value="demo_repo",
            placeholder=r"D:\Projects\MyApp",
        )
        st.caption("Local paths work when Streamlit is running on the same machine.")
    else:
        source_type = "zip"
        uploaded_zip = st.file_uploader("Upload repository ZIP", type=["zip"])

    if st.button("Load repository", type="primary", use_container_width=True):
        try:
            with st.status("Preparing repository...", expanded=True) as status:
                if source_type == "zip":
                    if uploaded_zip is None:
                        raise ValueError("Choose a ZIP file first.")
                    status.write("Saving uploaded archive")
                    source_value = str(
                        save_zip_upload(uploaded_zip.getvalue(), uploaded_zip.name)
                    )
                elif not source_value or not source_value.strip():
                    raise ValueError("Enter a repository source first.")

                status.write("Resolving repository source")
                status.write("Loading source files and repository index")
                load_selected_repository(source_value, source_type)
                status.update(label="Repository ready", state="complete", expanded=False)
        except Exception as error:
            st.session_state.load_error = str(error)
            st.error(str(error))

    st.divider()
    if st.session_state.repository is not None:
        repository = st.session_state.repository
        st.success(f"Ready: {repository.name}")
        st.caption(f"{repository.source_type} · {st.session_state.repository_file_count} source files")
        if st.button("Clear repository", use_container_width=True):
            st.session_state.repository = None
            st.session_state.repository_file_count = 0
            reset_workflow_state()
            st.rerun()

    st.divider()
    st.markdown("### Environment")
    api_ready = bool(os.getenv("GOOGLE_API_KEY"))
    if api_ready:
        st.caption("✓ Google model API key detected")
    else:
        st.caption("⚠ Add GOOGLE_API_KEY to `.env` before running an investigation")


st.markdown(
    """
    <div class="hero">
        <h1>Issue2Impact</h1>
        <p>Repository-grounded software investigation with agentic planning, reflection, self-healing, and human approval.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

render_repository_summary()

st.markdown("### Agent workflow")
render_workflow(st.session_state.result)

st.markdown("### Describe the issue")
query = st.text_area(
    "Issue or repository question",
    value=st.session_state.active_query,
    placeholder=(
        "Example: Users sometimes remain authenticated when an invalid token is supplied. "
        "Investigate token validation and logout, then propose a safer implementation plan."
    ),
    height=125,
    label_visibility="collapsed",
)

investigate_disabled = st.session_state.repository is None
if st.button(
    "Investigate repository",
    type="primary",
    use_container_width=True,
    disabled=investigate_disabled,
):
    if not query.strip():
        st.warning("Describe an issue or ask a repository-specific question first.")
    else:
        try:
            with st.status("Running multi-agent investigation...", expanded=True) as status:
                status.write("Routing request")
                status.write("Searching repository evidence")
                status.write("Planning and critic review")
                start_investigation(query.strip())
                if get_interrupt_payload(st.session_state.result):
                    status.update(
                        label="Investigation ready for human review",
                        state="complete",
                        expanded=False,
                    )
                else:
                    status.update(label="Investigation complete", state="complete", expanded=False)
            st.rerun()
        except Exception as error:
            st.error(f"Investigation failed: {error}")

if investigate_disabled:
    st.caption("Load a repository from the sidebar to enable investigation.")

if st.session_state.result is not None:
    st.divider()
    st.subheader("Analysis")
    render_results(st.session_state.result)
    render_human_approval(st.session_state.result)
