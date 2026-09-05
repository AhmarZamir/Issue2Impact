import argparse

from src.services.workflow import (
    build_repository_graph,
    extract_text,
    get_interrupt_payload,
    make_config,
    prepare_repository_runtime,
    resume_workflow,
    start_workflow,
)


DEFAULT_QUERY = """
Users sometimes remain authenticated when an invalid token is supplied.
Investigate how token validation and logout work in this repository and give me
an implementation plan to make authentication handling safer.
""".strip()


def print_trace(result, repository=None):
    if repository is not None:
        print("\n=== REPOSITORY ===")
        print("Name:", repository.name)
        print("Source type:", repository.source_type)
        print("Path:", repository.path)
        print("Repository ID:", repository.repository_id)

    print("\n=== ROUTING ===")
    print("Route:", result.get("route"))
    print("Reason:", result.get("route_reason"))

    print("\n=== INVESTIGATION ===")
    print(result.get("investigation", "N/A"))

    print("\n=== PLAN ===")
    print(result.get("plan", "N/A"))

    print("\n=== CRITIC ===")
    print("Approved:", result.get("plan_approved"))
    print("Needs more evidence:", result.get("needs_more_evidence"))
    print("Feedback:", result.get("critic_feedback", "N/A"))
    print("Retries:", result.get("retry_count", 0))

    print("\n=== HUMAN REVIEW ===")
    print("Approved:", result.get("human_approved"))
    print("Feedback:", result.get("human_feedback", "N/A"))

    print("\n=== MESSAGE TRACE ===")
    for message in result.get("messages", []):
        print(f"\n--- {type(message).__name__} ---")
        print(message)


def run(
    query: str,
    repository_source: str = "demo_repo",
    source_type: str = "auto",
    show_trace: bool = False,
    input_fn=input,
    thread_id: str | None = None,
):
    """Prepare a repository, run the workflow, and handle human approval interrupts."""
    runtime = prepare_repository_runtime(repository_source, source_type=source_type)
    config = make_config(thread_id=thread_id)
    result, config = start_workflow(runtime.graph, query, config=config)

    while get_interrupt_payload(result):
        interrupt_info = get_interrupt_payload(result)

        print("\n=== HUMAN APPROVAL REQUIRED ===")
        print(interrupt_info.get("message", "Review the proposed plan."))
        print("\nPlan:\n")
        print(interrupt_info.get("plan", "N/A"))
        print("\nCritic feedback:\n")
        print(interrupt_info.get("critic_feedback", "N/A"))

        answer = input_fn("\nApprove plan? [y/n]: ").strip().lower()
        approved = answer in {"y", "yes"}
        feedback = (
            "Approved by human reviewer."
            if approved
            else input_fn("Why are you rejecting it? ").strip()
        )

        result = resume_workflow(
            runtime.graph,
            config,
            approved=approved,
            feedback=feedback,
        )

    if show_trace:
        print_trace(result, repository=runtime.repository)

    messages = result.get("messages", [])
    if not messages:
        return "Workflow finished without a final message."
    return extract_text(messages[-1].content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Investigate a local, GitHub, or ZIP source-code repository."
    )
    parser.add_argument("query", nargs="?", default=DEFAULT_QUERY)
    parser.add_argument(
        "--repo",
        default="demo_repo",
        help=(
            "Repository source: local folder path, public GitHub repository URL, "
            "or path to a ZIP archive."
        ),
    )
    parser.add_argument(
        "--source-type",
        choices=["auto", "local", "github", "zip"],
        default="auto",
        help="Repository source type. 'auto' detects the type from --repo.",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help=(
            "Print repository metadata, routing, investigation, plan, critic state, "
            "human-review state, and every message in the workflow."
        ),
    )
    args = parser.parse_args()

    try:
        print(
            run(
                args.query,
                repository_source=args.repo,
                source_type=args.source_type,
                show_trace=args.trace,
            )
        )
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"Issue2Impact could not start: {error}") from error
