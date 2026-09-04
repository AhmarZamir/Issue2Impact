# Issue2Impact

Issue2Impact is a learning-focused Agentic AI project for investigating software issues using repository-grounded evidence.

It currently combines repository ingestion, code-aware chunking, local embeddings, Chroma vector search, cross-encoder reranking, retrieval evaluation, tool calling, routed LangGraph orchestration, repository investigation, structured implementation planning, critic review, reflection, bounded self-healing retries, and human-in-the-loop approval.

## Current phases

- Phase 1: repository loading and code-aware chunking
- Phase 2: Hugging Face embeddings and Chroma vector storage
- Phase 3: candidate retrieval, reranking, and retrieval evaluation
- Phase 4: conditional tool calling with repository search and safe file reading
- Phase 5: LangGraph state, agent/tool nodes, conditional routing, and cycles
- Phase 6: router agent, richer graph state, structured route decisions, general/unsupported branches, and context-aware execution
- Phase 7: repository investigator -> planner handoff, investigation state, structured implementation plans, and multi-agent specialization
- Phase 8: critic agent, reflection, plan revision, evidence re-investigation, bounded retries, and safe termination
- Phase 9: human-in-the-loop approval with LangGraph interrupts, checkpointed state, thread IDs, resume commands, and human-guided plan revision

## Phase 9 workflow

```text
START
  ↓
Router
  ├── repository
  │      ↓
  │  Repository Investigator
  │      ↓
  │   Tool needed?
  │    ↙       ↘
  │  Tools   Capture Investigation
  │    ↓             ↓
  │ Investigator   Planner
  │                  ↓
  │                Critic
  │          ┌────────┼──────────┐
  │          ↓        ↓          ↓
  │      Weak Plan  More       Critic Approved
  │          ↓      Evidence          ↓
  │       Planner      ↓        Human Approval
  │                    ↓        ↙            ↘
  │              Investigator Reject       Approve
  │                              ↓            ↓
  │                       Human Feedback   Finalize
  │                              ↓            ↓
  │                           Planner        END
  │
  ├── general → General Software Node → END
  └── unsupported → Deterministic Scope Response → END
```

The Critic is still responsible for technical quality: grounding, relevance, actionability, tests, risk, and evidence sufficiency. Once the Critic approves a plan, the graph no longer finishes automatically. It pauses with LangGraph `interrupt()` and waits for a human decision.

Human approval is implemented as a real workflow pause/resume boundary rather than a blocking `input()` inside a graph node. The CLI handles the user interface outside the node, then resumes the exact same graph execution with `Command(resume=...)`.

Phase 9 uses `InMemorySaver` and a `thread_id` so LangGraph can preserve the workflow state while paused. The same thread ID is reused for the resume command. This preserves the investigation, plan, critic feedback, retry count, and current graph position.

If the human rejects the plan, the feedback is sent back to the Planner. The revised plan is reviewed by the Critic again and, if technically approved, is presented to the human again. Human rejections share the existing bounded retry budget so the workflow cannot loop indefinitely.

The system still does not modify repository code. The approved output is an evidence-grounded implementation plan.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your Google API key, and select a Gemini model available to your account.

## Run

```bash
python main.py --trace
python main.py "Investigate token validation and logout and propose a safer implementation plan." --trace
```

When the Critic approves a repository plan, the terminal will display the plan and ask:

```text
Approve plan? [y/n]:
```

If you reject it, you can enter feedback. The graph resumes from the saved checkpoint, revises the plan, reviews it again, and can pause for another human decision.

General programming questions and unsupported requests do not enter the human-approval path.

## Tests

```bash
pytest
```

The graph tests use deterministic fake models, routers, planners, and critics. Phase 9 tests validate that critic approval pauses instead of finishing, approval resumes and finalizes the same thread, rejection feeds human feedback back into the Planner, repeated human rejection respects the retry limit, and non-repository routes bypass HITL.
