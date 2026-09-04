# Issue2Impact

Issue2Impact is a learning-focused Agentic AI project for investigating software issues using repository-grounded evidence.

It currently combines repository ingestion, code-aware chunking, local embeddings, Chroma vector search, cross-encoder reranking, retrieval evaluation, tool calling, routed LangGraph orchestration, repository investigation, structured implementation planning, critic review, reflection, and bounded self-healing retries.

## Current phases

- Phase 1: repository loading and code-aware chunking
- Phase 2: Hugging Face embeddings and Chroma vector storage
- Phase 3: candidate retrieval, reranking, and retrieval evaluation
- Phase 4: conditional tool calling with repository search and safe file reading
- Phase 5: LangGraph state, agent/tool nodes, conditional routing, and cycles
- Phase 6: router agent, richer graph state, structured route decisions, general/unsupported branches, and context-aware execution
- Phase 7: repository investigator -> planner handoff, investigation state, structured implementation plans, and multi-agent specialization
- Phase 8: critic agent, reflection, plan revision, evidence re-investigation, bounded retries, and safe termination

## Phase 8 workflow

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
  │      Approved   Revise   More Evidence
  │          ↓        ↓          ↓
  │         END   Plan Retry  Evidence Retry
  │                   ↓          ↓
  │                Planner   Investigator
  │
  ├── general → General Software Node → END
  └── unsupported → Deterministic Scope Response → END
```

The Phase 8 critic checks grounding, relevance, actionability, tests, risks, and whether the repository evidence is sufficient. A weak plan is sent back to the Planner with critic feedback. Missing evidence is sent back to the Repository Investigator with the critic's reason so the workflow can investigate again before replanning.

Retries are intentionally bounded with `MAX_RETRIES = 2`. LangGraph also uses a larger recursion limit as an emergency graph-level guard, while the workflow's own retry state is the primary stopping condition. This prevents uncontrolled reflection loops and avoids accepting weak plans indefinitely.

The planner still does not modify code. Phase 8 only investigates, plans, critiques, and self-corrects.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your Google API key, and select a Gemini model available to your account.

```bash
python main.py --trace
python main.py "Investigate token validation and logout and propose a safer implementation plan." --trace
python main.py "What is dependency injection?" --trace
python main.py "Write a romantic poem." --trace
```

`--trace` displays routing, investigation, plan, critic approval state, critic feedback, evidence requests, retry count, and the full message/tool sequence.

## Tests

```bash
pytest
```

The graph tests use deterministic fake models, routers, planners, and critics. They validate immediate approval, plan revision, evidence re-investigation, retry exhaustion, and the non-repository routes without requiring a live model API key or quota.
