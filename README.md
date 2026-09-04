# Issue2Impact

Issue2Impact is a learning-focused Agentic AI project for investigating software issues using repository-grounded evidence.

It currently combines repository ingestion, code-aware chunking, local embeddings, Chroma vector search, cross-encoder reranking, retrieval evaluation, tool calling, routed LangGraph orchestration, repository investigation, and structured implementation planning.

## Current phases

- Phase 1: repository loading and code-aware chunking
- Phase 2: Hugging Face embeddings and Chroma vector storage
- Phase 3: candidate retrieval, reranking, and retrieval evaluation
- Phase 4: conditional tool calling with repository search and safe file reading
- Phase 5: LangGraph state, agent/tool nodes, conditional routing, and cycles
- Phase 6: router agent, richer graph state, structured route decisions, general/unsupported branches, and context-aware execution
- Phase 7: repository investigator -> planner handoff, investigation state, structured implementation plans, and multi-agent specialization

## Phase 7 workflow

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
  │                 END
  │
  ├── general → General Software Node → END
  └── unsupported → Deterministic Scope Response → END
```

The repository branch now separates two responsibilities:

- **Repository Investigator**: searches and reads repository evidence, then produces a concise evidence-backed investigation.
- **Planner Agent**: receives only the original issue plus the investigation and produces a structured implementation plan with files, ordered steps, tests, and risks.

The planner does not modify code and does not receive repository tools. This keeps evidence gathering and solution planning separated and makes the handoff explicit in LangGraph state.

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

`--trace` displays the routing decision, captured investigation, final plan, and the full message/tool sequence. Graph execution uses a recursion limit to prevent uncontrolled tool loops.

## Tests

```bash
pytest
```

The graph tests use deterministic fake models, routers, and planners so the routing branches, repository tool execution, multi-step investigation, investigator-to-planner handoff, and structured plan formatting can be validated without a live API key or model quota.
