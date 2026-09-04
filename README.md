# Issue2Impact

Issue2Impact is a learning-focused Agentic AI project for investigating software issues using repository-grounded evidence.

It currently combines repository ingestion, code-aware chunking, local embeddings, Chroma vector search, cross-encoder reranking, retrieval evaluation, tool calling, and LangGraph orchestration.

## Current phases

- Phase 1: repository loading and code-aware chunking
- Phase 2: Hugging Face embeddings and Chroma vector storage
- Phase 3: candidate retrieval, reranking, and retrieval evaluation
- Phase 4: conditional tool calling with repository search and safe file reading
- Phase 5: LangGraph state, agent/tool nodes, conditional routing, and cycles
- Phase 6: router agent, richer graph state, structured route decisions, general/unsupported branches, and context-aware execution

## Phase 6 workflow

```text
START
  ↓
Router
  ├── repository → Repository Agent → Tools? → Agent → END
  ├── general → General Software Node → END
  └── unsupported → Deterministic Scope Response → END
```

The router classifies each request before repository reasoning begins. Repository-specific requests still use the existing agentic RAG loop, while general programming questions avoid repository retrieval entirely and out-of-scope requests exit deterministically.

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
python main.py "Where is invalid login behavior tested?" --trace
python main.py "What is dependency injection?" --trace
python main.py "Write a romantic poem." --trace
```

`--trace` displays the routing decision and the message/tool sequence so the workflow can be inspected directly. Graph execution uses a recursion limit to prevent uncontrolled tool loops.

## Tests

```bash
pytest
```

The graph tests use deterministic fake models and routers so routing, repository tool execution, multi-step tool use, and unsupported/general branches can be validated without a live API key or model quota.
