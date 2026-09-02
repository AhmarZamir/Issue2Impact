# Issue2Impact

Issue2Impact is a learning project that investigates software issues with
repository-grounded evidence. It currently combines code ingestion, semantic
retrieval, cross-encoder reranking, tool calling, and a LangGraph agent loop.

## Current phases

- Phase 1: repository loading and code-aware chunking
- Phase 2: Hugging Face embeddings and Chroma vector storage
- Phase 3: candidate retrieval, reranking, and retrieval evaluation
- Phase 4: conditional tool calling with repository search and safe file reading
- Phase 5: LangGraph state, agent/tool nodes, conditional routing, and cycles

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your Google API key, and select a Gemini
model available to your account. The key and local Chroma database are ignored
by Git.

```bash
python main.py --trace
python main.py "Where is invalid login behavior tested?" --trace
```

`--trace` displays the full `HumanMessage -> AIMessage -> ToolMessage` sequence
so the Phase 5 loop can be studied directly. Graph execution has a recursion
limit to prevent uncontrolled tool loops.

## Tests

```bash
pytest
```

The tests use deterministic fake models, so they validate routing and tool
execution without an API key or Gemini quota.
