# Architecture

Issue2Impact is organized as a repository-grounded agent workflow rather than a general chatbot.

```text
Repository source
  ↓
Source adapter (local / GitHub / ZIP)
  ↓
RepositoryContext
  ↓
Loader → chunker → embeddings → isolated Chroma index
  ↓
Repository tools
  ↓
Router
  ├── General software question → direct answer
  ├── Unsupported request → scope response
  └── Repository request
        ↓
      Investigator ↔ search/read tools
        ↓
      Planner
        ↓
      Critic
        ├── weak plan → Planner retry
        ├── weak evidence → Investigator retry
        └── approved → Human approval interrupt
                            ├── reject → Planner revision
                            └── approve → Final result
```

## Responsibilities

- `src/repository/`: resolves repository inputs into a safe local workspace.
- `src/ingestion/`: loads supported files and creates code-aware chunks.
- `src/retrieval/`: embeddings, repository-specific Chroma storage, candidate retrieval, and reranking.
- `src/tools/`: repository search and safe full-file reading.
- `src/agents/`: router, planner, and critic specializations.
- `src/graph/`: shared LangGraph state and orchestration.
- `src/services/workflow.py`: application-facing service layer used by CLI/UI code.
- `main.py`: terminal interface.
- `streamlit_app.py`: interactive product interface.

## Safety and reliability choices

Repository file reading is constrained to the selected repository root. ZIP extraction rejects path traversal. Generated vector indexes are separated by repository identity. Reflection and human-rejection cycles use bounded retries. Human approval uses LangGraph interrupts, checkpointing, thread IDs, and resume commands instead of blocking input inside graph nodes.

## Why the service layer matters

The workflow service centralizes repository preparation, graph construction, workflow configuration, interrupt extraction, and resume validation. This keeps the agent architecture independent from the user interface and makes it easier to add another client such as FastAPI, a desktop application, or a hosted service without rewriting the core workflow.
