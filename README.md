# Issue2Impact

Issue2Impact is an Agentic AI system that investigates software issues against real source-code repositories, builds an evidence-grounded implementation plan, critiques its own work, retries when evidence is weak, and pauses for human approval before finalizing the result.

It is designed as an engineering workflow rather than a generic chatbot.

## Why this project matters

Software issue analysis is rarely a one-shot retrieval problem. A useful system has to decide whether repository evidence is needed, search and read the right files, separate evidence gathering from planning, challenge weak conclusions, recover from missing evidence, and keep a human in control of the final plan.

Issue2Impact demonstrates that full lifecycle with LangGraph orchestration, repository-specific RAG, specialized agents, reflection, bounded self-healing, and human-in-the-loop approval.

## Product capabilities

- Inspect a local repository folder
- Clone and inspect a public GitHub repository
- Upload and inspect a repository ZIP
- Build an isolated Chroma index per repository/source fingerprint
- Search repository chunks with embeddings and cross-encoder reranking
- Safely read full repository files without escaping the selected repository root
- Route general, repository-specific, and unsupported requests
- Investigate repository behavior with tool-calling agents
- Produce structured implementation plans
- Critique grounding, relevance, actionability, tests, risks, and evidence sufficiency
- Retry planning or repository investigation when the Critic rejects a result
- Pause for human approval with LangGraph interrupts/checkpointing
- Accept human rejection feedback and revise the plan
- Use either a CLI or a polished Streamlit interface

## Architecture

```text
Repository source
      ↓
Local folder / GitHub / ZIP
      ↓
RepositoryContext
      ↓
Loader → Chunker → Embeddings → Repository-specific Chroma index
      ↓
Repository search/read tools
      ↓
Router
  ├── general → General Software Node → END
  ├── unsupported → Scope Response → END
  └── repository
          ↓
    Repository Investigator
          ↕
        Tools
          ↓
    Capture Investigation
          ↓
        Planner
          ↓
        Critic
     ┌────┼──────────────┐
     ↓    ↓              ↓
  Revise  More evidence  Approved
     ↓    ↓              ↓
  Planner Investigator  Human approval
                         ↙          ↘
                      Reject      Approve
                        ↓            ↓
                     Planner      Finalize
                                      ↓
                                     END
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the component-level breakdown.

## Project structure

```text
Issue2Impact/
├── main.py
├── streamlit_app.py
├── demo_repo/
├── src/
│   ├── agents/
│   ├── graph/
│   ├── ingestion/
│   ├── llm/
│   ├── prompts/
│   ├── repository/
│   ├── retrieval/
│   ├── services/
│   ├── tools/
│   └── ui/
├── tests/
├── docs/
└── requirements.txt
```

`src/services/workflow.py` is the application-facing service layer. It centralizes repository preparation, graph construction, workflow configuration, interrupt handling, and resume validation so the CLI and UI do not own core agent logic.

## Supported source files

The loader handles common Python, JavaScript/TypeScript, Java, C/C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, HTML/CSS, SQL, shell, configuration, Markdown, JSON, YAML, XML, Dockerfile, and Makefile content. Dependency folders, virtual environments, build outputs, cache folders, oversized files, and unreadable/binary content are skipped.

## Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add `GOOGLE_API_KEY`, and select a Gemini model available to your account.

## Streamlit application

```bash
streamlit run streamlit_app.py
```

The interface provides repository loading, repository status, visible agent workflow progress, investigation/plan/critic result views, and human approve/reject controls.

For hosted Streamlit, use a public GitHub URL or ZIP upload. A hosted server cannot directly access an arbitrary folder path on a user's computer.

## CLI examples

Bundled demo repository:

```bash
python main.py --trace
```

Local repository:

```bash
python main.py "Investigate the authentication flow and identify risky behavior." --repo "D:\Projects\MyApp" --source-type local --trace
```

Public GitHub repository:

```bash
python main.py "Find where authentication is implemented and propose a safer design." --repo "https://github.com/user/project" --source-type github --trace
```

ZIP repository:

```bash
python main.py "Inspect this repository for authentication-related impact." --repo "./project.zip" --source-type zip --trace
```

## Reliability and safety choices

- Repository file reads are restricted to the selected repository root.
- ZIP extraction rejects unsafe path traversal.
- Different repositories do not share a vector index.
- Local-source content changes produce a new repository fingerprint/index identity.
- Critic and human-revision loops are bounded to prevent uncontrolled recursion.
- Human approval uses LangGraph interrupts and resume commands rather than blocking inside graph nodes.
- Empty/unsupported repositories fail early with a clear error.
- Human rejection requires feedback before the workflow resumes.
- CLI startup errors are converted into concise user-facing failures.

## Tests

```bash
pytest
```

The test suite covers ingestion metadata, safe repository reading, routing, tool loops, investigator/planner handoff, critic reflection, evidence re-investigation, bounded retries, human approval/resume behavior, source detection, local/ZIP repository preparation, ZIP traversal protection, upload validation, and workflow-service edge cases.

Graph tests use deterministic fake models/agents so the core orchestration can be validated without consuming a live model quota.

## Deployment

See [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) for local and hosted Streamlit notes, environment requirements, workspace behavior, resource considerations, and production-hardening recommendations.

## Current limitations

- GitHub URL mode currently targets public repositories.
- The development checkpointer is in-memory; process restarts do not preserve interrupted sessions.
- Local embedding/reranking models can take time to download and index large repositories.
- This version investigates and plans changes; it does not automatically edit repository code.
- LangSmith observability was intentionally deferred and can be added later without changing the core workflow architecture.

## Portfolio summary

Issue2Impact demonstrates a progression from repository RAG to tool use, LangGraph state/cycles, specialized multi-agent handoffs, reflection, self-healing retries, human governance, dynamic repository ingestion, and a user-facing Streamlit application. The result is a complete end-to-end Agentic AI repository-analysis system rather than a single-prompt prototype.
