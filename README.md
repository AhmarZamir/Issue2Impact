# Issue2Impact

Issue2Impact is an Agentic AI system for investigating software issues using repository-grounded evidence.

It combines repository ingestion, code-aware chunking, local embeddings, Chroma vector search, cross-encoder reranking, tool calling, routed LangGraph orchestration, repository investigation, structured implementation planning, critic reflection, bounded self-healing retries, human approval gates, dynamic repository input, and a Streamlit interface.

## What it can inspect

Issue2Impact can prepare and inspect three repository sources:

- a local repository folder
- a public GitHub repository URL
- a ZIP archive containing a repository

Every source is resolved to a local repository path before the agent workflow starts. The repository reader and retrieval tools are then created specifically for that repository.

Repository indexes are isolated under `data/vector_stores/<repository-id>/`. The repository ID includes a source fingerprint, so different repositories do not share the same Chroma index and local source changes create a fresh index identity.

GitHub repositories are cloned into `workspace/repos/`. Existing cached clones are fast-forwarded when possible. Git must be installed and available on `PATH` for GitHub URL mode.

## Current capabilities

- repository loading and code-aware chunking
- Hugging Face embeddings and Chroma vector storage
- candidate retrieval, reranking, and retrieval evaluation
- conditional tool calling with repository search and safe file reading
- LangGraph state, routing, cycles, and multi-agent handoffs
- Repository Investigator and structured Planner Agent
- Critic Agent with plan revision and evidence re-investigation
- bounded retry/self-healing loops
- human-in-the-loop approval using LangGraph interrupts, checkpoints, thread IDs, and resume commands
- dynamic local/GitHub/ZIP repository input
- repository-specific vector indexes and read tools
- Streamlit UI with repository loading, workflow status, investigation results, critic review, and human approval controls

## Workflow

```text
Repository source
      ↓
Local folder / GitHub / ZIP
      ↓
RepositoryContext
      ↓
Repository-specific index + tools
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

## Supported source files

The loader handles common Python, JavaScript/TypeScript, Java, C/C++, C#, Go, Rust, PHP, Ruby, Swift, Kotlin, Scala, HTML/CSS, SQL, shell, configuration, Markdown, JSON, YAML, XML, Dockerfile, and Makefile content. Large files, binary/invalid UTF-8 files, dependency folders, build output, virtual environments, and common cache directories are skipped.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add your Google API key, and select a Gemini model available to your account.

## Streamlit application

Run the user interface with:

```bash
streamlit run streamlit_app.py
```

The sidebar lets you load a repository from a public GitHub URL, a local folder, or a ZIP upload. After the repository is prepared, enter an issue or repository question and click **Investigate repository**.

The UI shows:

- repository source and supported file count
- visible agent workflow status
- captured repository investigation
- structured implementation plan
- critic approval, retry count, and evidence feedback
- final workflow output
- human approval controls when LangGraph pauses at the approval gate

If the human rejects a plan, feedback is sent back into the same checkpointed LangGraph thread. The Planner revises the plan, the Critic reviews it again, and the interface presents another approval request when appropriate.

Local folder paths only work when Streamlit is running on the same computer that owns the folder. For a hosted Streamlit deployment, use a public GitHub URL or ZIP upload instead.

## CLI

The command-line interface remains available.

### Bundled demo repository

```bash
python main.py --trace
```

### Local folder

```bash
python main.py "Investigate the authentication flow and identify risky behavior." --repo "D:\Projects\MyApp" --source-type local --trace
```

### Public GitHub repository

```bash
python main.py "Find where authentication is implemented and propose a safer design." --repo "https://github.com/user/project" --source-type github --trace
```

### ZIP repository

```bash
python main.py "Inspect this repository for authentication-related impact." --repo "./project.zip" --source-type zip --trace
```

## Tests

```bash
pytest
```

Tests cover ingestion metadata, repository-safe file reading, routing, reflection, evidence re-investigation, bounded retries, human approval/resume behavior, local repository preparation, ZIP extraction, ZIP traversal protection, automatic source detection, and uploaded ZIP validation. Graph tests use deterministic fake agents so they do not require a live model API key.
