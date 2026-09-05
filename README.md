# Issue2Impact

Issue2Impact is a learning-focused Agentic AI system for investigating software issues using repository-grounded evidence.

It combines repository ingestion, code-aware chunking, local embeddings, Chroma vector search, cross-encoder reranking, tool calling, routed LangGraph orchestration, repository investigation, structured implementation planning, critic reflection, bounded self-healing retries, and human approval gates.

## What it can inspect

Issue2Impact can now prepare and inspect three repository sources:

- a local repository folder
- a public GitHub repository URL
- a local ZIP archive containing a repository

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

## Run against the bundled demo repository

```bash
python main.py --trace
```

## Run against a local folder

```bash
python main.py "Investigate the authentication flow and identify risky behavior." --repo "D:\Projects\MyApp" --source-type local --trace
```

`--source-type auto` is the default, so this also works:

```bash
python main.py "Where is token validation implemented?" --repo "D:\Projects\MyApp"
```

## Run against a public GitHub repository

```bash
python main.py "Find where authentication is implemented and propose a safer design." --repo "https://github.com/user/project" --source-type github --trace
```

## Run against a ZIP repository

```bash
python main.py "Inspect this repository for authentication-related impact." --repo "./project.zip" --source-type zip --trace
```

When the Critic approves a repository plan, the workflow pauses and asks for human approval. A rejection can include feedback, which is sent back to the Planner before another Critic review.

## Tests

```bash
pytest
```

Tests cover ingestion metadata, repository-safe file reading, routing, reflection, evidence re-investigation, bounded retries, human approval/resume behavior, local repository preparation, ZIP extraction, ZIP traversal protection, and automatic source detection. The graph tests use deterministic fake agents so they do not require a live model API key.
