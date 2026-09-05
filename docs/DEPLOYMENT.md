# Deployment Notes

## Local development

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run streamlit_app.py
```

Create `.env` from `.env.example` and provide `GOOGLE_API_KEY`. Git must be installed and available on `PATH` when using public GitHub repository URLs.

## Hosted Streamlit

For a hosted deployment, use public GitHub URLs or ZIP uploads. A hosted process cannot read an arbitrary folder path from a user's local computer.

Configure the model API key with the platform's secret-management feature rather than committing `.env`.

The application writes temporary/cached data to:

- `workspace/repos/` for GitHub clones and extracted repositories
- `workspace/uploads/` for uploaded ZIP archives
- `data/vector_stores/` for repository-specific Chroma indexes

These directories are ignored by Git. On ephemeral hosting they may disappear across restarts, which is acceptable for a demo but means repositories can need re-indexing.

## Resource considerations

The default embedding and reranking models run locally and can require noticeable memory and first-run download time. Large repositories also increase indexing time. For a public demo, start with small-to-medium repositories and consider limiting upload size at the hosting layer.

## Production considerations

The current checkpointer is in-memory, so interrupted sessions survive Streamlit reruns in the active process but not a process restart. A production deployment should use a durable LangGraph-compatible checkpoint store. It should also add authentication, rate limiting, persistent job/session storage, repository size quotas, and cleanup policies for workspaces and vector indexes.
