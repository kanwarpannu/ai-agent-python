# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Setup

```bash
python3 -m venv .venv          # Python 3.14
source .venv/bin/activate
pip install -r requirements.txt
```

Requires a `.env` in the project root:
```
OPENAI_API_KEY='...'
OPENAI_BASE_URL='...'   # e.g. https://api.openai.com/v1
```

## Running the Scripts

```bash
# Standalone POC scripts
python simple_agent_with_role.py
python agent_with_tool.py
python agent_with_tool_external_api.py

# RAG pipeline (requires ChromaDB via Docker)
docker-compose up -d
python -m rag_essentials.main health
python -m rag_essentials.main ingest
python -m rag_essentials.main ask "your question here"
python -m rag_essentials.main ask "your question" --json
```

There are no automated tests in this repo.

## Architecture

### Shared Library (`lib/`)

Three primitives used by all standalone scripts:

- **`lib/llm.py`** — Wraps the OpenAI client. `LLM.invoke()` accepts a string, a single message, or a list of messages. When tools are registered, it builds the `tools` payload for the API and returns an `AIMessage` that may contain `tool_calls`.
- **`lib/messages.py`** — Pydantic models for `SystemMessage`, `UserMessage`, `AIMessage`, `ToolMessage`. Each has a `dict()` method that serializes to the OpenAI API wire format.
- **`lib/tooling.py`** — `@tool` decorator that introspects a function's signature and type hints to auto-generate an OpenAI-compatible JSON schema. `_infer_json_schema_type` maps: `Literal` → string enum; unions → the inner non-`None` type (matched via `origin is Literal` / `origin is Union`, so keep `Literal` and `Union` imported); `list`/`dict` → array/object; primitives via a lookup table. Both `Optional[X]`/`Union[...]` and PEP 604 `X | None` are handled (on Python 3.14 `get_origin(X | None) is Union`). Decorated functions remain callable and also expose a `.dict()` for API registration.

### Tool Calling Pattern (standalone scripts)

The two `agent_with_tool*.py` scripts follow the same loop:
1. Instantiate `LLM` and register tools via `llm.tools = [my_tool]`
2. Call `llm.invoke(messages)` — if the LLM wants a tool, `response.tool_calls` is populated
3. Execute the tool locally, wrap the result in a `ToolMessage`
4. Append both the AI response and the `ToolMessage` to the message list
5. Call `llm.invoke(messages)` again for the final grounded response

### RAG Pipeline (`rag_essentials/`)

A stateful pipeline with clear layer separation:

- **`main.py`** — Click CLI; three commands delegate to `RagService`
- **`rag_service.py`** — Orchestrates the full flow: ingest (PDF → chunks → Chroma) and ask (retrieve → inject context → LLM)
- **`state_machine.py`** — Guards operations with state transitions: `IDLE → INGESTING → READY → RETRIEVING → ANSWERING`. Prevents e.g. asking before ingestion.
- **`pdf_loader.py`** — Reads PDFs via `pypdf`, splits into chunks with configurable size/overlap, attaches metadata (page number, chunk index), assigns UUIDs
- **`vector_store.py`** — HTTP client to local ChromaDB. Stores/retrieves chunks; uses cosine HNSW space by default
- **`llm_adapter.py`** — Formats retrieved chunks as context, builds the prompt, calls `lib.llm.LLM`
- **`config.py`** — Pydantic `BaseSettings` hierarchy loaded from `config.yaml` then env vars (prefix `RAG_`). Sections: `AppConfig`, `RetrievalConfig`, `ChromaConfig`, `LLMConfig`, `LoggingConfig`
- **`models.py`** — Pydantic data models: `DocumentChunk`, `RetrievalHit`, `RagAnswer`, `RAGContext`
- **`exceptions.py`** — `RAGError` base with subclasses: `ConfigurationError`, `IngestionError`, `RetrievalError`, `LLMInvocationError`

### Configuration

`config.yaml` is the primary config file for the RAG module. Settings can be overridden with env vars using the `RAG_` prefix. The `LLM` class in `lib/llm.py` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` directly from the environment.

## Conventions

- **Python 3.14** (pinned versions in `requirements.txt`, last tested on 3.14.6). The user invokes Python as `python3`.
- **Type hints** use built-in generics (`list[...]`, `dict[...]`) and PEP 604 unions (`X | None`) throughout — avoid reintroducing `typing.List`/`Dict`/`Optional`. Exception: `lib/tooling.py` still imports `Literal` and `Union` because it compares against them at runtime (see above).
- Use timezone-aware UTC (`datetime.now(timezone.utc)`), not the removed-path `datetime.utcnow()`.
