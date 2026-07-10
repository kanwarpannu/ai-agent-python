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

# RAG evaluation (requires ChromaDB up + an ingested collection)
python -m rag_essentials.main eval --dataset eval_dataset.yaml
python -m rag_essentials.main eval --dataset eval_dataset.yaml --report report.json
python -m rag_essentials.main eval --json
```

There are no `pytest`-style automated tests in this repo. The closest thing is the
RAG evaluation harness (`rag_essentials/eval/`, run via the `eval` command above),
which scores pipeline quality against a labelled dataset rather than asserting unit behavior.

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

- **`main.py`** — `argparse` CLI; four commands (`health`, `ingest`, `ask`, `eval`) delegate to `RAGService` / the eval harness
- **`rag_service.py`** — Orchestrates the full flow: ingest (PDF → chunks → Chroma) and ask (retrieve → inject context → LLM). `ask()` also resets and records a per-run `trajectory` on `RAGContext` for evaluation.
- **`state_machine.py`** — Guards operations with state transitions: `IDLE → INGESTING → READY → RETRIEVING → ANSWERING`. Prevents e.g. asking before ingestion. Each `transition()` appends a `TrajectoryStep` to `RAGContext.trajectory`.
- **`pdf_loader.py`** — Reads PDFs via `pypdf`, splits into chunks with configurable size/overlap, attaches metadata (page number, chunk index), assigns UUIDs
- **`vector_store.py`** — HTTP client to local ChromaDB. Stores/retrieves chunks; uses cosine HNSW space by default
- **`llm_adapter.py`** — Formats retrieved chunks as context, builds the prompt, calls `lib.llm.LLM`
- **`config.py`** — Pydantic `BaseSettings` hierarchy loaded from `config.yaml` then env vars (prefix `RAG_`). Sections: `AppConfig`, `RetrievalConfig`, `ChromaConfig`, `LLMConfig`, `LoggingConfig`
- **`models.py`** — Pydantic data models: `DocumentChunk`, `RetrievalHit`, `RagAnswer`, `RAGContext`, `TrajectoryStep`. `RAGContext` carries a `trajectory: list[TrajectoryStep]` (cleared at the start of each `ask()`) used by trajectory evaluation.
- **`exceptions.py`** — `RAGError` base with subclasses: `ConfigurationError`, `IngestionError`, `RetrievalError`, `LLMInvocationError`

### Evaluation Harness (`rag_essentials/eval/`)

A self-contained, hand-rolled evaluator (no external eval libraries — reuses `lib.llm.LLM`
as the LLM-as-judge). Scores pipeline runs against a labelled YAML dataset across three lenses:

- **Black-box** — treats the system as opaque (question → answer): `correctness` (vs. a
  reference answer) and `answer_relevancy`. Both via LLM-as-judge.
- **Single-step** — scores each component in isolation. *Retrieval:* `context_recall`,
  `context_precision`, `mrr`, `avg_distance` (computed from `RetrievalHit` page metadata +
  distances vs. the dataset's `relevant_pages`). *Generation:* `faithfulness` (LLM-as-judge:
  is every claim grounded in the retrieved context?).
- **Trajectory** — deterministic structural checks on the recorded `RAGContext.trajectory`:
  `correct_order`, `no_error_state`, `context_nonempty_before_answer`, `expected_path_match`.

Modules:
- **`runner.py`** — `run_eval(dataset_path, settings=None, judge_model=None) -> EvalReport`;
  builds one `RAGService`, runs each case, reads back `service.context.trajectory`, scores it.
- **`judge.py`** — `Judge` wraps `lib.llm.LLM`. Since `lib.llm` has no structured-output
  support, the judge prompts for a strict JSON object (`{"score", "reason"}`) and parses it
  defensively (json.loads → regex `{...}` fallback → `score=None` on failure).
- **`metrics.py`** — pure scoring functions + `aggregate()` (per-metric means, skipping `None`).
- **`dataset.py`** — loads/validates the YAML dataset into `EvalCase` models.
- **`report.py`** — `render_table()` for the summary table; `EvalReport.model_dump_json()` for `--report`.
- **`models.py`** — `EvalCase`, `JudgeScore`, `CaseResult`, `EvalReport`.

The dataset lives at the repo root (`eval_dataset.yaml`): a YAML list of cases, each with
`id`, `question`, optional `reference_answer`, and `relevant_pages` (1-indexed, matching the
`page` metadata from `pdf_loader.py`). Judge defaults to the RAG model (`gpt-4o-mini`).

Note: trajectory state paths differ between the first `ask` on a fresh service
(`[READY, RETRIEVING, ANSWERING, READY]`) and subsequent asks (`[RETRIEVING, ANSWERING, READY]`)
because the leading `IDLE → READY` only happens once — `metrics.py` accepts both as canonical.

### Configuration

`config.yaml` is the primary config file for the RAG module. Settings can be overridden with env vars using the `RAG_` prefix. The `LLM` class in `lib/llm.py` reads `OPENAI_API_KEY` and `OPENAI_BASE_URL` directly from the environment.

## Conventions

- **Python 3.14** (pinned versions in `requirements.txt`, last tested on 3.14.6). The user invokes Python as `python3`.
- **Type hints** use built-in generics (`list[...]`, `dict[...]`) and PEP 604 unions (`X | None`) throughout — avoid reintroducing `typing.List`/`Dict`/`Optional`. Exception: `lib/tooling.py` still imports `Literal` and `Union` because it compares against them at runtime (see above).
- Use timezone-aware UTC (`datetime.now(timezone.utc)`), not the removed-path `datetime.utcnow()`.
