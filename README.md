# Agentic AI — Python POCs

A collection of small, progressive proof-of-concepts exploring agentic AI patterns with the OpenAI API. Each module builds on the previous: starting from basic LLM calls with role-based prompting, through tool/function calling with both mock and live APIs, up to a full Retrieval-Augmented Generation (RAG) pipeline backed by a vector database.

---

## Prerequisites & Setup

**Requirements:** Python 3.14, Docker (RAG module only)

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Create a .env file in the project root
OPENAI_API_KEY='<your-api-key>'
OPENAI_BASE_URL='<openai-url>'   # e.g. https://api.openai.com/v1
```

---

## Project Structure

```
ai-agent-python/
├── lib/                          # Shared LLM primitives (used by all scripts)
│   ├── llm.py                    # OpenAI client wrapper with tool support
│   ├── messages.py               # Typed message classes (System, User, AI, Tool)
│   └── tooling.py                # @tool decorator — auto-generates function schemas
│
├── rag_essentials/               # Full RAG pipeline (CLI)
│   ├── main.py                   # Entry point: health / ingest / ask commands
│   ├── rag_service.py            # Orchestration layer
│   ├── pdf_loader.py             # PDF parsing and chunking
│   ├── vector_store.py           # ChromaDB integration
│   ├── llm_adapter.py            # Context injection and LLM invocation
│   ├── state_machine.py          # Pipeline state management
│   ├── config.py                 # Pydantic settings (YAML + env vars)
│   ├── models.py                 # Data models
│   └── exceptions.py             # Custom exception types
│
├── simple_agent_with_role.py     # POC 1 — role-based prompting
├── agent_with_tool.py            # POC 2 — tool calling (mock tool)
├── agent_with_tool_external_api.py  # POC 3 — tool calling (live API)
├── config.yaml                   # RAG configuration
├── docker-compose.yml            # ChromaDB service
├── TheGamingIndustry2024.pdf     # Sample document for RAG ingestion
└── requirements.txt
```

---

## Executables

### 1. Simple Agent with Role — `simple_agent_with_role.py`

**Demonstrates:** Role-based prompting. The script takes a role and a user prompt, sends them as a system + user message pair, and prints the LLM response.

```bash
python simple_agent_with_role.py
```

**What to expect:** Interactive prompts for a role (e.g. "You are a historian") and a question, then the LLM's response.

---

### 2. Agent with Internal Tool — `agent_with_tool.py`

**Demonstrates:** Tool / function calling with a mock internal tool. The agent is asked about NYC weather; it recognizes it needs the `get_weather()` tool, the script executes it, feeds the result back to the LLM, and a final grounded answer is printed.

```bash
python agent_with_tool.py
```

**What to expect:** The LLM requests the tool, the script calls it with mock temperature data, and the LLM returns a natural-language answer using that data.

---

### 3. Agent with External API Tool — `agent_with_tool_external_api.py`

**Demonstrates:** Tool calling against a real external API (PokéAPI). The agent calls `get_random_pokemon_facts()` which hits a live HTTP endpoint; the returned Pokémon data is passed back to the LLM to produce a final response.

```bash
python agent_with_tool_external_api.py
```

**What to expect:** A randomly selected Pokémon's name and flavor text, formatted by the LLM into a natural response.

---

### 4. RAG Pipeline — `rag_essentials/`

**Demonstrates:** A complete Retrieval-Augmented Generation pipeline — PDF ingestion → text chunking → vector storage in ChromaDB → semantic retrieval → LLM answer grounded in the document. The sample document is `TheGamingIndustry2024.pdf`.

**Requires Docker** (ChromaDB runs as a local container):

```bash
# Step 1 — Start ChromaDB
docker-compose up -d

# Step 2 — Verify connectivity
python -m rag_essentials.main health

# Step 3 — Ingest the sample PDF (chunks and indexes it)
python -m rag_essentials.main ingest

# Step 4 — Ask questions against the indexed document
python -m rag_essentials.main ask "Summarize the PDF"
python -m rag_essentials.main ask "What is the gaming industry outlook?" --json
```

**What to expect:** After ingestion, the `ask` command retrieves the most relevant chunks from the vector DB, injects them as context into the LLM prompt, and returns an answer grounded in the document. Use `--json` for machine-readable output.

---

## Shared Library — `lib/`

Reusable primitives shared across all standalone scripts:

| File | Purpose |
|------|---------|
| `lib/llm.py` | OpenAI client wrapper; handles tool registration and multi-turn message building |
| `lib/messages.py` | Pydantic message types: `SystemMessage`, `UserMessage`, `AIMessage`, `ToolMessage` |
| `lib/tooling.py` | `@tool` decorator that introspects function signatures and auto-generates OpenAI-compatible JSON schemas |

---

## Concepts Covered

| Concept | Where |
|---------|-------|
| Role-based prompting | `simple_agent_with_role.py` |
| Tool / function calling | `agent_with_tool.py` |
| External API tool integration | `agent_with_tool_external_api.py` |
| RAG (document Q&A) | `rag_essentials/` |
| Vector embeddings & semantic search | `rag_essentials/vector_store.py` |
| Document chunking | `rag_essentials/pdf_loader.py` |
| Automatic JSON schema generation | `lib/tooling.py` |
| Pydantic configuration management | `rag_essentials/config.py` |
| State machine for pipeline stages | `rag_essentials/state_machine.py` |
