# PRecog Memory Backend for LLMrouter

This document is the implementation contract for the PRecog agent that will add
RAG/memory endpoints consumed by LLMrouter.

## Goal

LLMrouter can now use a pluggable memory backend. The `local` backend stores
project memory in SQLite. The new `precog` backend calls PRecog before and after
chat completions:

1. LLMrouter identifies the project from request metadata, headers, or Cline
   workspace text.
2. LLMrouter sends the project and prompt text to PRecog.
3. PRecog returns relevant RAG/memory snippets.
4. LLMrouter injects those snippets as a compact `system` message.
5. After the model responds, LLMrouter posts the interaction back to PRecog.

LLMrouter is intentionally best-effort: if PRecog is unavailable or returns an
error, LLMrouter continues without memory.

## LLMrouter Configuration

Use these environment variables in `/home/vieli/LLMrouter/.env` when the PRecog
endpoints are ready:

```env
LLMROUTER_MEMORY__ENABLED="true"
LLMROUTER_MEMORY__BACKEND="precog"
LLMROUTER_MEMORY__DEFAULT_PROJECT="default"
LLMROUTER_MEMORY__TOP_K="4"
LLMROUTER_MEMORY__MIN_SCORE="0.08"
LLMROUTER_MEMORY__MAX_CONTEXT_CHARS="2400"
LLMROUTER_MEMORY__QUERY_PATH="/internal/rag/query"
LLMROUTER_MEMORY__RECORD_PATH="/internal/llmrouter/observations"

LLMROUTER_PRECOG__BASE_URL="http://localhost:8888"
LLMROUTER_PRECOG__API_KEY="<shared-internal-token>"
LLMROUTER_PRECOG__TIMEOUT="3.0"
```

`LLMROUTER_PRECOG__ENABLED` is not required for memory lookup. It only controls
the older observation publisher path. The memory backend reuses the PRecog base
URL, API key, and timeout.

## Project Identification

LLMrouter sends one project string to PRecog. It is resolved in this order:

1. `X-LLMrouter-Project` header
2. JSON payload `llmrouter.project`
3. JSON payload `metadata.project`
4. JSON payload `extra.project`
5. Cline prompt text patterns, for example:
   `Current Workspace Directory (/home/vieli/github/PRecog)`
6. `LLMROUTER_MEMORY__DEFAULT_PROJECT`

PRecog should treat project names as tenant/namespace boundaries. Do not return
memory from another project.

## Endpoint: Query RAG/Memory

Implement:

```http
POST /internal/rag/query
Authorization: Bearer <shared-internal-token>
Content-Type: application/json
```

Request body:

```json
{
  "project": "PRecog",
  "query": "full flattened prompt text from the current chat request",
  "top_k": 4,
  "min_score": 0.08,
  "max_context_chars": 2400,
  "source": "llmrouter"
}
```

Expected response:

```json
{
  "project": "PRecog",
  "memories": [
    {
      "id": 123,
      "project": "PRecog",
      "prompt": "Prior user request or short title",
      "response": "Useful memory text or retrieved RAG chunk",
      "score": 0.91,
      "metadata": {
        "kind": "decision",
        "source": "project_docs",
        "path": "docs/architecture.md"
      }
    }
  ]
}
```

LLMrouter also accepts `results` instead of `memories`, and accepts `content`,
`text`, or `response` as the snippet body. Prefer the canonical shape above.

Status codes:

- `200`: success
- `401` or `403`: invalid internal token
- `422`: invalid request
- `5xx`: transient PRecog failure

On non-2xx, LLMrouter logs a warning and continues without memory.

## Endpoint: Record Interaction

Implement:

```http
POST /internal/llmrouter/observations
Authorization: Bearer <shared-internal-token>
Content-Type: application/json
```

Request body:

```json
{
  "request_id": "llmrouter-...",
  "project": "PRecog",
  "source": "llmrouter",
  "prompt": "original prompt text, without injected memory context",
  "response": "assistant response text",
  "metadata": {
    "request_id": "llmrouter-...",
    "task_role": "review",
    "selected_model": "ollama/kimi-k2.7-code:cloud",
    "provider": "ollama",
    "provider_model": "kimi-k2.7-code:cloud",
    "retrieved_memory_ids": [123, 456]
  }
}
```

Expected response:

```json
{
  "status": "accepted",
  "request_id": "llmrouter-...",
  "memory_id": 789
}
```

LLMrouter only requires a 2xx response. The body can be extended.

## PRecog Indexing Policy

Do not blindly index every request as durable memory. Suggested rules:

- Index when a response contains durable project knowledge, decisions, fixes,
  architecture notes, file ownership, test results, API contracts, migration
  constraints, or debugging conclusions.
- Avoid indexing secrets, full credentials, tokens, raw `.env` contents, and
  transient logs unless redacted.
- Avoid indexing low-signal greetings, repeated retries, empty responses, or
  pure tool-call noise.
- Deduplicate by project plus semantic similarity plus normalized response.
- Store source attribution: `source=llmrouter`, selected model, timestamp,
  request id, retrieved source ids, and optional file paths.

## Retrieval Policy

PRecog should combine:

- vector similarity over project docs and prior memory,
- exact keyword/path matches,
- recency and validation boosts,
- project namespace filtering.

Return short snippets. LLMrouter will inject all returned snippets into a single
system message capped by `max_context_chars`. Avoid returning huge documents.

## Security

- Require bearer auth for both endpoints.
- Never log full prompts/responses at info level.
- Redact secrets before durable indexing.
- Enforce project namespace isolation.

## Acceptance Criteria

1. `POST /internal/rag/query` returns relevant memory for a known project.
2. `POST /internal/llmrouter/observations` accepts LLMrouter interactions.
3. A second LLMrouter request for the same project can retrieve memory created
   from the first request.
4. A request for another project does not retrieve that memory.
5. PRecog downtime does not break LLMrouter; LLMrouter should receive non-2xx or
   timeout and continue normally.

## Local LLMrouter Smoke Test

After PRecog endpoints are implemented and `.env` points to `backend=precog`,
send two calls through LLMrouter:

```json
{
  "model": "auto",
  "llmrouter": {"project": "PRecog"},
  "messages": [
    {
      "role": "user",
      "content": "Remember: PRecog stores project RAG in pgvector collection project_docs."
    }
  ]
}
```

Then:

```json
{
  "model": "auto",
  "llmrouter": {"project": "PRecog"},
  "messages": [
    {
      "role": "user",
      "content": "Where does PRecog store project RAG?"
    }
  ]
}
```

Expected LLMrouter metadata on the second response:

```json
{
  "llmrouter": {
    "memory": {
      "used": true,
      "project": "PRecog",
      "top_k": 1,
      "ids": [123]
    }
  }
}
```
