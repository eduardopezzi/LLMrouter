# API Surface — RAGFlow + Infinity (acoplamento ao LLMrouter)

**Tipo:** pesquisa da superfície de API e desenho de integração. As próximas
etapas abaixo preservam o plano original. O cliente e as rotas Lite já estão
no [registro de implementações](../implementado/CAPACIDADES_IMPLEMENTADAS.md);
decisão formal e validação operacional seguem no [roadmap](../ROADMAP.md).

**Versão alvo:** RAGFlow v0.19.1-full (validado em operação em `http://172.17.0.1:9380`)
**Data:** 2026-10-05
**Cards:** E5-A2 (`t_925d584a`)
**Próxima:** E5-A3 (ADR-0001 Lite vs Completo)

---

## 1. Componentes do stack

| Componente | Função | Endereço local | Padrão externo |
|---|---|---|---|
| **RAGFlow server** (Flask) | Orquestra: dataset, doc, chunk, retrieval, chat | `:9380` (ragflow-custom:v0191-full) | único |
| **Infinity** (Rust) | Embedder + ANN retrieval | `:16001` (interno ao RAGFlow, mas exposto via `v0.19.1` se configurado) | separado |
| **MySQL 8** | Metadata | `:5455` (interno) | n/a |
| **Elasticsearch 8.11** | Full-text + doc_index | `:9201` (interno) | n/a |
| **MinIO** | Storage de PDFs/imagens | `:9001` (interno) | n/a |
| **Valkey/Redis** | Cache de parse | `:6379` (interno) | n/a |

**Conclusão:** tudo o que precisamos acoplar é via **RAGFlow server (`:9380`)**. Infinity é interno do RAGFlow e não precisa ser exposto.

---

## 2. API surface — o que o RAGFlow oferece que o LLMrouter precisa

### 2.1 Datasets (CRUD de KBs)

| Método | Rota | Auth | Resumo da resposta |
|---|---|---|---|
| `POST` | `/api/v1/datasets` | Bearer `ragflow-...` | `{id, name, ...}` |
| `GET` | `/api/v1/datasets` | idem | `{data: [...], total}` |
| `PUT` | `/api/v1/datasets/{id}` | idem | `{id, name, ...}` |
| `DELETE` | `/api/v1/datasets/{id}` | idem | `{}` |

**Uso LLMrouter:** mapear `project_id` ↔ `dataset_id`; listar datasets disponíveis para `X-Project-ID`.

### 2.2 Documents (upload + parse)

| Método | Rota | Função |
|---|---|---|
| `POST` | `/api/v1/datasets/{id}/documents` | Upload + parse (multipart) |
| `GET` | `/api/v1/datasets/{id}/documents` | Listar com `progress` |
| `PUT` | `/api/v1/datasets/{id}/documents/{doc_id}` | Update metadata |
| `DELETE` | `/api/v1/datasets/{id}/documents/{doc_id}` | Remove |
| `POST` | `/api/v1/datasets/{id}/chunks` | Add chunk manual |
| `GET` | `/api/v1/datasets/{id}/documents/{doc_id}/chunks` | List chunks |
| `DELETE` | `/api/v1/datasets/{id}/documents/{doc_id}/chunks` | Remove chunks |

**Uso LLMrouter:** a indexação fica **fora** do LLMrouter (admin); LLMrouter só consome retrieval.

### 2.3 Retrieval — **o que interessa**

| Método | Rota | Auth | Body | Resumo |
|---|---|---|---|---|
| `POST` | `/api/v1/retrieval` | Bearer + Bearer | `{question, dataset_ids:[], top_k, similarity_threshold, vector_similarity_weight, page, page_size, rank_feature, cross_languages?}` | `{chunks:[{content, content_lt, docnm_kwd, document_id, dataset_id, similarity, vector_similarity, positions}, ...]}` |
| `POST` | `/api/v1/dify/retrieval` | **API-key (diferente)** | `{knowledge_id, query, retrieval_setting:{top_k, score_threshold}}` | `{records:[{content, score, title, metadata}]}` — **payload mínimo/compatível** |

**Decisão-chave:** `/api/v1/dify/retrieval` é um endpoint minimalista, feito para integração externa estilo Dify. Vai ser **mais barato de acoplar** do que `/retrieval` (sem pagination, sem ranks avançados).

### 2.4 Chat / completion

| Método | Rota | Função |
|---|---|---|
| `POST` | `/api/v1/chats` | Criar chat (com agent config) |
| `PUT` | `/api/v1/chats/{id}` | Update |
| `GET` | `/api/v1/chats` | List |
| `DELETE` | `/api/v1/chats/{id}` | Delete |
| `GET` | `/api/v1/conversations/getsse/{dialog_id}` | SSE streaming |

**Uso LLMrouter:** **descartado.** O LLMrouter **não vai proxyar chat do RAGFlow** — ele vai usar **retrieval** (chunks) como contexto, e a geração fica no LLMrouter.

### 2.5 Chunks admin (chunk_app.py)

`POST /list`, `GET /get`, `POST /set`, `POST /switch`, `POST /rm` — gestão de chunks via `/api/v1/chunks/*`.

**Uso LLMrouter:** não.

---

## 3. SDK Python

`api/apps/sdk/dataset.py`, `chat.py`, `doc.py` — **internos** ao RAGFlow (importam de `api.db.services`, `rag.app.tag`).

**Conclusão:** **não dá para usar o SDK Python fora do RAGFlow.** O acoplamento precisa ser **HTTP direto**.

Exemplo: o smoke test E5-A1 (`/opt/data/ragflow/smoke_test2.py`) usa `requests` puro, não o SDK.

---

## 4. Recomendações Lite vs Completo

| Cenário | Lite | Completo |
|---|---|---|
| **Acopla** | retrieval-only (1 endpoint) | retrieval + chunks admin + ingestion pipeline |
| **Endpoints** | `POST /api/v1/dify/retrieval` | `/retrieval` + `/datasets/*` + `/documents/*` + `/chunks/*` |
| **Auth** | API-key (única) | Bearer + API-key (por projeto) |
| **Code path LLMrouter** | `core/rag_query.py` (50 linhas) | `core/rag_query.py` + `core/rag_admin.py` (200+ linhas) |
| **Suporta ingestão?** | Não (admin via UI RAGFlow) | Sim (CRUD docs) |
| **Casos de uso** | LLMrouter **só consome** RAG | LLMrouter **gerencia** RAG (upload, delete, status) |
| **Latência p50 retrieval (estimada)** | < 500ms (mesma rota do Dify) | 200–800ms (retrieval completo) |

---

## 5. Recomendação

**Lite** (Dify-compatible retrieval-only) é a escolha certa para o E5:

- Endpoint único = `/api/v1/dify/retrieval` (já tem semântica mínima)
- Auth: API-key armazenada em `.env` (variável `RAGFLOW_API_KEY`)
- Code: **1 módulo novo** `core/rag_query.py` (~50 linhas) + 1 cliente `RagflowClient` (HTTP)
- **Não duplica** a UI de admin do RAGFlow — operação de dataset/doc fica na UI

**Razões:**
1. LLMrouter **não é gerenciador de conhecimento** — é gateway de LLM. Misturar admin de RAG incha o codebase.
2. O LLMrouter **já tem catálogo próprio** (`config/models.yaml`) — a abstração de "registro de fonte de contexto" pode ficar paralela.
3. O **scenario B (provider roteável)** usa Dify-style retrieval como base mesmo.
4. O cenário C (middle layer) é incremental sobre Lite.

### Trade-off aceito

- Se Eduardo quiser **upload via API** (sem acessar a UI RAGFlow), o caminho é o cenário B/C. **Não recomendado agora** — primeiro Lite, ver se atende.

---

## 6. Plano de E5-A3 (ADR-0001) — pré-requisito para E5-B/C/D

Antes de implementar `core/rag_query.py`, registrar **ADR-0001: Lite vs B vs C vs D** com:

- trade-offs quantificados desta tabela
- validação da recomendação Lite com dados de produção (latência, fallback)
- decisão explícita do Eduardo — não silenciosa

---

## Referências

- Repo RAGFlow v0.19.1: https://github.com/infiniflow/ragflow/tree/v0.19.1/api/apps/sdk
- Espérance local: `/opt/data/ragflow/` (Dockerfile.ragflow-custom)
- Spike E5-A1: `/opt/data/ragflow/smoke_test2.py` (17s E2E validado)
- Setup RAGFlow: `/opt/data/ragflow/.env` (RSA pub/priv)
- Épico: `/opt/data/backlog/ragflow-acoplamento-llmrouter.md`
