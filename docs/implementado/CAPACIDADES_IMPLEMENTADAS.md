# Capacidades implementadas — LLMrouter

**Referência documental:** 2026-10-06.

Este registro descreve as entregas do gateway e seus limites operacionais.
Código entregue não implica ativação ou validação em produção. As próximas
entregas e os gates de operação são mantidos no [roadmap único](../ROADMAP.md).

## 5. Contratos cross-repository — 100%

- `ContractRegistry` exporta snapshots JSON determinísticos em
  `contracts/llmrouter.contract.json`.
- `BreakingChangeDetector` identifica remoções e mudanças incompatíveis de
  endpoints, modelos, capabilities, roles, schemas e janelas de contexto.
- CLI: `export-contracts`, `check-contracts`, `diff-contracts` e
  `publish-contracts`; Makefile com os comandos equivalentes.
- O guia de CI documenta publicação no `phoenix_versions` e validação de uma
  baseline pelo repositório consumidor.

## 6. Health e performance por modelo — 100%

- `ModelHealthTracker` coleta latência P50/P95/P99, taxa de erro, qualidade,
  custo e volume por modelo, com backends em memória e SQLite.
- `HealthScore` influencia as estratégias de roteamento e as métricas são
  coletadas pelo proxy em sucessos e falhas.
- API de health e CLI estão disponíveis para inspeção operacional.

## 6.1. Estatísticas operacionais unificadas — 100%

- `MetricsCollector` agrega requests, distribuição por tier, fallback,
  falhas, streaming, percentis de latência e erros por provider/modelo.
- `GET /v1/llmrouter/stats` oferece a visão consolidada e autenticada.
- Os campos de cache e budget já existem como pontos de extensão do payload;
  devem ser preenchidos quando esses subsistemas evoluírem.

## 7. Roteamento semântico — 90%

- `SemanticPromptScorer` e `HybridScorer` são usados no runtime quando o
  recurso é habilitado, com fallback para regras se embeddings falharem.
- A inspeção sem chamada ao provider está disponível em
  `POST /v1/llmrouter/semantic/inspect` e `llmrouter semantic-inspect`.
- Roles iniciais cobrem arquitetura, segurança, revisão, correção,
  refatoração, testes, migração, documentação e sumarização.

**Pendente para concluir:** coletar feedback real de roteamento, calibrar
embeddings e thresholds por projeto/tipo de tarefa e definir métricas de
qualidade para detectar regressões da classificação.

## 8. Cache de respostas — 100%

**Entregue: cache exato (MVP) + cache semântico (opt-in).**

- `SQLiteCacheBackend` e `CacheManager` persistem respostas não-streaming.
- A chave normalizada considera prompt, modelo, `temperature`, `top_p` e
  `max_tokens`; streaming ignora o cache exato. O cache semântico tem replay próprio.
- TTL por tier, expiração, persistência e métricas de hit rate, tokens e custo
  economizados estão implementados.
- `GET /v1/llmrouter/cache/stats` expõe as estatísticas (incluindo os contadores
  semânticos `semantic_hits`/`semantic_misses`/`semantic_unavailable`).

**Cache semântico (entregue, desligado por padrão).**

- Reutiliza os embeddings do scorer híbrido e procura respostas por similaridade
  cosine com threshold conservador configurável (default `0.95`).
- Restringe candidatos por modelo, tier e parâmetros de sampling, mantendo o
  cache exato como fallback quando embeddings não estiverem disponíveis
  (circuit breaker abre após 3 falhas consecutivas do embedder).
- Embedder síncrono roda em `asyncio.to_thread` com timeout configurável
  (`embed_timeout_seconds`, default 5s) — nunca bloqueia o event loop.
- Store semântico opcionalmente em background task (`background_store`, default
  on) para não somar latência ao caminho de resposta.
- Habilitar via `llmrouter.semantic_cache.enabled=true` após validar hit rate e
  falsos positivos em produção; respostas erradas são um risco maior que um
  cache miss.

## 9. Rollout canary / blue-green — 100%

- `ModelInfo.rollout_percentage` e o filtro determinístico do router permitem
  expor modelos gradualmente sem alterar sua prioridade.
- CLI e API permitem consultar e alterar o rollout; a alteração recarrega o
  catálogo em runtime.
- `rollout_percentage=0` remove o modelo da seleção automática, inclusive no
  fallback para outros tiers. O router usa outro modelo elegível ou falha se
  nenhum respeitar o rollout configurado.


## 10. Budgets por tenant — MVP entregue

**Entregue (B1-B3, opt-in via `llmrouter.budgets.enabled`).**

- `BudgetManager` com SQLite como primeiro backend e interface que permite
  Redis em produção (`core/budget.py`).
- Consumo identificado por `X-Project-ID` e `X-User-ID`, com fallback seguro
  para `default`; limites diário e mensal independentes.
- Modo `soft` (header `X-Budget-Warning`) e `hard` (HTTP 402 no pré-flight);
  gravação de uso pós-resposta via `record_usage` com `cost_known=False`
  quando o modelo não tem preço no catálogo.
- Endpoints `GET /v1/llmrouter/budgets/{project_id}` e
  `POST /v1/llmrouter/budgets`; limites no snapshot de contrato.
- Streaming registra pré-flight apenas (usage pós-stream é limitação
  documentada); erro de budget nunca quebra a resposta de chat.

## 13. Cache semântico observável e streaming replay — código entregue

- E1: hit-log persistente, juiz local Ollama, endpoint de verificação P-CHR e
  contadores de precisão; job diário em `scripts/pchr_daily.py`.
- E2: armazenamento de streams completos, verificação k-token antes do replay
  SSE e fallback live em divergência, timeout ou indisponibilidade.
- E2.5: TTL específico, usage/status SSE quando solicitado, auditoria após a
  conclusão do gerador, aliases TL e contadores de abortos.
- Evidências e limitações estão no [relatório de QA](QA_E2_5_STREAMING_REPLAY.md).
- A janela de produção e a decisão sobre o default seguem pendentes no
  [plano E2.5](../ROADMAP.md#e2-5-streaming-replay). O guard/fingerprint de diff
  proposto para PRecog também permanece pendente em E2.

## 14. Retrieval RAGFlow Lite — base entregue

- Cliente HTTP em `src/llmrouter/core/ragflow_client.py` e configuração
  opt-in `RagflowConfig` em `src/llmrouter/config.py`.
- Rotas `POST /v1/llmrouter/rag/query` e `GET /v1/llmrouter/rag/health` em
  `src/llmrouter/api/routes.py`.
- Testes locais em `tests/test_ragflow_client.py` e
  `tests/test_ragflow_integration.py`.
- A [ADR](../desenvolvimento/ADR-0001-ragflow-coupling.md) continua registrada
  como proposta: a presença de código Lite não comprova uma decisão formal nem
  validação operacional. Essas pendências estão no item 14 do roadmap.
