# Roadmap

| Item | Status | Implantacao |
| --- | ---: | --- |
| **5. Cross-Repository** | 100% | `ContractRegistry`, `BreakingChangeDetector`, snapshot versionavel, scripts `make` e CLI implementados e cobertos por testes |
| **6. Model Health & Performance Tracking** | 100% | `ModelHealthTracker` com métricas em tempo real (latência P50/P95/P99, taxa de erro, qualidade média, custo real) e `HealthScore` para roteamento adaptativo |
| **7. Semantic Prompt Routing via Embeddings** | 90% | `SemanticPromptScorer` + `HybridScorer` ligados ao runtime quando `semantic.enabled=true`; API/CLI de inspect para calibracao operacional |
| **8. Response Caching com Chave Semântica** | 0% | Cache LRU/TTL com embedding do prompt como chave; reutiliza respostas para prompts semanticamente equivalentes (>0.95 cosine) |
| **9. Canary/Blue-Green Model Rollout** | 100% | `ModelInfo.rollout_percentage` (0-100) + `MultiModelRouter._apply_rollout()` com hash determinístico; CLI `--set-rollout`; endpoints API `/v1/llmrouter/rollout`; rollback instantâneo via `rollout_percentage=0` |
| **10. Cost Budgets & Alertas por Projeto/Usuário** | 0% | `BudgetManager` com backend Redis/SQLite; headers `X-Project-ID`/`X-User-ID`; auto-fallback para modelos gratuitos ao exceder budget |

---

## 5. Cross-Repository

O fluxo cross-repository esta operacional de ponta a ponta:

- `ContractRegistry` exporta snapshots JSON deterministicos em `contracts/llmrouter.contract.json`.
- `BreakingChangeDetector` classifica mudancas compativeis e breaking changes.
- CLI: `llmrouter export-contracts`, `llmrouter check-contracts`, `llmrouter diff-contracts`.
- Makefile: `make contracts-export`, `make contracts-check`, `make contracts-diff`.
- Testes: `tests/test_cross_repository.py`.

---

## 6. Model Health & Performance Tracking

**Status**: 100% concluído — suite de testes passando (87 passed).

**Objetivo**: Tornar o roteamento adaptativo baseado em performance real (latência, erro, qualidade) e não apenas configuração estática.

**Componentes entregues**:
- `src/llmrouter/core/health.py` — `ModelHealthTracker`, `ModelHealth` dataclass, `HealthScore` composite
- Backends `InMemoryHealthStore` (default/testes) e `SQLiteHealthStore` (persistente, com TTL)
- Integração nas estratégias (`BalancedStrategy`, `CostStrategy`, `QualityStrategy`, `LatencyStrategy`) & no `MultiModelRouter`
- Coleta automática de métricas no `ProviderProxy` para requisições de sucesso e erro
- `latency_ms` real preenchido nos `ChatResponse` normalizados
- Configuração via `HealthConfig` (`config.py`) e runtime (`runtime.py`)
- Endpoint `GET /health/models` e `GET /health/models/{model_name}`
- CLI `llmrouter health` + opção "10. Show model health" no painel interativo

**Métricas coletadas por modelo**:
- Latência P50, P95, P99 (ms)
- Taxa de erro (provider errors, timeouts, validation errors)
- Score médio de qualidade (via evaluator/PRecog feedback)
- Custo real por request (USD)
- Contagem de requests na janela deslizante (últimos N minutos)

**Peso padrão do HealthScore**: latência 30%, erro 35%, qualidade 25%, custo 10%. Configurável via
`LLMROUTER_HEALTH__LATENCY_WEIGHT`, `ERROR_WEIGHT`, `QUALITY_WEIGHT`, `COST_WEIGHT`.

**Uso da API**:
```bash
curl http://localhost:12345/health/models
# ou detalhe de um modelo específico
curl http://localhost:12345/health/models/gpt-4o
```

**Uso do CLI**:
```bash
llmrouter health                         # texto
llmrouter health --json                # JSON para scripts
llmrouter health --backend sqlite --db-path data/health.db --window-minutes 15
```

**Persistência**: o backend `sqlite` mantém eventos dentro da janela de TTL configurada (padrão 60 min)
e expira dados antigos automaticamente. O backend `redis` está reservado para implementação futura e
faz fallback para memória com aviso no log.

**Testes**: `tests/test_health.py` cobre percentis, score, expiração de janela, backends in-memory/SQLite
e integração das estratégias com health tracker.

---

## 7. Semantic Prompt Routing via Embeddings

**Status**: 90% concluído — componentes implementados, ligados ao runtime e cobertos por API/CLI de inspeção.

**Objetivo**: Substituir/complementar heurísticas de keywords por compreensão semântica real da tarefa.

**Componentes entregues**:
- `src/llmrouter/core/semantic_scorer.py`
  - `SemanticPromptScorer` com interface compatível com `PromptScorer`
  - Embedder lazy `sentence-transformers/all-MiniLM-L6-v2` (~80MB, CPU/GPU)
  - Role embeddings pré-computados: `architecture`, `security_audit`, `review`, `fix`, `refactoring`, `test_generation`, `migration`, `documentation`, `summarization`
  - Mapeamento similarity → tier com calibração de confiança
  - Cache de embeddings em `data/semantic_role_embeddings.json`
  - Fallback automático para score neutro se o modelo falhar
- `HybridScorer` combina `PromptScorer` (rule-based) + `SemanticPromptScorer` com pesos configuráveis
- Configuração via `.env`:
  ```env
  LLMROUTER_SEMANTIC__ENABLED=true
  LLMROUTER_SEMANTIC__DEVICE=cuda  # ou cpu/mps
  LLMROUTER_HYBRID__RULE_WEIGHT=0.3
  LLMROUTER_HYBRID__SEMANTIC_WEIGHT=0.7
  LLMROUTER_HYBRID__SEMANTIC_CONFIDENCE_THRESHOLD=0.35
  ```
- `MultiModelRouter._build_reason` inclui `role=<semantic_role>` nas decisões de roteamento
- `Settings` ganhou `SemanticConfig` e `HybridScorerConfig`

**Pendências / próximos passos**:
- Coletar feedback do roteamento real para ajustar os embeddings por role
- Calibrar thresholds por projeto/tipo de tarefa com dados de produção

**Testes**: `tests/test_semantic_scorer.py` cobre classificação, fallback, cache, integração com router e carregamento de config.

---

## 8. Response Caching com Chave Semântica

**Objetivo**: Eliminar chamadas redundantes para prompts iguais/semanticamente equivalentes.

**Componentes novos**:
- `src/llmrouter/core/cache.py` — `SemanticCache` com backend Redis (produção) ou SQLite/LRU (dev)
- Chave: `hash(embedding(prompt) + model + temperature + top_p + max_tokens)`
- Busca exata → busca semântica (cosine > 0.95) → compute + store
- TTL configurável por tier (T1: 1h, T2: 4h, T3: 24h)
- Métricas: hit rate, tokens saved, cost saved
- Endpoint `/cache/stats` e CLI `llmrouter cache stats`

---

## 9. Canary/Blue-Green Model Rollout

**Objetivo**: Permitir promoção gradual e segura de novos modelos no catálogo.

**Mudanças**:
- `ModelInfo.rollout_percentage: float = 100.0` (0-100)
- `MultiModelRouter._apply_rollout()` filtra candidatos por weighted random
- CLI panel: opção "Set model rollout percentage" + `llmrouter panel --set-rollout <model> <pct>`
- Log estruturado: `routing_decision.rollout_sampled=model_name:pct`
- Rollback instantâneo: `rollout_percentage = 0`

---

## 10. Cost Budgets & Alertas por Projeto/Usuário

**Objetivo**: Governança de custo multi-tenant com enforcement automático.

**Componentes novos**:
- `src/llmrouter/core/budget.py` — `BudgetManager` (Redis backend recomendado)
- Headers: `X-Project-ID`, `X-User-ID` (opcional, fallback para `default`)
- Configuração via `.env` ou API: `daily_limit_usd`, `monthly_limit_usd`, `alert_threshold_pct`
- Comportamento ao exceder:
  - `soft`: log warning + header `X-Budget-Warning`
  - `hard`: auto-fallback para modelos Ollama (custo 0) + header `X-Budget-Exceeded`
- Endpoints: `GET /v1/llmrouter/budgets/{project_id}`, `POST /v1/llmrouter/budgets`
- CLI: `llmrouter budget set <project> --daily 10 --monthly 200`, `llmrouter budget status <project>`
