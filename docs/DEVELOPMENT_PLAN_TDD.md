# Plano de Desenvolvimento TDD - LLMrouter

Este plano consolida a avaliacao do projeto, o roadmap em `docs/ROADMAP.md` e
o plano TDD anexado. A diretriz central e transformar recursos ja existentes em
capacidades operaveis em producao, mantendo a suite verde a cada etapa.

## Ciclo de Trabalho

Cada fase deve seguir o ciclo:

```text
teste falha -> implementacao minima -> teste passa -> refatora -> pytest completo -> contratos/docs
```

Quando a fase alterar API publica ou CLI, atualizar tambem:

- `ContractRegistry`
- `contracts/llmrouter.contract.json`
- `docs/ROADMAP.md`
- `README.md`
- exemplos de configuracao `.env`, quando aplicavel

## Status Atual

| Fase | Status | Resultado |
| --- | --- | --- |
| 0. Baseline verde | Concluida | Corrigido parsing de timestamp absoluto em cooldown; suite completa verde |
| 1. Semantic routing no runtime | Concluida | `build_app()` usa `HybridScorer` quando semantic esta habilitado |
| 2. Semantic inspect API/CLI | Concluida | `POST /v1/llmrouter/semantic/inspect` e `llmrouter semantic-inspect` implementados |
| 3. Stats operacionais | Concluida | `MetricsCollector` e `GET /v1/llmrouter/stats` implementados |
| 4. Cache exato SQLite | Concluida | `SQLiteCacheBackend`, `CacheManager` e `GET /v1/llmrouter/cache/stats` implementados |
| 5. Cache semantico | Pendente | Aguardando cache exato e semantic calibrado |
| 6. Budget Manager SQLite | Pendente | Aguardando metricas/custos mais consolidados |
| 7. Auto-rollback rollout | Pendente | Aguardando health/stats operacionais |

## Fase 0 - Baseline Verde

**Objetivo:** corrigir `tests/test_provider_cooldown.py::test_parses_reset_timestamp_as_utc`.

**Motivo:** a suite completa precisa estar verde antes de avancar em features.

**Escopo:**

- Investigar `src/llmrouter/core/cooldown.py`.
- Corrigir parsing de timestamp absoluto UTC.
- Rodar `pytest tests/test_provider_cooldown.py`.
- Rodar `pytest`.

**Criterio de aceite:** `pytest` completo verde.

## Fase 1 - Ativar Semantic Routing no Runtime

**Objetivo:** quando `LLMROUTER_SEMANTIC__ENABLED=true`, `build_app()` deve usar
`HybridScorer`.

**Implementacao:**

- Criar `_build_scorer(settings)` em `src/llmrouter/runtime.py`.
- Usar `PromptScorer` quando semantic estiver desabilitado.
- Usar `HybridScorer` quando semantic estiver habilitado.
- Passar pesos de `settings.hybrid`.
- Fazer fallback limpo para `PromptScorer` se o scorer semantico falhar.

**Testes:**

- Semantic disabled usa `PromptScorer`.
- Semantic enabled usa `HybridScorer`.
- Pesos customizados sao aplicados.
- Fallback funciona quando o scorer semantico esta indisponivel.

**Criterio de aceite:** semantic routing esta ligado no app real, nao apenas em
testes isolados.

## Fase 2 - Semantic Inspect API e CLI

**Objetivo:** dar visibilidade operacional a classificacao semantica de prompts.

**Endpoint:**

```text
POST /v1/llmrouter/semantic/inspect
```

**Resposta esperada:**

```json
{
  "score": 0.72,
  "tier": 3,
  "semantic_role": "review",
  "semantic_confidence": 0.81,
  "semantic_used": true,
  "signals": {}
}
```

**CLI:**

```bash
llmrouter semantic-inspect "review this architecture"
```

**Testes:**

- Exige auth quando API key esta configurada.
- Prompt vazio retorna role `none` e tier T1.
- Retorna role, confidence, tier e signals.
- Funciona com semantic disabled, expondo score rule-based.
- CLI imprime JSON ou saida estruturada.

**Criterio de aceite:** e possivel calibrar semantic routing sem chamar um
provedor externo.

## Fase 3 - Stats Operacionais Unificados

**Objetivo:** consolidar metricas hoje espalhadas em logs.

**Implementacao:**

- Criar `src/llmrouter/core/stats.py`.
- Criar `MetricsCollector` seguro para concorrencia.
- Integrar em `routes.py` e `ProviderProxy`.
- Expor `GET /v1/llmrouter/stats`.

**Metricas iniciais:**

- Requests totais.
- Distribuicao por tier.
- Fallback disponivel vs fallback usado.
- Erros por provider/modelo.
- Latencia P50/P95.
- Espaco reservado para cache e budget.

**Testes:**

- Stats vazio.
- Stats apos request.
- Fallback usado incrementa no proxy.
- Erros por provider aparecem.
- Endpoint exige auth.

**Criterio de aceite:** operador entende roteamento e falhas sem depender de
leitura manual de logs.

## Fase 4 - Response Cache MVP Exato

**Objetivo:** cache SQLite para respostas nao-streaming com chave exata
normalizada.

**Implementacao:**

- Criar `src/llmrouter/core/cache.py`.
- Implementar `SQLiteCacheBackend`.
- Implementar `CacheManager`.
- Gerar chave por prompt normalizado, modelo, temperature, top_p e max_tokens.
- TTL por tier.
- Bypass para `stream=true`.
- Expor `GET /v1/llmrouter/cache/stats`.

**Testes:**

- Miss seguido de hit.
- TTL expira.
- Parametros diferentes geram chaves diferentes.
- Streaming ignora cache.
- Persistencia SQLite.
- Stats de hit rate, tokens saved e cost saved.

**Criterio de aceite:** chamadas repetidas reduzem custo/latencia sem risco
semantico.

## Fase 5 - Cache Semantico — CONCLUIDA (opt-in)

**Objetivo:** reutilizar respostas para prompts semanticamente equivalentes.

**Implementacao (entregue nos commits A1-A3):**

- Reutilizar embeddings do semantic scorer (adapter com `asyncio.to_thread`,
  timeout e circuit breaker de 3 falhas).
- Similaridade cosine com threshold configuravel, default conservador `0.95`.
- Restringir por modelo, tier e parametros de sampling.
- Fallback para cache exato quando embeddings indisponiveis.
- Wiring via `llmrouter.semantic_cache.enabled` (default **off**).

**Testes:** `tests/test_semantic_cache.py` + `tests/test_semantic_wiring.py`.

**Criterio de aceite:** cache semantico nao gera resposta errada em cenarios
obvios.

## Fase 6 - Budget Manager MVP — CONCLUIDA (opt-in)

**Objetivo:** governanca de custo por projeto/usuario.

**Implementacao (entregue nos commits B1-B3):**

- `src/llmrouter/core/budget.py` com SQLite backend.
- Headers `X-Project-ID` e `X-User-ID` (fallback `default`).
- Limites diario e mensal; modo `soft` (header `X-Budget-Warning`) e `hard`
  (HTTP 402 no pre-flight).
- Endpoints `GET /v1/llmrouter/budgets/{project_id}` e
  `POST /v1/llmrouter/budgets`.
- Wiring via `llmrouter.budgets.enabled` (default **off**).

**Testes:** `tests/test_budget.py` + `tests/test_budget_api.py` (permite
dentro do limite, bloqueia excedido, soft nao bloqueia, budgets por
projeto/usuario independentes, reset diario/mensal, persistencia,
integracao com a rota chat, 503 sem manager).

**Criterio de aceite:** custo por tenant e governavel.

## Fase 7 - Rollout v2: Auto-Rollback

**Objetivo:** canary ruim volta automaticamente para `rollout_percentage=0`.

**Implementacao:**

- Criar `src/llmrouter/core/auto_rollback.py` ou componente equivalente.
- Gatilhos:
  - error rate acima do threshold.
  - health score abaixo do minimo.
  - P95 acima do limite.
- Usar `set_model_rollout_percentage`.
- Recarregar catalogo com `router.replace_registry`.

**Testes:**

- Rollback por erro alto.
- Rollback por latencia alta.
- Rollback por health baixo.
- Modelo estavel nao e afetado.
- Evento estruturado em log.
- Router deixa de selecionar canary apos rollback.

**Criterio de aceite:** canary pode rodar com menor supervisao manual.

## Ordem Recomendada

1. Baseline verde: cooldown.
2. Semantic routing no runtime.
3. Semantic inspect API/CLI.
4. Stats operacionais.
5. Cache exato SQLite.
6. Cache semantico.
7. Budget Manager SQLite.
8. Auto-rollback rollout.
9. Contratos e docs a cada mudanca publica.
