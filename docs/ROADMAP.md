# Roadmap — LLMrouter

Este roadmap consolida as capacidades implementadas e as próximas entregas
identificadas nos documentos de arquitetura, operação e TDD. O status mede a
implementação no repositório, e não apenas a existência de um plano.

## Visão geral

| Item | Status | Próximo resultado verificável |
| --- | ---: | --- |
| **5. Contratos cross-repository** | 100% | Manter compatibilidade e publicar contratos em releases |
| **6. Health e performance por modelo** | 100% | Usar os indicadores como base para automações de rollout |
| **6.1. Estatísticas operacionais unificadas** | 100% | Evoluir o payload conforme novos subsistemas forem adicionados |
| **7. Roteamento semântico** | 90% | Calibrar roles e thresholds com feedback de produção |
| **8. Cache de respostas** | 100% | Habilitar `llmrouter.semantic_cache.enabled` em produção com observação de hit rate e falsos positivos |
| **9. Rollout canary / blue-green** | 100% | Evoluir para rollout automatizado e sticky bucketing |
| **9.2. Controle de rollout na TUI** | 0% | Editar, validar e persistir `rollout_percentage` por modelo na interface interativa |
| **10. Budgets e alertas por tenant** | 100% | Evoluir para downgrade automático e alertas proativos (webhook/PRecog) |
| **11. Contratos para APIs customizadas** | 0% | Permitir declarar endpoints fora do perfil OpenAI-compatible |
| **12. Governança do catálogo de modelos** | 0% | Validar metadados, limites e fontes de forma repetível |
| **13. Otimização de tokens (ecossistema)** | 5% | Seguir `ROADMAP_TOKEN_OPTIMIZATION.md` (E1–E7) |

---

## Capacidades concluídas

### 5. Contratos cross-repository — 100%

- `ContractRegistry` exporta snapshots JSON determinísticos em
  `contracts/llmrouter.contract.json`.
- `BreakingChangeDetector` identifica remoções e mudanças incompatíveis de
  endpoints, modelos, capabilities, roles, schemas e janelas de contexto.
- CLI: `export-contracts`, `check-contracts`, `diff-contracts` e
  `publish-contracts`; Makefile com os comandos equivalentes.
- O guia de CI documenta publicação no `phoenix_versions` e validação de uma
  baseline pelo repositório consumidor.

### 6. Health e performance por modelo — 100%

- `ModelHealthTracker` coleta latência P50/P95/P99, taxa de erro, qualidade,
  custo e volume por modelo, com backends em memória e SQLite.
- `HealthScore` influencia as estratégias de roteamento e as métricas são
  coletadas pelo proxy em sucessos e falhas.
- API de health e CLI estão disponíveis para inspeção operacional.

### 6.1. Estatísticas operacionais unificadas — 100%

- `MetricsCollector` agrega requests, distribuição por tier, fallback,
  falhas, streaming, percentis de latência e erros por provider/modelo.
- `GET /v1/llmrouter/stats` oferece a visão consolidada e autenticada.
- Os campos de cache e budget já existem como pontos de extensão do payload;
  devem ser preenchidos quando esses subsistemas evoluírem.

### 7. Roteamento semântico — 90%

- `SemanticPromptScorer` e `HybridScorer` são usados no runtime quando o
  recurso é habilitado, com fallback para regras se embeddings falharem.
- A inspeção sem chamada ao provider está disponível em
  `POST /v1/llmrouter/semantic/inspect` e `llmrouter semantic-inspect`.
- Roles iniciais cobrem arquitetura, segurança, revisão, correção,
  refatoração, testes, migração, documentação e sumarização.

**Pendente para concluir:** coletar feedback real de roteamento, calibrar
embeddings e thresholds por projeto/tipo de tarefa e definir métricas de
qualidade para detectar regressões da classificação.

### 8. Cache de respostas — 100%

**Entregue: cache exato (MVP) + cache semântico (opt-in).**

- `SQLiteCacheBackend` e `CacheManager` persistem respostas não-streaming.
- A chave normalizada considera prompt, modelo, `temperature`, `top_p` e
  `max_tokens`; streaming sempre ignora o cache.
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

### 9. Rollout canary / blue-green — 100%

- `ModelInfo.rollout_percentage` e o filtro determinístico do router permitem
  expor modelos gradualmente sem alterar sua prioridade.
- CLI e API permitem consultar e alterar o rollout; a alteração recarrega o
  catálogo em runtime.
- `rollout_percentage=0` remove o modelo da seleção automática, inclusive no
  fallback para outros tiers. O router usa outro modelo elegível ou falha se
  nenhum respeitar o rollout configurado.

---

## Próximas entregas priorizadas

### 9.1. Automação e afinidade de rollout — 0%

**Objetivo:** reduzir a operação manual de canaries sem perder a possibilidade
de intervenção imediata.

- Auto-rollback para `rollout_percentage=0` quando um canary ultrapassar
  limites configuráveis de taxa de erro, HealthScore ou latência P95.
- Evento estruturado e auditável para cada rollback; confirmar que o router
  deixa de selecionar o canary após a atualização do catálogo.
- Sticky bucketing por `X-User-ID` e, posteriormente, `session_id`, em vez de
  somente pelo prompt, para experiências A/B consistentes.
- Auto-promoção deve permanecer desabilitada por padrão e só avançar por
  estágios explícitos (por exemplo, 5% → 25% → 50% → 100%) após janela mínima
  de amostras e métricas saudáveis.

**Dependências:** itens 6 e 6.1. **Critério de aceite:** canary degradado é
removido automaticamente, com motivo observável e sem reinício do serviço.

### 9.2. Controle de rollout na TUI — 0%

**Objetivo:** permitir que a operação ajuste o percentual de tráfego de cada
modelo sem sair da interface Textual do LLMrouter.

- Exibir o `rollout_percentage` do modelo selecionado e abrir um editor
  interativo para alterá-lo, com suporte a teclado e mouse.
- Validar valores entre `0` e `100`, deixando claros os significados de `0%`
  (fora do tráfego normal), percentual parcial (canary) e `100%` (sempre
  elegível).
- Manter alterações pendentes apenas em memória até `s`/Save, com indicador de
  estado modificado e possibilidade de recarregar sem salvar.
- Persistir o valor no catálogo usando a mesma função do CLI
  (`set_model_rollout_percentage`) e atualizar a tabela após o salvamento.
- Exibir confirmação ou erro de validação e manter o rollback manual rápido
  para `0%`.
- Cobrir o fluxo com testes Textual/CLI, incluindo modelos com e sem o campo
  explícito no YAML.

**Critério de aceite:** o operador seleciona um modelo na aba `Models`, altera
seu percentual, salva, recarrega a TUI e encontra o valor persistido; o serviço
passa a aplicar o novo rollout sem edição manual do YAML.

### 10. Budgets e alertas por tenant — 100%

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

**Próxima evolução:** downgrade automático para modelo local e alertas
proativos (webhook/PRecog).

**Dependências:** custos confiáveis do item 6. **Critério de aceite:** tenants
independentes têm consumo correto, resets de período e enforcement testados na
rota de chat.

### 11. Contratos para APIs customizadas — 0%

**Objetivo:** eliminar a geração manual de snapshots para serviços que não
usam os endpoints OpenAI-compatible embutidos na CLI.

- Permitir fornecer um manifesto de endpoints ou um arquivo de contrato-base
  para `export-contracts` e `publish-contracts`.
- Validar schema, nome do serviço e determinismo do snapshot antes de publicar.
- Manter as mesmas regras de breaking change para contratos gerados e
  declarados manualmente.

**Critério de aceite:** um serviço com endpoints próprios publica e valida seu
contrato no mesmo fluxo de CI, sem script JSON ad hoc.

### 12. Governança do catálogo de modelos — 0%

**Objetivo:** tornar repetível e auditável a manutenção de `max_tokens`,
`context_window`, capabilities e aliases de providers.

- Transformar a verificação hoje documentada em `MODEL_TOKEN_LIMITS.md` em um
  processo versionado: fonte, data de validação e decisão operacional por
  modelo.
- Criar validações de catálogo para limites coerentes (`max_tokens <=
  context_window` quando aplicável), aliases depreciados e campos obrigatórios.
- Adicionar uma checagem de CI que detecte metadados sem fonte/data ou mudanças
  incompatíveis no contrato; atualização externa deve continuar revisável, não
  automática e silenciosa.

**Critério de aceite:** toda alteração de capacidade de modelo é rastreável,
validada no CI e refletida no contrato exportado.

---

## Ordem de execução

1. Calibrar o roteamento semântico com observabilidade e feedback (item 7).
2. Concluir o cache semântico com rollout opt-in e validação de qualidade
   (item 8).
3. Implementar budgets persistentes e integrar suas métricas (item 10).
4. Automatizar rollback de canary; só então considerar auto-promoção (item 9.1).
5. Entregar o controle operacional de rollout na TUI (item 9.2).
6. Entregar contratos de endpoints customizados (item 11).
7. Instituir governança e checagens do catálogo (item 12).
8. Executar o roadmap de otimização de tokens (item 13,
   `ROADMAP_TOKEN_OPTIMIZATION.md`): E1 → E3 → E2 → E4 → E5 → E6 → E7.

## Qualidade transversal

Cada entrega deve seguir TDD: teste que falha, implementação mínima,
refatoração e suite completa. Mudanças de API pública ou CLI também exigem
atualização de contrato, README, exemplos de configuração e deste roadmap.
