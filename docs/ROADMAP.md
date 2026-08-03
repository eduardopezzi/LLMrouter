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
| **8. Cache de respostas** | 50% | Adicionar busca semântica conservadora sobre o cache exato existente |
| **9. Rollout canary / blue-green** | 100% | Evoluir para rollout automatizado e sticky bucketing |
| **10. Budgets e alertas por tenant** | 0% | Entregar controle de custo por projeto e usuário |
| **11. Contratos para APIs customizadas** | 0% | Permitir declarar endpoints fora do perfil OpenAI-compatible |
| **12. Governança do catálogo de modelos** | 0% | Validar metadados, limites e fontes de forma repetível |

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

### 8. Cache de respostas — 50%

**Entregue: cache exato (MVP).**

- `SQLiteCacheBackend` e `CacheManager` persistem respostas não-streaming.
- A chave normalizada considera prompt, modelo, `temperature`, `top_p` e
  `max_tokens`; streaming sempre ignora o cache.
- TTL por tier, expiração, persistência e métricas de hit rate, tokens e custo
  economizados estão implementados.
- `GET /v1/llmrouter/cache/stats` expõe as estatísticas.

**Próxima fase: cache semântico.**

- Reutilizar embeddings do scorer e procurar respostas por similaridade cosine
  com threshold configurável e conservador (inicialmente `0.95`).
- Restringir candidatos por modelo, tier e parâmetros de sampling, mantendo o
  cache exato como fallback quando embeddings não estiverem disponíveis.
- Validar explicitamente falsos positivos/negativos antes de habilitar por
  padrão; respostas erradas são um risco maior que um cache miss.

### 9. Rollout canary / blue-green — 100%

- `ModelInfo.rollout_percentage` e o filtro determinístico do router permitem
  expor modelos gradualmente sem alterar sua prioridade.
- CLI e API permitem consultar e alterar o rollout; a alteração recarrega o
  catálogo em runtime.
- `rollout_percentage=0` fornece rollback imediato, e o safety net evita que
  um filtro vazio interrompa o tráfego.

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

### 10. Budgets e alertas por tenant — 0%

**Objetivo:** governar custo por projeto e usuário de forma persistente.

- Implementar `BudgetManager` com SQLite como primeiro backend e uma interface
  que permita Redis em produção.
- Identificar consumo por `X-Project-ID` e `X-User-ID`, com fallback seguro
  para `default`; manter limites diário e mensal independentes.
- Oferecer modo `soft` (warning/header) e `hard` (bloqueio ou downgrade para
  modelo local previamente elegível), sem assumir que todo modelo Ollama tem
  custo zero.
- Expor configuração e consulta por API/CLI, persistir consumo e incorporar
  uso/custo aos dados operacionais.

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
5. Entregar contratos de endpoints customizados (item 11).
6. Instituir governança e checagens do catálogo (item 12).

## Qualidade transversal

Cada entrega deve seguir TDD: teste que falha, implementação mínima,
refatoração e suite completa. Mudanças de API pública ou CLI também exigem
atualização de contrato, README, exemplos de configuração e deste roadmap.
