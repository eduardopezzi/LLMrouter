# Roadmap — LLMrouter e ecossistema

**Consolidação documental:** 2026-10-06.

Este é o único roadmap do projeto. Reúne a evolução do gateway, a otimização
de tokens (E1–E7) e a reconciliação do streaming replay (E2.5). O status separa
código entregue de ativação, medição e aceite em produção. Baselines de hosts e
resultados de testes são registros datados; esta consolidação não os reexecuta.

- [Visão geral](#visão-geral)
- [Próximas entregas do gateway](#próximas-entregas-do-gateway)
- [Otimização de tokens](#13-otimização-de-tokens--llmrouter--precog):
  [E1](#e1), [E2](#e2), [E2.5](#e2-5-streaming-replay), [E3](#e3),
  [E4](#e4), [E5](#e5), [E6](#e6), [E7](#e7)
- [Integrações e estudos técnicos](#integrações-e-estudos-técnicos)
- [Ordem de execução](#ordem-de-execução)

## Visão geral

| Item | Código / desenho | Próximo resultado verificável |
| --- | --- | --- |
| **5. Contratos cross-repository** | Implementado | Manter compatibilidade e publicar contratos em releases |
| **6. Health e performance por modelo** | Implementado | Usar indicadores nas automações de rollout |
| **6.1. Estatísticas operacionais** | Implementado | Integrar novas métricas de subsistemas |
| **7. Roteamento semântico** | Implementado; calibração pendente (90% no plano original) | Calibrar roles e thresholds com feedback de produção |
| **8. Cache exato e semântico** | Implementado; semântico opt-in | Validar qualidade e hit rate em E1 |
| **9. Rollout canary / blue-green** | Implementado | Automatizar rollback e afinidade em 9.1 |
| **9.1. Automação e afinidade de rollout** | Pendente | Rollback observável, sticky bucketing e estágios de promoção |
| **9.2. Controle de rollout na TUI** | Pendente | Editar, validar e persistir o percentual na interface Textual |
| **10. Budgets por tenant** | MVP implementado; opt-in | Downgrade, alertas e usage pós-stream em 10.1 |
| **11. Contratos para APIs customizadas** | Pendente | Declarar endpoints fora do perfil OpenAI-compatible |
| **12. Governança do catálogo** | Pendente | Validar metadados, limites e fontes de forma repetível |
| **13 / E1. Cache observável** | Instrumentação implementada | ≥1 semana com precision ≥99% |
| **13 / E2. Streaming replay** | Núcleo implementado; guard de diff pendente | Entregar fingerprint e validar operação |
| **13 / E2.5. Reconciliação streaming** | Código entregue; operação pendente | Inventário, deploy, janela de 14 dias e decisão de default |
| **13 / E3–E6. Eficiência do ecossistema** | Planejado | Executar critérios e dependências de cada etapa |
| **13 / E7. Transferência KV** | Exploratório, condicionado | Spike se o serving expuser KV/cache compatível |
| **14. RAGFlow Lite** | Cliente e rotas implementados; ADR ainda proposta | Formalizar decisão e validar integração real |
| **15. CacheBlend para RAG** | Estudo futuro | Medir baseline e decidir piloto vLLM + LMCache |

## Entregas e documentação técnica

As capacidades entregues e seus limites estão em
[CAPACIDADES_IMPLEMENTADAS.md](implementado/CAPACIDADES_IMPLEMENTADAS.md).
O [fluxo de requisições](implementado/LLMROUTER_REQUEST_FLOW.md), o
[desenho de rollout](implementado/ROLL_FEATURE_DESIGN.md) e o
[QA de E2.5](implementado/QA_E2_5_STREAMING_REPLAY.md) detalham a implementação.
O [plano TDD](desenvolvimento/DEVELOPMENT_PLAN_TDD.md) define o método e os
critérios técnicos. O [índice](README.md) organiza os demais documentos.

## Próximas entregas do gateway

### 7. Calibração do roteamento semântico

Coletar feedback real de roteamento, calibrar embeddings e thresholds por
projeto/tipo de tarefa e definir métricas de qualidade para detectar regressões.
A inspeção API/CLI e o wiring do scorer já estão implementados.

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

### 10.1. Evolução de budgets por tenant — pendente

O MVP SQLite, limites diário/mensal, modos soft/hard e API estão entregues;
ver [capacidades implementadas](implementado/CAPACIDADES_IMPLEMENTADAS.md).

- Downgrade automático para modelo local ao atingir o budget.
- Alertas proativos via webhook/PRecog.
- Evoluir registro de usage pós-stream e integração das métricas de consumo.

**Dependências:** custos confiáveis do item 6. **Critério de aceite:** consumo
por tenant observável, downgrade e alertas verificáveis, sem misturar tenants.

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

- Transformar a verificação hoje documentada em [MODEL_TOKEN_LIMITS.md](informacoes/MODEL_TOKEN_LIMITS.md) em um
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

## 13. Otimização de tokens — LLMrouter + PRecog

**Branch de origem:** `Hermes-roadmap-token-optimization`
**Base do plano original:** main `2df1fba` (Onda 2 mergeada, PR #6)
**Data do plano original:** 2026-09-27

**Situação na consolidação (2026-10-06):** E1 tem instrumentação entregue;
E2 e E2.5 têm código de streaming replay entregue. Os gates de produção e o
fingerprint de diff não estão concluídos. E3–E7 permanecem como planejamento
ou exploração; dependências externas devem ser revalidadas antes da execução.
As referências de pesquisa abaixo são preservadas do plano original.

**Motivação estratégica:** os planos assinados (ZAI/DeepSeek) têm **limites de
token**. Cada token poupado é headroom para mais trabalho antes do teto —
economia de tokens é prioridade estratégica do ecossistema Vieli-Tech
(Hermes, PRecog, agente VSCode), não otimização cosmética de custo.

**Fonte:** consolidação de três pesquisas de literatura (arXiv, ago–set/2026):

1. **Streaming & cache semântico** — LaCache (`2608.01718`), Similarity
   Gates (`2608.10216`), P-CHR (`2606.19719`), FinCacheServe
   (`2607.26076`), FreshCache (`2607.04281`), eviction de prefixo
   (`2609.28870`), PrefixBench (`2609.19657`).
2. **Contexto, memória longa e compressão** — CAPC (`2607.15516`),
   cost attribution de gateways de compressão (`2609.22114`), custo de
   interação da compressão (`2608.16370`), LycheeMemory V2
   (`2608.12990`), RPMem (`2609.23466`), Harness the Memory
   (`2608.15008`), reprodução LightMem (`2607.29104`), Compact-Memory
   (`2609.04915`), Token Optimization em multi-agentes (`2608.17188`),
   Context Codec (`2605.17304`).
3. **Trabalho local antes do prompt** — TRIAGE/TaaS (`2609.01428`),
   Symbolic Separation (`2609.17107`), PYTHALAB-MERA (`2605.08468`),
   ECK (`2608.16295`), LaMR (`2605.15315`), InflationAgent
   (`2608.13571`), R2V (`2605.16604`), RelayLLM (`2601.05167`),
   skill rewriting (`2606.09421`, `2607.03048`).

---

### Princípios extraídos da literatura (regem todas as fases)

1. **Medir custo total por sessão, não por requisição** (`2608.16370`):
   compressão/cache que força re-aquisição de estado pode **aumentar** o
   consumo total (retrieval calls 21→64 com completion estável). Toda fase
   abaixo exige métrica agregada por sessão antes/depois.
2. **Similaridade cosseno não é verificação semântica** (`2608.10216`):
   inversões de significado passam com 0.96. Threshold alto **não dispensa**
   verificação barata (k-token, hashes, determinismo).
3. **Cache e compressão devem ser query-agnostic** (`2607.15516`):
   prefixos que mudam por query invalidam caches. Estabilidade do prefixo
   > agressividade da compressão.
4. **Primeiro o determinístico, depois o LLM** (`2609.17107`, `2609.01428`):
   tudo que um script valida ou replaya não gasta token. 0 token > modelo
   barato > modelo caro.
5. **LRU + recência é difícil de bater** em workloads agênticos
   (`2609.28870`): não superengenheirar evicção.
6. **TDD e contrato**: cada fase segue o padrão do repo (RED→GREEN,
   suíte completa, contrato atualizado no mesmo PR quando a API pública
   muda).

---

<a id="e1"></a>

### ETAPA 1 — Cache semântico em produção, observável (sem streaming)

**Escopo LLMrouter · esforço: baixo · risco: baixo · economia: moderada**

**Status:** cache e instrumentação P-CHR implementados; aceitação operacional
pendente. O baseline do Yoda registrado em E2.5 já tinha o cache semântico
habilitado, mas não comprova uma semana de precisão ≥99%. As configurações e
similaridades abaixo são registros do plano original, não defaults universais;
calibrar por embedder antes de adotá-las.

1. **Config de produção** (host Yoda):
   - `llmrouter.semantic_cache.enabled=true`
   - `threshold=0.85` (calibrado empiricamente para `embeddinggemma`: paráfrases
     legítimas scoram 0.85–0.95; inversões de significado scoram ≤0.73.
     O paper `2608.10216` adverte que cosseno ≥0.96 pode aprovar inversões
     em outros embedders — o threshold deve ser **medido por embedder**,
     não assumido)
   - `ttl_seconds=1800` (janela curta: pega repetição imediata, limita
     dano de hit errado)
   - `background_store=true`, `embed_timeout_seconds=5` (defaults já OK)
2. **Verificação de qualidade com 2 prompts gêmeos**: mesmos 97% do texto,
   instrução final invertida (estilo `2608.10216`) — o hit NÃO deve
   acontecer; documentar no runlog. **Conforme observado em operação:
   `embeddinggemma` produz similaridades 0.85–0.95 para paráfrases
   legítimas e ≤0.73 para inversões — threshold 0.85 discrimina
   corretamente para este embedder.**
3. **Métrica P-CHR artesanal** (`2606.19719`) — **IMPLEMENTADO (E1, onda 2)**:
   hit-log persistido (`semantic_cache_hit_log`) + verificação por juiz LLM
   local (`OllamaJudge`, POST `/api/chat` do Ollama, `temperature=0`,
   veredicto yes/no na 1ª palavra) acionado via
   `POST /v1/llmrouter/cache/verify` (body opcional `{sample_size}`,
   default `verify_sample_size=20`; requer API key; 503 sem cache semântico).
   Contadores de verificação (`pchr_pending`, `pchr_verified_ok`,
   `pchr_verified_mismatch`, `pchr_precision`, `pchr_last_verified_ts`)
   ficam no `GET /v1/llmrouter/cache/stats` (derivados do log, sobrevivem a
   restart); o detalhamento por bucket de similaridade
   ([0.80,0.85)/[0.85,0.90)/[0.90,0.95)/[0.95,1.01]) volta no payload do
   endpoint de verify. Erros de juiz (`verified=2`) são re-tentados na
   execução seguinte (só ok/mismatch são finais), então um outage do Ollama
   não queima a amostra. **Nota:** o re-gen com chamada real ao provider
   (replay do prompt contra o modelo e comparação com a resposta servida)
   ficou como opt-in futuro — hoje o juiz avalia prompt×resposta do log,
   sem gastar tokens de provider.
4. **Critério de aceite:** ≥1 semana de operação com precision de hits
   ≥99% e hit-rate estável; qualquer resposta errada detectada → threshold
   sobe para 0.98 ou flag desliga sem redeploy.

**Dependências:** nenhuma. **Contrato:** contadores novos no snapshot.

---

<a id="e2"></a>

### ETAPA 2 — Streaming cache replay com verificação k-token

**Escopo LLMrouter · esforço: médio · risco: médio · economia: ALTA**
**Status:** persistência, replay SSE, probe k-token e métricas implementados.
A reconciliação de contrato/auditoria está detalhada em [E2.5](#e2-5-streaming-replay).
O fingerprint de diff/`cache_guard` para PRecog segue pendente; a aceitação em
produção depende dos gates de E1/E2.5. O bypass de 100% era o cenário anterior
à implementação, não o comportamento atual.

Desenho de referência (LaCache `2608.01718` adaptado ao proxy):

1. **Persistir a resposta final montada** das requisições streaming
   (tokens + usage + fim normal) no mesmo cache semântico, chaveada pelo
   prompt normalizado + scope existente (model+tier+temp+top_p+max_tokens).
2. **Hit em requisição `stream=true`:**
   - replay como SSE (tokens do cache no ritmo do clock) — TTFT melhora;
   - `cache_status: semantic_hit` no evento final.
3. **Verificação k-token obrigatória** (mitiga o risco do `2608.10216`):
   antes do replay, pedir ao modelo-alvo os primeiros `k` tokens
   (`k=8` inicial, `max_tokens=k`); se casarem com o prefixo em cache
   (comparação exata de texto), replay; senão, descarta o candidato e
   segue live. Custo do probe: ~k tokens de output no pior caso.
4. **Amp escopo para o caso PRecog** (FinCacheServe `2607.26076`): chave
   de cache de reviews inclui **fingerprint do diff** (hash do texto do
   diff no prompt) — dois PRs diferentes nunca colidem mesmo com cosseno
   alto. Implementar como campo opcional `cache_guard` no prompt scorer.
5. **Métricas:** `stream_replays`, `stream_probes_ok/fail`, tokens
   economizados por replay; tudo em `/v1/llmrouter/cache/stats`.

**Testes TDD:** replay byte-a-byte igual ao live; probe divergente cai
para live; fingerprint diferente → miss garantido; abort do cliente
durante replay não corrompe estado.

**Dependências:** Etapa 1 em operação (precision validada).
**Contrato:** contadores + campos no snapshot.

---

<a id="e2-5-streaming-replay"></a>

### E2.5 — Streaming replay

**Issue:** [#11 — E2.5: reconciliar implementação com desenho TL (streaming replay)](https://github.com/eduardopezzi/LLMrouter/issues/11)

**Base da avaliação:** implementação E2 após PR #10 e follow-ups de QA round 3.
**Objetivo:** fechar as divergências do replay streaming sem quebrar clientes atuais e tornar cada decisão de rollout observável.

#### E2.5 — Decisões de escopo

1. A decisão original do item 3 da issue era adicionar auditoria de replay após a conclusão da emissão SSE, pois o caminho streaming não gravava no `semantic_cache_hit_log`. Essa auditoria foi entregue na Fase 4; não se tratava de mover uma gravação existente.
2. Métricas TL serão introduzidas com compatibilidade. Os nomes atuais são parte do contrato de `/v1/llmrouter/cache/stats`; não removê-los na mesma mudança que introduzir aliases.
3. `stream_probe_tokens_spent` será documentado como estimativa baseada no limite `k` até o provider devolver usage real para `first_tokens()`. Não apresentar essa estimativa como tokens faturados.
4. O percentil p50 exige amostras de latência com limite de memória ou um estimador de quantis. Uma soma ou média não serve para produzir p50.
5. A conclusão do gerador pode confirmar que o servidor emitiu o chunk `[DONE]`; não prova que o cliente recebeu os bytes pela rede. O hit-log usará essa definição explícita de sucesso.
6. A mudança do default para `stream_cache_enabled=false` fica condicionada a dados de uso e plano de rollout. O cache semântico principal já é opt-in, mas quem o habilita hoje também habilita streaming por padrão.

#### E2.5 — Fase 0 — Fechar decisões de contrato e obter baseline

**Escopo:** especificação e observabilidade, sem mudar comportamento de replay.

- [x] Definir o chunk SSE final com `choices: []`, `usage`, `cache_status` (`live` ou `semantic_hit`) e `usage_source` (`provider`, `cached` ou `estimated`). Preservar os headers existentes.
- [x] Emitir o chunk de usage apenas quando `stream_options.include_usage=true`, tanto no replay quanto no modo live. No live, usar usage do provider quando disponível; caso contrário, estimar e identificar a origem.
- [x] Registrar baseline do Yoda em 2026-10-06: o único host SSH configurado é Yoda; `LLMROUTER_SEMANTIC_CACHE__ENABLED=true`; override de streaming ausente (default efetivo `true`). Às 16:13:46 UTC, `/health` e `/v1/llmrouter/cache/stats` responderam HTTP 200 e os dez contadores legados de streaming estavam zerados. O processo havia iniciado às 13:41:23 UTC; ainda não havia aliases TL.
- [ ] Confirmar inventário global além do Yoda. O `~/.ssh/config` disponível contém apenas o alvo `yoda`, mas isso não prova inexistência de instalações fora desse inventário.
- [x] Atualizar a issue #11 com o resultado e o link do [PR #20](https://github.com/eduardopezzi/LLMrouter/pull/20); o item 3 agora descreve a nova auditoria de replay.

**Status:** contrato, baseline inicial do Yoda e sincronização da issue concluídos; inventário global de instalações pendente. O baseline cobre 2h32 antes do deploy e não sustenta a decisão do default.

#### E2.5 — Fase 1 — Fixar invariantes e lacunas de QA

**Escopo:** testes de regressão antes das mudanças de comportamento.

- [x] N2a: circuito aberto com `SemanticCache` real; lookup e probe não são chamados.
- [x] N2b: `max_tokens=None` persiste como `NULL`, usa `-1` na chave de unicidade e migra o schema antigo.
- [x] Interrupção live antes do terminal não armazena resposta.
- [x] Stream vazio e terminal sem conteúdo não armazenam; exceção do iterador após terminal também não armazena.
- [x] GeneratorExit em replay e live não registra hit concluído; novo contador de abortos é separado e o contador legado continua incluindo abortos durante a compatibilidade.

**Status:** concluída. Os testes focados de replay/cache/wiring passaram (72 testes).

#### E2.5 — Fase 2 — Conclusão limpa e TTL específico para streaming

**Escopo:** endurecer o caminho live e permitir retenção independente.

- [x] Persistir somente depois que a iteração termina normalmente com finish reason terminal válido e conteúdo útil.
- [x] Abort, erro do provider ou truncamento não gravam resposta.
- [x] Adicionar `stream_ttl_seconds: float | None` à configuração e ao `SemanticCache`.
- [x] `None` herda `ttl_seconds`; valores explícitos precisam ser positivos. Variável: `LLMROUTER_SEMANTIC_CACHE__STREAM_TTL_SECONDS`.

**Status:** concluída e coberta por testes de TTL herdado e específico.

#### E2.5 — Fase 3 — Completar o payload SSE do replay

**Escopo:** uso e status de cache por evento, mantendo headers existentes.

- [x] Lookup retorna usage completo e a origem dos dados.
- [x] Replay emite usage/status imediatamente antes de `[DONE]` quando solicitado.
- [x] Preservar `X-LLMrouter-Stream-Cache` e `X-LLMrouter-Cache-Status`.
- [x] Normalização preserva chunks OpenAI com `choices: []`.
- [x] Aplicar a mesma política de usage no caminho live.

**Status:** concluída; contrato e testes cobrem a ordem `chunks → usage/cache_status → [DONE]`.

#### E2.5 — Fase 4 — Auditoria de replay após emissão concluída

**Escopo:** trilha P-CHR/audit específica para hits streaming.

- [x] Criar registro de hit com prompt/resposta, modelo, restrições, similaridade e threshold do candidato.
- [x] Retornar metadata de auditoria no resultado do lookup sem modificar os chunks.
- [x] Registrar somente depois que o gerador retoma após emitir `[DONE]`; abortos não geram hit.
- [x] Falha da auditoria é best-effort e não invalida o replay.
- [x] Reutilizar o hit-log e a retenção já configurados para o cache semântico.

**Status:** concluída; integração com SQLite verifica registro único, abort e falha de escrita.

#### E2.5 — Fase 5 — Métricas TL e migração do contrato

**Escopo:** nomes, semântica e distribuição de métricas.

- [x] Adicionar aliases TL sem remover os contadores atuais.
- [x] Adicionar `stream_aborts_total` agregado e contadores `stream_replay_aborts_total` / `stream_live_aborts_total` para separar desconexões. O contador legado `stream_replay_error_total` continua incluindo abortos durante a janela de compatibilidade.
- [x] Manter amostra limitada a 1.000 probes e expor `stream_probe_latency_ms_p50` em milissegundos.
- [x] Expor `stream_probe_tokens_spent_estimated` e o alias `stream_probe_tokens_spent` como estimativa do limite `k`, não como faturamento.
- [x] Atualizar `/v1/llmrouter/cache/stats`, snapshot em `contracts/llmrouter.contract.json` e testes de schema.
- [x] Depreciar nomes antigos somente após pelo menos uma versão minor e 90 dias desde a primeira release com aliases, valendo o prazo maior; remover apenas em release major, com evidência de migração dos consumidores e notas de migração.

**Status:** implementação, schema e política de depreciação concluídos.

Definição dos contadores: `stream_hits` conta candidatos que passam pelo threshold do lookup; `stream_replays_served` conta replay depois que o gerador retoma após `[DONE]`; `stream_replay_tokens_saved` soma completion tokens apenas nesse mesmo ponto de conclusão; `stream_probe_tokens_spent` soma o limite `k` de probes executados (estimativa, não faturamento); `stream_probe_latency_ms_p50` é o p50 em milissegundos das amostras limitadas aos últimos 1.000 probes. `stream_aborts_total` conta todo GeneratorExit; `stream_replay_aborts_total` e `stream_live_aborts_total` separam o caminho interrompido. O contador legado `stream_replay_error_total` inclui esses abortos e exceções de replay.

#### E2.5 — Fase 6 — Rollout e decisão sobre o default

**Escopo:** ativação controlada e decisão informada para `stream_cache_enabled`.

- [x] Manter o default atual (`stream_cache_enabled=true`) e registrar as métricas necessárias para o rollout.
- [x] Definir a janela e o gate: observar 14 dias completos após publicar a versão instrumentada, com pelo menos 100 probes e 30 replays concluídos. Se a amostra não for atingida, estender a janela e manter o default atual.
- [x] Definir limites para considerar saudável: taxa de probe correspondente (`ok / (ok + fail)`) ≥50%, p50 do probe ≤500 ms, taxa de abortos de replay (`replay_aborts / (replays_served + replay_aborts)`) ≤5% e zero erros de replay excluídos os abortos. Registrar a decisão e os valores observados antes de qualquer mudança de default.
- [ ] Observar a janela em produção. O baseline do Yoda cobre apenas 2h32 antes do deploy e foi medido antes da versão com aliases; ainda não permite aplicar o gate.
- [ ] Decidir o default após a janela de observação; manter `true` até haver evidência para uma mudança.
- [x] Replay pode ser desativado imediatamente por `LLMROUTER_SEMANTIC_CACHE__STREAM_CACHE_ENABLED=false`.
- [x] Rollback: definir `LLMROUTER_SEMANTIC_CACHE__STREAM_CACHE_ENABLED=false` no ambiente da instância, reiniciar o serviço `llmrouter` e validar `/health` e `/v1/llmrouter/cache/stats`; reverter a variável somente após estabilização.

**Status:** plano, critérios e rollback documentados; código mergeado via PR #20, mas implantação e observação por 14 dias ainda pendentes. A janela só começa após implantar a versão instrumentada.

#### E2.5 — Verificação registrada da implementação (2026-10-06)

- [x] Testes focados de replay, cache e wiring: 72 passaram.
- [x] Suíte completa: 950 passaram e 4 foram ignorados.
- [x] Cobertura final do código E2.5 alterado: linhas executáveis 204/204 (100%); ramos condicionais nas linhas alteradas 58/58 (100%). A cobertura global do repositório inclui módulos fora deste roadmap.
- [x] Ruff e `git diff --check` passaram.
- [x] Revalidação local em 2026-10-06: os 72 testes focados passaram com medição de ramos; a suíte completa passou com 950 testes e 4 skips. Ruff nos arquivos de E2.5 e `git diff --check` passaram.
- [x] Análise de QA requisito por requisito registrada em [QA_E2_5_STREAMING_REPLAY.md](implementado/QA_E2_5_STREAMING_REPLAY.md).
- [x] Baseline de produção atualizado em 2026-10-06 16:13:46 UTC no Yoda pela porta correta (12345): `/health` e `/v1/llmrouter/cache/stats` responderam 200; o processo ainda usa `da9c781` e reportou os dez contadores legados de streaming em zero. Prometheus-01/02 seguem reiniciando: apontam para um ID de rede bridge removido, sem IP/endpoint, e falham ao conectar ao Kafka em `172.17.0.1:9092`. O endpoint direto de stats continua disponível para coleta manual.
- [x] Publicar o código na `main` via [PR #20](https://github.com/eduardopezzi/LLMrouter/pull/20), atualizar a issue #11 e fazer pull fast-forward local; suíte pós-pull passou.
- [ ] Implantar a versão instrumentada no Yoda e executar a janela do rollout. Uma worktree isolada em `/home/vieli/LLMrouter-release-ee030f7` foi criada no commit `ee030f7`; os 72 testes focados passaram em Python 3.12.3 sem carregar o `.env` de produção. Também confirmei que `PYTHONPATH` resolve o pacote para o `src/` da worktree quando executado do `WorkingDirectory` atual. O drop-in preparado só define esse `PYTHONPATH`, mantendo `.env`, `config/`, `data/` e o `WorkingDirectory` existentes. Revalidação em 2026-10-06 16:21 UTC via `ssh vieli@yoda`: serviço ativo desde 13:41 UTC, ainda em `da9c781`; worktree `ee030f7` presente; checkout `main` continua 2 commits atrás, com alterações locais staged e arquivos não rastreados preservados; drop-in ainda ausente. `sudo -n true` retorna que a senha é necessária, então instalar o drop-in e reiniciar ainda depende de acesso administrativo interativo.

Para ativar a worktree após obter acesso administrativo:

```sh
sudo install -D -m 0644 /home/vieli/llmrouter-e25.service.conf /etc/systemd/system/llmrouter.service.d/10-e25-release.conf
sudo systemctl daemon-reload
sudo systemctl restart llmrouter
sudo systemctl show llmrouter --property=ActiveState,WorkingDirectory,ExecStart --no-pager
curl -fsS http://127.0.0.1:12345/health
```

Para reverter a versão do código, remova o drop-in e reinicie o serviço; para desativar somente o replay, use `LLMROUTER_SEMANTIC_CACHE__STREAM_CACHE_ENABLED=false` no `.env` e reinicie.

#### E2.5 — Ordem sugerida de entrega

1. Fase 0 e Fase 1 podem avançar juntas; a Fase 0 deve fechar o contrato SSE antes da implementação do chunk final.
2. Fase 2 pode ser entregue separadamente e antes das mudanças de payload.
3. Fases 3 e 4 dependem da definição dos metadados e da semântica de conclusão.
4. Fase 5 deve acompanhar a implementação de cada contador, mas a migração de nomes pode ser um PR separado para facilitar revisão de contrato.
5. Fase 6 começa após haver telemetria suficiente e termina com a decisão explícita sobre o default.

Cada PR deve atualizar este roadmap e a issue #11 com o que foi concluído, links dos PRs e decisões ainda pendentes. Alterações em contrato público devem incluir atualização do snapshot e testes correspondentes no mesmo PR.

---

<a id="e3"></a>

### ETAPA 3 — TaaS: cache de trajetórias operacionais (0 token)

**Escopo Hermes + agente VSCode (infra no control_panel; integração via
gateway) · esforço: médio · risco: baixo · economia: MUITO ALTA no
trabalho operacional** *(TRIAGE `2609.01428`: 56–62% das queries
operacionais são repetição)*

Ferramentas de agente para **resolver localmente antes de pedir prompt**:

1. **Skill cache determinístico** (`TaaS`): cada tarefa operacional
   concluída com sucesso (deploy, suite+lint+push, recriar container,
   aplicar fix recorrente) grava uma **trajetória** (comandos + ordem +
   condições) no runlog existente do control_panel. Nova tarefa similar →
   casa por embedding da descrição (threshold alto, ex. 0.93) + hash dos
   parâmetros mutáveis → **replay com substituição de parâmetros** e
   verificação de pré-condições locais (gate 2). Só escalar para LLM se
   pré-condição falhar.
2. **Gates determinísticos pré-prompt** (Symbolic Separation
   `2609.17107`): biblioteca de verificações locais que o agente roda
   antes de qualquer chamada: lint/typecheck/testes, manifest consistency,
   assinatura de contratos, dead imports. O prompt ao LLM só leva o
   resíduo que os gates não resolveram.
3. **Loop de reparo local fail-fast** (PYTHALAB-MERA `2605.08468`):
   após cada edição do agente: gates locais; nova tentativa do agente só
   com o erro específico no contexto; limite de N tentativas locais antes
   de escalar.
4. **File-reader com AST pruning** (LaMR `2605.15315`): tool MCP que
   devolve skeleton (assinaturas) + spans relevantes em vez de arquivo
   inteiro — o maior consumidor de tokens de coding agents é leitura de
   arquivos.

**Implementação incremental:** (a) instrumentar runlog com schema de
trajetória; (b) matcher + replay com dry-run; (c) gates como skill
Hermes reutilizável; (d) tool MCP do file-reader.

**Critério de aceite:** 2 semanas de uso com ≥30% das tarefas
operacionais servidas por replay (contador no runlog) e zero replay
incorreto (verificação de saída dos gates pós-replay).

---

<a id="e4"></a>

### ETAPA 4 — Tool-schema filter + compressão query-agnostic no gateway

**Escopo LLMrouter · esforço: médio · risco: baixo-médio · economia: ALTA
e linear** *(cost attribution `2609.22114`: schemas de tools = 21–57K
tokens/turno; compressão estável acumula quadraticamente)*

1. **Tool-schema filter** no proxy (opt-in por tenant via header):
   - requisições com `tools:` recebem apenas os schemas cujos nomes
     aparecem no histórico recente da sessão ou num allowlist por
     `X-Project-ID`;
   - diff de schemas mantido num registry local (o gateway já observa
     todas as chamadas — fonte natural do allowlist).
2. **CAPC — compressão query-agnostic** (`2607.15516`): comprimir
   **apenas** blocos estáveis (system prompt, docs embutidos) com
   estratégia determinística (mesmo input → mesmo output), preservando
   o prefixo idêntico entre chamadas (princípio 3). Proibir compressão
   do trecho volátil (última mensagem).
3. **Context stratification** (`2608.17188`): marcar no payload os blocos
   `evergreen` vs `ephemeral`; o gateway comprime/limpa só o evergreen e
   nunca reordena o ephemeral.
4. **Métrica de guarda** (princípio 1): tokens totais por sessão antes/
   depois (o `MetricsCollector` já agrega por request; adicionar
   correlação por `session_id`).

**Testes TDD:** prefixo estável byte-a-byte sob compressão; fallback
transparente se filter quebrar; A/B por rollout_percentage.

**Dependências:** nenhuma para o filter; Etapa 2 para reuso do mesmo
mechanismo de background store.

---

<a id="e5"></a>

### ETAPA 5 — Preflight local: modelo pequeno resolve o trivial

**Escopo LLMrouter · esforço: médio-alto · risco: médio · economia: ALTA
nos volumes de baixa complexidade** *(InflationAgent `2608.13571`,
R2V `2605.16604`, RelayLLM `2601.05167`)*

1. **Sinal de dificuldade local** (CBE): o scorer semântico já embute
   embeddings — derivar dele um score de complexidade calibrado por
   task_role (classificação de roles já existe no item 7 deste roadmap). Tarefas triviais (sumarização curta, formatamento, queries
   de health) → `ollama/glm-5.2` local (0 token de assinatura).
2. **Escalonamento mid-trajectory**: se o modelo local sinalizar baixa
   confiança (logprobs indisponíveis → usar verificador de formato +
   gates locais da Etapa 3), o gateway re-roteia para ZAI com o mesmo
   contexto (o cache de prefixo do provider ajuda no custo).
3. **Inflação de retry no routing** (`2608.13571`): registrar custo real
   por tarefa (não por token) para calibrar o limiar de escalonamento —
   evita o modelo local "caro por retry".
4. **Rollout canary** para a política (5%→25%→100%, item 9.1 deste roadmap), com auto-rollback se precisão da classe local cair.

**Dependências:** Etapa 1 (scorer estável); Etapa 3 (gates como
verificador local). **Contrato:** campo `routing_signals.preflight`.

---

<a id="e6"></a>

### ETAPA 6 — Memória de longo prazo e consolidação (PRecog)

**Escopo PRecog · esforço: médio-alto · risco: baixo · economia:
indireta (qualidade por token) + direta (menos encoding)**

1. **Consolidação segment-level das observações** (LycheeMemory V2
   `2608.12990`): agrupar observações de uma review inteira num registro
   tipado quando a review fecha — hoje quase tudo vira `indexable=0`
   por falta de `outcome`; o segmento fechado tem outcome natural
   (merge/reject/comments). Menos encoding LLM, registros RAG melhores.
2. **Retriever antes de memória** (LightMem `2607.29104`, Harness
   `2608.15008`): continuar investimento no canal structure (T5) e
   avaliar reranker antes de qualquer subsistema novo de memória; medir
   qualidade **por token de contexto** (Compact-Memory `2609.04915`
   define o Pareto: 83% da qualidade com 32% dos tokens).
3. **T7 (LoRA paramétrica) reafirmado** (RPMem `2609.23466`): memória
   paramétrica transferível entre backbones é direção validada; conduzir
   T7 conforme plano STAIR (condicional ao re-measurement pós-deploy
   T5/T6 no 8888).
4. **Auditoria de skills com custo** (`2606.09421`): revisar SKILL.md
   dos agentes com o critério âncoras-de-API > brevidade; medir custo
   por tarefa antes/depois da reescrita.

**Dependências:** deploy T5/T6 no 8888 (pendente, fora deste roadmap);
outcome marcado nas observações.

---

<a id="e7"></a>

### ETAPA 7 — Exploratório: cross-model KV transfer

**Escopo LLMrouter · esforço: alto · risco: alto · economia: potencial
ALTA em cascades** (`2608.03893`)

Mapeamento linear (ridge, por cabeça, RoPE-stripped) transfere KV cache
14B→32B dentro de família (Qwen3). Aplicável ao nosso cascade
`glm-5.2 → glm-5.3` **se** os providers expuserem KV interno — hoje não
expostos (API remota). Registrar como **spike** condicionado a: (a)
adoção de serving self-hosted (vLLM), ou (b) provider expondo API de
cache de prefixo compatível. Sem prazo.

---

### Ordem de execução e dependências

```
E1 (cache semântico on + P-CHR)
 └─> E2 (streaming replay + k-token + fingerprint diff)
E3 (TaaS + gates locais)          [independente, começa em paralelo com E1]
E4 (tool-schema filter + CAPC)    [independente]
E5 (preflight local)              [após E1 + E3]
E6 (memória PRecog)               [após deploy T5/T6 no 8888]
E7 (KV transfer)                  [spike sem prazo]
```

**Priorização por ROI/esforço:** E1 → E3 → E2 → E4 → E5 → E6 → E7.

### Métricas de sucesso do roadmap (ecossistema)

| Métrica | Baseline | Meta Etapa |
| --- | --- | --- |
| Tokens de assinatura/tarefa operacional (Hermes/VSCode) | atual | **−40%** (E3) |
| Tokens de review PRecog/PR | atual | **−30%** (E2 fingerprint + E6.1) |
| Hit-rate cache semântico com precision ≥99% | 0 (baseline do plano original; flag então off) | ≥25% nas rotas não-streaming (E1) |
| % de tarefas 0-token (replay/gates) | 0% | ≥30% (E3) |
| Tokens/turno de tool-schemas | 21–57K | −70% (E4.1) |
| Qualidade (suíte PRecog, revisões aceitas) | atual | **sem regressão** (todas) |

### Riscos transversais e mitigação

| Risco | Mitigação (origem literária) |
| --- | --- |
| Hit errado de cache (significado invertido) | threshold calibrado por embedder (0.97 no plano original) + k-token probe + fingerprint (E2; `2608.10216`, `2608.01718`) |
| Compressão que aumenta custo total | métrica por sessão + rollout canary (E4; `2608.16370`) |
| Replay de trajetória desatualizado | gates de pré-condição + verificação pós-replay (E3; `2609.01428`) |
| SQLite bloqueando event loop sob carga | to_thread + background store (já na Onda 2), monitorar p99 |
| Escalada de escopo (superengenharia) | LRU simples; só adicionar política com evidência (`2609.28870`) |

### Processo

Cada etapa entra como branch `Hermes-<etapa>`, com TDD, contrato
atualizado no mesmo PR quando a API pública muda, e item deste roadmap
atualizado (status + evidência de medição). Merge exclusivamente via PR.

## Integrações e estudos técnicos

### 14. RAGFlow Lite e evolução do acoplamento

**Código entregue:** cliente HTTP opt-in, rota de consulta e rota de health;
ver [capacidades implementadas](implementado/CAPACIDADES_IMPLEMENTADAS.md).

**Pendências:** formalizar a decisão da
[ADR-0001](desenvolvimento/ADR-0001-ragflow-coupling.md), que permanece proposta;
validar credenciais/configuração, smoke E2E e latência contra o RAGFlow real;
avaliar adoção antes de promover os cenários B/C. A análise de superfície está
em [RAGFLOW_API_SURFACE.md](informacoes/RAGFLOW_API_SURFACE.md).

**Critério de aceite:** decisão registrada e integração real validada com
fallback e métricas de latência. Não interpretar testes com mocks como deploy
ou aprovação formal da ADR.

### 15. CacheBlend para reduzir latência de RAG

**Status:** estudo futuro, condicionado a serving autohospedado compatível.

A [proposta técnica](desenvolvimento/CACHEBLEND_RAG_LATENCY_PROPOSAL.md)
preserva arquitetura, riscos e critérios do piloto: medir TTFT e reuso de
chunks, estabilizar chunks/prefixos, testar vLLM + LMCache isoladamente,
comparar qualidade/latência em A/B e só então avaliar integração gradual.

**Critério de aceite:** baseline e piloto reproduzíveis, ganhos medidos no
workload local e fallback validado. Não extrapolar ganhos publicados para o
gateway atual nem confundir CacheBlend/KV com cache de respostas.

## Ordem de execução

1. Validar qualidade do cache semântico em E1 e calibrar o roteamento (item 7).
2. Fechar inventário/deploy de E2.5, iniciar sua janela instrumentada e registrar
   a decisão sobre o default apenas após cumprir o gate.
3. Completar o guard de diff de E2; E3 pode avançar em paralelo a E1 no Hermes/VSCode.
4. Automatizar rollback de canary (9.1), antes da auto-promoção; entregar edição
   de rollout na TUI (9.2) e evolução dos budgets (10.1).
5. Entregar contratos de endpoints customizados (11) e governança do catálogo (12).
6. Executar E4 → E5 → E6 → E7 respeitando dependências e métricas. A prioridade
   original por ROI é E1 → E3 → E2 → E4 → E5 → E6 → E7; E1/E2 já têm código entregue.
7. Formalizar a decisão do RAGFlow (14) e validar operação antes de ampliar o
   acoplamento; CacheBlend (15) permanece condicionado à medição e ao serving.

## Qualidade transversal

Cada entrega de código deve seguir TDD: teste que falha, implementação mínima,
refatoração e suíte completa. Mudanças de API pública ou CLI exigem atualização
do contrato, README, exemplos de configuração e deste roadmap. Cada etapa entra
em branch própria e merge via PR; registrar status e evidência de medição aqui.
Os guias e propostas preservam desenho e critérios técnicos; prioridades e
acompanhamento das entregas ficam neste arquivo.
