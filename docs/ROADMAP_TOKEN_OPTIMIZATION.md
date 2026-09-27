# Roadmap de Otimização de Tokens — LLMrouter + PRecog

**Branch:** `Hermes-roadmap-token-optimization`
**Base:** main `2df1fba` (Onda 2 mergeada, PR #6)
**Data:** 2026-09-27
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

## Princípios extraídos da literatura (regem todas as fases)

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

## ETAPA 1 — Cache semântico em produção, observável (sem streaming)

**Escopo LLMrouter · esforço: baixo · risco: baixo · economia: moderada**

O código da Onda 2 já entrega tudo; falta ativação disciplinada com
observabilidade de qualidade (não só volume).

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
3. **Métrica P-CHR artesanal** (`2606.19719`): job diário que re-gera
   (chamada real) uma amostra de N hits e compara; reportar `precision`
   por bucket de threshold. Endpoint `GET /v1/llmrouter/cache/stats` já
   expõe `semantic_hits/misses`; adicionar os contadores de verificação.
4. **Critério de aceite:** ≥1 semana de operação com precision de hits
   ≥99% e hit-rate estável; qualquer resposta errada detectada → threshold
   sobe para 0.98 ou flag desliga sem redeploy.

**Dependências:** nenhuma. **Contrato:** contadores novos no snapshot.

---

## ETAPA 2 — Streaming cache replay com verificação k-token

**Escopo LLMrouter · esforço: médio · risco: médio · economia: ALTA**
*(o tráfego agêntico é majoritariamente streaming — hoje 100% bypass)*

Design (LaCache `2608.01718` adaptado ao nosso proxy):

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

## ETAPA 3 — TaaS: cache de trajetórias operacionais (0 token)

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

## ETAPA 4 — Tool-schema filter + compressão query-agnostic no gateway

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

## ETAPA 5 — Preflight local: modelo pequeno resolve o trivial

**Escopo LLMrouter · esforço: médio-alto · risco: médio · economia: ALTA
nos volumes de baixa complexidade** *(InflationAgent `2608.13571`,
R2V `2605.16604`, RelayLLM `2601.05167`)*

1. **Sinal de dificuldade local** (CBE): o scorer semântico já embute
   embeddings — derivar dele um score de complexidade calibrado por
   task_role (classificação de roles já existe no item 7 do ROADMAP
   principal). Tarefas triviais (sumarização curta, formatamento, queries
   de health) → `ollama/glm-5.2` local (0 token de assinatura).
2. **Escalonamento mid-trajectory**: se o modelo local sinalizar baixa
   confiança (logprobs indisponíveis → usar verificador de formato +
   gates locais da Etapa 3), o gateway re-roteia para ZAI com o mesmo
   contexto (o cache de prefixo do provider ajuda no custo).
3. **Inflação de retry no routing** (`2608.13571`): registrar custo real
   por tarefa (não por token) para calibrar o limiar de escalonamento —
   evita o modelo local "caro por retry".
4. **Rollout canary** para a política (5%→25%→100%, item 9.1 do ROADMAP
   principal), com auto-rollback se precisão da classe local cair.

**Dependências:** Etapa 1 (scorer estável); Etapa 3 (gates como
verificador local). **Contrato:** campo `routing_signals.preflight`.

---

## ETAPA 6 — Memória de longo prazo e consolidação (PRecog)

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

## ETAPA 7 — Exploratório: cross-model KV transfer

**Escopo LLMrouter · esforço: alto · risco: alto · economia: potencial
ALTA em cascades** (`2608.03893`)

Mapeamento linear (ridge, por cabeça, RoPE-stripped) transfere KV cache
14B→32B dentro de família (Qwen3). Aplicável ao nosso cascade
`glm-5.2 → glm-5.3` **se** os providers expuserem KV interno — hoje não
expostos (API remota). Registrar como **spike** condicionado a: (a)
adoção de serving self-hosted (vLLM), ou (b) provider expondo API de
cache de prefixo compatível. Sem prazo.

---

## Ordem de execução e dependências

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

## Métricas de sucesso do roadmap (ecossistema)

| Métrica | Baseline | Meta Etapa |
| --- | --- | --- |
| Tokens de assinatura/tarefa operacional (Hermes/VSCode) | atual | **−40%** (E3) |
| Tokens de review PRecog/PR | atual | **−30%** (E2 fingerprint + E6.1) |
| Hit-rate cache semântico com precision ≥99% | 0 (flag off) | ≥25% nas rotas não-streaming (E1) |
| % de tarefas 0-token (replay/gates) | 0% | ≥30% (E3) |
| Tokens/turno de tool-schemas | 21–57K | −70% (E4.1) |
| Qualidade (suíte PRecog, revisões aceitas) | atual | **sem regressão** (todas) |

## Riscos transversais e mitigação

| Risco | Mitigação (origem literária) |
| --- | --- |
| Hit errado de cache (significado invertido) | threshold 0.97 + k-token probe + fingerprint (E2; `2608.10216`, `2608.01718`) |
| Compressão que aumenta custo total | métrica por sessão + rollout canary (E4; `2608.16370`) |
| Replay de trajetória desatualizado | gates de pré-condição + verificação pós-replay (E3; `2609.01428`) |
| SQLite bloqueando event loop sob carga | to_thread + background store (já na Onda 2), monitorar p99 |
| Escalada de escopo (superengenharia) | LRU simples; só adicionar política com evidência (`2609.28870`) |

## Processo

Cada etapa entra como branch `Hermes-<etapa>`, com TDD, contrato
atualizado no mesmo PR quando a API pública muda, e item deste roadmap
atualizado (status + evidência de medição). Merge exclusivamente via PR.
