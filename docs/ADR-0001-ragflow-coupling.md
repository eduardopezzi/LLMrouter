# ADR-0001: Acoplamento RAGFlow → LLMrouter (Lite vs B vs C vs D)

**Status:** PROPOSED → **Aguardando decisão do Eduardo**
**Data:** 2026-10-05
**Cards:** E5-A3 (`t_5c61947b`)
**Pré-requisito:** [RAGFLOW_API_SURFACE.md](RAGFLOW_API_SURFACE.md) (E5-A2)

---

## Contexto

PRecog/LLMrouter têm deficit em retrieval semântico sobre corpus longo
(documentação Phoenix: ~280 documentos em Markdown, contratos Avro/JSON,
schemas PG, runbooks). Hoje usam RAG fused (8 canais RRF, verificado
R@1=0.20 R@3=0.40 contra o live PG pós-deploy T5/T6).

RAGFlow local v0.19.1 foi spikeado em E5-A1 (smoke E2E 17s validado).
**A decisão é: como acoplar ao LLMrouter?**

Quatro cenários foram brainstormados (sessões anteriores). Decidir
qual(is) avança(m).

---

## Opções

### Cenário A — Lite (Dify-compatible retrieval-only)

**O que é:** o LLMrouter adiciona **um único endpoint** (`POST /v1/llmrouter/rag/query`)
que faz proxy/transforma para `POST /api/v1/dify/retrieval` do RAGFlow.

**Trade-offs:**
- ✅ **Simples:** ~50 linhas de código + 1 cliente HTTP
- ✅ **Não duplica UI admin** (RAGFlow UI cuida de datasets/docs/chunks)
- ✅ **Independente do SDK Python** (que é interno ao RAGFlow)
- ✅ **Tempo de implementação:** 1 PR (B1 + B2)
- ⚠️ **Não tem CRUD de datasets/documents** (gerenciado só pela UI)
- ⚠️ **Cache tem que morar no LLMrouter** (não no RAGFlow)
- ❌ **Não suporta múltiplos datasets no mesmo request** (Dify endpoint = 1 knowledge_id)

**Latência:** proxy adiciona ~10–30ms; RAGFlow retrieval <500ms p50.

---

### Cenário B — Provider roteável (RagflowCompatibleProvider)

**O que é:** o LLMrouter trata o RAGFlow como **mais um provider** —
registra um `RagflowCompatibleProvider(BaseProvider)` que faz retrieval
no RAGFlow e injeta como system prompt antes de chamar o LLM real.

**Trade-offs:**
- ✅ **Transparente para o cliente:** qualquer chamada `/v1/chat/completions`
  pode acionar RAG automaticamente
- ✅ **Cache do gateway já cobre** (sem cache semântico E1 do LLMrouter)
- ⚠️ **Mistura concerns:** LLMrouter faz retrieval agora, não só roteamento
- ⚠️ **Cache deve ser por dataset_id+query** (não por model+tier+temperature
  como o cache atual)
- ❌ **Tempo:** 2 PRs (C1 + C2)
- ❌ **Resultado: retrieval passa a ser parte do proxy** (que historicamente
  é só roteamento)

**Latência:** mesma base de Lite + latência de chamada do LLM (chat).

---

### Cenário C — Middle layer (detector de intenção)

**O que é:** o LLMrouter adiciona um **detector de intenção RAG** (flag
explícita vence; heurística palavras-chave ≥80% em probe rotulado).
Quando detecta, redireciona para o RAGFlow + responde direto sem LLM.

**Trade-offs:**
- ✅ **Cliente escolhe o caminho** (`X-Use-RAG: true` flag explícita)
- ✅ **Cache hash(prompt+datasets)** TTL 1h — reuso de queries similares
- ✅ **Reduz tokens** quando responde direto pelo RAG (não chama LLM)
- ⚠️ **Heurística + ML juntos** = complexidade
- ❌ **Tempo:** 2–3 PRs (D1 + D2)
- ❌ **Mais código, mais testes, mais suporte**

**Latência:** depende do match — quando bate (cache + heurística), 50–100ms;
quando erra, adiciona latência total do caminho.

---

### Cenário D — Não acoplar

**O que é:** ignorar RAGFlow, melhorar o RAG atual do PRecog (que já tem
8 canais RRF, structure ativo, working memory).

**Trade-offs:**
- ✅ **Zero código novo**
- ✅ **Caminho endêmico** (mais profundo que Lite)
- ❌ **Perde a infra de parse/UI do RAGFlow** (que é boa para docs longos)
- ❌ **Não atende a expectativa** que originou o spike E5-A1

---

## Comparação quantitativa

| Cenário | LOC novo | PRs | Latência p50 | Latência p95 | Cobre admin? | Cobre RAG auto? |
|---|---|---|---|---|---|---|
| **A — Lite** | ~150 (1 cliente + 1 rota + testes) | 2 (B1+B2) | <500ms | <1.5s | ❌ | ❌ |
| **B — Provider** | ~400 (provider + bench + adapter) | 2 (C1+C2) | <800ms | <2s | ❌ | ✅ |
| **C — Middle** | ~700 (detector + middleware + cache) | 3 (D1+D2+QA) | <300ms (cache hit) | <1.8s | ❌ | ✅ |
| **D — Não acoplar** | 0 | 0 | n/a | n/a | n/a | n/a |

---

## Recomendação

**Recomendado: Cenário A (Lite).** Razões:

1. **PRecog/LLMrouter já tem retrieval bom** (structure+relations R@3=0.40).
   O gap não é em retrieval — é em **cobertura do corpus** (Phoenix docs).
   O Lite entrega isso sem reescrever o pipeline.
2. **Lite é a fundação dos cenários B e C.** Sem Lite, B e C não fazem
   sentido — eles assumem o cliente do Lite como peça central.
3. **Menor risco:** 1 endpoint novo, 1 cliente HTTP. Se quebrar, é fácil
   reverter.
4. **Não amarra o time:** cenários B/C podem vir depois sem refazer Lite.
5. **Eduardo pode pilotar Lite com o RAGFlow local** (172.17.0.1:9380)
   sem depender de provider externo.

**Ordem sugerida pós-decisão:**

```
E5-A3 (decisão) → E5-B1 (rota + contrato, TDD com mock)
              → E5-B2 (RagflowClient real + bench)
              → avaliar uso real → decidir se B/C valem a pena
```

**Pontos de retorno:** se Lite ficar subutilizado em produção (<5% das
chamadas usam RAG), considerar D (não acoplar) e voltar para o RAG endêmico.

---

## Trade-offs aceitos no Lite

- CRUD de corpus fica na **UI do RAGFlow** (não há admin via API no Lite).
  Aceitável enquanto o time não precisar de automação de corpus.
- Cliente precisa saber o `dataset_id` (ou resolver via project_id mapping).
  Mapeamento `project_id → dataset_id` será no config do LLMrouter.
- Retrieval RAGFlow tem latência variável; **cliente aceita risco de
  timeout** (3s default, retry 1x).

---

## Próximos passos após decisão

1. Merge de E5-A2 (PR #14).
2. Implementar **B1** (rota `/v1/llmrouter/rag/query` + contrato + testes
   com mock do RagflowClient).
3. Implementar **B2** (RagflowClient real + integration test contra
   RAGFlow local + bench p50<3s p95<10s).
4. Health endpoint `/v1/llmrouter/rag/health` para sondagem.
5. Adicionar `RAGFLOW_API_KEY` e `RAGFLOW_BASE_URL` no `.env.example`.
6. Smoke E2E real (cliente real → LLMrouter → RAGFlow local).

---

## Referências

- API surface: [RAGFLOW_API_SURFACE.md](RAGFLOW_API_SURFACE.md)
- Smoke E5-A1: `/opt/data/ragflow/smoke_test2.py` (17s E2E)
- Épico: `/opt/data/backlog/ragflow-acoplamento-llmrouter.md`
- ROADMAP token-opt: [ROADMAP_TOKEN_OPTIMIZATION.md](ROADMAP_TOKEN_OPTIMIZATION.md) E4 (RAG como
  proxy de contexto) e E5 (RAGFlow já validado)