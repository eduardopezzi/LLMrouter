# Auditoria de benchmarks e classificação — YODA

**Coleta:** 4 de agosto de 2026, somente leitura  
**Servidor:** `vieli@yoda:/home/vieli/LLMrouter`  
**Commit em execução:** `ffa077a feat: automate benchmark research and Ollama embeddings`  
**Catálogo de benchmarks:** gerado em `2026-08-03T16:30:17Z`

## Resumo

O YODA possui 17 modelos configurados, mas somente um deles tem notas públicas
no catálogo local: `ollama/kimi-k2.7-code:cloud`, com seis testes.

O modo semântico e o ranking dinâmico por benchmarks estão habilitados no
`.env` do servidor. Porém, os seis nomes de benchmarks com notas **não existem**
em `benchmark_knowledge_base.py`, que é a base usada para converter o prompt em
pesos de benchmark. Em sentido inverso, os 19 benchmarks conhecidos pela base
semântica não possuem notas para nenhum modelo no catálogo atual.

Consequência: no commit atualmente implantado no YODA, as notas do Kimi podem
ser exibidas no painel, mas não há cobertura de benchmark para a afinidade
gerada a partir do prompt. Portanto, o ranking automático de benchmark preserva
a ordem definida pela estratégia, em vez de usar essas seis notas.

```text
benchmarks com notas no catálogo       = 6
benchmarks conhecidos semanticamente   = 19
interseção entre os dois conjuntos     = 0
modelos com alguma nota                = 1 de 17
```

## Configuração que afeta o ranking no YODA

| Configuração | Valor observado | Efeito |
| --- | --- | --- |
| `LLMROUTER_SEMANTIC__ENABLED` | `true` | Ativa embeddings e afinidade de benchmark. |
| `LLMROUTER_ROUTING__DYNAMIC_BENCHMARK_ROUTING` | `true` | Permite reordenar candidatos quando há afinidade e cobertura. |
| `LLMROUTER_ROUTING__STRATEGY` | `quality` | Sem cobertura efetiva, ordena principalmente por `priority`, health e `max_tokens`. |
| `LLMROUTER_ROUTING__FALLBACK_COUNT` | `1` | Mantém um fallback após o modelo primary. |
| Threshold simples/complexo | `0,33` / `0,66` | Define os tiers T1, T2 e T3 do prompt. |

## Testes com notas no catálogo

Todas as notas abaixo foram coletadas da model card oficial do Kimi K2.7 Code.
Cada teste tem somente um candidato; logo, não representa um ranking comparativo
entre modelos.

| Benchmark | Modelo com nota | Nota bruta | Normalizada | Está na base semântica? | Pode afetar o ranking automático atual? |
| --- | --- | ---: | ---: | --- | --- |
| Kimi Claw 24/7 Bench | `ollama/kimi-k2.7-code:cloud` | 46,9 | 0,469 | Não | Não |
| Kimi Code Bench v2 | `ollama/kimi-k2.7-code:cloud` | 62,0 | 0,620 | Não | Não |
| MCP Atlas | `ollama/kimi-k2.7-code:cloud` | 76,0 | 0,760 | Não | Não |
| MCP Mark Verified | `ollama/kimi-k2.7-code:cloud` | 81,1 | 0,811 | Não | Não |
| MLS Bench Lite | `ollama/kimi-k2.7-code:cloud` | 35,1 | 0,351 | Não | Não |
| Program Bench | `ollama/kimi-k2.7-code:cloud` | 53,6 | 0,536 | Não | Não |

## Testes reconhecidos pela classificação semântica

`benchmark_knowledge_base.py` no YODA reconhece estes testes ao comparar o
vetor do prompt com descrições de capacidade. Nenhum deles possui score para um
modelo no `data/model_benchmarks.yaml` atualmente implantado.

| Benchmark conhecido semanticamente | Modelos com notas no catálogo YODA |
| --- | ---: |
| BFCL | 0 |
| BrowseComp | 0 |
| Codeforces | 0 |
| CorpusQA 1M | 0 |
| GPQA Diamond | 0 |
| GSM8K | 0 |
| HMMT | 0 |
| Humanity's Last Exam (HLE) | 0 |
| IFEval | 0 |
| IMOAnswerBench | 0 |
| LiveCodeBench | 0 |
| MATH | 0 |
| MMLU-Pro | 0 |
| MRCR 1M | 0 |
| MuSR | 0 |
| SWE-Bench Pro | 0 |
| SWE-Bench Verified | 0 |
| SimpleQA | 0 |
| TerminalBench 2.0 | 0 |

## Modelos configurados e classificação efetiva

O tier abaixo é o efetivamente calculado pelo código em execução no YODA. Como
o arquivo daquele servidor ainda não declara `tier` para os modelos, o registry
o infere por nome, papéis, `max_tokens` e `priority`.

| Modelo | Provider | Tier efetivo | Priority | Papéis declarados | Notas de benchmark | Situação de cobertura |
| --- | --- | --- | ---: | --- | ---: | --- |
| `ollama/kimi-k3:cloud` | Ollama | T3 | 1 | architecture, fix, migration, refactoring, review, test_generation | 0 | Sem notas |
| `zhipu/glm-5.2` | Z.AI | T3 | 1 | fix, migration, refactoring, review, test_generation | 0 | Sem notas |
| `ollama/minimax-m3:cloud` | Ollama | T3 | 2 | documentation, fix, refactoring, review, summarization, test_generation | 0 | Sem notas |
| `ollama/kimi-k2.7-code:cloud` | Ollama | T3 | 3 | fix, migration, review | 6 | Notas armazenadas, mas fora da base semântica |
| `ollama/deepseek-v4-pro:cloud` | Ollama | T3 | 4 | architecture, fix, review, security_audit | 0 | Sem notas |
| `ollama/deepseek-v4-flash:0731-cloud` | Ollama | T3 | 5 | refactoring, summarization, test_generation | 0 | Sem notas |
| `ollama/qwen3-coder:480b-cloud` | Ollama | T2 | 6 | refactoring, test_generation | 0 | Sem notas |
| `ollama/glm-5.2:cloud` | Ollama | T3 | 7 | fix, migration, test_generation | 0 | Sem notas |
| `ollama/north-mini-code-1.0:cloud` | Ollama | T3 | 8 | fix, refactoring | 0 | Sem notas |
| `ollama/qwen3.6-27b:cloud` | Ollama | T3 | 9 | documentation, test_generation | 0 | Sem notas |
| `ollama/gemma4:31b` | Ollama | T3 | 10 | fix, refactoring, review | 0 | Sem notas |
| `ollama/qwen2.5-coder:3b` | Ollama | T1 | 11 | documentation, summarization | 0 | Sem notas |
| `ollama/deepseek-v3.2:cloud` | Ollama | T3 | 12 | fix, review | 0 | Sem notas |
| `ollama/deepseek-v3.1:cloud` | Ollama | T3 | 13 | summarization, test_generation | 0 | Sem notas |
| `deepseek/deepseek-chat` | DeepSeek | T3 | 14 | documentation, fix, refactoring, review, summarization, test_generation | 0 | Sem notas |
| `deepseek/deepseek-reasoner` | DeepSeek | T3 | 15 | architecture, fix, review, security_audit | 0 | Sem notas |
| `zhipu/glm-5.1` | Z.AI | T3 | 16 | documentation, fix, review, summarization, test_generation | 0 | Sem notas |

### Distribuição de tiers no YODA

| Tier | Quantidade | Modelos |
| --- | ---: | --- |
| T1 | 1 | Qwen 2.5 Coder 3B |
| T2 | 1 | Qwen 3 Coder 480B |
| T3 | 15 | Todos os demais modelos |

Essa concentração em T3 acontece porque a versão do catálogo em execução deixa
o tier implícito e a regra promove modelos por papéis de alta complexidade,
`max_tokens >= 128000` ou `priority <= 4`. Ela reduz a capacidade de usar
modelos intermediários para prompts moderados.

## Como o ranking funciona de fato hoje

Para uma requisição com `model=auto` no YODA:

1. As regras e embeddings classificam o prompt em T1/T2/T3 e inferem a tarefa.
2. O router busca candidatos no tier. Para T2, há somente o Qwen 3 Coder; para
   T1, somente o Qwen 2.5 Coder 3B; para T3, há 15 candidatos.
3. A estratégia `quality` ordena inicialmente os candidatos por menor
   `priority`, depois health e, por fim, maior `max_tokens`.
4. O ranking por benchmark não muda a ordem, porque os nomes de benchmarks
   inferidos pelo prompt não têm notas no catálogo e os seis benchmarks com
   notas não podem ser inferidos pela knowledge base.
5. A afinidade da tarefa move modelos com o papel correspondente para a frente.
6. O primeiro modelo é o primary; somente o próximo fica como fallback, pois o
   servidor usa `fallback_count=1`.

Assim, em T3 e com estratégia `quality`, `priority` e papéis declarados têm
mais influência prática que os benchmarks no YODA atual.

## Ações necessárias para tornar benchmarks efetivos

1. Implantar no YODA o commit mais recente que amplia fontes, scores e a
   knowledge base de benchmarks. O servidor está no commit `ffa077a`, anterior
   ao commit local `3311569`.
2. Garantir que cada benchmark com score em `data/model_benchmarks.yaml` tenha
   a mesma chave, ou um alias, em `benchmark_knowledge_base.py`.
3. Coletar notas comparáveis para mais de um modelo por benchmark; uma nota
   isolada deve permanecer marcada como cobertura insuficiente.
4. Declarar tiers explicitamente no catálogo do YODA para distribuir modelos
   leves, intermediários e avançados conforme o objetivo do roteamento.
5. Executar `llmrouter semantic-inspect "<prompt>" --json` após a implantação e
   verificar `benchmark_top`, `benchmark_affinities` e `benchmark_used`.

## Observação de configuração

O nome do modelo DeepSeek Flash no `config/models.yaml` do YODA contém dois
espaços ao final: `ollama/deepseek-v4-flash:0731-cloud··`. Isso pode impedir a
correspondência com catálogos ou chamadas de provider que usem o nome sem
espaços; deve ser corrigido junto da próxima implantação.

