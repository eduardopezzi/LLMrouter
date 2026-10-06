# Guia de Eficiência de Contexto e Tokens para LLMRouter / PRecog

**Data:** 21/09/2026\
**Objetivo:** reduzir tokens, custo e latência sem degradar
assertividade, taxa de sucesso ou capacidade de correção de código.

------------------------------------------------------------------------

## 1. Resumo executivo

O estudo **HarnessTax** traz uma conclusão especialmente relevante para
o LLMRouter e o PRecog: o *harness* --- instruções, ferramentas,
schemas, histórico, resultados de tools e loop agentic --- pode alterar
muito mais o **custo** do que a **taxa de sucesso**.

Nos experimentos do HarnessTax, foram avaliadas 21 combinações de 7
modelos com 3 harnesses (Claude Code, Codex CLI e Pi), usando SWE-bench
Lite e Terminal-Bench 2.0. A diferença média de sucesso entre harnesses
ficou pequena em comparação com a diferença de custo, que chegou a
aproximadamente **5× para o mesmo modelo**. O Pi, com apenas quatro
ferramentas centrais --- `read`, `write`, `edit` e `bash` --- permaneceu
competitivo na fronteira custo × sucesso.

Uma análise dos dados publicados do HarnessTax reporta aproximadamente:

  -----------------------------------------------------------------------
  Harness          Contexto inicial      Caracteres de        Ferramentas
                              médio   schemas de tools 
  -------------- ------------------ ------------------ ------------------
  Pi                 \~1.972 tokens            \~2.873                  4

  Codex CLI         \~11.308 tokens           \~18.114   \~3--9, conforme
                                                                   modelo

  Claude Code       \~27.011 tokens           \~76.995                 23
  -----------------------------------------------------------------------

Isso não significa que "menos contexto sempre é melhor". Significa que
**cada token permanente precisa justificar sua presença**. Contexto
irrelevante aumenta custo, ocupa capacidade de atenção e pode até
reduzir a qualidade.

### Recomendação principal

Para o PRecog/LLMRouter, a arquitetura recomendada é:

**Minimal Core Context + Tool Discovery + Retrieval on Demand + Bounded
Tool Output + Structured Working Memory + Adaptive Model/Reasoning
Routing + Prompt Cache.**

Em termos práticos:

1.  manter o prompt-base curto e estável;
2.  não enviar todos os tools/schemas em toda requisição;
3.  carregar ferramentas por intenção/domínio;
4.  recuperar código/documentação somente quando necessário;
5.  limitar e indexar outputs grandes;
6.  resumir histórico em estado estruturado;
7.  preservar evidências e decisões importantes sem preservar todo o
    diálogo;
8.  medir **custo por tarefa resolvida**, não apenas tokens por chamada;
9.  escalar modelo/contexto/reasoning apenas quando sinais objetivos
    indicarem necessidade.

------------------------------------------------------------------------

# 2. O que o HarnessTax ensina

## 2.1 Harness é parte do custo do modelo

O modelo não recebe apenas a pergunta do usuário. Em um agente de código
ele normalmente recebe:

-   system/developer prompt;
-   políticas;
-   descrição das ferramentas;
-   JSON Schemas;
-   AGENTS.md / regras do projeto;
-   histórico;
-   resultados anteriores de tools;
-   arquivos recuperados;
-   planos;
-   mensagens de erro;
-   estado do agente.

Esse conjunto forma o **context tax**.

O HarnessTax mostra que duas aplicações usando exatamente o mesmo modelo
podem obter sucesso semelhante e, ainda assim, consumir quantidades
muito diferentes de tokens.

### Consequência para o LLMRouter

Hoje o roteamento não deveria decidir apenas:

> "qual modelo é melhor para esta tarefa?"

Deveria decidir:

> "qual combinação de modelo + reasoning + contexto + tools + retrieval
> budget resolve esta tarefa com menor custo esperado?"

A unidade de otimização passa a ser:

`Task → Context Policy → Tool Policy → Model → Reasoning → Validation`

e não apenas:

`Task → Model`.

------------------------------------------------------------------------

# 3. Métrica correta: custo por sucesso

Reduzir tokens isoladamente pode piorar o sistema. Se uma redução de 40%
nos tokens aumentar retries ou diminuir a taxa de sucesso, a economia
pode desaparecer.

A métrica principal recomendada é:

``` text
Cost per Successful Task =
    Total inference cost
  + embedding/retrieval cost
  + auxiliary model cost
  + retry cost
  / successful tasks
```

Acompanhar também:

``` text
Success Rate
First-pass Success Rate
Tokens / Successful Task
Input Tokens / Task
Output Tokens / Task
Cached Input Tokens / Task
Tool-output Tokens / Task
Context-overhead Tokens / Task
Retries / Task
LLM Calls / Task
Tool Calls / Task
Latency p50 / p95
Regression Escape Rate
Human Intervention Rate
```

Para PRecog, adicionar:

``` text
Patch Acceptance Rate
Tests Passing After Patch
False-positive Review Rate
False-negative Review Rate
Coverage Claim Accuracy
Relevant Files Retrieved / Files Sent
Relevant Symbols Retrieved / Symbols Sent
```

------------------------------------------------------------------------

# 4. Contexto: tratar como orçamento, não como depósito

A literatura sobre *Lost in the Middle* demonstra que uma janela maior
não implica uso igualmente eficaz de toda a informação. Informação
relevante pode ser prejudicada quando enterrada em grandes quantidades
de conteúdo.

Portanto, não usar:

``` text
"Temos 200k tokens disponíveis, então podemos enviar 150k."
```

Usar:

``` text
"Qual é o menor conjunto de evidências suficiente para esta decisão?"
```

## 4.1 Classificação proposta do contexto

Separar o contexto em cinco classes.

### A. Core --- sempre presente

Somente:

-   objetivo do agente;
-   regras críticas;
-   formato esperado;
-   regras de segurança;
-   protocolo de uso das ferramentas;
-   critérios de conclusão.

**Meta inicial:** 800--2.500 tokens.

### B. Task --- específico da tarefa

-   descrição do PR;
-   issue;
-   diff resumido;
-   objetivo;
-   restrições.

Não deve contaminar o prompt-base/cache.

### C. Retrieved --- recuperado sob demanda

-   símbolos;
-   arquivos;
-   dependências;
-   testes relacionados;
-   contratos;
-   histórico de PRs semelhantes;
-   documentação.

### D. Working Memory --- estado resumido

Exemplo:

``` yaml
goal: corrigir regressão no cálculo de OEE
hypotheses:
  - id: H1
    text: timezone incorreto na agregação
    status: rejected
  - id: H2
    text: intervalo cruza meia-noite
    status: active

files_examined:
  - src/oee/service.py
  - tests/test_oee.py

evidence:
  - E1: test_midnight_shift falha
  - E2: aggregation usa date local

changes:
  - src/oee/service.py: normalize shift interval

remaining:
  - executar testes relacionados
  - executar regressão do módulo
```

### E. Archive --- fora do prompt

Tudo que pode ser recuperado posteriormente:

-   logs completos;
-   outputs de testes;
-   diffs antigos;
-   traces;
-   documentos;
-   histórico completo;
-   respostas intermediárias.

Guardar em banco/objeto/índice e trazer apenas quando necessário.

------------------------------------------------------------------------

# 5. Ferramentas: menos superfície, mais capacidade

O HarnessTax indica que um harness mínimo pode competir com stacks muito
maiores. Isso sugere evitar dezenas de tools simultaneamente visíveis ao
modelo.

## 5.1 Toolset mínimo para coding agent

Um núcleo forte pode ser:

``` text
read
edit
write
execute
search
```

No PRecog, eu manteria diretamente disponíveis:

``` text
repo.search
repo.read
repo.edit
shell.execute
test.run
```

E carregaria sob demanda:

``` text
git.*
github.*
rag.*
dependency_graph.*
coverage.*
security.*
database.*
observability.*
web.*
```

## 5.2 Tool Discovery

Em vez de enviar 40 schemas:

``` text
tools:
  github_create_pr
  github_get_pr
  github_comments
  github_files
  postgres_query
  rag_query
  graph_query
  ...
```

enviar inicialmente namespaces compactos:

``` text
repo      - código, símbolos e arquivos
test      - descoberta e execução de testes
git       - diff, histórico e branches
github    - PRs, issues e checks
memory    - histórico e conhecimento recuperável
graph     - dependências e impacto
```

O modelo pede a capacidade necessária e o router injeta somente os
schemas daquele domínio.

A documentação atual da OpenAI chama esse padrão de **tool
search/deferred loading**. Para catálogos grandes, carregar ferramentas
sob demanda reduz tokens de schema e preserva melhor o cache.

### Regra sugerida

-   ≤ 5 tools frequentes: carregar diretamente.

-   6--15: avaliar namespace.

-   15: discovery/deferred loading.

-   Tool usada em \< 10% das tarefas: forte candidata a carregamento sob
    demanda.

------------------------------------------------------------------------

# 6. Tool output é uma das maiores oportunidades de economia

Não basta otimizar o prompt. Um comando como:

``` bash
pytest -vv
git log -1000
docker logs
grep -R
cat arquivo_grande
```

pode devolver dezenas de milhares de tokens.

A Anthropic recomenda paginação, seleção de intervalo, filtros e
truncamento para tools com respostas grandes. Projetos como
`context-compress` aplicam exatamente esse conceito: o output completo
fica indexado e apenas uma representação reduzida entra no contexto.

## 6.1 Padrão recomendado

Nunca retornar automaticamente o stdout completo.

Tool:

``` json
{
  "command": "pytest",
  "intent": "identify failing tests",
  "max_output_tokens": 1500
}
```

Resposta:

``` yaml
exit_code: 1
summary:
  passed: 842
  failed: 2
  skipped: 17

failures:
  - test_shift_cross_midnight
  - test_timezone_conversion

important_lines:
  - "AssertionError: expected 480, got 420"
  - "service.py:184"

artifact_id: log_8fc19
truncated: true
```

Se o agente precisar:

``` text
artifact.search(log_8fc19, "timezone")
artifact.read(log_8fc19, lines=440:500)
```

Isso é muito mais eficiente que reenviar todo o log.

------------------------------------------------------------------------

# 7. Retrieval: recuperar evidência, não documentos

No PRecog, RAG não deveria responder prioritariamente com "top 10
chunks".

O ideal para código é uma recuperação hierárquica:

``` text
Task
 ↓
Repo map
 ↓
Symbol search
 ↓
Dependency graph
 ↓
Candidate files
 ↓
Relevant functions/classes
 ↓
Exact ranges
```

## 7.1 Retrieval em duas etapas

### Stage 1 --- Discovery barato

Retornar:

``` text
file
symbol
signature
relevance
reason
dependency_distance
changed?
test_relation?
```

Sem corpo completo.

### Stage 2 --- Expansion

O modelo escolhe quais itens abrir.

Exemplo:

``` text
search("calculate_oee")

1. src/oee/service.py::calculate_oee      0.94
2. src/oee/model.py::OEERecord            0.82
3. tests/test_oee.py::test_midnight       0.78
```

Depois:

``` text
read_symbol("src/oee/service.py::calculate_oee")
```

Isso evita carregar arquivos inteiros apenas porque possuem uma palavra
semelhante.

------------------------------------------------------------------------

# 8. Usar o grafo do PRecog como compressor de contexto

O PRecog já possui uma vantagem arquitetural importante:
indexador/grafo/manifesto e análise de acoplamento.

Esse grafo deve ser usado não apenas para **análise**, mas para
**seleção de contexto**.

Para um arquivo alterado:

``` text
changed symbol
    ↓
direct callers
    ↓
interfaces/contracts
    ↓
tests touching symbol
    ↓
high-coupling dependencies
```

Gerar um **Context Pack**:

``` yaml
task:
  ...

changed_symbols:
  ...

contracts:
  ...

direct_dependencies:
  ...

likely_tests:
  ...

historical_failures:
  ...

risk:
  ...

recommended_reads:
  ...
```

Meta: entregar ao modelo **mapa + evidências**, não o repositório.

------------------------------------------------------------------------

# 9. Context Budget adaptativo

Não usar um orçamento fixo para todas as tarefas.

Exemplo inicial:

  Classe             Contexto-alvo Modelo
  ---------------- --------------- ------------------------------------
  T0 trivial                2k--6k modelo barato
  T1 simples               6k--12k modelo rápido
  T2 médio                12k--30k modelo intermediário
  T3 complexo             30k--60k modelo forte
  T4 excepcional              60k+ modelo forte + expansão controlada

**Importante:** estes números são pontos iniciais para experimento, não
limites universais.

Para uma tarefa T3, não começar com 60k. Começar, por exemplo, com
15--20k e permitir expansão.

``` text
initial_budget = 18k

if evidence_insufficient:
    +8k

if cross-module dependency:
    +10k

if test failure introduces new area:
    +8k
```

Assim o agente "compra contexto" conforme necessidade.

------------------------------------------------------------------------

# 10. Adaptive Retrieval / Context Escalation

Criar três níveis.

## Level 0 --- Minimal

``` text
task
diff
repo map
critical rules
```

## Level 1 --- Local evidence

``` text
changed symbols
direct dependencies
related tests
recent related commits
```

## Level 2 --- Expanded investigation

``` text
transitive dependencies
historical PRs
RAG
architecture docs
logs
broader test output
```

O agente só sobe de nível quando houver motivo.

Sinais:

``` text
confidence < threshold
test failed unexpectedly
symbol unresolved
contract ambiguity
multiple plausible implementations
security-sensitive change
cross-service dependency
```

------------------------------------------------------------------------

# 11. Compaction sem destruir informação importante

Resumo livre pode apagar detalhes críticos.

Preferir **structured compaction**.

## Não:

``` text
"O agente analisou vários arquivos e parece que o problema está no serviço..."
```

## Usar:

``` yaml
facts:
  - F1: test X falha com input Y
  - F2: function Z recebe UTC

decisions:
  - D1: manter API pública inalterada

rejected:
  - R1: alterar schema DB
    reason: quebra compatibilidade

modified:
  - file: src/x.py
    lines: 81-104
    reason: ...

open_questions:
  - Q1: comportamento esperado quando ...

next_actions:
  - run test_x
```

Essa estrutura reduz tokens e preserva rastreabilidade.

------------------------------------------------------------------------

# 12. Prompt caching

Para workloads agentic repetitivos, caching deve fazer parte da
arquitetura.

A documentação atual da OpenAI informa que o cache depende de **prefixos
idênticos**. Portanto:

``` text
[STATIC]
system
developer rules
tool definitions
project invariants

[DYNAMIC]
task
diff
retrieved context
tool outputs
```

Evitar inserir no começo:

``` text
timestamp
request_id
user_id
branch SHA variável
task description
```

Esses dados devem ficar depois do prefixo estável.

## Para GPT-5.6+

A documentação atual da OpenAI descreve:

-   mínimo de 1.024 tokens para prefixo elegível;
-   cache explícito ou implícito;
-   leitura de cache a uma fração do preço normal;
-   pontos explícitos de cache;
-   TTL configurável de 30 minutos no mecanismo atual.

Medir:

``` text
cache_hit_tokens / cache_eligible_tokens
```

Meta inicial sugerida para fluxos repetitivos:

``` text
> 70% bom
> 85% excelente
> 90% investigar se alcançável no workload
```

Não aumentar artificialmente prompts só para atingir cache sem validar
custo total.

------------------------------------------------------------------------

# 13. Otimizar schemas das tools

Schemas podem ser uma despesa invisível.

## Evitar

``` json
{
  "name": "execute_command_in_repository_and_return_the_results",
  "description": "This tool allows the assistant to execute...",
  ...
}
```

## Preferir

``` json
{
  "name": "exec",
  "description": "Run a command in the repo.",
  ...
}
```

Mas não comprimir até ficar ambíguo.

### Princípio

**mínimo texto que mantém seleção e chamada corretas.**

Criar um benchmark específico de tool selection:

``` text
100 intents
→ tool escolhida
→ parâmetros
→ sucesso
```

Testar schema original vs reduzido.

Aceitar a redução apenas se:

``` text
Tool Selection Accuracy não cair significativamente
Invalid Tool Calls não aumentar
Retries não aumentar
```

------------------------------------------------------------------------

# 14. Evitar tools redundantes

Exemplo ruim:

``` text
search_file
search_code
grep_code
find_symbol
find_text
repository_search
semantic_search
```

O modelo precisa gastar raciocínio decidindo qual usar.

Preferir:

``` text
repo.search(mode=text|symbol|semantic)
```

ou poucas ferramentas claramente distintas.

Tool overlap aumenta:

-   tokens de schema;
-   erro de seleção;
-   retries;
-   chamadas desnecessárias;
-   complexidade do prompt.

------------------------------------------------------------------------

# 15. Modelo pequeno como filtro, não necessariamente como executor

O LLMRouter pode usar modelos baratos para tarefas de baixa entropia:

``` text
classificação da tarefa
seleção do domínio de tools
extração estruturada
ranking de chunks
deduplicação
classificação de logs
detecção de erro conhecido
```

E reservar o modelo forte para:

``` text
arquitetura
debugging ambíguo
security audit
patch complexo
reasoning cross-module
revisão final
```

Porém, cada chamada auxiliar possui overhead.

Portanto testar:

``` text
1 chamada grande
vs
classifier + retrieval + executor
```

O pipeline multi-model só é vantajoso quando o trabalho economizado na
chamada cara supera seu próprio custo e latência.

------------------------------------------------------------------------

# 16. Estratégia específica para o LLMRouter

Hoje o router deveria produzir algo semelhante a:

``` json
{
  "task_type": "security_audit",
  "complexity": "T3",
  "model": "deepseek-v4-pro",
  "fallback": "deepseek-v4-flash",
  "reasoning": "high",
  "context_budget": 24000,
  "tool_profile": "security_code_review",
  "retrieval_depth": 2,
  "max_tool_output": 2000,
  "memory_policy": "structured",
  "cache_profile": "project_static_v3"
}
```

Isso transforma o router em **Resource Policy Engine**.

## Score proposto

``` text
utility =
    success_probability
    - λ1 * normalized_cost
    - λ2 * normalized_latency
    - λ3 * retry_probability
```

Ou otimizar diretamente:

``` text
Expected Cost per Success =
    Expected Total Cost / P(success)
```

------------------------------------------------------------------------

# 17. Estratégia específica para PRecog

Pipeline recomendado:

``` text
PR / Issue
   ↓
Task classifier
   ↓
Minimal repo map
   ↓
Diff parser
   ↓
Changed symbols
   ↓
Dependency graph
   ↓
Test Impact Analysis
   ↓
Context Pack
   ↓
Review/Fix Agent
   ↓
Targeted tests
   ↓
Failure evidence
   ├─ success → final validation
   └─ failure → context expansion
                     ↓
                 retry
```

Não enviar inicialmente:

-   todos os arquivos alterados completos;
-   todos os testes;
-   histórico completo do PR;
-   documentação inteira;
-   grafo inteiro;
-   logs completos.

Enviar referências recuperáveis.

------------------------------------------------------------------------

# 18. Memória histórica / RAG

A memória de PRs deve armazenar unidades pequenas e úteis.

Exemplo:

``` yaml
problem_signature:
root_cause:
affected_symbols:
fix_pattern:
tests_that_caught_it:
regression_risk:
commit:
pr:
embedding:
```

Retrieval:

``` text
current error signature
+ changed symbols
+ dependency neighborhood
+ task type
```

Retornar inicialmente 3--5 memórias.

Não 20--50 chunks.

Depois permitir:

``` text
memory.expand(id)
```

------------------------------------------------------------------------

# 19. Prevenindo "alucinação de cobertura"

Para o problema já identificado no PRecog de **alucinação de cobertura
de testes**, não pedir ao modelo:

``` text
"esses testes cobrem a alteração?"
```

sem evidência.

Gerar evidência mecanicamente:

``` text
changed symbol
 ↓
static dependency
 ↓
test references
 ↓
runtime coverage, quando disponível
 ↓
test execution result
```

Contexto entregue:

``` yaml
changed:
  - calculate_total

tests:
  direct:
    - test_calculate_total_discount
  indirect:
    - test_invoice_pipeline

runtime_coverage:
  calculate_total:
    covered: true
    lines: 88-121
```

O LLM interpreta a evidência; não inventa a evidência.

Isso simultaneamente melhora assertividade e reduz tokens.

------------------------------------------------------------------------

# 20. Observabilidade necessária

Registrar por chamada:

``` text
request_id
task_id
task_type
complexity
model
reasoning
prompt_version
tool_profile
input_tokens
cached_tokens
output_tokens
reasoning_tokens
tool_schema_tokens
system_tokens
retrieved_tokens
tool_output_tokens
history_tokens
context_total
latency
cost
tool_calls
retries
result
validation_result
```

Derivar:

``` text
Harness Tax =
(system + developer + tool schemas) / total input

Retrieval Efficiency =
relevant retrieved tokens / retrieved tokens

Tool Output Efficiency =
tokens ultimately referenced / tool-output tokens

Token Efficiency =
successful tasks / 1M tokens

Cache Efficiency =
cached input / cache-eligible input
```

------------------------------------------------------------------------

# 21. Experimentos recomendados

Não implementar tudo de uma vez. Criar um benchmark interno congelado.

## Dataset inicial

Sugestão:

-   25 bugs simples;
-   25 bugs cross-module;
-   20 reviews;
-   10 security audits;
-   10 refactors;
-   10 problemas de testes/CI.

Total: **100 tarefas**.

Usar tarefas históricas reais do PRecog e de projetos Vieli, desde que a
resposta esperada possa ser validada.

Executar idealmente 3 vezes por configuração para reduzir efeito de
variância.

------------------------------------------------------------------------

# 22. Experimento A --- Baseline

Congelar:

``` text
model
temperature
reasoning
task set
tool permissions
timeout
max turns
```

Medir stack atual.

Esse resultado é `BASELINE_V1`.

Nenhuma otimização deve ser aprovada sem comparação com ele.

------------------------------------------------------------------------

# 23. Experimento B --- Toolset mínimo

Comparar:

``` text
A = tools atuais
B = read/write/edit/bash/search
C = core + tool discovery
```

Medir:

``` text
success
tokens
cost
latency
invalid tool calls
retries
```

Hipótese:

> `core + discovery` tende a manter capacidade com menor custo que
> carregar o catálogo inteiro.

------------------------------------------------------------------------

# 24. Experimento C --- Tool output compression

Comparar:

``` text
A = stdout completo
B = truncate
C = summarize
D = index + summary + on-demand read
```

Minha expectativa para PRecog é que **D** seja o melhor desenho, pois
mantém recuperabilidade da evidência.

Critério:

``` text
>= baseline success - margem estatística aceitável
e
>= 30% redução em tool-output tokens
```

------------------------------------------------------------------------

# 25. Experimento D --- Retrieval progressivo

Comparar:

``` text
A = arquivos completos
B = top-k chunks
C = symbol-first
D = graph → symbol → exact range
```

Medir:

``` text
retrieved tokens
relevant-token ratio
success
retries
```

Para PRecog, **D** é a hipótese principal.

------------------------------------------------------------------------

# 26. Experimento E --- Compaction

Comparar:

``` text
A = histórico completo
B = resumo textual
C = structured state
D = structured state + archive searchable
```

Avaliar tarefas longas, principalmente com \>10 chamadas.

Hipótese:

> `structured state + archive searchable` reduz crescimento do contexto
> sem apagar evidências.

------------------------------------------------------------------------

# 27. Experimento F --- Cache

Comparar:

``` text
A = prompt atual variável
B = static prefix + dynamic suffix
C = B + tools estáveis/deferred
D = C + cache breakpoints
```

Medir:

``` text
cache hit
uncached tokens
cached tokens
cost
latency
```

------------------------------------------------------------------------

# 28. Experimento G --- Routing adaptativo

Comparar:

``` text
A = sempre modelo forte
B = router atual
C = router + context budget
D = router + context + tools + reasoning policy
```

O objetivo não é minimizar preço por chamada.

É minimizar:

``` text
cost / validated success
```

------------------------------------------------------------------------

# 29. Experimento H --- Context ablation

Esse é um dos testes mais importantes.

Para cada componente:

``` text
AGENTS.md
repo map
historical memory
dependency graph
tool descriptions
previous messages
full diff
test output
```

executar:

``` text
baseline
baseline - componente
```

Se remover um componente:

-   reduz tokens;
-   não altera sucesso;
-   não aumenta retries;

ele não deveria estar no contexto padrão.

Essa técnica cria um **Context Value Map** baseado em evidência.

------------------------------------------------------------------------

# 30. Matriz de aprovação

Uma otimização só entra em produção quando atender:

  Métrica              Regra
  -------------------- ------------------------------------------
  Success Rate         não degradar além da margem definida
  Cost / Success       melhorar ≥ 15%
  Input tokens         preferencialmente reduzir ≥ 20%
  p95 latency          não piorar \> 10%, salvo ganho relevante
  Retries              não aumentar significativamente
  Invalid tool calls   não aumentar
  Regression escapes   não aumentar

Para mudanças grandes de arquitetura, exigir ganho maior, por exemplo
25--30% em custo por sucesso.

------------------------------------------------------------------------

# 31. Ordem recomendada de implementação

## Fase 1 --- Instrumentação

Implementar primeiro:

``` text
token breakdown
tool schema size
tool output size
retrieval size
cache tokens
cost
success
latency
```

Sem isso, qualquer otimização vira opinião.

## Fase 2 --- Quick wins

1.  reduzir schemas;
2.  remover tools redundantes;
3.  limitar outputs;
4.  evitar arquivos completos;
5.  separar prefixo estável;
6.  preservar cache.

## Fase 3 --- Context Engine

Implementar:

``` text
ContextPackBuilder
ContextBudget
ArtifactStore
ArtifactSearch
SymbolRetriever
GraphRetriever
StructuredWorkingMemory
```

## Fase 4 --- Dynamic Tool Loading

Criar:

``` text
ToolRegistry
ToolNamespace
ToolSearch
ToolPolicy
```

## Fase 5 --- Adaptive Router

Expandir LLMRouter para escolher:

``` text
model
reasoning
context budget
tool profile
retrieval depth
compression policy
fallback
```

## Fase 6 --- Continuous optimization

Executar benchmark automaticamente quando houver mudança em:

``` text
prompt
tools
router
retrieval
memory
model
reasoning
compaction
```

------------------------------------------------------------------------

# 32. Arquitetura proposta

``` text
                    ┌─────────────────┐
                    │      TASK       │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Task Classifier │
                    └────────┬────────┘
                             │
                 ┌───────────▼───────────┐
                 │ Resource Policy Engine │
                 │ model / budget / tools │
                 └───────────┬───────────┘
                             │
                  ┌──────────▼──────────┐
                  │ ContextPackBuilder  │
                  └──────────┬──────────┘
                             │
          ┌──────────────────┼──────────────────┐
          │                  │                  │
     Symbol Search       Graph Search       Memory/RAG
          │                  │                  │
          └──────────────────┼──────────────────┘
                             │
                     ┌───────▼───────┐
                     │     Agent     │
                     └───────┬───────┘
                             │
                     ┌───────▼───────┐
                     │ Tool Gateway  │
                     │ bounded output│
                     └───────┬───────┘
                             │
                       Artifact Store
                             │
                     search/read on demand
                             │
                     ┌───────▼───────┐
                     │  Validation   │
                     └───────┬───────┘
                             │
                  success ───┴─── failure
                                  │
                           expand context
                                  │
                                retry
```

------------------------------------------------------------------------

# 33. Interfaces sugeridas

## ContextPack

``` python
class ContextPack:
    task
    constraints
    repo_map
    changed_symbols
    evidence
    related_tests
    contracts
    memories
    token_budget
```

## ContextBudget

``` python
class ContextBudget:
    core_tokens
    task_tokens
    retrieval_tokens
    tool_output_tokens
    history_tokens
    reserve_tokens
```

## ToolResult

``` python
class ToolResult:
    summary
    important_items
    artifact_id
    truncated
    token_count
```

## ResourcePolicy

``` python
class ResourcePolicy:
    model
    fallback_model
    reasoning_effort
    context_budget
    tool_profile
    retrieval_depth
    max_tool_output_tokens
    compaction_policy
```

------------------------------------------------------------------------

# 34. Targets iniciais para PRecog

Não tratar estes valores como metas definitivas. São hipóteses para o
primeiro ciclo.

``` text
First-call harness context:
    alvo < 5k tokens

Core permanent prompt:
    alvo < 2.5k tokens

Tool schemas iniciais:
    alvo < 4k tokens

Tool output:
    default < 1.5k tokens/chamada

Retrieved context:
    T1 < 8k
    T2 < 15k
    T3 < 25k inicialmente

Cache hit:
    > 70% inicialmente

Context utilization:
    > 50% dos itens recuperados devem ser usados/citados
```

Objetivo global inicial:

``` text
-30% tokens / successful task
sem redução estatisticamente relevante de sucesso
```

Depois:

``` text
-50%
```

------------------------------------------------------------------------

# 35. O que eu NÃO faria

Não começaria por:

-   resumir tudo automaticamente com outro LLM;
-   cortar contexto arbitrariamente;
-   usar sempre modelo pequeno;
-   aumentar `top_k` do RAG;
-   enviar o repositório inteiro;
-   criar dezenas de micro-tools;
-   colocar toda a documentação no system prompt;
-   usar 100k+ tokens só porque o modelo suporta;
-   medir apenas custo por chamada;
-   otimizar benchmark sem validar PRs reais.

Também não assumiria que a configuração mais barata do HarnessTax será
automaticamente a melhor no PRecog. O próprio estudo é limitado a dois
benchmarks e amostras relativamente pequenas. O valor principal do
trabalho é demonstrar que **harness e contexto precisam ser medidos como
variáveis independentes**.

------------------------------------------------------------------------

# 36. Prioridade prática para o projeto

Se fosse implementar no PRecog agora, eu faria nesta ordem:

1.  **Token telemetry por componente.**
2.  **Benchmark interno de 100 tarefas.**
3.  **Tool output bounded + ArtifactStore.**
4.  **Context Pack baseado no grafo existente.**
5.  **Symbol-first retrieval.**
6.  **Tool Registry + lazy/deferred tools.**
7.  **Structured Working Memory.**
8.  **Static prefix + cache optimization.**
9.  **Context Budget adaptativo no LLMRouter.**
10. **Routing conjunto de model + reasoning + context + tools.**
11. **Context ablation automático.**
12. **Pareto dashboard custo × sucesso × latência.**

Os itens 3--6 provavelmente oferecem a melhor relação entre complexidade
de implementação e redução de tokens no PRecog, porque atacam justamente
os maiores desperdícios de agentes de código: **schemas permanentes,
outputs de shell/testes e recuperação excessiva de arquivos**.

------------------------------------------------------------------------

# 37. Hipótese arquitetural final

O objetivo não deve ser construir o agente que "sabe tudo" em cada
chamada.

Deve ser construir o agente que consegue **encontrar rapidamente o que
precisa**.

Em vez de:

``` text
Large Context Agent
```

usar:

``` text
Small Active Context
+ Large Searchable Memory
+ Precise Tools
+ Evidence-driven Expansion
```

Essa abordagem preserva a informação disponível no sistema sem obrigar o
modelo a processá-la em cada turno.

Para o PRecog, isso combina particularmente bem com o que já existe:

``` text
indexador
+ grafo
+ manifesto
+ Test Impact Analysis
+ histórico de PR
+ RAG
+ LLMRouter
```

A evolução natural é transformar esses componentes em um **Context
Operating System** para o agente.

------------------------------------------------------------------------

# 38. Plano de teste de 30 dias

## Semana 1 --- Medir

-   instrumentar tokens por componente;
-   congelar benchmark;
-   capturar baseline;
-   medir tamanho real dos schemas;
-   medir maiores tool outputs;
-   medir cache hit;
-   medir custo por sucesso.

## Semana 2 --- Reduzir desperdício

-   bounded tool outputs;
-   artifact storage;
-   schema cleanup;
-   remover tools redundantes;
-   implementar profiles de tools.

## Semana 3 --- Context Engine

-   ContextPackBuilder;
-   symbol-first retrieval;
-   graph expansion;
-   structured working memory;
-   context budget.

## Semana 4 --- A/B e rollout

Comparar:

``` text
baseline
minimal tools
lazy tools
bounded outputs
graph context
full optimized
```

Gerar Pareto:

``` text
X = cost / successful task
Y = success rate
bubble = latency
```

Promover a configuração que reduzir custo sem degradação relevante de
qualidade.

------------------------------------------------------------------------

# 39. Critério de sucesso do projeto

Eu definiria o primeiro milestone como:

> **Reduzir em pelo menos 30% os tokens médios por tarefa validamente
> resolvida no PRecog, mantendo a taxa de sucesso dentro de ±2 pontos
> percentuais do baseline e sem aumento relevante de regressões.**

Segundo milestone:

> **Atingir 40--50% de redução de custo por tarefa resolvida usando
> seleção dinâmica de contexto, tools e modelo.**

O ganho deve ser medido sobre workloads reais do PRecog, e não inferido
diretamente de benchmarks externos.

------------------------------------------------------------------------

# 40. Referências principais

1.  HarnessTax --- *How Much Does the Harness Matter for Coding
    Agents?*\
    https://harnesstax.github.io/

2.  Arena --- HarnessTax research announcement\
    https://arena.ai/blog

3.  Anthropic --- *Effective context engineering for AI agents*\
    https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents

4.  Anthropic --- *Writing effective tools for AI agents*\
    https://www.anthropic.com/engineering/writing-tools-for-agents

5.  OpenAI --- Prompt Caching\
    https://developers.openai.com/api/docs/guides/prompt-caching

6.  OpenAI --- Tool Search / deferred loading\
    https://developers.openai.com/api/docs/guides/tools-tool-search

7.  Liu et al. --- *Lost in the Middle: How Language Models Use Long
    Contexts*\
    https://arxiv.org/abs/2307.03172

8.  Jiang et al. --- *LongLLMLingua: Accelerating and Enhancing LLMs in
    Long Context Scenarios via Prompt Compression*\
    https://arxiv.org/abs/2310.06839

9.  context-compress --- indexed compression of agent tool outputs\
    https://github.com/Open330/context-compress

------------------------------------------------------------------------

## Conclusão

A principal oportunidade no LLMRouter/PRecog não parece ser simplesmente
trocar de modelo. É **controlar melhor o que chega ao modelo**.

A combinação com maior potencial é:

``` text
minimal permanent context
+ stable cached prefix
+ lazy tool loading
+ graph/symbol-first retrieval
+ bounded tool outputs
+ searchable artifacts
+ structured memory
+ adaptive context budget
+ model/reasoning routing
+ validation-driven retries
```

A pergunta que o sistema deve responder antes de cada chamada deixa de
ser:

> "Quanto contexto cabe?"

e passa a ser:

> **"Qual é o menor conjunto de contexto e ferramentas que maximiza a
> probabilidade de resolver esta tarefa corretamente?"**

Essa deve ser a função central do próximo estágio do LLMRouter.
