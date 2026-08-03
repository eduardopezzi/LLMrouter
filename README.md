# LLMrouter

LLMrouter is an OpenAI-compatible gateway that routes chat requests across a configured model catalog and records observations for local self-evaluation with Ollama.

## Docker

Build:

```bash
docker build -t llmrouter .
```

Run with API authentication:

```bash
docker run --rm -p 12345:12345 \
  -e LLMROUTER_SERVER__API_KEY="your-secret-key" \
  -e LLMROUTER_PROVIDERS__OLLAMA__BASE_URL="http://host.docker.internal:11434" \
  -e LLMROUTER_EVALUATOR__OLLAMA__BASE_URL="http://host.docker.internal:11434" \
  llmrouter
```

Use:

```bash
curl -H "Authorization: Bearer your-secret-key" http://localhost:12345/v1/models
```

## Atualizar os modelos utilizados

O catálogo carregado pelo LLMrouter fica em `config/models.yaml` (ou no caminho
definido por `LLMROUTER_MODELS_FILE`). Para adicionar, remover ou atualizar um
modelo, edite a lista `models` desse arquivo. Exemplo:

```yaml
models:
  - name: "ollama/exemplo:latest"
    provider: "ollama"
    api_base: "http://localhost:11434"
    roles: ["review", "fix"]
    priority: 1
    rollout_percentage: 100
    description: "Modelo usado para revisão e correções."
    max_tokens: 32768
    context_window: 131072
    prompt_cost_per_1m_tokens: 0
    completion_cost_per_1m_tokens: 0
    benchmark_scores:
      MMLU-Pro: 84.2
      LiveCodeBench: 71.5
      Codeforces: 2816
```

Os campos `name` e `provider` são obrigatórios. `roles` define para quais tarefas
o modelo pode ser selecionado, `priority` desempata modelos equivalentes (menor
número tem preferência) e `rollout_percentage` controla quanto tráfego ele pode
receber (`0` desativa e `100` libera totalmente). Use em `provider` um provedor
suportado pelo projeto e configure sua credencial correspondente no `.env`.

`benchmark_scores` é opcional e recebe as medições reais publicadas para o
modelo. Percentuais podem ser informados em `0–100` ou `0–1`; o rating de
`Codeforces` pode permanecer na escala Elo original. Não preencha notas
estimadas: sem dados para um modelo ou benchmark, o router usa tier, custo,
prioridade e saúde como fallback.

### Atualização semanal dos benchmarks

As notas coletadas ficam separadas do catálogo de modelos em
`data/model_benchmarks.yaml`. Cada valor traz fonte, data de coleta e
metodologia; as URLs, tabelas e colunas aceitas ficam em
`data/benchmark_sources.yaml`. Cadastre somente fontes oficiais/model cards e
nunca faça scraping genérico de páginas de terceiros.

```bash
make benchmarks-refresh # baixa, valida e atualiza o catálogo local
make benchmarks-check   # verifica se há mudança sem gravar arquivos
```

Com `LLMROUTER_BENCHMARKS__REFRESH_ENABLED=true` (padrão), o próprio processo
do LLMrouter executa a primeira verificação em background ao iniciar e repete a
cada 168 horas. Uma fonte que não valide a tabela declarada falha sem alterar o
catálogo anterior. Quando há mudança, o router recarrega as notas em memória,
sem reinício. Para persistir atualizações em Docker, monte `data/` como volume.

Para modelos Ollama locais, disponibilize o modelo antes de reiniciar:

```bash
ollama pull exemplo:latest
```

Valide o YAML e o catálogo sem iniciar o servidor:

```bash
PYTHONPATH=src python -c \
  'from llmrouter.core.registry import load_model_registry; print([m.name for m in load_model_registry("config/models.yaml").all()])'
```

Depois da edição, reinicie o LLMrouter, pois alterações manuais no YAML não são
recarregadas automaticamente:

```bash
# execução local: encerre o processo atual e rode novamente
llmrouter

# serviço systemd
sudo systemctl restart llmrouter
```

Se estiver usando Docker sem montar `config/models.yaml` como volume, reconstrua
a imagem antes de recriar o container. Por fim, confirme o catálogo carregado:

```bash
curl -H "Authorization: Bearer $LLMROUTER_SERVER__API_KEY" \
  http://localhost:12345/v1/models
```

Mantenha também `config/models.example.yaml` atualizado quando a alteração deve
virar o padrão do projeto. Se `config/models.yaml` não existir, o LLMrouter cria
esse arquivo copiando o catálogo de exemplo; ele não sobrescreve um catálogo
ativo já existente.

## Integração com PRecog

O LLMrouter pode ser usado pelo PRecog como backend OpenAI-compatible. Neste
modo, o PRecog mantém RAG, memória, pgvector, análise de testes e contexto; o
LLMrouter fica responsável por escolher o modelo/provedor e aplicar fallback.

Endpoint principal:

```text
POST http://localhost:12345/v1/chat/completions
```

Exemplo:

```bash
curl -X POST http://localhost:12345/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-key" \
  -d '{
    "task_role": "review",
    "messages": [
      {"role": "system", "content": "You are a senior code reviewer."},
      {"role": "user", "content": "Review this diff..."}
    ],
    "temperature": 0.1
  }'
```

`task_role` é opcional, mas recomendado para chamadas vindas do PRecog. Valores
úteis hoje incluem `review`, `test_generation`, `fix`, `summarization`,
`documentation`, `refactoring`, `security_audit`, `architecture` e `migration`.
Também é possível enviar o papel em `llmrouter.task_role` ou `extra.task_role`.

### Diretivas no prompt

Para clientes como o Cline, onde nem sempre é prático enviar metadados JSON, o
LLMrouter também aceita diretivas curtas no começo do prompt. Elas devem aparecer
nas primeiras 5 linhas de uma mensagem:

```text
{{project:PRecog}} {{task:deep_research}} {{model:zhipu/glm-5.1}}

Investigue como melhorar o pipeline de memória/RAG.
```

Aliases aceitos:

| Diretiva | Aliases | Uso |
| -------- | ------- | --- |
| `project` | `p` | Namespace de memória/RAG do projeto |
| `task` | `t`, `task_role`, `role` | Papel da tarefa para roteamento |
| `model` | `m`, `preferred_model` | Modelo preferido quando `model=auto` |

As diretivas `project`/`p` e `model`/`m` são tolerantes a erro de grafia. O
LLMrouter usa `difflib.get_close_matches(..., cutoff=0.0)` para aproximar o
valor digitado ao modelo mais próximo do catálogo e ao projeto/repositório mais
próximo encontrado localmente. Valores exatos continuam tendo prioridade.

Exemplos:

```text
{{p:LLMrouter}} {{t:review}} {{m:ollama/kimi-k2.7-code:cloud}}
Revise a mudança antes do deploy.
```

```text
{{project:PRecog}} {{task:refactoring}}
Refatore este módulo mantendo compatibilidade com a API atual.
```

Metadados explícitos no payload têm prioridade sobre as diretivas do prompt. Ou
seja, `llmrouter.project`, `task_role` e `model` enviados em JSON vencem o texto
quando ambos existirem. O parser só lê as primeiras linhas de cada mensagem para
evitar conflito com código, Markdown, templates e diffs.

### Publicação de observações no PRecog

O LLMrouter também pode enviar observações e feedback para os endpoints internos
do PRecog. Habilite no `.env`:

```env
LLMROUTER_PRECOG__ENABLED=true
LLMROUTER_PRECOG__BASE_URL=http://localhost:8888
LLMROUTER_PRECOG__API_KEY=mesmo-token-configurado-no-precog
LLMROUTER_PRECOG__PROJECT=llmrouter
```

Após cada chamada, o LLMrouter envia em modo best-effort:

```text
POST /internal/llmrouter/observations
```

A resposta OpenAI-compatible inclui `llmrouter.request_id`. Para chamadas
streaming, o mesmo valor é exposto no header `X-LLMrouter-Request-Id`.

Para registrar feedback posterior:

```bash
curl -X POST http://localhost:12345/v1/llmrouter/feedback \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "llmrouter-request-id",
    "outcome": {
      "accepted": true,
      "tests_passed": true,
      "validated": true,
      "rating": 5
    }
  }'
```

Esse endpoint encaminha para:

```text
PATCH /internal/llmrouter/observations/{request_id}
```

Para descobrir os papéis disponíveis no catálogo carregado:

```bash
curl http://localhost:12345/health
```

### Inspecionar roteamento semântico

Quando `LLMROUTER_SEMANTIC__ENABLED=true`, o runtime usa o `HybridScorer`
para combinar heurísticas e embeddings. Para calibrar a classificação sem chamar
provedores externos:

Além da role, o scorer compara o prompt com a base
`benchmark_knowledge_base.py`. As afinidades acima do limiar são normalizadas
para somar `1.0` e usadas como pesos sobre `benchmark_scores` dos modelos. Isso
permite que um prompt misto distribua peso entre, por exemplo, contexto longo,
engenharia de software e terminal. O ranking dinâmico só é aplicado quando há
notas compatíveis; caso contrário, o comportamento anterior é preservado.

Instale as dependências opcionais de embeddings antes de habilitar o recurso:

```bash
pip install -e '.[ml]'
```

```env
LLMROUTER_SEMANTIC__ENABLED=true
LLMROUTER_SEMANTIC__BENCHMARK_KNOWLEDGE_BASE_PATH=benchmark_knowledge_base.py
LLMROUTER_SEMANTIC__BENCHMARK_SIMILARITY_THRESHOLD=0.30
LLMROUTER_SEMANTIC__BENCHMARK_TOP_K=5
LLMROUTER_ROUTING__DYNAMIC_BENCHMARK_ROUTING=true
```

```bash
curl -X POST http://localhost:12345/v1/llmrouter/semantic/inspect \
  -H "Authorization: Bearer your-secret-key" \
  -H "Content-Type: application/json" \
  -d '{"prompt":"Review this architecture and identify security risks."}'
```

Via CLI:

```bash
llmrouter semantic-inspect "Review this architecture and identify security risks." --json
```

## Integração com o Cline

O [Cline](https://github.com/cline/cline) é um agent de coding autônomo para VS Code.
Como o LLMrouter implementa a API OpenAI (`/v1/chat/completions`, `/v1/models`)
com **streaming SSE** e **function calling** (tool calls), basta usar o provider
**OpenAI Compatible** do Cline.

### Pré-requisitos

1. **LLMrouter rodando** na porta `12345` (ver [Local Server](#local-server)).
2. **Ollama** rodando em `http://localhost:11434` com os modelos do catálogo
   local (`config/models.yaml`). Se ele não existir, o LLMrouter cria uma cópia
   a partir de `config/models.example.yaml`.
3. **API Key** configurada no LLMrouter via `LLMROUTER_SERVER__API_KEY` no `.env`.

### Configuração no Cline (VS Code)

1. Abra o Cline: `Cmd/Ctrl + Shift + P` → `Cline: Focus on View`
2. Clique no ícone de **Configurações** (⚙️)
3. Em **API Provider**, selecione: **OpenAI Compatible**
4. Preencha:

   | Campo        | Valor                                      |
   | ------------ | ------------------------------------------ |
   | **Base URL** | `http://localhost:12345/v1`                |
   | **API Key**  | Valor de `LLMROUTER_SERVER__API_KEY`       |
   | **Model**    | `auto` (roteamento) ou modelo do catálogo  |

5. Clique em **Let's go!**

> **Dica:** `auto` ativa o roteamento inteligente. Para respostas instantâneas em
> testes rápidos, use um modelo local: `ollama/qwen2.5-coder:3b`.

### Modelos recomendados para o Cline

| Uso                | Model                          | Tipo   |
| ------------------ | ------------------------------ | ------ |
| Roteamento auto    | `auto`                         | Auto   |
| Rápido / local     | `ollama/qwen2.5-coder:3b`      | Local  |
| Coding pesado      | `ollama/qwen3-coder:480b-cloud`| Cloud  |
| Code review / fix  | `ollama/kimi-k2.7-code:cloud`  | Cloud  |
| Arquitetura        | `ollama/deepseek-v4-pro:cloud` | Cloud  |

### Validação rápida

```bash
# Streaming (como o Cline usa)
curl -N -X POST http://localhost:12345/v1/chat/completions \
  -H "Authorization: Bearer $LLMROUTER_SERVER__API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"auto","messages":[{"role":"user","content":"Olá!"}],"stream":true}'
```

Guia completo com troubleshooting: [`docs/CLINE_SETUP.md`](docs/CLINE_SETUP.md).

## Cross-Repository Contracts

O item **Cross-Repository** publica um contrato JSON versionavel para que repos
consumidores validem compatibilidade antes de atualizar o LLMrouter. O snapshot
inclui endpoints, schema resumido de requests/responses, catalogo de modelos e
roles de roteamento disponiveis.

O repositorio central planejado e `Vieli-Tech/phoenix_versions`. Cada projeto
deve ter uma pasta propria e os JSONs vigentes ficam na raiz dessa pasta. A CLI
resolve o nome da pasta de forma case-insensitive; por exemplo, `llmrouter`,
`LLMRouter` e `LLMROUTER` apontam para a mesma pasta existente.

Exportar o contrato atual:

```bash
make contracts-export
```

Exportar direto para o repositorio central:

```bash
llmrouter export-contracts \
  --contracts-root ../phoenix_versions \
  --project llmrouter \
  --filename llmrouter.contract.json
```

Publicar direto no GitHub usando `GITHUB_TOKEN` do `.env`:

```bash
make contracts-publish
```

ou:

```bash
llmrouter publish-contracts \
  --repo https://github.com/Vieli-Tech/phoenix_versions.git \
  --project llmrouter \
  --filename llmrouter.contract.json
```

Comparar um snapshot anterior com o atual, falhando em breaking changes:

```bash
make contracts-check \
  PREVIOUS_CONTRACT=contracts/previous.llmrouter.contract.json \
  CONTRACT=contracts/llmrouter.contract.json
```

Ver diferencas sem falhar:

```bash
make contracts-diff \
  PREVIOUS_CONTRACT=contracts/previous.llmrouter.contract.json \
  CONTRACT=contracts/llmrouter.contract.json
```

Tambem e possivel chamar a CLI diretamente:

```bash
llmrouter export-contracts --output contracts/llmrouter.contract.json
llmrouter check-contracts old.json new.json
llmrouter diff-contracts old.json new.json
```

O `BreakingChangeDetector` marca como **breaking** remocao de endpoint, modelo ou
role, mudanca de metodo/schema de endpoint, troca de provider/modelo interno,
remocao de capability e reducao de janela de contexto. Adicoes de endpoints,
modelos, roles, capabilities ou aumento de contexto sao tratadas como
compativeis.

## Routing Preference

Por padrao, o LLMrouter usa a estrategia `cost`. Primeiro ele escolhe o tier e
as capabilities necessarias para a tarefa; dentro desses candidatos, prefere o
menor custo. Quando os custos numericos do catalogo empatam ou estao zerados, o
desempate segue a ordem comercial atual:

```text
Zhipu -> Ollama -> NVIDIA
```

Para trocar a estrategia:

```env
LLMROUTER_ROUTING__STRATEGY=quality
```

Tambem existe um painel CLI para configurar a priorizacao e ver estatisticas:

```bash
make panel
```

Ver somente o resumo atual:

```bash
make panel-stats
```

Alterar configuracoes sem menu interativo:

```bash
llmrouter panel --set-strategy cost
llmrouter panel --set-fallback-count 3
llmrouter panel --set-provider-cost-order nvidia,zai,ollama
```

O painel grava essas preferencias no `.env`:

```env
LLMROUTER_ROUTING__STRATEGY=cost
LLMROUTER_ROUTING__FALLBACK_COUNT=3
LLMROUTER_ROUTING__PROVIDER_COST_ORDER=["nvidia", "zai", "ollama"]
```

## Local Server

> **Importante:** Este projeto usa *src layout* (`src/llmrouter/`). Portanto, o
> comando deve incluir `PYTHONPATH=src` ou o pacote deve ser instalado antes.

### Opção 1 — Instalar o pacote (recomendado para desenvolvimento)

```bash
pip install -e .          # ou: make install-dev  (inclui deps de dev)
llmrouter                 # usa o entrypoint, porta 12345 por padrão
```

### Opção 2 — Rodar com PYTHONPATH (sem instalar)

```bash
PYTHONPATH=src python -m uvicorn llmrouter.main:app --host 0.0.0.0 --port 12345
```

### Workers

Para distribuir requests entre múltiplos núcleos, rode o LLMrouter com múltiplos
workers Uvicorn:

```bash
llmrouter --workers 4
```

ou:

```bash
make run WORKERS=4
```

Tambem e possivel configurar por ambiente:

```env
LLMROUTER_SERVER__WORKERS=4
```

Use `--reload` apenas em desenvolvimento; reload e múltiplos workers não rodam
juntos. Workers aumentam concorrência entre requests, mas uma única request
CPU-bound ainda pode ocupar um núcleo enquanto estiver sendo processada.

### Atalhos via Makefile

| Comando          | Descrição                                  |
| ---------------- | ------------------------------------------ |
| `make help`      | Lista todos os comandos disponíveis        |
| `make install`   | Instala o pacote em modo editável          |
| `make install-dev` | Instala com dependências de desenvolvimento |
| `make run`       | Inicia o servidor (porta 12345, use `WORKERS=4`) |
| `make run-reload`| Inicia com auto-reload                     |
| `make panel` | Abre painel CLI de roteamento e estatisticas |
| `make panel-stats` | Mostra estatisticas do painel CLI |
| `make contracts-export` | Exporta contrato cross-repository em JSON |
| `make contracts-check` | Valida compatibilidade entre snapshots |
| `make contracts-diff` | Mostra diferencas entre snapshots |
| `make contracts-publish` | Publica contrato vigente no repo GitHub central |
| `make test`      | Executa os testes                          |
| `make lint`      | Executa o linter (ruff)                    |
| `make format`    | Formata o código                           |

A porta padrão é `12345` e pode ser alterada via variável de ambiente
`LLMROUTER_SERVER__PORT` ou pelo parâmetro `PORT` do Makefile (`make run PORT=8080`).

### Logs

Se o LLMrouter estiver instalado como serviço systemd chamado `llmrouter`, acompanhe
os logs em tempo real com:

```bash
journalctl -u llmrouter -f
```

Para ver as últimas linhas e continuar acompanhando:

```bash
journalctl -u llmrouter -n 100 -f
```
