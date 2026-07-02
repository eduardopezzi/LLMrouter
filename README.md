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
nas primeiras 5 linhas:

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
quando ambos existirem. O parser só lê as primeiras linhas para evitar conflito
com código, Markdown, templates e diffs.

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
