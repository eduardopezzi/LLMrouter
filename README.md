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
   (`config/models.yaml`).
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

### Atalhos via Makefile

| Comando          | Descrição                                  |
| ---------------- | ------------------------------------------ |
| `make help`      | Lista todos os comandos disponíveis        |
| `make install`   | Instala o pacote em modo editável          |
| `make install-dev` | Instala com dependências de desenvolvimento |
| `make run`       | Inicia o servidor (porta 12345)            |
| `make run-reload`| Inicia com auto-reload                     |
| `make test`      | Executa os testes                          |
| `make lint`      | Executa o linter (ruff)                    |
| `make format`    | Formata o código                           |

A porta padrão é `12345` e pode ser alterada via variável de ambiente
`LLMROUTER_SERVER__PORT` ou pelo parâmetro `PORT` do Makefile (`make run PORT=8080`).
