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
