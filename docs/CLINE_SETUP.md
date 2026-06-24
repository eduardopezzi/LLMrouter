# Configurando o Cline com LLMrouter

O LLMrouter é um gateway **OpenAI-compatible** com roteamento automático de modelos.
Esta guia mostra como configurá-lo como provider do [Cline](https://github.com/cline/cline).

## Pré-requisitos

1. **LLMrouter rodando** na porta `12345` (padrão):

```bash
# Opção A — via Makefile
make run

# Opção B — manual
PYTHONPATH=src python -m uvicorn llmrouter.main:app --host 0.0.0.0 --port 12345
```

2. **Ollama** rodando localmente em `http://localhost:11434` com os modelos do catálogo (`config/models.yaml`).

3. **API Key** configurada no LLMrouter (variável `LLMROUTER_SERVER__API_KEY` no `.env`).

## Configuração no Cline

### Passo a passo na interface do Cline (VS Code)

1. Abra o Cline no VS Code (`Cmd/Ctrl + Shift + P` → `Cline: Focus on View`)
2. Clique no ícone de **Configurações** (⚙️) no painel do Cline
3. Em **API Provider**, selecione: **OpenAI Compatible**
4. Preencha os campos:

| Campo              | Valor                                    |
| ------------------ | ---------------------------------------- |
| **Base URL**       | `http://localhost:12345/v1`              |
| **API Key**        | (sua `LLMROUTER_SERVER__API_KEY` do `.env`) |
| **Model**          | `auto` (deixe o roteador decidir)        |

5. Clique em **Let's go!**

> **Dica:** O campo `Model` pode ser `auto` para roteamento automático, ou um modelo específico do catálogo (ex: `ollama/qwen3-coder:480b-cloud`). Veja os modelos disponíveis em `GET /v1/models`.

### Por que "OpenAI Compatible"?

O Cline não tem um provider nativo para o LLMrouter, mas como o LLMrouter implementa a API OpenAI (`/v1/chat/completions`, `/v1/models`) **com streaming SSE** e **function calling** (tool calls), o tipo "OpenAI Compatible" funciona perfeitamente.

## Validação rápida

Teste se o gateway está respondendo:

```bash
# Listar modelos disponíveis
curl -H "Authorization: Bearer SUA_API_KEY" \
  http://localhost:12345/v1/models | jq .

# Testar chat completion (não-streaming)
curl -X POST http://localhost:12345/v1/chat/completions \
  -H "Authorization: Bearer SUA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Olá!"}],
    "stream": false
  }' | jq .

# Testar streaming SSE
curl -N -X POST http://localhost:12345/v1/chat/completions \
  -H "Authorization: Bearer SUA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Olá!"}],
    "stream": true
  }'
```

## Como funciona o roteamento

Quando o Cline envia `"model": "auto"`:

1. O **PromptScorer** analisa a complexidade do prompt
2. O **MultiModelRouter** seleciona o modelo ideal do catálogo (`config/models.yaml`)
3. O **ProviderProxy** executa a requisição com fallback automático
4. A resposta retorna no formato OpenAI padrão, com metadados extras em `llmrouter`:

```json
{
  "llmrouter": {
    "selected_model": "ollama/qwen3-coder:480b-cloud",
    "provider_model": "qwen3-coder:480b-cloud",
    "score": 0.72,
    "tier": 2,
    "reason": "Code detection: high complexity"
  }
}
```

## Troubleshooting

### Erro: "Connection refused"
- Verifique se o LLMrouter está rodando: `curl http://localhost:12345/health`
- Confirme a porta (padrão `12345`)

### Erro: "401 Unauthorized"
- A API Key do Cline deve ser igual à `LLMROUTER_SERVER__API_KEY` do `.env`

### Streaming não funciona / Cline trava
- O LLMrouter agora suporta streaming SSE nativamente (`stream: true`)
- Se havia um erro 501 antes, atualize o código e reinicie o servidor

### Modelos não aparecem
- Verifique se o Ollama está rodando: `ollama list`
- Confirme os modelos em `config/models.yaml`

### Tool calls (function calling) não funcionam
- O LLMrouter agora faz passthrough de `tools`, `tool_choice` e `response_format`
- Garanta que o modelo upstream suporta tool calls (ex: Qwen Coder, Kimi)