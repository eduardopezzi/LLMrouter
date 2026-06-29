# Integração PRecog + LLMrouter

Este documento orienta a implementação, no PRecog, do ciclo completo de uso do
LLMrouter como gateway de modelos. A responsabilidade principal do PRecog deve
continuar sendo contexto, RAG, memória e feedback. O LLMrouter deve continuar
como gateway OpenAI-compatible, responsável por roteamento, fallback, custo,
logs e estatísticas de uso de modelos.

## Objetivo

Implementar no PRecog:

- cliente OpenAI-compatible apontando para o LLMrouter;
- decisão de quando usar RAG;
- montagem de contexto recuperado;
- envio de metadados RAG para o LLMrouter;
- ingestão de observações/feedback vindas do ciclo de execução;
- política de memória para decidir o que entra no RAG.

## Arquitetura Desejada

```text
Cline / usuário
  -> PRecog
      -> decide se precisa RAG
      -> busca contexto no pgvector/memória
      -> monta prompt enriquecido
      -> chama LLMrouter
  -> LLMrouter
      -> escolhe modelo/provider
      -> executa fallback
      -> registra observação
      -> retorna resposta
  -> PRecog
      -> avalia resultado
      -> salva feedback/memória quando apropriado
```

## Regra de Responsabilidade

O PRecog deve ser o dono de:

- documentos do projeto;
- embeddings;
- pgvector;
- memória;
- histórico de decisões;
- contexto semântico;
- critérios para indexar ou descartar informações.

O LLMrouter deve ser o dono de:

- seleção de modelo;
- seleção de provider;
- fallback;
- custo;
- logs por request;
- estatísticas de uso;
- compatibilidade OpenAI `/v1/chat/completions`.

Evitar escrever diretamente no banco do PRecog a partir do LLMrouter. Quando
for necessário integrar feedback, usar API interna do PRecog.

## Cliente LLMrouter no PRecog

Configurar o PRecog para chamar:

```text
POST http://llmrouter:12345/v1/chat/completions
```

Ou, em desenvolvimento local:

```text
POST http://localhost:12345/v1/chat/completions
```

Headers:

```http
Content-Type: application/json
Authorization: Bearer <LLMROUTER_SERVER__API_KEY>
```

Payload base:

```json
{
  "model": "auto",
  "task_role": "review",
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "Você é um assistente técnico. Use o contexto fornecido quando ele existir."
    },
    {
      "role": "user",
      "content": "Pergunta do usuário..."
    }
  ],
  "llmrouter": {
    "source": "precog",
    "project": "nome-do-projeto",
    "task_role": "review",
    "rag": {
      "used": false,
      "collection": null,
      "top_k": 0,
      "context_tokens": 0
    }
  }
}
```

## Decisor de RAG

Antes de chamar o LLMrouter, o PRecog deve decidir se precisa recuperar
contexto.

Usar RAG quando:

- a pergunta depende de código/projeto;
- o usuário menciona arquivo, classe, função, endpoint, bug, teste ou diff;
- `task_role` é `review`, `fix`, `refactoring`, `test_generation`,
  `security_audit`, `architecture` ou `migration`;
- o usuário pede algo como "neste projeto", "nesse repo", "com base no código";
- há necessidade de histórico, decisão técnica anterior ou documentação interna.

Não usar RAG quando:

- a pergunta é genérica;
- o pedido é simples e autocontido;
- o contexto já veio completo no prompt;
- é conversa casual;
- o custo de recuperar contexto não compensa.

Pseudo-lógica:

```python
def should_use_rag(task_role: str | None, prompt: str, has_full_context: bool) -> bool:
    if has_full_context:
        return False
    if task_role in {
        "review",
        "fix",
        "refactoring",
        "test_generation",
        "security_audit",
        "architecture",
        "migration",
    }:
        return True
    keywords = ["arquivo", "classe", "função", "endpoint", "bug", "diff", "teste", "repo"]
    return any(keyword in prompt.lower() for keyword in keywords)
```

## Retriever e Context Builder

Quando RAG for usado, o PRecog deve:

1. gerar embedding da pergunta;
2. buscar no pgvector/memória;
3. deduplicar trechos por arquivo/origem;
4. limitar por `top_k`;
5. limitar por `max_context_tokens`;
6. montar contexto com metadados de origem.

Formato recomendado do contexto:

```text
Contexto recuperado:

[fonte: src/foo.py | score: 0.83]
...

[fonte: docs/api.md | score: 0.77]
...

Pergunta:
...
```

Payload com RAG:

```json
{
  "model": "auto",
  "task_role": "fix",
  "stream": true,
  "messages": [
    {
      "role": "system",
      "content": "Use o contexto do projeto. Se o contexto não for suficiente, diga isso claramente."
    },
    {
      "role": "user",
      "content": "Contexto recuperado:\n...\n\nPergunta:\nCorrija o bug..."
    }
  ],
  "llmrouter": {
    "source": "precog",
    "project": "precog",
    "task_role": "fix",
    "rag": {
      "used": true,
      "collection": "project_docs",
      "top_k": 6,
      "context_tokens": 4200
    }
  }
}
```

## Metadados Obrigatórios

Sempre que o PRecog chamar o LLMrouter, enviar `llmrouter` com:

```json
{
  "source": "precog",
  "project": "nome-do-projeto",
  "task_role": "fix",
  "rag": {
    "used": true,
    "collection": "project_docs",
    "top_k": 6,
    "context_tokens": 4200
  }
}
```

Quando não houver RAG:

```json
{
  "source": "precog",
  "project": "nome-do-projeto",
  "task_role": "documentation",
  "rag": {
    "used": false,
    "collection": null,
    "top_k": 0,
    "context_tokens": 0
  }
}
```

## Feedback e Observações

O PRecog deve ter uma API interna para receber observações de uso ou registrar
resultados posteriores.

Endpoint sugerido:

```text
POST /internal/llmrouter/observations
```

Schema sugerido:

```json
{
  "request_id": "uuid",
  "project": "precog",
  "task_role": "review",
  "prompt_hash": "sha256...",
  "selected_model": "nvidia_nim/...",
  "provider": "nvidia",
  "provider_model": "...",
  "latency_ms": 1234,
  "prompt_tokens": 1000,
  "completion_tokens": 400,
  "cost_usd": 0.01,
  "rag": {
    "used": true,
    "collection": "project_docs",
    "top_k": 6,
    "context_tokens": 4200
  },
  "outcome": {
    "accepted": null,
    "tests_passed": null,
    "rating": null,
    "error": null
  }
}
```

Para atualizar o resultado depois:

```text
PATCH /internal/llmrouter/observations/{request_id}
```

Exemplo:

```json
{
  "outcome": {
    "accepted": true,
    "tests_passed": true,
    "rating": 5
  }
}
```

## Política de Memória/RAG

Não indexar tudo automaticamente.

Indexar no RAG/memória quando:

- o usuário aceitou a resposta;
- os testes passaram;
- a resposta contém decisão técnica reutilizável;
- a resposta documenta comportamento real do projeto;
- a correção foi validada.

Não indexar quando:

- contém segredo, token, senha ou dado sensível;
- a resposta foi rejeitada;
- a resposta causou erro;
- é contexto temporário;
- é uma hipótese não validada;
- é conteúdo duplicado ou de baixa qualidade.

## Integração com Cline

Para uso via Cline, o PRecog deve preservar tool calls e streaming sempre que
possível. O LLMrouter já expõe endpoint OpenAI-compatible e deve ser chamado com
`stream: true`.

O PRecog deve mapear intents comuns do Cline para `task_role`:

| Intent | `task_role` |
| --- | --- |
| revisar diff/código | `review` |
| corrigir bug | `fix` |
| gerar testes | `test_generation` |
| refatorar | `refactoring` |
| explicar/gerar docs | `documentation` |
| avaliar segurança | `security_audit` |
| discutir arquitetura | `architecture` |
| migração de stack | `migration` |

## Checklist de Implementação

- [ ] Criar cliente HTTP/OpenAI-compatible para o LLMrouter.
- [ ] Adicionar configuração `LLMROUTER_BASE_URL`.
- [ ] Adicionar configuração `LLMROUTER_API_KEY`.
- [ ] Implementar `should_use_rag`.
- [ ] Implementar montagem de contexto com limite de tokens.
- [ ] Enviar metadados `llmrouter.source`, `project`, `task_role` e `rag`.
- [ ] Criar endpoint interno `POST /internal/llmrouter/observations`.
- [ ] Criar endpoint interno `PATCH /internal/llmrouter/observations/{request_id}`.
- [ ] Implementar política de indexação em memória/RAG.
- [ ] Adicionar filtros para segredos/dados sensíveis antes de indexar.
- [ ] Adicionar testes de integração com LLMrouter mockado.
- [ ] Adicionar métricas por projeto, task_role, modelo e provider.

## Critérios de Aceite

- PRecog consegue chamar o LLMrouter com `model: auto`.
- PRecog decide corretamente quando usar RAG.
- Chamadas com RAG enviam metadados completos ao LLMrouter.
- Chamadas sem RAG também enviam metadados indicando `used: false`.
- Observações de uso podem ser registradas no PRecog.
- Feedback posterior pode atualizar observações.
- Apenas informações validadas entram no RAG/memória.
- Não há escrita direta do LLMrouter no banco do PRecog.
