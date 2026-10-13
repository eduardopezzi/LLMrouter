# Model Token Limits

Validated on 2026-10-09.

The catalog keeps two token fields:

- `max_tokens`: maximum generated/output tokens when the provider documents it;
  otherwise a conservative operational cap.
- `context_window`: maximum total context/input window used for routing and
  contract checks.

## Sources

- Ollama context length docs: https://docs.ollama.com/context-length
- Ollama model pages/tags:
  - https://ollama.com/library/kimi-k2.7-code
  - https://ollama.com/library/deepseek-v4-pro/tags
  - https://ollama.com/library/deepseek-v4-flash
  - https://ollama.com/library/qwen3-coder/tags
  - https://ollama.com/library/glm-5.2/tags
  - https://ollama.com/library/north-mini-code-1.0/tags
  - https://ollama.com/library/qwen3.6/tags
  - https://ollama.com/library/gemma4/tags
  - https://ollama.com/library/deepseek-v3.2
  - https://ollama.com/library/deepseek-v3.1
- DeepSeek API docs:
  - https://api-docs.deepseek.com/api/list-models/
  - https://api-docs.deepseek.com/quick_start/pricing
- Current Ollama model cards:
  - https://ollama.com/library/deepseek-v4.1-flash
  - https://ollama.com/library/mistral-large-4
  - https://ollama.com/library/qwen3.8
  - https://ollama.com/library/gemma4
- Z.AI docs:
  - https://docs.z.ai/guides/llm/glm-5.3
  - https://docs.z.ai/guides/vlm/glm-5.3-flash
  - https://docs.z.ai/api-reference/llm/chat-completion
- Qwen official notes:
  - https://qwenlm.github.io/blog/qwen3-coder/
  - https://ollama.com/library/qwen2.5

## Catalog Decisions

| Model | max_tokens | context_window | Notes |
| --- | ---: | ---: | --- |
| `ollama/kimi-k2.7-code:cloud` | 128000 | 262144 | Ollama documents 256K context; output cap not separately published. |
| `ollama/deepseek-v4-pro:cloud` | 393216 | 1000000 | DeepSeek V4 API max output is 384K; Ollama context is 1M. |
| `ollama/deepseek-v4-flash:0731-cloud` | 393216 | 1000000 | Ollama still lists the 0731 tag; 1M context and 384K provider output cap. |
| `ollama/deepseek-v4.1-flash:cloud` | 393216 | 1000000 | Current multimodal Flash model; provider API documents 384K output and 1M context. |
| `ollama/mistral-large-4:cloud` | 131072 | 1000000 | General-purpose multimodal model; provider output cap not separately published. |
| `ollama/qwen3-coder:480b-cloud` | 65536 | 262144 | Ollama/Qwen document 256K native context; Qwen recommends 65,536 output. |
| `ollama/glm-5.2:cloud` | 131072 | 999424 | Z.AI output is 128K; Ollama tag reports 976K context. |
| `ollama/north-mini-code-1.0` | 65536 | 262144 | Official model card documents 64K output and 256K context. |
| `ollama/qwen3.8:27b` | 128000 | 262144 | Ollama documents 256K context; 128K is a conservative catalog output cap. |
| `ollama/gemma4:31b` | 262144 | 262144 | Ollama tag reports 256K context; output cap not separately published. |
| `ollama/qwen2.5-coder:3b` | 8192 | 32768 | Qwen 2.5 can generate up to 8K; 3B coder catalog is 32K context. |
| `ollama/deepseek-v3.1` | 128000 | 163840 | Local Ollama tag reports 160K context; disabled pending local deployment. |
| `deepseek/deepseek-flash` | 393216 | 1048576 | Current API ID for V4.1 Flash; legacy V4 Flash IDs route to this model. |
| `deepseek/deepseek-v4-pro` | 393216 | 1048576 | Current API model ID; output limit and context from the official model list. |

