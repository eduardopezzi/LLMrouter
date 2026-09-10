# Modelos GLM via Z.ai

O LLMrouter usa a API OpenAI-compatible da Z.ai em
`https://api.z.ai/api/paas/v4`. Configure `ZAI_API_KEY` no ambiente. Os nomes
canônicos no catálogo são `zhipu/glm-5.3-flash` e `zhipu/glm-5.3`; o prefixo
`zhipu/` é removido antes de enviar o nome à Z.ai.

## Política de roteamento

| Perfil | Modelo | Aplicações principais |
|---|---|---|
| Simples/rápido | `zhipu/glm-5.3-flash` | Resumos, documentação, classificação, extração, análise de arquivos/imagens, UI/frontend, automações curtas e geração de testes simples. |
| Complexo/long-horizon | `zhipu/glm-5.3` | Arquitetura, auditoria de segurança, debugging, migrações, refatorações grandes, coding agentic, terminal e implementação/testes multi-etapa. |

O Flash está no tier 1 com prioridade 1. O GLM-5.3 está no tier 3 com
prioridade 1. Assim, a classificação de complexidade seleciona o Flash para
prompts simples e o GLM-5.3 para prompts complexos; ambos continuam disponíveis
como fallback. O provedor envia automaticamente `thinking.type=enabled` e
`reasoning_effort=max` para esses dois IDs, como exigido pela documentação atual.

## Benchmark publicado

Valores abaixo são os resultados publicados nos model cards da Z.ai/Ollama e
foram carregados em `data/model_benchmarks.yaml`. Eles não são uma medição local
do LLMrouter e podem variar conforme harness, temperatura, ferramentas e
ambiente de execução.

| Benchmark | GLM-5.3 | GLM-5.3-Flash |
|---|---:|---:|
| Terminal Bench 2.1 | 88.2 | 84.3 |
| DeepSWE v1.1 | 66.9 | 63.4 |
| NL2Repo | 58.0 | 56.3 |
| Toolathlon Verified | 73.0 | 78.4 |
| AutomationBench v1.0.6 | 48.2 | 48.8 |
| Agents' Last Exam (CLI) | 28.5 | 26.3 |
| HLE with Tools | 62.5 | 55.3 |
| GDPval-AA v2 | 1769 | 1773 |

### Ranking

1. `zhipu/glm-5.3` — melhor escolha geral para qualidade e tarefas complexas;
   lidera TerminalBench, DeepSWE, NL2Repo, HLE com tools e Agents' Last Exam.
2. `zhipu/glm-5.3-flash` — melhor escolha custo/velocidade para tarefas simples;
   lidera Toolathlon, AutomationBench e GDPval-AA v2, além de manter desempenho
   próximo ao GLM-5.3 nos demais benchmarks comparáveis.

O ranking operacional por aplicação prevalece sobre um ranking único: o Flash é
intencionalmente preferido para baixa complexidade, enquanto o GLM-5.3 recebe o
trabalho que exige raciocínio profundo. A pontuação composta do roteador também
usa os benchmarks agentic atuais, além dos sinais de tier, contexto, custo e
provedor.

## Fontes

- [Quick Start da API Z.ai](https://docs.z.ai/guides/overview/quick-start)
- [Documentação do GLM-5.3](https://docs.z.ai/guides/llm/glm-5.3)
- [Documentação do GLM-5.3-Flash](https://docs.z.ai/guides/vlm/glm-5.3-flash)
- [Model card do GLM-5.3](https://huggingface.co/zai-org/GLM-5.3/raw/main/README.md)
- [Model card do GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash/raw/main/README.md)
- [Tabela de benchmarks do registro Ollama para o Flash](https://ollama.com/library/glm-5.3-flash)
