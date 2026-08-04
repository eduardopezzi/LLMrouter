# Pesquisa: Open LLM Leaderboard como fonte de benchmarks

Data da consulta: 2026-08-04

## Resultado executivo

O Open LLM Leaderboard pode ser uma fonte complementar de métricas públicas
atualizadas para o LLMrouter. A consulta ao endpoint de dados formatados retornou
4.576 registros de avaliação. Porém, nenhum dos identificadores configurados no
servidor YODA possui uma correspondência exata no catálogo consultado.

Por isso, **nenhuma nota foi importada automaticamente** nesta etapa. Associar
uma nota de uma variante de 7B, 14B ou 32B a `ollama/qwen2.5-coder:3b`, por
exemplo, produziria um ranking incorreto. O mesmo vale para modelos cloud cujo
nome comercial não identifica inequivocamente o checkpoint avaliado.

Fontes consultadas:

- [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)
- [Endpoint formatado do leaderboard](https://open-llm-leaderboard-open-llm-leaderboard.hf.space/api/leaderboard/formatted)
- [Código do endpoint no Space](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard/blob/main/backend/app/api/endpoints/leaderboard.py)

## Campos úteis para o LLMrouter

| Campo do leaderboard | Uso possível no router | Situação na base de conhecimento |
| --- | --- | --- |
| `model.name` / `model.id` | Chave de correspondência explícita com o modelo local | Necessária antes da importação |
| `model.average_score` | Sinal geral, apenas complementar | Não deve substituir métricas por capacidade |
| `evaluations.ifeval.normalized_score` | Seguimento de instruções | Já existe como `IFEval` |
| `evaluations.bbh.normalized_score` | Raciocínio amplo | Exige novo benchmark `BBH` |
| `evaluations.math.normalized_score` | Matemática | Corresponde a `MATH` (Level 5 no leaderboard) |
| `evaluations.gpqa.normalized_score` | Ciência/raciocínio difícil | Não equivale automaticamente a `GPQA Diamond` |
| `evaluations.musr.normalized_score` | Raciocínio multi-etapa | Já existe como `MuSR` |
| `evaluations.mmlu_pro.normalized_score` | Conhecimento e raciocínio geral | Já existe como `MMLU-Pro` |
| Metadados (parâmetros, licença, arquitetura, data) | Auditoria, filtros e rastreabilidade | Úteis como metadados da fonte |

As métricas normalizadas expostas pelo endpoint estão na escala percentual de
0 a 100. O catálogo do LLMrouter já converte percentuais para a escala interna
de 0 a 1 ao calcular rankings; a importação deve preservar o valor bruto e sua
origem.

## Verificação contra o inventário do YODA

Foram buscados os nomes e famílias dos 17 modelos configurados no inventário
auditado do YODA. Não houve correspondência exata para Kimi K3/K2.7, MiniMax M3,
GLM 5.x, DeepSeek V3/V4, Qwen3 Coder, Qwen3.6, Gemma 4 ou North Mini Code.

Também não há uma entrada para o modelo exato `ollama/qwen2.5-coder:3b`. O
leaderboard contém variantes Qwen2.5-Coder de outros tamanhos, como
`Qwen/Qwen2.5-Coder-14B-Instruct` e `Qwen/Qwen2.5-Coder-32B-Instruct`; elas não
podem ser usadas como substitutas do modelo 3B.

Exemplos de registros encontrados que devem permanecer apenas como referência:

| Modelo avaliado | Média | IFEval | BBH | MATH | GPQA | MuSR | MMLU-Pro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen2.5-Coder-14B-Instruct | 32,12 | 69,08 | 44,22 | 32,48 | 7,27 | 7,03 | 32,66 |
| Qwen2.5-Coder-32B-Instruct | 39,89 | 72,65 | 52,27 | 49,55 | 13,20 | 13,72 | 37,92 |

Esses números demonstram a cobertura da fonte, não uma classificação dos
modelos instalados no YODA.

## Integração recomendada

1. Manter o Open LLM Leaderboard como fonte complementar, sem substituir as
   fontes oficiais de cada provedor ou modelo cloud.
2. Criar uma tabela de mapeamento revisada por humano, com a relação explícita
   `id_do_catalogo -> model.id_do_hugging_face`.
3. Aceitar a importação somente quando o checkpoint, variante (Base/Instruct),
   tamanho, precisão e versão forem compatíveis. Não usar busca por substring,
   família ou "modelo mais próximo".
4. Importar inicialmente apenas `IFEval`, `MATH`, `MuSR` e `MMLU-Pro`, que têm
   correspondentes diretos na base de conhecimento. Tratar `GPQA` e `GPQA
   Diamond` como benchmarks diferentes.
5. Caso `BBH` seja adicionado, incluir antes sua descrição e exemplos de tarefa
   na base de conhecimento. Assim, o classificador semântico poderá associar
   prompts a essa capacidade de forma explicável.
6. Salvar, para cada nota, a URL/identificador da fonte, data de coleta,
   identificação exata do modelo e versão do benchmark. Uma atualização
   agendada deve gerar uma proposta para revisão humana, não alterar rankings
   publicados por conta própria.

## Decisão

Vale implementar um conector de leitura para o endpoint quando existirem
mapeamentos aprovados. No estado atual, a ação correta é não importar dados:
há cobertura útil de benchmarks, mas ainda não há identidade verificável entre
os registros do leaderboard e os modelos que o YODA encaminha.
