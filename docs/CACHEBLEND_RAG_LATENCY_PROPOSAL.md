# Proposta técnica: CacheBlend para reduzir a latência de RAG

**Status:** estudo para desenvolvimento futuro  
**Data da análise:** 4 de agosto de 2026  
**Decisão recomendada:** preparar observabilidade e executar uma prova de conceito;
não ativar CacheBlend em produção no estado atual do projeto.

## 1. Resumo executivo

CacheBlend reduz o tempo que um LLM leva para processar os textos recuperados
por RAG antes de produzir o primeiro token. A técnica guarda o *KV cache* de
cada chunk, reutiliza esses caches mesmo quando o chunk não está no início do
prompt e recompõe seletivamente os tokens mais afetados pela atenção entre
chunks.

O ganho publicado é relevante, mas não deve ser transferido diretamente para o
LLMrouter. No experimento do paper, CacheBlend reduziu o *time to first token*
(TTFT) em 2,2 a 3,3 vezes e aumentou o throughput em 2,8 a 5 vezes. Esses
resultados foram obtidos com seis chunks de 512 tokens, três modelos específicos,
GPUs NVIDIA A40 e um conjunto limitado de tarefas e datasets. O próprio paper
reporta uma pequena perda de qualidade, embora muito inferior à reutilização
integral de caches sem recomputação.

No LLMrouter atual, o contexto da memória/RAG é limitado por padrão a 2.400
caracteres e é enviado principalmente a modelos Ollama locais ou serviços de
nuvem. CacheBlend, por sua vez, precisa ser instalado junto ao servidor de
inferência e foi implementado sobre vLLM/LMCache. Ele não pode ser implementado
somente dentro do gateway nem imposto a provedores de nuvem que não exponham
essa capacidade.

Portanto:

- **vale a pena investigar** se o projeto passar a usar RAG com milhares de
  tokens, chunks repetidos entre requisições e um modelo Transformer
  autohospedado em vLLM;
- **não vale a pena implantar agora** sem medir TTFT, tamanho de contexto e taxa
  real de reutilização dos chunks;
- o próximo passo de baixo risco é instrumentar o pipeline e montar um piloto
  isolado com vLLM + LMCache, mantendo o caminho atual como controle e fallback.

## 2. Problema que a técnica resolve

Uma requisição RAG possui, de forma simplificada, quatro etapas:

```text
pergunta -> embedding -> busca/reranking -> prefill do contexto -> geração
                                     milissegundos ^          ^ primeiro token
                                                   potencialmente segundos
```

A busca vetorial pode ser rápida e ainda assim a resposta demorar. Antes de
gerar, o Transformer precisa processar os tokens de entrada em todas as camadas.
Essa etapa é o **prefill** e produz as matrizes de chaves e valores usadas pela
atenção, chamadas de **KV cache**. O tempo até terminar o prefill é um componente
central do TTFT.

CacheBlend não acelera embeddings nem a busca no banco vetorial. Ele evita
repetir grande parte do prefill de chunks que já foram processados anteriormente.

Também não deve ser confundido com os dois caches já relacionados ao projeto:

- o cache de respostas do LLMrouter devolve uma resposta anterior inteira;
- os arquivos de embeddings semânticos evitam recalcular vetores de
  classificação;
- CacheBlend reutiliza estados intermediários do Transformer e ainda executa o
  modelo para responder à nova pergunta.

## 3. Por que prefix caching não é suficiente

O cache de prefixo tradicional só é reutilizável quando todos os tokens
anteriores também são iguais. Isso preserva exatamente o resultado porque o KV
cache de um token depende do seu prefixo.

Considere dois prompts:

```text
requisição A: instrução | chunk A | chunk B | chunk C | pergunta 1
requisição B: instrução | chunk A | chunk B | chunk D | pergunta 2
```

O prefixo comum pode ser reaproveitado, mas a divergência em `chunk C/D`
impede o reaproveitamento dos blocos seguintes. Em RAG, os chunks recuperados e
sua ordem mudam frequentemente; por isso, apenas o primeiro trecho comum tende a
se beneficiar.

Uma concatenação direta de caches pré-calculados também não é uma solução
correta. O KV cache de um chunk calculado isoladamente não contém a atenção com
os chunks que vieram antes dele. O paper mostra que ignorar essa atenção entre
chunks pode degradar respostas que dependem de comparação ou raciocínio
multi-hop.

## 4. Como CacheBlend funciona

O processo descrito pelo paper é:

1. O KV cache de cada chunk reutilizável é calculado e armazenado.
2. Quando o retriever escolhe os chunks, seus caches são carregados e combinados
   nas posições correspondentes do novo prompt.
3. Em cada camada do Transformer, CacheBlend identifica uma pequena fração de
   tokens com maior desvio causado pela ausência de atenção entre chunks.
4. Somente os KVs desses tokens são recalculados; os demais KVs são reutilizados.
5. O carregamento do cache da próxima camada é sobreposto à recomputação da
   camada atual.

O paper chama os tokens selecionados de *High-KV-Deviation tokens* (HKVD). A
seleção prática usa duas observações medidas pelos autores:

- poucos tokens concentram grande parte do desvio, em razão da esparsidade da
  atenção;
- tokens com alto desvio em uma camada tendem a continuar relevantes na camada
  seguinte.

CacheBlend usa filtragem gradual entre as camadas. O paper avaliou razões de
recomputação entre 5% e 18% em parte dos experimentos e indica 10% a 20% como a
faixa que, nos cenários estudados, reduz bastante o desvio de atenção. Esses
valores são resultados experimentais, não parâmetros que devam ser adotados sem
calibração no nosso corpus.

O ganho de sobreposição depende da relação entre duas velocidades: carregar os
KVs do armazenamento e recalcular os tokens selecionados. O controlador do
CacheBlend estima ambas para escolher a razão de recomputação e o nível de
armazenamento. Assim, um SSD mais lento não implica necessariamente o mesmo
resultado observado com RAM, e uma GPU/modelo diferente precisa ser perfilada.

## 5. Evidência disponível e limites dos ganhos

### 5.1 Resultados publicados

O paper avaliou Mistral-7B, Yi-34B e Llama-70B em até duas GPUs NVIDIA A40, com
128 GB de RAM e SSD NVMe medido em 4,8 GB/s. Os datasets foram 2WikiMQA,
Musique, SAMSum e MultiNews. Na comparação principal, cada consulta recebeu seis
chunks de 512 tokens.

Resultados reportados em relação à recomputação completa:

| Métrica | Resultado do paper |
| --- | --- |
| TTFT | redução de 2,2 a 3,3 vezes |
| Throughput | aumento de 2,8 a 5 vezes |
| Qualidade | queda de até 0,01 a 0,03 em F1/Rouge-L no resumo da avaliação |

Em outra análise do mesmo paper, a diferença de F1/Rouge-L em relação à
recomputação completa ficou dentro de 0,02. A forma correta de interpretar a
alegação de “sem comprometer a qualidade” do resumo é, portanto, **perda pequena
nos testes executados**, e não equivalência garantida para qualquer modelo ou
tarefa.

### 5.2 O que não pode ser prometido ao LLMrouter

Não há base nas fontes para prometer 2,2 a 3,3 vezes de redução no nosso
ambiente. O ganho efetivo depende, no mínimo, de:

- o prefill representar uma parcela relevante da latência total;
- os mesmos chunks aparecerem em requisições diferentes;
- o cache estar aquecido e não ter sido removido;
- tamanho e quantidade dos chunks;
- modelo, quantização, GPU, RAM/SSD e largura de banda;
- compatibilidade entre a versão do modelo, vLLM e LMCache;
- razão de recomputação necessária para preservar a qualidade local.

Cache frio, conteúdo muito dinâmico ou baixa repetição reduzem o benefício. Se
o contexto for curto, o custo de localizar e transferir KVs pode não compensar o
prefill evitado. A resposta para o nosso projeto deve vir de medição, não da
extrapolação dos números do paper.

## 6. Situação atual do LLMrouter

O fluxo relevante hoje é:

```text
Cliente OpenAI-compatible
        |
        v
FastAPI / LLMrouter
        |
        +-- recupera memória local SQLite ou PRecog
        +-- limita/renderiza contexto (padrão: 2.400 caracteres, top_k=4)
        +-- adiciona o contexto como mensagem system
        +-- escolhe modelo e provider
        v
Ollama ou provider de nuvem
```

Pontos já favoráveis:

- `MemoryEntry` preserva identificador, projeto, score e metadados;
- o contexto recuperado é separado da pergunta em uma mensagem de sistema;
- o provider OpenAI-compatible já interpreta contadores de tokens em cache
  quando o upstream os informa;
- o proxy já possui fallback de modelo/provider e métricas de latência total.

Lacunas antes de avaliar CacheBlend:

- não há métrica explícita de TTFT;
- a latência de retrieval, renderização e prefill não é separada;
- caracteres de contexto não equivalem a tokens e não medimos sua distribuição;
- não existe métrica de reutilização de IDs/hashes dos chunks;
- a memória local armazena interações completas, não chunks documentais
  canônicos preparados para cache;
- o endpoint PRecog precisa garantir IDs e versões estáveis de chunks;
- o LLMrouter não possui um provider vLLM explícito nem gerenciamento de
  CacheBlend;
- CacheBlend precisa do controle do servidor de inferência; os modelos Ollama
  cloud e outros provedores remotos não oferecem esse controle ao gateway.

Com `max_context_chars=2400`, o contexto atual tende a ser muito menor que os
3.072 tokens recuperados no cenário principal do paper. Não é possível afirmar
o tamanho em tokens sem instrumentação, mas essa diferença reforça que uma
implantação imediata seria prematura.

## 7. Arquitetura proposta para um piloto

CacheBlend deve ficar atrás do LLMrouter, junto ao modelo autohospedado:

```text
                         +---------------------------+
pergunta -> LLMrouter -> | PRecog / retriever        |
             |           | chunks estáveis + versões |
             |           +-------------+-------------+
             |                         |
             |   prompt com chunks e delimitadores  |
             v                         v
        provider vLLM ----------> vLLM + LMCache/CacheBlend
                                      |       |
                                      |       +-- RAM / NVMe para KV caches
                                      +---------- GPU para recomputação/decode
```

Responsabilidades sugeridas:

### PRecog ou camada RAG

- produzir chunks textuais estáveis;
- retornar `chunk_id`, versão/hash do conteúdo, origem e projeto;
- invalidar ou versionar chunks alterados;
- preservar as referências necessárias para avaliação e citações.

### LLMrouter

- continuar responsável por roteamento, autenticação e fallback;
- montar o contexto com limites claros e delimitadores compatíveis com a versão
  fixada do LMCache;
- encaminhar um `cache_salt` derivado do domínio de isolamento do projeto ou
  tenant, sem expor segredos no valor;
- escolher CacheBlend apenas para modelos/endpoints declarados como compatíveis;
- cair para prefill normal quando CacheBlend estiver indisponível;
- coletar TTFT, tokens/chunks reutilizados e métricas de qualidade.

### vLLM + LMCache

- calcular, armazenar, buscar e remover os KV caches;
- corrigir posição e recomputar seletivamente os tokens;
- controlar RAM/SSD e política de remoção;
- expor métricas operacionais e de cache.

O modo multiprocess do LMCache é atualmente recomendado pela própria
documentação por oferecer isolamento do processo, compartilhamento e
observabilidade. A configuração exata deve ser fixada durante o piloto conforme
uma combinação de versões testada; exemplos de configuração da documentação não
devem ser copiados para produção sem esse teste.

## 8. Plano de implementação futura

### Fase 0 — baseline e decisão baseada em dados

Adicionar, sem CacheBlend:

- TTFT P50/P95 para respostas em streaming;
- latência de retrieval e latência total;
- quantidade de chunks e tokens de contexto por requisição;
- frequência de reutilização de cada `chunk_id` por projeto;
- `cached_tokens` quando informado pelo provider;
- taxa de cache frio/quente e qualidade por tipo de tarefa.

Saída esperada: confirmar se o gargalo é prefill e se existe reutilização
suficiente. Sem essa confirmação, encerrar o estudo sem implantar infraestrutura.

### Fase 1 — tornar os chunks cacheáveis

- definir contrato de chunk entre PRecog e LLMrouter;
- tornar ID, versão e conteúdo determinísticos;
- definir invalidação quando conteúdo, modelo ou tokenização mudar;
- definir separador e template estáveis para o prompt;
- preservar o vínculo de cada chunk com sua fonte;
- definir isolamento por projeto/tenant.

### Fase 2 — piloto isolado

- selecionar um único modelo Transformer suportado e autohospedado;
- subir vLLM e LMCache em ambiente Linux/GPU compatível;
- fixar todas as versões e registrar a configuração;
- comparar três modos: prefill completo, cache de prefixo e CacheBlend;
- manter o tráfego de produção fora do piloto inicialmente.

### Fase 3 — avaliação A/B reproduzível

Reexecutar um corpus representativo de perguntas reais, com a mesma seleção e
ordem de chunks em todos os modos. Medir separadamente cache frio e aquecido.

Critérios que precisam ser acordados antes do teste:

- redução mínima aceitável de TTFT P50 e P95;
- tolerância de perda de qualidade por tarefa;
- throughput mínimo;
- orçamento de RAM, VRAM e NVMe;
- taxa de erros e comportamento de fallback;
- ausência de reutilização entre projetos não autorizados.

Não são definidos valores artificiais neste documento porque o projeto ainda
não possui o baseline necessário.

### Fase 4 — integração gradual

Se os critérios forem atendidos:

- adicionar capacidade de modelo `kv_cache_blending` ao catálogo;
- adicionar configuração de feature flag e rollout;
- rotear apenas requisições RAG elegíveis ao endpoint vLLM/LMCache;
- manter prefill normal como fallback;
- iniciar com tráfego pequeno e rollback automático por TTFT, erro e qualidade.

## 9. Riscos e controles

| Risco | Controle proposto |
| --- | --- |
| Perda de qualidade por atenção incompleta | Avaliar tarefas multi-hop e ajustar recomputação; fallback para prefill completo |
| Vazamento ou inferência entre tenants | `cache_salt` por domínio de confiança, quotas e isolamento de armazenamento |
| Cache incompatível após troca de modelo | Incluir modelo/versão/tokenização na identidade e invalidar o cache |
| Pressão de RAM/VRAM/SSD | Orçamento, métricas, LRU/quota e teste de carga |
| Cache frio ou baixo reuso | Habilitar somente quando o perfil real justificar |
| Falha do serviço de cache | CacheBlend opcional; vLLM executa o caminho normal |
| Lock-in de infraestrutura | Manter API OpenAI-compatible e integração atrás de um provider |
| Regressões entre versões | Fixar versões e executar testes de compatibilidade antes de atualizar |

Há um cuidado adicional de maturidade: em maio de 2026 foi reportado no
repositório do LMCache um problema aberto de reutilização não-prefixada no
conector vLLM V1 em determinadas versões. Isso não prova que todas as versões ou
o modo multiprocess estejam quebrados, mas impede assumir compatibilidade sem um
teste da combinação exata escolhida.

## 10. Alternativas e medidas complementares

As seguintes ações não substituem CacheBlend, mas podem reduzir latência com
menor complexidade:

- limitar e reranquear melhor os chunks antes do LLM;
- manter instruções estáveis no início do prompt para aproveitar prefix caching;
- usar streaming e medir TTFT em vez de apenas latência total;
- usar o cache exato existente para requisições idênticas;
- implementar o cache semântico já previsto no roadmap para perguntas
  equivalentes, com avaliação de segurança e qualidade;
- direcionar RAG simples a modelos menores quando a qualidade observada permitir.

Essas opções atuam em pontos diferentes e podem coexistir com CacheBlend.

## 11. Recomendação final

**Recomendação: PoC condicional, não implantação imediata.**

CacheBlend é tecnicamente adequado para o problema de prefill em RAG
multi-chunk e possui evidência experimental forte o bastante para justificar um
piloto. A integração no LMCache e o reconhecimento no EuroSys 2025 indicam que
não se trata apenas de uma ideia conceitual.

Entretanto, a arquitetura atual do LLMrouter ainda não demonstra o perfil em
que o benefício aparece: o contexto padrão é curto, a taxa de repetição de
chunks não é medida e a maior parte dos modelos está atrás de Ollama ou
provedores remotos. O custo operacional de adicionar vLLM, LMCache e
armazenamento de KVs não se justifica antes do baseline.

A decisão de produção deve ser tomada somente depois das Fases 0 a 3. Se o
prefill não dominar o TTFT ou os chunks tiverem pouco reuso, a recomendação deve
ser não implementar. Se contextos longos e repetitivos forem comuns e o piloto
preservar a qualidade, CacheBlend passa a ser uma otimização de alto potencial.

## 12. Fontes

- [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion — paper, versão 3](https://arxiv.org/abs/2405.16444)
- [CacheBlend — texto integral do paper no arXiv](https://arxiv.org/html/2405.16444v3)
- [University of Chicago — CacheBlend e o prêmio Best Paper do EuroSys 2025](https://cs.uchicago.edu/news/cacheblend-university-of-chicagos-game-changer-in-ai-speed-and-precision/)
- [Repositório oficial LMCache](https://github.com/LMCache/LMCache)
- [Documentação LMCache — Blending](https://docs.lmcache.ai/kv_cache_optimizations/blending.html)
- [Documentação LMCache — arquitetura](https://docs.lmcache.ai/developer_guide/architecture.html)
- [Documentação vLLM — Automatic Prefix Caching e `cache_salt`](https://docs.vllm.ai/en/v0.14.1/design/prefix_caching/)
- [Issue LMCache #3238 — compatibilidade do CacheBlend com conector vLLM V1](https://github.com/LMCache/LMCache/issues/3238)

