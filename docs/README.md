# Documentação — LLMrouter

A documentação está organizada pela finalidade de cada arquivo. O
[roadmap único](ROADMAP.md) reúne prioridades, dependências, status e critérios
de aceite do gateway e do ecossistema, incluindo E1–E7 e E2.5.

## Informações e referências

Em `informacoes/` ficam guias de uso, pesquisas, fontes de catálogo e análises
datadas. Recomendações nesses estudos não comprovam implementação; baselines
descrevem a data da coleta.

| Documento | Conteúdo |
| --- | --- |
| [Configuração do Cline](informacoes/CLINE_SETUP.md) | Integração, validação e troubleshooting |
| [Contratos cross-repository](informacoes/CROSS_REPOSITORY_CI.md) | Publicação e consumo de contratos no CI |
| [Limites de tokens](informacoes/MODEL_TOKEN_LIMITS.md) | Fontes e decisões para o catálogo |
| [Modelos GLM via Z.ai](informacoes/ZAI_GLM_MODELS.md) | Política de roteamento e benchmarks publicados |
| [Open LLM Leaderboard](informacoes/OPEN_LLM_LEADERBOARD_RESEARCH_2026-08-04.md) | Pesquisa de fontes de benchmarks de 2026-08-04 |
| [Auditoria de benchmarks do Yoda](informacoes/YODA_BENCHMARK_AUDIT_2026-08-04.md) | Inventário e classificação observados em 2026-08-04 |
| [Superfície da API RAGFlow](informacoes/RAGFLOW_API_SURFACE.md) | Pesquisa de endpoints e opções de integração |
| [Eficiência de contexto e tokens](informacoes/guia_otimizacao_tokens_llmrouter_precog.md) | Estudo de estratégias, experimentos e interfaces propostas |

## Documentos técnicos de desenvolvimento

Em `desenvolvimento/` ficam método de trabalho, propostas e decisões ainda
abertas. Documentos mistos identificam quais partes já foram entregues;
acompanhar próximas entregas no roadmap.

| Documento | Conteúdo e situação |
| --- | --- |
| [Plano TDD](desenvolvimento/DEVELOPMENT_PLAN_TDD.md) | Método e critérios por fase; fases 0–6 entregues, auto-rollback pendente |
| [ADR-0001: acoplamento RAGFlow](desenvolvimento/ADR-0001-ragflow-coupling.md) | Comparação de cenários; decisão formal ainda registrada como proposta |
| [Proposta CacheBlend](desenvolvimento/CACHEBLEND_RAG_LATENCY_PROPOSAL.md) | Arquitetura e critérios para estudo/piloto futuro |

## Recursos já implementados

Em `implementado/` ficam descrições do sistema entregue e evidências de QA.
Implementação concluída não implica deploy, ativação ou aceite em produção;
essas pendências continuam no roadmap.

| Documento | Conteúdo |
| --- | --- |
| [Capacidades implementadas](implementado/CAPACIDADES_IMPLEMENTADAS.md) | Entregas do gateway e limitações operacionais |
| [Fluxo de requisições](implementado/LLMROUTER_REQUEST_FLOW.md) | Arquitetura do caminho de chat e mapeamento para o código |
| [Desenho de rollout](implementado/ROLL_FEATURE_DESIGN.md) | Desenho original da base canary/blue-green já entregue; extensões identificadas |
| [QA de streaming replay E2.5](implementado/QA_E2_5_STREAMING_REPLAY.md) | Evidências de revisão/testes e limites da validação de produção |

## Manutenção

- Acrescentar cada documento neste índice e escolher a pasta pela finalidade.
- Registrar prioridades e status de entregas somente em `ROADMAP.md`.
- Manter desenhos, métodos e critérios detalhados nos documentos técnicos.
- Ao concluir código, atualizar seu registro em `implementado/` e distinguir
  eventuais gates de produção ainda pendentes.
- Ao mover arquivos, atualizar links no README, nos documentos e nas
  referências de scripts/código.
