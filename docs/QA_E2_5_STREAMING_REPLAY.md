# QA — E2.5 Streaming Replay

**Data da revisão:** 2026-10-06
**Escopo:** mudanças locais de reconciliação do streaming replay com TL, comparadas com `docs/ROADMAP_E2_5_STREAMING_REPLAY.md` e a issue [#11](https://github.com/eduardopezzi/LLMrouter/issues/11).

## Parecer

QA independente revisou o diff em três rodadas. A primeira apontou lacunas nos testes da migração legada e na validação do TTL; elas foram corrigidas com fixtures de dados anteriores, validação na classe e testes de TTL por configuração e por chamada (incluindo valores não finitos e conversão com overflow). As rodadas seguintes confirmaram as correções. A implementação local está aprovada para revisão de código. O roadmap completo e o rollout continuam pendentes: ainda faltam inventário global, sincronização da issue, publicação/implantação e a janela de produção de 14 dias.

## Verificações por requisito

| Requisito | Evidência de QA | Resultado |
|---|---|---|
| Circuito aberto evita lookup e probe | Teste N2a com `SemanticCache` real e spy | Aprovado |
| `max_tokens=None` é armazenado como `NULL`, com chave `-1` | Teste de persistência/unicidade; migração semeada com linhas legadas NULL duplicadas, sentinel `-1` e limite explícito | Aprovado |
| TTL de streaming inválido não cria entrada expirada | Testes de configuração, construtor e override por chamada com zero, negativo, infinito, NaN e overflow de conversão | Aprovado |
| Só streams live completos e com conteúdo podem ser armazenados | Testes de stream vazio, terminal sem conteúdo, abort e exceção após terminal | Aprovado |
| SSE inclui usage/status somente quando solicitado e antes de `[DONE]` | Testes live/replay, normalização de `choices: []` e `stream_options` | Aprovado |
| Hit-log só é gravado após retomada normal depois de `[DONE]` | Teste de integração SQLite para sucesso, abort e falha de escrita | Aprovado |
| Métricas representam replay concluído e probe estimado | Teste desconecta após `[DONE]` antes da retomada; contadores permanecem zerados | Aprovado |
| Aborto de live não se confunde com aborto de replay | Asserções verificam contadores agregados e específicos nos dois caminhos | Aprovado |
| Aliases e snapshot do contrato estão sincronizados | Teste do endpoint e verificação do `contracts/llmrouter.contract.json` | Aprovado |
| Retenção e proteção de dados do hit-log | Implementação reutiliza `hit_log_enabled` e retenção já existentes | Aprovado no escopo atual |

## Execução local

- Suíte focada (`.venv/bin/python -m pytest tests/test_stream_replay.py tests/test_semantic_stream_cache.py tests/test_semantic_wiring.py -q`): 72 testes passaram.
- Suíte completa (`.venv/bin/python -m pytest -q`, Python 3.11.9): 950 passaram, 4 foram ignorados; cobertura global do repositório: 83%.
- Cobertura das linhas executáveis alteradas em `src/llmrouter`: 204/204 (100%). Cobertura dos ramos condicionais nas linhas alteradas: 58/58 (100%), aferida com `--cov-branch` nos 72 testes focados.
- Ruff e `git diff --check` passaram; mypy isolado de `src/llmrouter/core/semantic_cache.py` passou.
- O `pytest` global do shell está vinculado a Python 3.10, abaixo do requisito do projeto (>=3.11), e falha na coleta por imports de `datetime.UTC`/`enum.StrEnum`. Os comandos acima usam o ambiente local compatível `.venv`.
- Mypy isolado de `semantic_cache.py` passou. A verificação ampla continua encontrando erros em trechos não alterados de `routes.py` e `runtime.py`, além de erros transitivos pré-existentes em outros módulos.

## Validação de produção e riscos restantes

Na única instalação acessível, Yoda, a API autenticada pela porta 12345 respondeu HTTP 200. A configuração observada tem cache semântico habilitado e streaming no default `true`; os contadores legados estavam zerados em 2026-10-06 15:36 UTC, menos de duas horas após o início do processo. Isso é apenas um baseline inicial, não a janela de rollout. Os contêineres Prometheus-01 e Prometheus-02 estavam reiniciando.

O serviço do Yoda continua executando `da9c781`; o checkout principal mantém alterações staged em `src/llmrouter/config.py` e `tests/test_rollout.py`, além de `docs/adr/` e `src/llmrouter/core/auto_rollback.py` não rastreados. Para preservar esse trabalho, foi criada uma worktree isolada em `/home/vieli/LLMrouter-release-ee030f7`, no merge `ee030f7`. O import aponta para o `src/` dessa worktree, enquanto `data/` e `.env` são links para os dados e configurações existentes. Os 72 testes focados passaram no servidor depois de definir variáveis que isolam os defaults do `.env` de produção.

A mudança do serviço ainda não foi aplicada. `sudo -n true` foi recusado porque é necessária senha; nenhuma unidade systemd ou processo de produção foi alterado. O drop-in pronto para instalação está em `/home/vieli/llmrouter-e25.service.conf`; instruções de ativação e rollback estão no roadmap. O contador de 14 dias só começa depois de reiniciar o serviço usando a worktree instrumentada.

O gate de rollout está definido no roadmap: 14 dias da versão instrumentada em produção, pelo menos 100 probes e 30 replays concluídos; taxa de correspondência de probes ≥50%, p50 ≤500 ms, abortos ≤5% e zero erros de replay fora dos abortos. A avaliação do default permanece pendente até haver amostra suficiente. A issue #11 foi atualizada e permanece aberta enquanto o deploy e o gate não forem concluídos; o código foi mergeado pelo [PR #20](https://github.com/eduardopezzi/LLMrouter/pull/20).
