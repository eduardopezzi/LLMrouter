# Roadmap E2.5 — Reconciliar streaming replay com o desenho TL

**Issue:** [#11 — E2.5: reconciliar implementação com desenho TL (streaming replay)](https://github.com/eduardopezzi/LLMrouter/issues/11)

**Base da avaliação:** implementação E2 após PR #10 e follow-ups de QA round 3.
**Objetivo:** fechar as divergências do replay streaming sem quebrar clientes atuais e tornar cada decisão de rollout observável.

## Decisões de escopo

1. O item 3 da issue deve ser reescrito: o caminho streaming ainda não grava no `semantic_cache_hit_log`. O trabalho é adicionar auditoria de replay, após a conclusão da emissão SSE; não mover uma gravação existente.
2. Métricas TL serão introduzidas com compatibilidade. Os nomes atuais são parte do contrato de `/v1/llmrouter/cache/stats`; não removê-los na mesma mudança que introduzir aliases.
3. `stream_probe_tokens_spent` será documentado como estimativa baseada no limite `k` até o provider devolver usage real para `first_tokens()`. Não apresentar essa estimativa como tokens faturados.
4. O percentil p50 exige amostras de latência com limite de memória ou um estimador de quantis. Uma soma ou média não serve para produzir p50.
5. A conclusão do gerador pode confirmar que o servidor emitiu o chunk `[DONE]`; não prova que o cliente recebeu os bytes pela rede. O hit-log usará essa definição explícita de sucesso.
6. A mudança do default para `stream_cache_enabled=false` fica condicionada a dados de uso e plano de rollout. O cache semântico principal já é opt-in, mas quem o habilita hoje também habilita streaming por padrão.

## Fase 0 — Fechar decisões de contrato e obter baseline

**Escopo:** especificação e observabilidade, sem mudar comportamento de replay.

- [x] Definir o chunk SSE final com `choices: []`, `usage`, `cache_status` (`live` ou `semantic_hit`) e `usage_source` (`provider`, `cached` ou `estimated`). Preservar os headers existentes.
- [x] Emitir o chunk de usage apenas quando `stream_options.include_usage=true`, tanto no replay quanto no modo live. No live, usar usage do provider quando disponível; caso contrário, estimar e identificar a origem.
- [x] Registrar baseline do Yoda em 2026-10-06: o único host SSH configurado é Yoda; `LLMROUTER_SEMANTIC_CACHE__ENABLED=true`; override de streaming ausente (default efetivo `true`). Às 16:13:46 UTC, `/health` e `/v1/llmrouter/cache/stats` responderam HTTP 200 e os dez contadores legados de streaming estavam zerados. O processo havia iniciado às 13:41:23 UTC; ainda não havia aliases TL.
- [ ] Confirmar inventário global além do Yoda. O `~/.ssh/config` disponível contém apenas o alvo `yoda`, mas isso não prova inexistência de instalações fora desse inventário.
- [x] Atualizar a issue #11 com o resultado e o link do [PR #20](https://github.com/eduardopezzi/LLMrouter/pull/20); o item 3 agora descreve a nova auditoria de replay.

**Status:** contrato, baseline inicial do Yoda e sincronização da issue concluídos; inventário global de instalações pendente. O baseline cobre 2h32 antes do deploy e não sustenta a decisão do default.

## Fase 1 — Fixar invariantes e lacunas de QA

**Escopo:** testes de regressão antes das mudanças de comportamento.

- [x] N2a: circuito aberto com `SemanticCache` real; lookup e probe não são chamados.
- [x] N2b: `max_tokens=None` persiste como `NULL`, usa `-1` na chave de unicidade e migra o schema antigo.
- [x] Interrupção live antes do terminal não armazena resposta.
- [x] Stream vazio e terminal sem conteúdo não armazenam; exceção do iterador após terminal também não armazena.
- [x] GeneratorExit em replay e live não registra hit concluído; novo contador de abortos é separado e o contador legado continua incluindo abortos durante a compatibilidade.

**Status:** concluída. Os testes focados de replay/cache/wiring passaram (72 testes).

## Fase 2 — Conclusão limpa e TTL específico para streaming

**Escopo:** endurecer o caminho live e permitir retenção independente.

- [x] Persistir somente depois que a iteração termina normalmente com finish reason terminal válido e conteúdo útil.
- [x] Abort, erro do provider ou truncamento não gravam resposta.
- [x] Adicionar `stream_ttl_seconds: float | None` à configuração e ao `SemanticCache`.
- [x] `None` herda `ttl_seconds`; valores explícitos precisam ser positivos. Variável: `LLMROUTER_SEMANTIC_CACHE__STREAM_TTL_SECONDS`.

**Status:** concluída e coberta por testes de TTL herdado e específico.

## Fase 3 — Completar o payload SSE do replay

**Escopo:** uso e status de cache por evento, mantendo headers existentes.

- [x] Lookup retorna usage completo e a origem dos dados.
- [x] Replay emite usage/status imediatamente antes de `[DONE]` quando solicitado.
- [x] Preservar `X-LLMrouter-Stream-Cache` e `X-LLMrouter-Cache-Status`.
- [x] Normalização preserva chunks OpenAI com `choices: []`.
- [x] Aplicar a mesma política de usage no caminho live.

**Status:** concluída; contrato e testes cobrem a ordem `chunks → usage/cache_status → [DONE]`.

## Fase 4 — Auditoria de replay após emissão concluída

**Escopo:** trilha P-CHR/audit específica para hits streaming.

- [x] Criar registro de hit com prompt/resposta, modelo, restrições, similaridade e threshold do candidato.
- [x] Retornar metadata de auditoria no resultado do lookup sem modificar os chunks.
- [x] Registrar somente depois que o gerador retoma após emitir `[DONE]`; abortos não geram hit.
- [x] Falha da auditoria é best-effort e não invalida o replay.
- [x] Reutilizar o hit-log e a retenção já configurados para o cache semântico.

**Status:** concluída; integração com SQLite verifica registro único, abort e falha de escrita.

## Fase 5 — Métricas TL e migração do contrato

**Escopo:** nomes, semântica e distribuição de métricas.

- [x] Adicionar aliases TL sem remover os contadores atuais.
- [x] Adicionar `stream_aborts_total` agregado e contadores `stream_replay_aborts_total` / `stream_live_aborts_total` para separar desconexões. O contador legado `stream_replay_error_total` continua incluindo abortos durante a janela de compatibilidade.
- [x] Manter amostra limitada a 1.000 probes e expor `stream_probe_latency_ms_p50` em milissegundos.
- [x] Expor `stream_probe_tokens_spent_estimated` e o alias `stream_probe_tokens_spent` como estimativa do limite `k`, não como faturamento.
- [x] Atualizar `/v1/llmrouter/cache/stats`, snapshot em `contracts/llmrouter.contract.json` e testes de schema.
- [x] Depreciar nomes antigos somente após pelo menos uma versão minor e 90 dias desde a primeira release com aliases, valendo o prazo maior; remover apenas em release major, com evidência de migração dos consumidores e notas de migração.

**Status:** implementação, schema e política de depreciação concluídos.

Definição dos contadores: `stream_hits` conta candidatos que passam pelo threshold do lookup; `stream_replays_served` conta replay depois que o gerador retoma após `[DONE]`; `stream_replay_tokens_saved` soma completion tokens apenas nesse mesmo ponto de conclusão; `stream_probe_tokens_spent` soma o limite `k` de probes executados (estimativa, não faturamento); `stream_probe_latency_ms_p50` é o p50 em milissegundos das amostras limitadas aos últimos 1.000 probes. `stream_aborts_total` conta todo GeneratorExit; `stream_replay_aborts_total` e `stream_live_aborts_total` separam o caminho interrompido. O contador legado `stream_replay_error_total` inclui esses abortos e exceções de replay.

## Fase 6 — Rollout e decisão sobre o default

**Escopo:** ativação controlada e decisão informada para `stream_cache_enabled`.

- [x] Manter o default atual (`stream_cache_enabled=true`) e registrar as métricas necessárias para o rollout.
- [x] Definir a janela e o gate: observar 14 dias completos após publicar a versão instrumentada, com pelo menos 100 probes e 30 replays concluídos. Se a amostra não for atingida, estender a janela e manter o default atual.
- [x] Definir limites para considerar saudável: taxa de probe correspondente (`ok / (ok + fail)`) ≥50%, p50 do probe ≤500 ms, taxa de abortos de replay (`replay_aborts / (replays_served + replay_aborts)`) ≤5% e zero erros de replay excluídos os abortos. Registrar a decisão e os valores observados antes de qualquer mudança de default.
- [ ] Observar a janela em produção. O baseline do Yoda cobre apenas 2h32 antes do deploy e foi medido antes da versão com aliases; ainda não permite aplicar o gate.
- [ ] Decidir o default após a janela de observação; manter `true` até haver evidência para uma mudança.
- [x] Replay pode ser desativado imediatamente por `LLMROUTER_SEMANTIC_CACHE__STREAM_CACHE_ENABLED=false`.
- [x] Rollback: definir `LLMROUTER_SEMANTIC_CACHE__STREAM_CACHE_ENABLED=false` no ambiente da instância, reiniciar o serviço `llmrouter` e validar `/health` e `/v1/llmrouter/cache/stats`; reverter a variável somente após estabilização.

**Status:** plano, critérios e rollback documentados; código mergeado via PR #20, mas implantação e observação por 14 dias ainda pendentes. A janela só começa após implantar a versão instrumentada.

## Verificação desta implementação

- [x] Testes focados de replay, cache e wiring: 72 passaram.
- [x] Suíte completa: 950 passaram e 4 foram ignorados.
- [x] Cobertura final do código E2.5 alterado: linhas executáveis 204/204 (100%); ramos condicionais nas linhas alteradas 58/58 (100%). A cobertura global do repositório inclui módulos fora deste roadmap.
- [x] Ruff e `git diff --check` passaram.
- [x] Revalidação local em 2026-10-06: os 72 testes focados passaram com medição de ramos; a suíte completa passou com 950 testes e 4 skips. Ruff nos arquivos de E2.5 e `git diff --check` passaram.
- [x] Análise de QA requisito por requisito registrada em [QA_E2_5_STREAMING_REPLAY.md](QA_E2_5_STREAMING_REPLAY.md).
- [x] Baseline de produção atualizado em 2026-10-06 16:13:46 UTC no Yoda pela porta correta (12345): `/health` e `/v1/llmrouter/cache/stats` responderam 200; o processo ainda usa `da9c781` e reportou os dez contadores legados de streaming em zero. Prometheus-01/02 seguem reiniciando: apontam para um ID de rede bridge removido, sem IP/endpoint, e falham ao conectar ao Kafka em `172.17.0.1:9092`. O endpoint direto de stats continua disponível para coleta manual.
- [x] Publicar o código na `main` via [PR #20](https://github.com/eduardopezzi/LLMrouter/pull/20), atualizar a issue #11 e fazer pull fast-forward local; suíte pós-pull passou.
- [ ] Implantar a versão instrumentada no Yoda e executar a janela do rollout. Uma worktree isolada em `/home/vieli/LLMrouter-release-ee030f7` foi criada no commit `ee030f7`; os 72 testes focados passaram em Python 3.12.3 sem carregar o `.env` de produção. Também confirmei que `PYTHONPATH` resolve o pacote para o `src/` da worktree quando executado do `WorkingDirectory` atual. O drop-in preparado só define esse `PYTHONPATH`, mantendo `.env`, `config/`, `data/` e o `WorkingDirectory` existentes. Revalidação em 2026-10-06 16:21 UTC via `ssh vieli@yoda`: serviço ativo desde 13:41 UTC, ainda em `da9c781`; worktree `ee030f7` presente; checkout `main` continua 2 commits atrás, com alterações locais staged e arquivos não rastreados preservados; drop-in ainda ausente. `sudo -n true` retorna que a senha é necessária, então instalar o drop-in e reiniciar ainda depende de acesso administrativo interativo.

Para ativar a worktree após obter acesso administrativo:

```sh
sudo install -D -m 0644 /home/vieli/llmrouter-e25.service.conf /etc/systemd/system/llmrouter.service.d/10-e25-release.conf
sudo systemctl daemon-reload
sudo systemctl restart llmrouter
sudo systemctl show llmrouter --property=ActiveState,WorkingDirectory,ExecStart --no-pager
curl -fsS http://127.0.0.1:12345/health
```

Para reverter a versão do código, remova o drop-in e reinicie o serviço; para desativar somente o replay, use `LLMROUTER_SEMANTIC_CACHE__STREAM_CACHE_ENABLED=false` no `.env` e reinicie.

## Ordem sugerida de entrega

1. Fase 0 e Fase 1 podem avançar juntas; a Fase 0 deve fechar o contrato SSE antes da implementação do chunk final.
2. Fase 2 pode ser entregue separadamente e antes das mudanças de payload.
3. Fases 3 e 4 dependem da definição dos metadados e da semântica de conclusão.
4. Fase 5 deve acompanhar a implementação de cada contador, mas a migração de nomes pode ser um PR separado para facilitar revisão de contrato.
5. Fase 6 começa após haver telemetria suficiente e termina com a decisão explícita sobre o default.

Cada PR deve atualizar este roadmap e a issue #11 com o que foi concluído, links dos PRs e decisões ainda pendentes. Alterações em contrato público devem incluir atualização do snapshot e testes correspondentes no mesmo PR.
