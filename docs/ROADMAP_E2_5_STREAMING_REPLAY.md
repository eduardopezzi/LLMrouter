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

- Definir o formato do chunk final SSE: `usage` no formato OpenAI (`choices: []`) e onde `cache_status` será representado. Manter os headers atuais durante a transição.
- Definir quando emitir usage no modo live e no replay, incluindo o comportamento de `stream_options.include_usage` se suportado pela API.
- Registrar baseline das métricas atuais e do número de instalações/configurações com cache semântico e streaming habilitados, antes de decidir o novo default.
- Atualizar a issue para deixar claro que item 3 adiciona auditoria específica para streaming.

**Aceite:** formato SSE documentado, estratégia de compatibilidade aprovada no próprio desenho técnico e baseline disponível para a decisão do default.

## Fase 1 — Fixar invariantes e lacunas de QA

**Escopo:** testes de regressão antes das mudanças de comportamento.

- N2a: exercitar o circuito aberto com o `SemanticCache` real ou um spy fiel, provando que `lookup_stream_response` não é chamado quando o circuito está aberto.
- N2b: cobrir persistência e unicidade quando `max_tokens=None`, verificando a sentinela `-1` na chave e `NULL` no campo persistido.
- Cobrir interrupção do caminho live antes de `finish_reason`: nenhuma entrada deve ser armazenada.
- Cobrir o sinal de conclusão limpa no caminho live, inclusive stream vazio, `finish_reason` terminal e encerramento do iterador com exceção.
- Cobrir GeneratorExit durante replay e durante live, separando erro de replay de abort do cliente na semântica dos contadores.

**Aceite:** os testes reproduzem as duas notas N2 e demonstram que apenas streams live completos e válidos podem ser armazenados.

## Fase 2 — Conclusão limpa e TTL específico para streaming

**Escopo:** endurecer o caminho live e permitir retenção independente.

- Substituir `saw_finish_reason` por estado explícito de conclusão limpa, definido somente após iteração normal e observação de um chunk terminal válido.
- Continuar exigindo conteúdo útil antes de armazenar; abort, erro do provider ou truncamento não podem gravar resposta.
- Adicionar `stream_ttl_seconds: float | None` à configuração e ao `SemanticCache`.
- Quando `stream_ttl_seconds` for `None`, herdar `ttl_seconds`; validar valores explícitos como positivos e documentar a variável de ambiente correspondente.

**Aceite:** fluxos incompletos nunca são armazenados; TTL omitido mantém o comportamento existente; TTL configurado controla apenas novas entradas streaming.

## Fase 3 — Completar o payload SSE do replay

**Escopo:** uso e status de cache por evento, mantendo headers existentes.

- Retornar do lookup os dados de usage completos, não só `completion_tokens`.
- No replay, emitir um chunk final de usage e `cache_status` conforme o contrato definido na Fase 0, imediatamente antes de `[DONE]`.
- Preservar os headers `X-LLMrouter-Stream-Cache` e `X-LLMrouter-Cache-Status` durante a migração.
- Garantir que normalização e encaminhamento de chunks não descartem o novo chunk final. Hoje chunks com `choices: []` são filtrados pela normalização.
- Definir a mesma política de usage para resposta live, evitando que os clientes precisem tratar o replay como um protocolo distinto.

**Aceite:** teste de integração compara os eventos SSE do replay com o contrato, valida a ordem `chunks → usage/cache_status → [DONE]` e cobre compatibilidade do comportamento live.

## Fase 4 — Auditoria de replay após emissão concluída

**Escopo:** trilha P-CHR/audit específica para hits streaming.

- Criar uma operação de registro de hit para streaming com os dados disponíveis do candidato: prompt, resposta, modelo, restrições, similaridade e threshold.
- Devolver do lookup a metadata do candidato necessária ao registro sem alterar o conteúdo dos chunks.
- Só registrar o hit depois que o gerador tiver retomado após emitir `[DONE]` e concluído a sequência normalmente. Aborts e exceções não geram hit concluído.
- Manter a gravação best-effort: erro de auditoria não pode interromper nem invalidar uma resposta já emitida.
- Definir retenção e política de dados do texto de prompt/resposta de acordo com o hit-log existente.

**Aceite:** replay concluído cria exatamente um registro; miss, divergência do probe, circuito aberto, abort e erro não criam registro; falha de escrita não afeta a resposta.

## Fase 5 — Métricas TL e migração do contrato

**Escopo:** nomes, semântica e distribuição de métricas.

- Adicionar aliases TL para hits, replays servidos e tokens economizados, mantendo contadores atuais durante a janela de compatibilidade.
- Separar `stream_aborts_total` de `stream_replay_error_total`; documentar se o contador legado continua acumulando aborts durante a depreciação.
- Instrumentar latência do probe com uma amostra limitada ou estimador de quantis e expor p50 com unidade explícita.
- Expor `stream_probe_tokens_spent` como estimativa `k` para probes executados enquanto não houver usage real do provider. Se necessário, renomear o campo para explicitar que é estimado.
- Atualizar o endpoint, snapshot do contrato e testes de schema juntos. Publicar período de depreciação e critério para remover aliases antigos em uma versão futura.

**Aceite:** cada métrica tem definição, unidade e ponto de incremento documentados; aliases coexistem com os nomes atuais; contrato e implementação têm o mesmo schema.

## Fase 6 — Rollout e decisão sobre o default

**Escopo:** ativação controlada e decisão informada para `stream_cache_enabled`.

- Implantar inicialmente sem alterar o default e observar taxa de lookup, replay concluído, probe divergente, aborts, latência p50, tokens estimados e precisão auditada.
- Definir janela e limites de qualidade/latência antes de avaliar o default. Se os dados forem insuficientes, manter o default existente e tratar a decisão como pendente.
- Se aprovado, mudar para `false`, atualizar descrições, exemplos de configuração e notas de migração; explicitar como habilitar streaming em ambientes que já usam cache semântico.
- Manter uma forma rápida de desativar replay por configuração e documentar o procedimento de rollback.

**Aceite:** decisão registrada com dados observados; rollout e rollback documentados; mudança de default acompanhada por cobertura de configuração e comunicação de migração.

## Ordem sugerida de entrega

1. Fase 0 e Fase 1 podem avançar juntas; a Fase 0 deve fechar o contrato SSE antes da implementação do chunk final.
2. Fase 2 pode ser entregue separadamente e antes das mudanças de payload.
3. Fases 3 e 4 dependem da definição dos metadados e da semântica de conclusão.
4. Fase 5 deve acompanhar a implementação de cada contador, mas a migração de nomes pode ser um PR separado para facilitar revisão de contrato.
5. Fase 6 começa após haver telemetria suficiente e termina com a decisão explícita sobre o default.

Cada PR deve atualizar este roadmap e a issue #11 com o que foi concluído, links dos PRs e decisões ainda pendentes. Alterações em contrato público devem incluir atualização do snapshot e testes correspondentes no mesmo PR.
