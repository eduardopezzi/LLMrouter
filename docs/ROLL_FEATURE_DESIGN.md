# Canary/Blue-Green Model Rollout — Plano Executivo (Item #9 do Roadmap)

## 1. Resumo Executivo

A feature **Canary/Blue-Green Model Rollout** permite promoção gradual e segura de novos modelos no catálogo LLMrouter. Cada modelo recebe um campo `rollout_percentage` (0.0–100.0). Antes da estratégia de seleção finalizar o candidato, o router aplica um filtro de rollout baseado em hash determinístico do prompt. O objetivo é:

- Reduzir risco de regressão ao introduzir um novo modelo.
- Permitir rollback instantâneo sem restart (basta setar `rollout_percentage = 0`).
- Reutilizar o `ModelHealthTracker` já existente (item #6) para observabilidade e futura auto-promoção/auto-rollback.

---

## 2. Pesquisa de Referências e Boas Práticas

| Referência | Conceito Chave | Aplicabilidade no LLMrouter |
|---|---|---|
| **Istio / Envoy VirtualService** | Traffic splitting via `weight` entre destinos | Weighted random/deterministic selection, percentuais exatos |
| **Argo Rollouts / Flagger** | Progressive delivery com 5%→25%→50%→100% e análise automática de métricas | Integrar com `ModelHealthTracker` para auto-promoção |
| **LaunchDarkly / Unleash** | Feature flags com rollout % e sticky bucketing | Futuro: adicionar `X-User-ID` para bucketing consistente |
| **OpenRouter API** | Model catalog com `priority` + `weight` | Schema de catálogo já usa `priority`; adicionamos `rollout_percentage` |
| **LiteLLM Gateway** | Fallback chain com weighted routing | Alinhado com `ProviderProxy` existente |
| **Netflix Spinnaker Canary** | Métricas: error rate, latency + rollback automático se thresholds violados | Reutilizar P95 e error rate do HealthTracker |
| **Google SRE — Canary Release Book** | "Push one, watch one, rollback one" + error budget | Definir budget de erro durante rollout |

**Decisões de design recomendadas:**

1. **Determinístico (hash-based)** por padrão: mesmo prompt sempre mapeia ao mesmo bucket, facilitando A/B consistente e debugging.
2. **Safety net**: se após filtragem todos os candidatos forem removidos, retornar à lista original.
3. **Rollback instantâneo**: `rollout_percentage = 0` equivale a remover o modelo do tráfego sem alterar `priority` ou apagar do YAML.
4. **Sticky bucketing futuro**: via `X-User-ID`/`session_id` no hash.

---

## 3. Arquitetura do Sistema Atual Relevante

### 3.1 Fluxo de roteamento existente

```
POST /v1/chat/completions
  └── routes.py:chat_completions()
        ├── PromptScorer.score(prompt) → tier
        ├── MultiModelRouter.route(request, constraints)
        │     ├── _get_candidates(tier)  → lista de ModelInfo
        │     ├── strategy.select(candidates) → ordenado
        │     └── fallback chain
        └── ProviderProxy.chat_completion(decision)
              └── tenta primary → fallbacks
```

### 3.2 Estrutura central

- `src/llmrouter/core/types.py` — `ModelInfo` (frozen dataclass), `RoutingDecision`.
- `src/llmrouter/core/registry.py` — carrega catálogo YAML em `ModelInfo`.
- `src/llmrouter/core/router.py` — `MultiModelRouter` e estratégias (`CostStrategy`, `QualityStrategy`, etc.).
- `src/llmrouter/core/proxy.py` — `ProviderProxy`, fallback chain.
- `src/llmrouter/cli_panel.py` — painel interativo e funções de persistência no YAML.
- `src/llmrouter/main.py` — entrypoint CLI `llmrouter panel --set-strategy ...`.
- `src/llmrouter/api/routes.py` — rotas HTTP.
- `src/llmrouter/config.py` — configurações `pydantic-settings`.

---

## 4. Mudanças Planejadas

### 4.1 Domain Types — `src/llmrouter/core/types.py`

Adicionar a `ModelInfo`:

```python
rollout_percentage: float = 100.0
```

- Validar no `__post_init__`: `0.0 <= value <= 100.0`.
- Manter `frozen=True` para segurança em async tasks.

Estender `RoutingDecision` com metadados de rollout:

```python
@dataclass(frozen=True)
class RoutingDecision:
    primary: ModelInfo
    fallbacks: list[ModelInfo]
    score: float
    tier: Tier
    reason: str
    rollout_sampled: str | None = None  # "model_name:pct"
```

### 4.2 Registry Loader — `src/llmrouter/core/registry.py`

No `_model_from_mapping`, ler:

```python
rollout_percentage = max(0.0, min(100.0, float(item.get("rollout_percentage", 100.0))))
```

### 4.3 Router — `src/llmrouter/core/router.py`

Novo método `MultiModelRouter._apply_rollout(candidates, request)`:

```python
def _apply_rollout(self, candidates, request):
    # Fast path: todos com 100%, retorna como está
    # Hash determinístico do prompt + model.name
    # Bucket = hash % 100
    # Se bucket < rollout_percentage, mantém o modelo
    # Se nenhum sobreviver, retorna lista original
```

Integrar em `route()` entre `_get_candidates` e `_strategy.select`.

Log estruturado:

```
routing_decision.rollout_sampled=model_name:percentage
```

### 4.4 CLI Panel — `src/llmrouter/cli_panel.py`

- Nova opção `11. Set model rollout percentage` em `run_interactive_panel`.
- Nova função `set_model_rollout_percentage(models_file, model_name, percentage)`:
  - Parse do YAML via regex similar a `promote_model_priority`.
  - Atualiza/insere `rollout_percentage`.
  - Reescreve o arquivo preservando comentários e formatação.
- Função auxiliar `_prompt_rollout_percentage`.

### 4.5 CLI Entrypoint — `src/llmrouter/main.py`

Adicionar ao `panel` subparser:

```bash
llmrouter panel --set-rollout <model> <pct>
```

Handler chama `set_model_rollout_percentage`, recarrega registry e imprime resumo.

### 4.6 API Routes — `src/llmrouter/api/routes.py`

Endpoints administrativos:

- `GET /v1/llmrouter/rollout` — percentual de rollout de todos os modelos.
- `POST /v1/llmrouter/rollout/{model_name}?percentage=...` — atualização em runtime.

Atualização em runtime: atualiza o YAML, chama `load_model_registry` e `router.replace_registry()`.

### 4.7 Config — `src/llmrouter/config.py`

Adicionar `RolloutConfig` dentro de `RoutingConfig` ou como sub-config:

```python
class RolloutConfig(BaseModel):
    enabled: bool = True
    deterministic: bool = True
    critical_threshold_pct: float = 5.0  # abaixo disso é considerado canary
    auto_promote: bool = False           # futuro
    auto_rollback_error_rate: float = 20.0  # % erro durante canary
```

### 4.8 Catálogo YAML — `config/models.example.yaml`

Adicionar exemplo:

```yaml
- name: "ollama/deepseek-v4-pro:cloud"
  provider: "ollama"
  ...
  priority: 3
  rollout_percentage: 100
```

Exemplo de canary:

```yaml
- name: "ollama/new-experimental:canary"
  provider: "ollama"
  ...
  priority: 10
  rollout_percentage: 10
```

---

## 5. Diagrama de Sequência

```
Client        API Routes        MultiModelRouter    SelectionStrategy    ProviderProxy
  | POST /v1/chat/completions      |                   |                    |
  |------------------------------->|                   |                    |
  |              |                 |                   |                    |
  |              | route(request)  |                   |                    |
  |              |------------------------------------>|                    |
  |              |                 | _get_candidates   |                    |
  |              |                 | (tier)            |                    |
  |              |                 |                   |                    |
  |              |                 | _apply_rollout()  |                    |
  |              |                 | (hash%100 < pct)  |                    |
  |              |                 |                   |                    |
  |              |                 | select(candidates)|                    |
  |              |                 |------------------------------------->|
  |              |                 |                   |                    |
  |              |                 | chat_completion(decision)            |
  |              |                 |------------------------------------------------>|
```

---

## 6. Plano de Testes

| Arquivo | Cobertura |
|---|---|
| `tests/test_rollout.py` (novo) | Unidade de `_apply_rollout`: 0%, 100%, 50/50, determinismo, safety net, fallback chain |
| `tests/test_router.py` (extensão) | Integração rollout no `route()` |
| `tests/test_registry_loader.py` (extensão) | Parse de `rollout_percentage` do YAML |
| `tests/test_cli_panel_full.py` (extensão) | `set_model_rollout_percentage`, `--set-rollout` |
| `tests/test_api.py` (extensão) | Endpoints GET/POST rollout |

---

## 7. Entregáveis e Estimativa

| Sprint | Entrega | Arquivos Principais | Dias |
|---|---|---|---|
| S1: Core rollout | `_apply_rollout`, `ModelInfo`, registry parser | `types.py`, `router.py`, `registry.py` | 2 |
| S2: Observabilidade | Log estruturado, `RoutingDecision` metadata | `router.py`, `routes.py` | 1 |
| S3: CLI | `--set-rollout`, opção 11, persistência YAML | `cli_panel.py`, `main.py` | 1 |
| S4: API | GET/POST rollout, hot reload | `routes.py` | 1 |
| S5: Config/docs | `RolloutConfig`, sample YAML, este documento | `config.py`, `models.example.yaml`, `docs/` | 1 |
| S6: Testes | Suite completa | `tests/` | 1 |

**Total estimado: 7 dias.**

---

## 8. Riscos e Mitigações

| Risco | Mitigação |
|---|---|
| Todos os modelos bloqueados por rollout=0 | Safety net: retorna lista original se `filtered` fica vazio |
| Falha de provider no canary | Rollback instantâneo via CLI/API sem restart |
| Inconsistência de A/B | Hash determinístico por padrão |
| Corrupção do YAML ao editar | Reutilizar parser respeitando comentários; testes de YAML round-trip |
| Registry não atualizar em runtime | Usar `router.replace_registry()` já existente |

---

## 9. Rollback Instantâneo

Dois caminhos:

1. **CLI:** `llmrouter panel --set-rollout modelo_problematico 0`
2. **API:** `POST /v1/llmrouter/rollout/modelo_problematico?percentage=0`

Ambos atualizam o arquivo YAML e hot-reload o registry sem reiniciar o servidor.

---

## 10. Métricas e Sucintas Métricas de Sucesso

- Tempo para rollback: **<5 segundos** (editar YAML + hot-reload).
- Cobertura de teste da feature: **>90%**.
- Determinismo do hash: 10.000 prompts idênticos → mesmo bucket para o mesmo modelo.
- Simulação de canary 10% sobre 10.000 requisições → ±1% do percentual alvo.

---

## 11. Checklist de Implementação

- [ ] Adicionar `rollout_percentage` a `ModelInfo`
- [ ] Validar range no dataloader
- [ ] Implementar `MultiModelRouter._apply_rollout()`
- [ ] Integrar filtro no fluxo de `route()`
- [ ] Adicionar `rollout_sampled` a `RoutingDecision`
- [ ] Log estruturado `routing_decision.rollout_sampled=model:pct`
- [ ] Implementar `set_model_rollout_percentage()`
- [ ] Adicionar opção 11 no painel interativo
- [ ] Adicionar `--set-rollout` no CLI
- [ ] Criar endpoints GET/POST rollout
- [ ] Adicionar `RolloutConfig` em config.py
- [ ] Atualizar `models.example.yaml`
- [ ] Escrever `tests/test_rollout.py`
- [ ] Atualizar testes existentes
- [ ] Documentar no README/ROADMAP