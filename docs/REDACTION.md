# Política de Redação de PII/Segredos no LLMRouter

## Por que

- PII (LGPD) e segredos de tenant podem aparecer no `prompt` do usuário.
- O `ObservationCollector` persiste esses prompts num banco local e pode sincronizar com sistemas externos.
- Rollout com tenants reais exige redação determinística antes da persistência.

## Estratégia

Substituição determinística por regex + hashing opcional. Sem LLM:

- Latência <1ms
- Zero custo
- Reproduzível (mesmo input → mesmo output)
- Auditável

## Tabela de redação

| Padrão | Exemplo | Placeholder |
|---|---|---|
| EMAIL | `eduardo@vielitech.com.br` | `[EMAIL]` |
| PHONE | `+55 11 99999-8888` | `[PHONE]` |
| CPF / CNPJ (c/ dígito verificador) | `123.456.789-00` | `[DOC_ID]` |
| PAYMENT (cartão Luhn) | `4111 1111...` | `[PAYMENT]` |
| IPV4 | `192.168.0.1` | `[IP]` |
| TOKEN / BEARER / JWT | `ghp_…`, `sk-…`, `ragflow-…`, JWT, `Bearer …` | `[REDACTED:TOKEN]` |
| PASSWORD / API_KEY= | `password=…`, `token=…`, `api_key=…` | `[REDACTED:KV]` |
| PATH (STRICT) | `/opt/data/file`, `C:\\Users\\` | `[PATH]` |

`hash_mode=True` substitui o placeholder por SHA-256 truncado (12 chars) — preserva capacidade de agregação/contagem sem expor valor. Determinístico e não reversível.

## Níveis (cumulativos)

- `NONE` — nada (dev só)
- `PII` — EMAIL, PHONE, CPF/CNPJ, PAYMENT, IPV4
- `SECRETS` — PII + tokens + JWT + password/token=
- `STRICT` — SECRETS + paths absolutos

**Default**: `SECRETS` no ObservationCollector (persistência local). Recomendado:

- `SECRETS` p/ o endpoint `/v1/llmrouter/feedback` (publisher externo PRecog)
- `PII` p/ observações usadas pelo Judge/Grader (não precisam de secrets)
- `STRICT` p/ sync externo (PRecog / telemetry upstream)

## Onde aplicar

1. **`ObservationCollector.flush()`** (implementado aqui) — redige `item.prompt` antes do INSERT, grava `redaction_level` no registro.
2. **`/v1/llmrouter/feedback` → `precog_publisher.update_observation`** — ainda não implementado; p/ rollout fazer o mesmo (redação na borda).
3. **`/v1/llmrouter/semantic/inspect`** — já não persiste; se algum dia logar, aplicar redação.

## Migração de dados existentes

- Não destruir dados antigos (provas de auditoria).
- Coluna `redaction_level` default `'none'` p/ dados antigos; novos persistem o nível aplicado.
- Dashboard / export deve expor somente `redaction_level >= 'secrets'` por default; opt-in explícito p/ ver raw.

## Próximos passos (sugestão E6)

- Config por tenant no models.yaml / config yaml (`redaction: {level: secrets, hash_mode: true}`)
- Backfill 1x p/ registros antigos (reaplicar redactor)
- Integration tests: garantir que `/v1/llmrouter/feedback` não devolve o prompt raw
