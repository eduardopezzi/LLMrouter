# Roadmap

| Item | Status | Implantacao |
| --- | ---: | --- |
| **5. Cross-Repository** | 100% | `ContractRegistry`, `BreakingChangeDetector`, snapshot versionavel, scripts `make` e CLI implementados e cobertos por testes |

## 5. Cross-Repository

O fluxo cross-repository esta operacional de ponta a ponta:

- `ContractRegistry` exporta snapshots JSON deterministicos em `contracts/llmrouter.contract.json`.
- `BreakingChangeDetector` classifica mudancas compativeis e breaking changes.
- CLI: `llmrouter export-contracts`, `llmrouter check-contracts`, `llmrouter diff-contracts`.
- Makefile: `make contracts-export`, `make contracts-check`, `make contracts-diff`.
- Testes: `tests/test_cross_repository.py`.
