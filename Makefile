## LLMrouter – comandos de desenvolvimento
## Uso: make <alvo>

PYTHONPATH := src
HOST ?= 0.0.0.0
PORT ?= 12345

.PHONY: help install install-dev run run-reload test lint format typecheck clean docker-build docker-run

help: ## Mostra os comandos disponíveis
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Instala o pacote em modo editável (recomendado para dev)
	pip install -e .

install-dev: ## Instala o pacote com dependências de desenvolvimento
	pip install -e ".[dev]"

run: ## Inicia o servidor (porta 12345)
	PYTHONPATH=$(PYTHONPATH) python -m uvicorn llmrouter.main:app --host $(HOST) --port $(PORT)

run-reload: ## Inicia o servidor com auto-reload
	PYTHONPATH=$(PYTHONPATH) python -m uvicorn llmrouter.main:app --host $(HOST) --port $(PORT) --reload

test: ## Executa os testes
	pytest

lint: ## Executa o linter (ruff)
	ruff check src tests

format: ## Formata o código (ruff)
	ruff format src tests
	ruff check --fix src tests

typecheck: ## Verificação de tipos (mypy)
	mypy src

clean: ## Remove artefatos de build e cache
	rm -rf build dist *.egg-info .pytest_cache .mypy_cache .ruff_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +

docker-build: ## Constrói a imagem Docker
	docker build -t llmrouter .

docker-run: ## Executa o container Docker
	docker run --rm -p $(PORT):$(PORT) llmrouter