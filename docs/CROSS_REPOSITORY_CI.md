<!-- markdownlint-disable MD024 -->
# Contratos cross-repository — Guia de CI

Este documento define como os repositórios da Vieli-Tech devem publicar e
consumir contratos no repositório central de versões:
`Vieli-Tech/phoenix_versions`.

## O que é o contrato

O contrato é um snapshot JSON versionável que descreve a interface pública de
um serviço:

- endpoints com método e schemas resumidos;
- catálogo de modelos/capabilities (quando aplicável);
- roles/papéis de roteamento;
- versão do schema e nome do serviço.

Exemplo de contrato: `contracts/llmrouter.contract.json`.

## Repositório central

- URL: `https://github.com/Vieli-Tech/phoenix_versions.git`
- Cada projeto tem sua própria pasta na raiz do repositório.
- O nome da pasta é resolvido de forma **case-insensitiva**; `Gump`, `gump` e
  `GUMP` apontam para a mesma pasta existente.
- O arquivo vigente fica na raiz da pasta do projeto.

## Publicar contrato no release

Todo repositório que expõe uma API deve publicar seu contrato durante o
workflow de release. A pasta no `phoenix_versions` deve ter o nome do projeto.

### Pré-requisitos

1. Token `VERSOES_REPO_TOKEN` (ou `GITHUB_TOKEN`) com permissão de **push**
   no repositório `Vieli-Tech/phoenix_versions`.
2. CLI do LLMrouter instalada no CI:

   ```bash
   pip install git+https://github.com/Vieli-Tech/LLMrouter.git
   ```

### Opção A — Publicar com `publish-contracts` (push isolado)

Adicione um step no workflow de release:

```yaml
      - name: Publicar contrato no phoenix_versions
        env:
          GITHUB_TOKEN: ${{ secrets.VERSOES_REPO_TOKEN }}
        run: |
          llmrouter publish-contracts \
            --repo https://github.com/Vieli-Tech/phoenix_versions.git \
            --project Gump \
            --filename gump.contract.json \
            --service gump
```

A CLI:

1. Clona o `phoenix_versions`;
2. Cria a pasta `Gump` se não existir;
3. Gera o snapshot em `Gump/gump.contract.json`;
4. Committa e faz push.

> Use esta opção quando o contrato é o único artefato publicado no
> `phoenix_versions` por aquele workflow.

### Opção B — Gerar o contrato localmente e commitar junto com outras mudanças

Use esta opção quando o mesmo workflow também atualiza outros arquivos do
`phoenix_versions` (ex.: `imagens.py`).

```yaml
      - name: Instalar LLMrouter CLI
        run: pip install git+https://github.com/Vieli-Tech/LLMrouter.git

      - name: Clonar repositório versoes
        run: |
          git clone https://x-access-token:${{ secrets.VERSOES_REPO_TOKEN }}@github.com/Vieli-Tech/phoenix_versions.git

      - name: Gerar contrato na pasta do projeto
        run: |
          llmrouter export-contracts \
            --contracts-root phoenix_versions \
            --project Gump \
            --filename gump.contract.json \
            --service gump

      - name: Atualizar outras configurações
        run: |
          # exemplo: atualização de imagens.py, helm charts, etc.

      - name: Commit e push
        run: |
          cd phoenix_versions
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"
          git add .
          git commit -m "chore: update gump to ${{ github.ref_name }}" || exit 0
          git push
```

## Variáveis de configuração

| Variável | Descrição | Exemplo |
| -------- | --------- | ------- |
| `--project` | Pasta do projeto no `phoenix_versions` | `Gump` |
| `--filename` | Nome do arquivo de contrato | `gump.contract.json` |
| `--service` | Nome lógico do serviço dentro do snapshot | `gump` |
| `--repo` | URL do repositório central | `https://github.com/Vieli-Tech/phoenix_versions.git` |
| `--branch` | Branch de destino | `main` (padrão) |

## Serviços que não usam os endpoints padrão do LLMrouter

A CLI `llmrouter export-contracts` usa endpoints OpenAI-compatible definidos
internamente. Se o seu serviço expõe endpoints diferentes, ainda não é possível
sobrescrever os endpoints via CLI. Neste caso, gere o arquivo JSON por conta
própria com a estrutura esperada e copie-o para a pasta do projeto no
`phoenix_versions`.

Estrutura mínima do JSON:

```json
{
  "schema_version": "1.0",
  "service": "gump",
  "endpoints": [
    {
      "path": "/v1/status",
      "method": "GET",
      "auth_required": false,
      "response_schema": {"status": "str"}
    }
  ],
  "models": [],
  "routing_roles": []
}
```

## Consumir contrato de outro serviço

Repositórios que consomem um serviço publicado no `phoenix_versions` devem
validar compatibilidade antes de aceitar atualizações. Exemplo de job de CI:

```yaml
name: Check upstream contract

on:
  schedule:
    - cron: '0 6 * * *'
  workflow_dispatch:

jobs:
  contract-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install LLMrouter CLI
        run: pip install git+https://github.com/Vieli-Tech/LLMrouter.git

      - name: Download current contract
        run: |
          curl -L \
            -o /tmp/current.contract.json \
            https://raw.githubusercontent.com/Vieli-Tech/phoenix_versions/main/Gump/gump.contract.json

      - name: Check for breaking changes
        run: |
          llmrouter check-contracts \
            contracts/baseline.gump.contract.json \
            /tmp/current.contract.json
```

## Regras de breaking change

O comando `check-contracts` sai com código `1` quando detecta:

- remoção de endpoint;
- mudança de método ou schema de endpoint;
- remoção de modelo;
- mudança de `provider`, `provider_model` ou `tier` de um modelo;
- remoção de capability de um modelo;
- redução do `context_window` de um modelo;
- remoção de routing role;
- mudança de `schema_version`.

As seguintes alterações são consideradas **compatíveis**:

- adição de endpoint, modelo, role ou capability;
- aumento de `context_window`.

## Exemplo completo para repositório Gump

Workflow de release publicando imagem Docker no DockerHub, atualizando
`imagens.py` e gerando o contrato na pasta `Gump`:

```yaml
name: Create and publish a Docker image

on:
  release:
    types:
      - created

jobs:
  build-and-push-image:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - name: Check out repository code
        uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies and LLMrouter CLI
        run: |
          python -m pip install --upgrade pip
          pip install git+https://github.com/Vieli-Tech/LLMrouter.git
          pip install -r requirements.txt --no-cache

      - name: Clonar repositório versoes
        run: |
          git clone https://x-access-token:${{ secrets.VERSOES_REPO_TOKEN }}@github.com/Vieli-Tech/phoenix_versions.git

      - name: Gerar contrato na pasta do projeto
        run: |
          llmrouter export-contracts \
            --contracts-root phoenix_versions \
            --project Gump \
            --filename gump.contract.json \
            --service gump

      - name: Atualizar arquivo de imagens
        shell: python
        env:
          CLEAN_VERSION: ${{ github.ref_name }}
        run: |
          import re
          import os

          file_path = "phoenix_versions/imagens.py"
          service = "Gump"
          image = "vielitech/gump"
          new_version = os.environ['CLEAN_VERSION'].lstrip('v')

          with open(file_path, "r", encoding="utf-8") as f:
              content = f.read()

          pattern = rf'("{service}"\s*:\s*"{image}:)([^"]+)"'
          new_content, count = re.subn(pattern, lambda m: f'{m.group(1)}{new_version}"', content)

          if count == 0:
              print(f"Erro: {service} não encontrado em {file_path}")
              exit(1)

          with open(file_path, "w", encoding="utf-8") as f:
              f.write(new_content)

          print(f"Sucesso: {service} atualizado para {new_version}")

      - name: Commit e push no repo versoes
        run: |
          cd phoenix_versions
          git config user.name "github-actions"
          git config user.email "github-actions@github.com"
          git add .
          git commit -m "chore: update gump to ${{ github.ref_name }}" || exit 0
          git push
```

## Checklist para novos repositórios

- [ ] Definir o nome da pasta do projeto no `phoenix_versions`.
- [ ] Definir o nome do arquivo de contrato (`<servico>.contract.json`).
- [ ] Garantir que o token `VERSOES_REPO_TOKEN` tenha permissão de push.
- [ ] Adicionar step de publicação do contrato no workflow de release.
- [ ] Se o serviço for consumido por outros, manter uma baseline local do
      contrato e validar breaking changes em CI.
- [ ] Se os endpoints forem diferentes do padrão LLMrouter, gerar o JSON
      manualmente até a CLI suportar endpoints customizados.
