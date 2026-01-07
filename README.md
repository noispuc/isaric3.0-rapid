# 📦 ISARIC HUB SA - RAPID

Este repositório contém o pacote isaric, uma implementação modular da metodologia RAPID (Reproducible Analytics for Predictive modeling, Inference, and Data preprocessing), voltada para projetos de pesquisa clínica e análise preditiva.

## 🧠 O que você encontra aqui:
- Estrutura modular baseada em namespace packages
- Implementação das etapas do framework RAPID:
-- Data Cleaning
-- Preprocessing
-- Modeling
-- Model Evaluation
-- Validation
-- Visualization
- Documentação com MkDocs
- Testes automatizados
- Padrões de `.env`, `.gitignore`, `pyproject.toml`, etc.
- Checklist de limpeza para novos projetos

Ao criar um novo projeto, use este repositório como **template** e ajuste conforme necessário.

## 📦 Estrutura

- `src/`: Código fonte principal
- `tests/`: Testes automatizados com `unittest`
- `docs/`: Documentação gerada com MkDocs
- `config/`: Arquivos `.env` para ambientes
- `requirements.txt`: Lista de dependências do projeto
- `pyproject.toml`: Configuração do pacote Python

## 📚 Documentação

A documentação é gerada com [MkDocs](https://www.mkdocs.org/) e inclui:

- Referência de código com `mkdocstrings`
- Diagramas em Mermaid
- Guia de início rápido

## ⚙️ Requisitos

- Python 3.10+
- [pip](https://pip.pypa.io/en/stable/)
- Ambiente virtual recomendado

```bash
python -m venv .rapid
source .rapid/bin/activate  # ou .rapid\Scripts\activate no Windows
pip install -r requirements.txt

## ✅ Checklist de Limpeza Pós-Clonagem

Após criar seu repositório a partir deste template:

- [ ] Remover testes em `tests/` se não forem usados
- [ ] Ajustar estrutura em `src/` conforme sua lógica de negócio
- [ ] Atualizar ou apagar arquivos em `docs/` se necessário
- [ ] Configurar `.env` a partir do `config/`
- [ ] Revisar `README.md` com a descrição específica do projeto
- [ ] Validar dependências e versões no `requirements.txt`
