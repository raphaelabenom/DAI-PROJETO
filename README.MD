# Projeto de Recuperação de Informação

Este projeto realiza a recuperação de informações a partir de um corpus de textos utilizando técnicas de TF-IDF e similaridade de cosseno.

## Estrutura do Projeto
antigo - Vector Template.ipynb
antigo-main.ipynb
app-backup.py
app.py
Corpus/
    EdFisica_txt/
    Geografia_txt/
    Historia_txt/
    Linguistica_txt/
mlruns/
requirements.txt

## ré-requisitos
Python 3.8+
pip (gerenciador de pacotes do Python)
Virtualenv (opcional, mas recomendado)

## Instalação

1. Crie e ative um ambiente virtual (opcional, mas recomendado):
python -m venv venv
source venv/bin/activate  # No Windows use `venv\Scripts\activate`

2. Instale as dependências do projeto:

pip install -r requirements.txt

## Executando o Projeto

1. Execute o script principal:
python app.py

2. Certifique-se de que o servidor MLflow está em execução:
mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns
mlflow ui

## Estrutura do Corpus
O corpus de textos está organizado em subdiretórios dentro da pasta Corpus, cada um representando uma categoria diferente:

EdFisica_txt/
Geografia_txt/
Historia_txt/
Linguistica_txt/
Cada subdiretório contém arquivos de texto que serão processados pelo script.