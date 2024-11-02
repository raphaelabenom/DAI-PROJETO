import pandas as pd
import time
import nltk
import re
import os
import glob
import string
import mlflow

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from string import punctuation
from nltk.tokenize import word_tokenize
from nltk.text import Text
from unidecode import unidecode
from collections import Counter
from tqdm import tqdm

# nltk.download('stopwords')
# nltk.download('punkt_tab')
# nltk.download('wordnet')

def carregar_arquivos(diretorio: str) -> list:
    arquivos_conteudo = []

    if os.path.isdir(diretorio):
        for arquivo in glob.glob(os.path.join(diretorio, '*.txt')):
            try:
                with open(arquivo, 'r', encoding='utf-8') as txt:
                    conteudo = txt.read().lower()
                    conteudo = conteudo.replace("-\n", "")
                    arquivos_conteudo.append(conteudo)
            except Exception as e:
                print(f"Erro ao ler o arquivo {arquivo}: {e}")
    else:
        print(f"O diretório {diretorio} não existe.")

    return arquivos_conteudo

# def tokenizador(texto: str) -> list:

#     tokens = word_tokenize(texto)
#     stop_words = set(stopwords.words('portuguese'))
#     tokens = [token for token in tokens if token not in stop_words]
#     tokens = [unidecode(token) for token in tokens]
#     tokens = [token for token in tokens if not token.isdigit()]
#     tokens = [token for token in tokens if token.isalpha()]
    
#     return tokens

def tokenizador(texto: str) -> list:
    tokens = word_tokenize(texto)
    stop_words = set(stopwords.words('portuguese'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words and token.isalpha()]
    return tokens

def processar_arquivos(diretorio: str) -> list:
    textos = carregar_arquivos(diretorio)
    lista_de_tokens = [tokenizador(conteudo) for conteudo in textos]
    return lista_de_tokens


def calcular_similaridade_consulta(consulta: str, corpora: dict) -> dict:
    # Tokeniza a consulta
    tokens = tokenizador(consulta)
    
    texto_consulta = ' '.join(tokens)
    
    similaridades = {}
    resultados = {}

    for categoria, textos in corpora.items():

        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, use_idf=True, norm='l2')
        # Configuração dos parâmetros do vectorizer
        # ngram_range=(1, 2),
        # max_df=0.95,
        # use_idf=True

        vectorizer.fit([' '.join(tokens) for tokens in textos])
        
        tfidf_corpus = vectorizer.transform([' '.join(tokens) for tokens in textos])
        tfidf_consulta = vectorizer.transform([texto_consulta])
        
        similaridade = cosine_similarity(tfidf_consulta, tfidf_corpus).flatten()
        similaridades[categoria] = similaridade.mean()

        documentos_ordenados = sorted(enumerate(similaridade), key=lambda x: x[1], reverse=True)
        resultados[categoria] = documentos_ordenados
    
    return similaridades, resultados

# Caminho dos repositórios
caminho_EdFisica_txt = "./Corpus/EdFisica_txt/"
caminho_Geografia_txt = "./Corpus/Geografia_txt/"
caminho_Historia_txt = "./Corpus/Historia_txt/"
caminho_Linguistica_txt = "./Corpus/Linguistica_txt/"

# Dados limpos e adaptados
dados_EdFisica_txt = processar_arquivos(caminho_EdFisica_txt)
dados_Geografia_txt = processar_arquivos(caminho_Geografia_txt)
dados_Historia_txt = processar_arquivos(caminho_Historia_txt)
dados_Linguistica_txt = processar_arquivos(caminho_Linguistica_txt)

corpora = {
    "EdFisica": dados_EdFisica_txt,
    "Geografia": dados_Geografia_txt,   
    "Historia": dados_Historia_txt,
    "Linguistica": dados_Linguistica_txt
}

# Inicia um experimento MLflow
mlflow.set_experiment("retrieval-dev")
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id=119349917168293976) # Colocar o experiment_id do mlflow
mlflow.autolog()

with mlflow.start_run():

    print("\n")

    consulta = input("Digite sua consulta: ")

    similaridades, resultados = calcular_similaridade_consulta(consulta, corpora)

    for categoria, documentos in resultados.items():
        print(f"\nDocumentos mais relevantes para {categoria}:")
        for doc_idx, similaridade in documentos[:3]: 
            print(f"Documento {doc_idx} - Similaridade: {similaridade:.4f}")

    print("\n")
    print("Média dos resultados")
    print("\n")

    for categoria, similaridade in similaridades.items():
        print(f"Similaridade média com {categoria}: {similaridade:.4f}")

    # Loga as similaridades médias como métricas
    for categoria, similaridade in similaridades.items():
        mlflow.log_metric(f"similaridade_{categoria}", similaridade)

    print("\n")

    categoria_mais_similar = max(similaridades, key=similaridades.get)
    print(f"A consulta é mais similar ao corpus: {categoria_mais_similar}")

    print("\n")

mlflow.end_run()