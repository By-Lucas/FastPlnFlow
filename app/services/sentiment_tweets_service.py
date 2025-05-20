import re
import os
import spacy
import random
import string
import pandas as pd
from spacy.training import Example
from spacy.lang.pt.stop_words import STOP_WORDS


# Caminho do modelo salvo
MODEL_PATH = "models/sentiment_tweets_model"
pln = spacy.load("pt_core_news_sm")  # Lematização

# Pré-processamento igual ao Colab
def preprocess(texto):
    texto = texto.lower()
    texto = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', texto)
    texto = re.sub(r"https?://[A-Za-z0-9./]+", ' ', texto)
    texto = re.sub(r" +", ' ', texto)

    lista_emocoes = {
        ':)': 'emocaopositiva',
        ':d': 'emocaopositiva',
        ':(': 'emocaonegativo'
    }
    for emocao in lista_emocoes:
        texto = texto.replace(emocao, lista_emocoes[emocao])

    documento = pln(texto)
    lista = [token.lemma_ for token in documento]
    lista = [palavra for palavra in lista if palavra not in STOP_WORDS and palavra not in string.punctuation]
    lista = ' '.join([str(palavra) for palavra in lista if not palavra.isdigit()])
    return lista

def train_model_tweets(
    csv_path: str = "data/Train50.csv",
    limit: int = 100_000,
    n_epochs: int = 5,
    batch_size: int = 512
) -> dict:
    """
    Treina um modelo spaCy TextCat para classificação de sentimento usando dados rotulados em um CSV.
    O modelo é salvo em MODEL_PATH.

    Args:
        csv_path: Caminho do arquivo CSV com as colunas 'tweet_text' e 'sentiment'.
        limit: Quantidade máxima de linhas para treinar (para datasets grandes).
        n_epochs: Quantidade de épocas de treino.
        batch_size: Tamanho dos lotes para atualização do modelo.

    Returns:
        dict: Mensagem e caminho do modelo salvo.
    """
    df = pd.read_csv(csv_path, delimiter=';')
    if not {'tweet_text', 'sentiment'}.issubset(df.columns):
        raise ValueError("CSV deve conter as colunas 'tweet_text' e 'sentiment'")

    df = df.dropna(subset=["tweet_text", "sentiment"]).head(limit)
    df['sentiment'] = df['sentiment'].astype(str).str.upper().str.strip()
    df['sentiment'] = df['sentiment'].map({'POSITIVO': 1, 'NEGATIVO': 0, '1': 1, '0': 0})
    df = df.dropna(subset=["sentiment"])
    df['text'] = df['tweet_text'].apply(preprocess)

    # Monta o dataset no formato spaCy
    training_data = []
    for text, label in zip(df['text'], df['sentiment']):
        cats = {'POSITIVO': label == 1, 'NEGATIVO': label == 0}
        training_data.append((text, cats.copy()))

    print(f"[INFO] Total de exemplos para treino: {len(training_data)}")

    # Criação e configuração do pipeline spaCy
    nlp = spacy.blank("pt")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("POSITIVO")
    textcat.add_label("NEGATIVO")

    nlp.begin_training()
    historico = []

    for epoca in range(n_epochs):
        random.shuffle(training_data)
        losses = {}
        for batch in spacy.util.minibatch(training_data, batch_size):
            docs = [nlp(text) for text, cats in batch]
            annotations = [{'cats': cats} for _, cats in batch]
            examples = [Example.from_dict(doc, ann) for doc, ann in zip(docs, annotations)]
            nlp.update(examples, losses=losses)
            historico.append(losses)
        print(f"[EPOCH {epoca+1}] Loss: {losses}")

    os.makedirs(MODEL_PATH, exist_ok=True)
    nlp.to_disk(MODEL_PATH)

    print(f"[SUCESSO] Modelo treinado e salvo em {MODEL_PATH}")
    return {"message": "Modelo treinado com sucesso", "path": MODEL_PATH}



# ✅ Predição com validação e debug
def predict_sentiment_by_parts(text: str):
    """
    Classifica o sentimento de cada parte/frase separada por vírgula,
    retornando lista de previsões, totais e as frases processadas.
    """
    try:
        nlp = spacy.load(MODEL_PATH)
    except Exception as e:
        return {"error": f"Modelo não treinado ou não encontrado: {e}"}

    partes = [frase.strip() for frase in text.split(",") if frase.strip()]
    previsoes_final = []

    for parte in partes:
        doc = nlp(preprocess(parte))
        if doc.cats.get("POSITIVO", 0) > doc.cats.get("NEGATIVO", 0):
            previsoes_final.append("positivo")
        else:
            previsoes_final.append("negativo")

    return {
        "frases": partes,
        "previsoes": previsoes_final,
        "total_positivo": previsoes_final.count("positivo"),
        "total_negativo": previsoes_final.count("negativo")
    }
