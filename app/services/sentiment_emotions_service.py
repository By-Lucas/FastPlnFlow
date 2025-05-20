import os
import spacy
import string
import random
import pandas as pd
from spacy.training import Example
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.metrics import confusion_matrix, accuracy_score


# Caminho do modelo treinado no Colab
MODEL_PATH = "models/sentiment_text_model"
pln = spacy.load("pt_core_news_sm")

# Lista de stopwords e pontuação
stop_words = STOP_WORDS
pontuacoes = string.punctuation

# Pré-processamento igual ao do Colab
def preprocessamento(texto):
    texto = texto.lower()
    documento = pln(texto)
    lista = [token.lemma_ for token in documento if token.text not in stop_words and token.text not in pontuacoes]
    lista = ' '.join([str(elemento) for elemento in lista if not elemento.isdigit()])
    return lista

# Treinamento do modelo (mesmo que o Colab)
def train_emotion_model_v2(txt_path='data/base_treinamento.txt'):
    df = pd.read_csv(txt_path, encoding='utf-8')
    df.columns = ["texto", "emocao"]
    df.dropna(inplace=True)
    df['texto'] = df['texto'].apply(preprocessamento)

    base_final = []
    for texto, emocao in zip(df['texto'], df['emocao']):
        if emocao == 'alegria':
            dic = {'ALEGRIA': True, 'MEDO': False}
        elif emocao == 'medo':
            dic = {'ALEGRIA': False, 'MEDO': True}
        base_final.append([texto, dic.copy()])

    modelo = spacy.blank('pt')
    textcat = modelo.add_pipe("textcat")
    textcat.add_label("ALEGRIA")
    textcat.add_label("MEDO")
    historico = []

    modelo.begin_training()
    for epoca in range(1000):
        random.shuffle(base_final)
        losses = {}
        for batch in spacy.util.minibatch(base_final, 30):
            textos = [modelo(texto) for texto, entities in batch]
            annotations = [{'cats': entities} for texto, entities in batch]
            examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(textos, annotations)]
            modelo.update(examples, losses=losses)
        if epoca % 100 == 0:
            print(f"Epoch {epoca}, Losses: {losses}")
            historico.append(losses)

    os.makedirs(MODEL_PATH, exist_ok=True)
    modelo.to_disk(MODEL_PATH)

    return {"message": "Modelo 100% treinado com sucesso", "path": MODEL_PATH}

# Predição direta, com mesmo comportamento do Colab
def predict_emotion_by_parts(text: str):
    try:
        modelo = spacy.load(MODEL_PATH)
    except Exception as e:
        return {"error": f"Modelo não treinado ou não encontrado: {e}"}

    partes = [frase.strip() for frase in text.split(",") if frase.strip()]
    previsoes_final = []

    for parte in partes:
        doc = modelo(preprocessamento(parte))
        if doc.cats["ALEGRIA"] > doc.cats["MEDO"]:
            previsoes_final.append("alegria")
        else:
            previsoes_final.append("medo")

    return {
        "frases": partes,
        "previsoes": previsoes_final,
        "total_alegria": previsoes_final.count("alegria"),
        "total_medo": previsoes_final.count("medo")
    }
