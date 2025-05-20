import os
import spacy
import string
import random
import pandas as pd
from spacy.training import Example
from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.metrics import confusion_matrix, accuracy_score

# Path to the trained model
MODEL_PATH = "models/sentiment_text_model"
nlp = spacy.load("pt_core_news_sm")

# List of stopwords and punctuation
stop_words = STOP_WORDS
punctuation = string.punctuation

# Preprocessing function (same logic as in Colab)
def preprocess_text(text):
    text = text.lower()
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in stop_words and token.text not in punctuation
    ]
    tokens = ' '.join([str(element) for element in tokens if not element.isdigit()])
    return tokens

# Model training function (same as in Colab)
def train_emotion_model_v2(txt_path='data/base_treinamento.txt'):
    df = pd.read_csv(txt_path, encoding='utf-8')
    df.columns = ["text", "emotion"]
    df.dropna(inplace=True)
    df['text'] = df['text'].apply(preprocess_text)

    final_base = []
    for text, emotion in zip(df['text'], df['emotion']):
        if emotion == 'alegria':
            cats = {'JOY': True, 'FEAR': False}
        elif emotion == 'medo':
            cats = {'JOY': False, 'FEAR': True}
        final_base.append([text, cats.copy()])

    model = spacy.blank('pt')
    textcat = model.add_pipe("textcat")
    textcat.add_label("JOY")
    textcat.add_label("FEAR")
    history = []

    model.begin_training()
    for epoch in range(1000):
        random.shuffle(final_base)
        losses = {}
        for batch in spacy.util.minibatch(final_base, 30):
            texts = [model(text) for text, entities in batch]
            annotations = [{'cats': entities} for text, entities in batch]
            examples = [Example.from_dict(doc, annotation) for doc, annotation in zip(texts, annotations)]
            model.update(examples, losses=losses)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Losses: {losses}")
            history.append(losses)

    os.makedirs(MODEL_PATH, exist_ok=True)
    model.to_disk(MODEL_PATH)

    return {"message": "Model successfully trained!", "path": MODEL_PATH}

# Direct prediction, same as Colab
def predict_emotion_by_parts(text: str):
    try:
        model = spacy.load(MODEL_PATH)
    except Exception as e:
        return {"error": f"Model not trained or not found: {e}"}

    parts = [phrase.strip() for phrase in text.split(",") if phrase.strip()]
    predictions = []

    for part in parts:
        doc = model(preprocess_text(part))
        if doc.cats["JOY"] > doc.cats["FEAR"]:
            predictions.append("joy")
        else:
            predictions.append("fear")

    return {
        "phrases": parts,
        "predictions": predictions,
        "total_joy": predictions.count("joy"),
        "total_fear": predictions.count("fear")
    }
