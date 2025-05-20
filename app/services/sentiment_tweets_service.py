import re
import os
import spacy
import random
import string
import pandas as pd
from spacy.training import Example
from spacy.lang.pt.stop_words import STOP_WORDS

# Path to the saved model
MODEL_PATH = "models/sentiment_tweets_model"
nlp_pt = spacy.load("pt_core_news_sm")  # Lemmatization

# Preprocessing (same as Colab)
def preprocess(text):
    text = text.lower()
    text = re.sub(r"@[A-Za-z0-9$-_@.&+]+", ' ', text)
    text = re.sub(r"https?://[A-Za-z0-9./]+", ' ', text)
    text = re.sub(r" +", ' ', text)

    emotion_map = {
        ':)': 'positiveemotion',
        ':d': 'positiveemotion',
        ':(': 'negativeemotion'
    }
    for emoji, replacement in emotion_map.items():
        text = text.replace(emoji, replacement)

    doc = nlp_pt(text)
    lemmas = [token.lemma_ for token in doc]
    lemmas = [word for word in lemmas if word not in STOP_WORDS and word not in string.punctuation]
    lemmas = ' '.join([str(word) for word in lemmas if not word.isdigit()])
    return lemmas

def train_model_tweets(
    csv_path: str = "data/Train50.csv",
    limit: int = 100_000,
    n_epochs: int = 5,
    batch_size: int = 512
) -> dict:
    """
    Trains a spaCy TextCat model for sentiment classification using labeled CSV data.
    The model is saved to MODEL_PATH.

    Args:
        csv_path: Path to the CSV file with 'tweet_text' and 'sentiment' columns.
        limit: Maximum number of rows to use for training (for large datasets).
        n_epochs: Number of training epochs.
        batch_size: Batch size for model updates.

    Returns:
        dict: Message and path of the saved model.
    """
    df = pd.read_csv(csv_path, delimiter=';')
    if not {'tweet_text', 'sentiment'}.issubset(df.columns):
        raise ValueError("CSV must contain 'tweet_text' and 'sentiment' columns")

    df = df.dropna(subset=["tweet_text", "sentiment"]).head(limit)
    df['sentiment'] = df['sentiment'].astype(str).str.upper().str.strip()
    df['sentiment'] = df['sentiment'].map({'POSITIVO': 1, 'NEGATIVO': 0, '1': 1, '0': 0})
    df = df.dropna(subset=["sentiment"])
    df['text'] = df['tweet_text'].apply(preprocess)

    # Build the dataset in spaCy format
    training_data = []
    for text, label in zip(df['text'], df['sentiment']):
        cats = {'POSITIVE': label == 1, 'NEGATIVE': label == 0}
        training_data.append((text, cats.copy()))

    print(f"[INFO] Total training examples: {len(training_data)}")

    # Create and configure the spaCy pipeline
    nlp = spacy.blank("pt")
    textcat = nlp.add_pipe("textcat")
    textcat.add_label("POSITIVE")
    textcat.add_label("NEGATIVE")

    nlp.begin_training()
    history = []

    for epoch in range(n_epochs):
        random.shuffle(training_data)
        losses = {}
        for batch in spacy.util.minibatch(training_data, batch_size):
            docs = [nlp(text) for text, cats in batch]
            annotations = [{'cats': cats} for _, cats in batch]
            examples = [Example.from_dict(doc, ann) for doc, ann in zip(docs, annotations)]
            nlp.update(examples, losses=losses)
            history.append(losses)
        print(f"[EPOCH {epoch+1}] Loss: {losses}")

    os.makedirs(MODEL_PATH, exist_ok=True)
    nlp.to_disk(MODEL_PATH)

    print(f"[SUCCESS] Model trained and saved at {MODEL_PATH}")
    return {"message": "Model trained successfully", "path": MODEL_PATH}


# ✅ Prediction with validation and debug
def predict_sentiment_by_parts(text: str):
    """
    Classifies the sentiment of each part/phrase separated by commas,
    returning a list of predictions, totals, and processed phrases.
    """
    try:
        nlp = spacy.load(MODEL_PATH)
    except Exception as e:
        return {"error": f"Model not trained or not found: {e}"}

    parts = [phrase.strip() for phrase in text.split(",") if phrase.strip()]
    final_predictions = []

    for part in parts:
        doc = nlp(preprocess(part))
        if doc.cats.get("POSITIVE", 0) > doc.cats.get("NEGATIVE", 0):
            final_predictions.append("positive")
        else:
            final_predictions.append("negative")

    return {
        "phrases": parts,
        "predictions": final_predictions,
        "total_positive": final_predictions.count("positive"),
        "total_negative": final_predictions.count("negative")
    }
