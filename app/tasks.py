from app.services.celery_worker import celery

@celery.task
def exemplo_tarefa(x, y):
    return x + y

# Exemplo real para treinar modelo em background
@celery.task
def treinar_sentimento_task(caminho_csv="data/Train50.csv", limit=100_000):
    from app.services.sentiment_tweets_service import train_model_tweets
    return train_model_tweets(csv_path=caminho_csv, limit=limit)

@celery.task
def train_emotion_model_v2_task(caminho_txt="data/base_treinamento.txt"):
    from app.services.sentiment_emotions_service import train_emotion_model_v2
    return train_emotion_model_v2(txt_path=caminho_txt)
