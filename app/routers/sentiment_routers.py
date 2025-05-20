from fastapi import APIRouter, Query

from app.tasks import train_emotion_model_v2_task, treinar_sentimento_task
from app.schemas.nlp_schemas import EmotionRequest
from app.services import sentiment_emotions_service, sentiment_tweets_service


router = APIRouter()

def read_file_text(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


@router.post("/sentiment/emotions/", summary="Classificação de emoção", description="Classifica a emoção do texto com base em modelo treinado (alegria, medo, etc).")
def emotion_analysis(request: EmotionRequest):
    return sentiment_emotions_service.predict_emotion_by_parts(request.text)

@router.post("/sentiment/train/", summary="Treinamento do modelo de sentimento", description="Treina um modelo spaCy para classificar sentimentos (ex: positivo, negativo, neutro) com base em dados anotados.")
async def train_sentiment_model():
    task = train_emotion_model_v2_task.delay("data/base_treinamento.txt")
    return {"task_id": task.id, "status": "Treinamento de sentimentos de texto em processamento "}

@router.post(
    "/sentiment/tweets/train/",
    summary="Treinar modelo de sentimento com Tweets",
    description="Treina um modelo spaCy usando dados rotulados de Tweets presentes no arquivo `data/Train50.csv`. O modelo identifica sentimentos como 'POSITIVO' e 'NEGATIVO'."
)
async def train_tweets(limit: int = 100_000):
    task = treinar_sentimento_task.delay("data/Train50.csv", limit)
    return {"task_id": task.id, "status": "Treinamento de sentimentos de Tweets em processamento"}

@router.post(
    "/sentiment/tweets/",
    summary="Classificar sentimento de um tweet",
    description="Classifica o sentimento de um texto curto (ex: tweet) com base no modelo treinado. Retorna se é POSITIVO ou NEGATIVO com o score de confiança."
)
def classify_tweet_sentiment(
    text: str = Query(..., description="Texto do tweet ou frase a ser classificada."),
    context: str = Query("Comentário", description="Contexto opcional (ex: produto, marca, evento)")
):
    return sentiment_tweets_service.predict_sentiment_by_parts(text)
