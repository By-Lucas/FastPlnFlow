from fastapi import FastAPI
from app.routers import nlp_router
from app.routers import sentiment_routers


app = FastAPI(title="FastPlnFlow - API de PLN com FastAPI")

# Inclui as rotas de NLP
app.include_router(nlp_router.router, prefix="/nlp", tags=["NLP"])
app.include_router(sentiment_routers.router, prefix="/sentiment", tags=["SENTIMENTO / EMOÇÕES"])


@app.get("/")
async def root():
    return {"message": "🚀 FastPlnFlow API ativa! Vá para /docs para ver a documentação."}
