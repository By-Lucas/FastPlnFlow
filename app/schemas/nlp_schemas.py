from pydantic import BaseModel
from typing import Optional


class TextRequest(BaseModel):
    text: str


class SentimentRequest(BaseModel):
    filename: str
    context: Optional[str] = None


class EmotionRequest(BaseModel):
    text: str
