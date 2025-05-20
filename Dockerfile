FROM python:3.11-slim

WORKDIR /app

# Instala dependências
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Baixa o modelo spaCy após instalar o spaCy
RUN python -m spacy download pt_core_news_sm

# Copia código fonte
COPY ./app /app

# Exponha portas (8000 para API, 5555 para Flower monitor)
EXPOSE 8000 5555

# Comando padrão: não starta nada aqui, quem faz é o docker-compose
CMD ["bash"]
