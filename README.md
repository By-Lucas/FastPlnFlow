# FastPlnFlow

[🇺🇸 Read in English](./README-EN.md)

> **API de Processamento de Linguagem Natural (PLN) moderna, rápida e plug-and-play!**

---

## 🚀 Tecnologias de Destaque

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal?logo=fastapi)
![Celery](https://img.shields.io/badge/Celery-Task%20Queue-green?logo=celery)
![RabbitMQ](https://img.shields.io/badge/RabbitMQ-Messaging-orange?logo=rabbitmq)
![Docker](https://img.shields.io/badge/Docker-DevContainer-blue?logo=docker)
![spaCy](https://img.shields.io/badge/spaCy-NLP-blueviolet?logo=spacy)
![NLTK](https://img.shields.io/badge/NLTK-Linguistics-yellow?logo=nltk)
![VSCode](https://img.shields.io/badge/VSCode-Ready-0078d7?logo=visual-studio-code)

---
- **Análise gramatical (POS tagging)**
- **Lematização**
- **Extração de entidades**
- **Sentimento em texto/tweet**
- **Nuvem de palavras**
- **Treinamento assíncrono de modelos com Celery/RabbitMQ**
- Pronto para desenvolvimento moderno (VSCode, DevContainer ou Docker Compose).

---

## 🚀 Instalação Local (Recomendado para Testes Rápidos)

```bash
git clone https://github.com/By-lucas/fastplnflow.git
cd fastplnflow

# Ambiente virtual Python (opcional, mas recomendado)
python -m venv .venv
source .venv/bin/activate  # ou .venv\Scripts\activate no Windows

# Instale as dependências
pip install -r requirements.txt

# Baixe recursos necessários (apenas uma vez)
python -m nltk.downloader rslp
python -m spacy download pt_core_news_sm
```

---

## ▶️ Execução Local

```bash
uvicorn app.main:app --reload
```

- **Instale o celery para executar**

Acesse a documentação interativa:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Rodando com Docker/DevContainer

### Pré-requisitos

- Docker instalado  
- VSCode com extensão "Remote - Containers" (DevContainers)

### Como rodar

1. Abra o projeto no VSCode
2. Clique no botão verde (canto inferior esquerdo) e selecione **“Reopen in Container”**
3. Aguarde subir o ambiente (pode demorar na primeira vez)
4. Acesse:  
   - **API:** [http://localhost:8000/docs](http://localhost:8000/docs)  
   - **RabbitMQ:** [http://localhost:15672](http://localhost:15672)  
     (Usuário: `fastplnflow` — Senha: `fastplnflow2025`)

#### Comandos para rodar manualmente dentro do DevContainer:

```bash
# Suba a API FastAPI (se não estiver rodando):
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Suba o worker do Celery (para treinos/tarefas assíncronas):
celery -A app.services.celery_worker.celery worker --loglevel=INFO
```
> Obs: normalmente o Docker Compose já sobe tudo automaticamente.

---

## 📚 Endpoints disponíveis

**Veja exemplos prontos no Swagger:**  
[http://localhost:8000/docs](http://localhost:8000/docs)

### 🔹 Processamento de texto via arquivo

- `/pos/` — Classe gramatical de cada palavra  
  (POST `/pos/?filename=meutexto.txt`)
- `/lemmatization/` — Lematização  
  (POST `/lemmatization/?filename=meutexto.txt`)
- `/stemming/` — Stemização  
  (POST `/stemming/?filename=meutexto.txt`)
- `/entities/` — Entidades nomeadas  
  (POST `/entities/?filename=meutexto.txt`)
- `/search/` — Busca termo no texto  
  (POST `/search/?filename=meutexto.txt&term=palavra`)
- `/wordcloud/` — Nuvem de palavras (retorna imagem base64/png)
- `/wikipedia/save/` — Salva artigo da Wikipedia  
  (GET `/wikipedia/save/?url=...&filename=wikipedia.txt`)
- `/save-text/` — Salva texto recebido (JSON) em arquivo .txt

### 🔹 Sentimento e Emoção (Tweets/texto curto)

- `/sentiment/tweets/train/`  
  POST `/sentiment/tweets/train/?limit=10000`  
  Treina modelo spaCy em background (Celery+RabbitMQ).
- `/sentiment/tweets/`  
  POST `/sentiment/tweets/?text=Texto aqui`  
  Classifica uma ou mais frases (separadas por vírgula) como POSITIVO ou NEGATIVO.
- `/sentiment/train/`  
  Treina modelo de emoções (alegria, medo, etc.) em background.
- `/sentiment/emotions/`  
  Prediz emoções no texto.

---

## 📦 Estrutura do Projeto

```
fastplnflow/
├── app/
│   ├── main.py
│   ├── routers/
│   ├── services/
│   ├── utils/
│   └── models/
├── requirements.txt
├── Dockerfile
├── data
├── docker-compose.yml
├── .devcontainer/
│   └── devcontainer.json
└── README.md
```

---

## ⚡️ Dicas de uso

- Todos os arquivos lidos/salvos são relativos ao diretório do container `/app`.
- Use a aba de arquivos do VSCode para subir/baixar `.txt` e `.png`.
- O modelo de sentimento treinado é salvo em `/models/sentiment_tweets_model`.
- Para grandes volumes, use sempre os endpoints de treinamento via Celery para não travar a API.

---

## 📬 Contribuições

Pull requests, issues e sugestões são super bem-vindos!

---

## 📄 Licença

MIT
![License: SR - Software Restrito](https://img.shields.io/badge/license-SR%20(Restricted)-red)

