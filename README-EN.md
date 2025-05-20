
# FastPlnFlow

[🇧🇷 Read in Portuguese](./README.md)

> **A modern, fast, and plug-and-play Natural Language Processing (NLP) API for Portuguese!**

---

## 🚀 Main Technologies

![Python](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-Framework-teal?logo=fastapi)
![Celery](https://img.shields.io/badge/Celery-Task%20Queue-green?logo=celery)
![RabbitMQ](https://img.shields.io/badge/RabbitMQ-Messaging-orange?logo=rabbitmq)
![Docker](https://img.shields.io/badge/Docker-DevContainer-blue?logo=docker)
![spaCy](https://img.shields.io/badge/spaCy-NLP-blueviolet?logo=spacy)
![NLTK](https://img.shields.io/badge/NLTK-Linguistics-yellow?logo=nltk)
![VSCode](https://img.shields.io/badge/VSCode-Ready-0078d7?logo=visual-studio-code)

---

## Quick Overview

Python API for advanced NLP in Portuguese, with:
- POS tagging (grammatical analysis)
- Lemmatization
- Named Entity Recognition (NER)
- Sentiment & emotion analysis
- Word cloud generation
- Async model training via Celery/RabbitMQ
- Ready for Docker/DevContainer/VSCode

---

## 🚀 Local Installation (Recommended for Quick Tests)

```bash
git clone https://github.com/By-lucas/fastplnflow.git
cd fastplnflow

# (Optional but recommended) Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Download required resources (one time only)
python -m nltk.downloader rslp
python -m spacy download pt_core_news_sm
```

---

## ▶️ Local Execution

```bash
uvicorn app.main:app --reload
```

Access the interactive docs at:  
[http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🐳 Running with Docker/DevContainer

### Prerequisites

- Docker installed  
- VSCode with "Remote - Containers" (DevContainers) extension

### How to run

1. Open the project in VSCode
2. Click the green button (bottom left) and select **“Reopen in Container”**
3. Wait for the environment to set up (may take a few minutes the first time)
4. Access:  
   - **API:** [http://localhost:8000/docs](http://localhost:8000/docs)  
   - **RabbitMQ:** [http://localhost:15672](http://localhost:15672)  
     (User: `fastplnflow` — Password: `fastplnflow2025`)

#### Commands to run manually inside the DevContainer:

```bash
# Start the FastAPI server (if not running):
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start the Celery worker (for async training/tasks):
celery -A app.services.celery_worker.celery worker --loglevel=INFO
```
> Note: Docker Compose usually starts everything automatically.

---

## 📚 Available Endpoints

**Check ready-to-use examples in Swagger:**  
[http://localhost:8000/docs](http://localhost:8000/docs)

### 🔹 File-based text processing

- `/pos/` — POS tagging (POST `/pos/?filename=mytext.txt`)
- `/lemmatization/` — Lemmatization (POST `/lemmatization/?filename=mytext.txt`)
- `/stemming/` — Stemming (POST `/stemming/?filename=mytext.txt`)
- `/entities/` — Named Entity Recognition (POST `/entities/?filename=mytext.txt`)
- `/search/` — Search for a term (POST `/search/?filename=mytext.txt&term=word`)
- `/wordcloud/` — Word cloud (returns base64/png image)
- `/wikipedia/save/` — Save Wikipedia article  
  (GET `/wikipedia/save/?url=...&filename=wikipedia.txt`)
- `/save-text/` — Save text received (JSON) as .txt file

### 🔹 Sentiment & Emotion (Tweets/short text)

- `/sentiment/tweets/train/`  
  POST `/sentiment/tweets/train/?limit=10000`  
  Train a spaCy model in the background (Celery+RabbitMQ).
- `/sentiment/tweets/`  
  POST `/sentiment/tweets/?text=Your text here`  
  Classify one or more sentences (comma-separated) as POSITIVE or NEGATIVE.
- `/sentiment/train/`  
  Train emotion model (joy, fear, etc.) in the background.
- `/sentiment/emotions/`  
  Predict emotions in the text.

---

## 📦 Project Structure

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
├── docker-compose.yml
├── .devcontainer/
│   └── devcontainer.json
└── README.md
```

---

## ⚡️ Usage Tips

- All files are read/saved relative to the container directory `/app`.
- Use VSCode file explorer to upload/download `.txt` and `.png`.
- The trained sentiment model is saved at `/models/sentiment_tweets_model`.
- For heavy loads, always use the training endpoints via Celery to keep the API responsive.

---

## 📬 Contributions

Pull requests, issues and suggestions are super welcome!

---

## 📄 License

MIT
![License: SR - Software Restrito](https://img.shields.io/badge/license-SR%20(Restricted)-red)
