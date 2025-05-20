import nltk
import spacy
import urllib.request
from bs4 import BeautifulSoup
from nltk.stem import RSLPStemmer
from spacy.matcher import PhraseMatcher

from app.utils.wordcloud_util import generate_wordcloud_base64


#nltk.download('rslp') # Descomente este codigo se deseja baixar o pacote sozinho ou rode manualmente se desejar
pln = spacy.load('pt_core_news_sm')
stemmer = RSLPStemmer()


def get_pos_tags(text: str):
    doc = pln(text)
    return [{"text": token.text, "pos": token.pos_} for token in doc]


def get_lemmatization(text: str):
    doc = pln(text)
    return [{"text": token.text, "lemma": token.lemma_} for token in doc]


def get_stemming(text: str):
    doc = pln(text)
    return [{"text": token.text, "stem": stemmer.stem(token.text)} for token in doc]


def get_entities(text: str):
    doc = pln(text)
    return [{"text": ent.text, "label": ent.label_} for ent in doc.ents]


def search_term_in_text(text: str, term: str):
    matcher = PhraseMatcher(pln.vocab)
    token_pesquisa = pln(term)
    matcher.add("SEARCH", [token_pesquisa])

    doc = pln(text)
    matches = matcher(doc)

    result = []
    for match_id, start, end in matches:
        result.append({
            "match": doc[start:end].text,
            "context": doc[start-5:end+10].text
        })
    return {"count": len(matches), "results": result}


def get_wikipedia_content(url: str):
    dados = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(dados, 'lxml')
    texto = ' '.join([p.text for p in soup.find_all('p')])
    return {"content": texto}


def generate_wordcloud(text: str):
    doc = pln(text.lower())
    palavras = [token.text for token in doc if not token.is_stop and token.is_alpha]
    return generate_wordcloud_base64(palavras)
