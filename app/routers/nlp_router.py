import io
import base64

from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse

from app.services import nlp_service
from app.schemas.nlp_schemas import TextRequest


router = APIRouter()

def read_file_text(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()

@router.post("/pos/", summary="Análise gramatical (POS tagging)", description="Retorna a classe gramatical de cada palavra no texto (substantivo, verbo, etc).")
def pos_tagging(filename: str = Query(..., description="Nome do arquivo .txt")):
    text = read_file_text(filename)
    return nlp_service.get_pos_tags(text)

@router.post("/lemmatization/", summary="Lematização", description="Retorna a forma base (lema) de cada palavra no texto.")
def lemmatization(filename: str = Query(..., description="Nome do arquivo .txt")):
    text = read_file_text(filename)
    return nlp_service.get_lemmatization(text)

@router.post("/stemming/", summary="Stemização (radical)", description="Retorna o radical de cada palavra com o stemmer RSLP do NLTK.")
def stemming(filename: str = Query(..., description="Nome do arquivo .txt")):
    text = read_file_text(filename)
    return nlp_service.get_stemming(text)

@router.post("/entities/", summary="Entidades nomeadas (NER)", description="Extrai nomes de pessoas, organizações, datas, locais e outras entidades do texto.")
def named_entities(filename: str = Query(..., description="Nome do arquivo .txt")):
    text = read_file_text(filename)
    return nlp_service.get_entities(text)

@router.post("/search/", summary="Busca de termo com contexto", description="Procura um termo específico no texto e retorna o contexto ao redor dele.")
def search_phrase(
    term: str = Query(..., description="Termo a ser procurado"),
    filename: str = Query(..., description="Nome do arquivo .txt")
):
    text = read_file_text(filename)
    return nlp_service.search_term_in_text(text, term)

@router.post("/wordcloud/", summary="Geração de nuvem de palavras", description="Gera uma imagem base64 com a nuvem de palavras do texto, excluindo stopwords.")
def wordcloud_generator(filename: str = Query(..., description="Nome do arquivo .txt")):
    text = read_file_text(filename)
    result = nlp_service.generate_wordcloud(text)
    img_data = base64.b64decode(result["wordcloud_image"].split(",")[1])
    return StreamingResponse(io.BytesIO(img_data), media_type="image/png", headers={"Content-Disposition": "inline; filename=wordcloud.png"})

@router.get("/wikipedia/save/", summary="Salvar conteúdo da Wikipedia", description="Busca o conteúdo de uma URL da Wikipedia e salva em um arquivo local.")
def save_wikipedia_article(
    url: str = Query(..., description="URL do artigo da Wikipédia"),
    filename: str = Query("wikipedia_text.txt", description="Nome do arquivo para salvar")
):
    content_dict = nlp_service.get_wikipedia_content(url)
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content_dict["content"])
    return {"message": f"Conteúdo salvo em {filename}"}

@router.post("/save-text/", summary="Salvar texto enviado", description="Recebe texto via JSON e salva localmente como .txt.")
def save_text_to_file(
    request: TextRequest,
    filename: str = Query("texto_salvo.txt", description="Nome do arquivo para salvar")
):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(request.text)
    return {"message": f"Texto salvo em {filename}"}