from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

from llm import llm_engine

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# chroma db呼び出し
kokubun_directory = "./docs/kokubun_chroma"
kokubun_db = Chroma(collection_name="langchain_store", persist_directory=kokubun_directory, embedding_function=embedding)

class RequestData(BaseModel):
    text: str

class LLMResponse(BaseModel):
    text: str
    title_1: str
    url_1: str
    title_2: str
    url_2: str
    title_3: str
    url_3: str
    title_4: str
    url_4: str

def ask_question(query: str) -> str:
    docs = llm_engine.get_as_retriever_answer(query, kokubun_db)
    answer = docs["result"]
    source_documents = docs["source_documents"]
    titles_urls = [(doc.metadata['title'], doc.metadata['url']) for doc in source_documents[:4]]
    return (answer, *sum(titles_urls, ()))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/kokubun")
async def run_llm(request_data: RequestData):
    answer, title_1, url_1, title_2, url_2, title_3, url_3, title_4, url_4 = ask_question(request_data.text)
    return LLMResponse(text=answer,
        title_1=title_1, url_1=url_1,
        title_2=title_2, url_2=url_2,
        title_3=title_3, url_3=url_3,
        title_4=title_4, url_4=url_4
    )