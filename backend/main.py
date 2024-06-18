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
persist_directory = "./docs/chroma"
db = Chroma(collection_name="langchain_store", persist_directory=persist_directory, embedding_function=embedding)

class RequestData(BaseModel):
    text: str

class LLMResponse(BaseModel):
    text: str
    title: str
    url: str

def ask_question(query: str) -> str:
    docs = llm_engine.get_as_retriever_answer(query, db)
    answer = docs["result"]
    source_documents = docs["source_documents"]
    answer_title = source_documents[0].metadata['title']
    answer_url = source_documents[0].metadata['url']
    return answer, answer_title, answer_url

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/send")
async def run_llm(request_data: RequestData):
    answer, title, url = ask_question(request_data.text)
    return LLMResponse(text=answer, title=title, url=url)