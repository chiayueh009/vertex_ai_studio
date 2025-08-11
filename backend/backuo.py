"""
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from contextlib import asynccontextmanager
from google import genai
from google.genai.types import GenerateContentConfig, Tool, Retrieval, VertexAISearch

load_dotenv() 

PROJECT_ID = os.getenv("PROJECT_ID")
DATASTORE_ID = os.getenv("DATASTORE_ID")
REGION = os.getenv("REGION", "global")
MODEL = os.getenv("MODEL", "gemini-1.5-flash")

DATASTORE_PATH = f"projects/wondersone/locations/global/collections/default_collection/dataStores/company-knowledge-base_1754880489912"

app = FastAPI()
client = None

class ChatIn(BaseModel):
    query: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    client = genai.Client()
    yield

app = FastAPI(lifespan=lifespan)

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/chat")
def chat(body: ChatIn):
    resp = client.models.generate_content(
        model=MODEL,
        contents=body.query,
        config=GenerateContentConfig(
            tools=[Tool(
                retrieval=Retrieval(
                    vertex_ai_search=VertexAISearch(datastore=DATASTORE_PATH)
                )
            )]
        ),
    )
    # 最簡：回文字（Grounding 來源可之後再解析）
    return {"answer": resp.text}
"""