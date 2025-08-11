from fastapi import FastAPI
import os, logging

app = FastAPI()
logger = logging.getLogger("uvicorn")
_client = None  # 延遲建立

def get_client():
    global _client
    if _client is None:
        # 真的要用到再建立；建不成就丟給呼叫端處理
        from google import genai
        _client = genai.Client()
    return _client

@app.get("/health")
def health():
    return {"ok": True, "port": os.environ.get("PORT")}

@app.get("/chat")
def chat(q: str):
    client = get_client()
    return {"answer": "ok"}
