# backend/main.py
from __future__ import annotations
import os
import logging
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- 日誌 ---
logger = logging.getLogger("vertex-bot")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="Vertex AI Bot")

# CORS（開發階段先全開，之後請收斂網域）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ==== 環境變數設定（請在部署時用 --set-env-vars 或 Cloud Run console 設定）====
PROJECT_ID    = os.getenv("PROJECT_ID") or os.getenv("GOOGLE_CLOUD_PROJECT")
LOCATION      = os.getenv("REGION", "asia-east1")        # Vertex/Gemini 所在區域
DATA_STORE_ID = os.getenv("DATASTORE_ID")                # Discovery Engine DataStore ID
SEARCH_LOCATION = os.getenv("SEARCH_LOCATION", "global") # Discovery Engine 多半是 global
MODEL_NAME    = os.getenv("MODEL", "gemini-1.5-flash-002")
CONTEXT_TOPK  = int(os.getenv("CONTEXT_TOPK", "5"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.4"))

# 延遲初始化：用到才載入/建立
_vertex_inited = False
_gemini_model = None
_search_client = None
_search_serving_config = None


def init_vertex_once() -> None:
    """初始化 Vertex（Gemini）與 Discovery Engine（Vertex AI Search）。"""
    global _vertex_inited, _gemini_model, _search_client, _search_serving_config

    if _vertex_inited:
        return

    # ---- Gemini 初始化（Vertex AI SDK 的高階介面）----
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel

        if not PROJECT_ID:
            raise ValueError("缺少 PROJECT_ID/GOOGLE_CLOUD_PROJECT 環境變數")
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        _gemini_model = GenerativeModel(MODEL_NAME)
        logger.info("Initialized Vertex AI GenerativeModel: %s", MODEL_NAME)
    except Exception as e:
        logger.exception("初始化 Vertex AI 失敗")
        raise RuntimeError(f"初始化 Vertex AI 失敗：{e}")

    # ---- Vertex AI Search（Discovery Engine）初始化 ----
    # 使用 discoveryengine_v1 的 SearchServiceClient
    if DATA_STORE_ID:
        try:
            from google.cloud import discoveryengine_v1 as discovery

            _search_client = discovery.SearchServiceClient()
            _search_serving_config = _search_client.serving_config_path(
                project=PROJECT_ID,
                location=SEARCH_LOCATION,        # 通常為 "global"
                data_store=DATA_STORE_ID,
                serving_config="default_search",
            )
            logger.info("Initialized Discovery Engine search; data_store=%s location=%s",
                        DATA_STORE_ID, SEARCH_LOCATION)
        except Exception as e:
            logger.exception("初始化 Vertex AI Search (Discovery Engine) 失敗")
            # 非致命，沒有就只走純生成
            _search_client = None
            _search_serving_config = None

    _vertex_inited = True


def search_vertex_ai(query: str, top_k: int = CONTEXT_TOPK) -> List[str]:
    """呼叫 Vertex AI Search (Discovery Engine) 回傳文字片段列表。若未設定則回傳空清單。"""
    if not _search_client or not _search_serving_config:
        return []

    from google.cloud import discoveryengine_v1 as discovery

    request = discovery.SearchRequest(
        serving_config=_search_serving_config,
        query=query,
        page_size=top_k,
        query_expansion_spec=discovery.SearchRequest.QueryExpansionSpec(
            condition=discovery.SearchRequest.QueryExpansionSpec.Condition.AUTO
        ),
        spell_correction_spec=discovery.SearchRequest.SpellCorrectionSpec(
            mode=discovery.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )

    result_texts: List[str] = []
    resp = _search_client.search(request=request)

    for result in resp:
        # 優先用 snippet，其次用原文
        if result.document:
            snips = []
            if result.document.derived_struct_data:
                # 新格式（部分客體可能在 structured data）
                pass
            if result.document.derived_struct_data and "snippets" in result.document.derived_struct_data:
                # 不一定有這欄，保守處理
                snips = result.document.derived_struct_data.get("snippets", [])
            # 通用 snippet 欄位
            if result.document.snippets:
                snips.extend([s.snippet for s in result.document.snippets if getattr(s, "snippet", "")])

            # 文件原文（有時候 snippets 為空）
            page_content = ""
            if result.document.content:
                page_content = result.document.content

            # 整理文字
            if snips:
                result_texts.append(" ".join(snips)[:2000])
            elif page_content:
                result_texts.append(page_content[:2000])

    return result_texts


def build_prompt(user_question: str, contexts: List[str]) -> str:
    """把檢索到的片段組成提示詞，交給 Gemini。"""
    ctx_block = "\n\n---\n\n".join(contexts) if contexts else "（無檢索到的相關資料）"
    return (
        "你是企業內部 FAQ/法規助理，請根據提供的資料回答問題；"
        "若資料不足，請明確說明不足之處並給出可能的下一步。\n\n"
        f"【使用者問題】\n{user_question}\n\n"
        f"【參考資料片段】\n{ctx_block}\n\n"
        "請用繁體中文、簡潔條列回答；必要時標注資料來源關鍵詞即可。"
    )


class ChatIn(BaseModel):
    query: str
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    model: Optional[str] = None


@app.get("/health")
def health():
    return {
        "ok": True,
        "port": os.environ.get("PORT"),
        "project": PROJECT_ID,
        "region": LOCATION,
        "model": MODEL_NAME,
        "search_enabled": bool(DATA_STORE_ID),
    }


@app.post("/chat")
def chat(body: ChatIn):
    if not body.query or not body.query.strip():
        raise HTTPException(status_code=400, detail="query 不可為空")

    # 初始化（只做一次；失敗會回 500）
    init_vertex_once()

    # 1) 檢索
    k = body.top_k or CONTEXT_TOPK
    contexts = []
    try:
        contexts = search_vertex_ai(body.query.strip(), top_k=k)
    except Exception as e:
        # 檢索失敗不致命，記 log、照樣走純生成
        logger.exception("Vertex AI Search 失敗：%s", e)

    # 2) 組提示
    prompt = build_prompt(body.query.strip(), contexts)

    # 3) 生成
    try:
        model_name = body.model or MODEL_NAME
        temperature = body.temperature if body.temperature is not None else TEMPERATURE

        # 使用 vertexai.generative_models 的 generate_content
        # 語法在不同版位會略有差異，這裡使用 messages/parts 單純文字
        result = _gemini_model.generate_content(
            [prompt],
            generation_config={"temperature": float(temperature)},
        )
        text = getattr(result, "text", "") or str(result)
    except Exception as e:
        logger.exception("Gemini 生成失敗")
        raise HTTPException(status_code=502, detail=f"Gemini 生成失敗：{e}")

    return {
        "model": model_name,
        "temperature": temperature,
        "context_used": len(contexts),
        "answer": text.strip(),
        "debug": {
            "has_project": bool(PROJECT_ID),
            "has_datastore": bool(DATA_STORE_ID),
        },
    }
