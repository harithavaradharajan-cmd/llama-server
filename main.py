import os
import asyncio
import json
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, WebSocket, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from llama_cpp import Llama
import requests

load_dotenv()

#MODEL_PATH = os.environ.get("LLAMA_MODEL_PATH", "/models/model.gguf")
CHAT_FORMAT = os.environ.get("LLAMA_CHAT_FORMAT", "llama-2")
DEFAULT_MAX_TOKENS = int(os.environ.get("DEFAULT_MAX_TOKENS", "512"))

# Download a small Llama model (Mistral 7B is faster and works great)
MODEL_NAME = "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_URL = "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf"
MODEL_PATH = f"/Users/apple/Downloads/llamaServer/models/{MODEL_NAME}"

print("📥 Downloading Llama model (this may take a few minutes)...")
if not os.path.exists(MODEL_PATH):
    response = requests.get(MODEL_URL, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(MODEL_PATH, 'wb') as f:
        downloaded = 0
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                downloaded += len(chunk)
                percent = (downloaded / total_size) * 100
                print(f"\r⏳ Progress: {percent:.1f}%", end="")
    print("\n✅ Model downloaded!")
else:
    print("✅ Model already exists!")

llm = Llama(model_path=MODEL_PATH, chat_format=CHAT_FORMAT)

app = FastAPI(title="FastAPI + Llama Chat Server", version="0.1")

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_tokens: Optional[int] = DEFAULT_MAX_TOKENS
    temperature: Optional[float] = 0.2
    top_p: Optional[float] = 0.95
    stop: Optional[List[str]] = None

def validate_messages(messages: List[Dict[str, Any]]):
    if not messages:
        raise HTTPException(status_code=400, detail="messages must be a non-empty array")
    for m in messages:
        if "role" not in m or "content" not in m:
            raise HTTPException(status_code=400, detail="each message must have role and content")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    message_dicts = [m.dict() for m in req.messages]
    validate_messages(message_dicts)
    resp = llm.create_chat_completion(
        messages=message_dicts,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        stop=req.stop,
    )
    return JSONResponse(content=resp)

@app.post("/v1/stream")
async def stream_chat(request: Request):
    body = await request.json()
    messages = body.get("messages")
    if messages is None:
        raise HTTPException(status_code=400, detail="`messages` field required")
    validate_messages(messages)
    max_tokens = body.get("max_tokens", DEFAULT_MAX_TOKENS)
    temperature = body.get("temperature", 0.2)
    top_p = body.get("top_p", 0.95)
    stop = body.get("stop")

    async def event_stream():
        stream_iter = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
        )
        try:
            for chunk in stream_iter:
                data = json.dumps(chunk, default=str)
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)
        except Exception as e:
            yield f"data: {{\"error\": \"{str(e)}\"}}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")

@app.websocket("/ws")
async def websocket_chat(ws: WebSocket):
    await ws.accept()
    try:
        raw = await ws.receive_text()
        payload = json.loads(raw)
        messages = payload.get("messages")
        validate_messages(messages)
        max_tokens = payload.get("max_tokens", DEFAULT_MAX_TOKENS)
        temperature = payload.get("temperature", 0.2)
        top_p = payload.get("top_p", 0.95)
        stop = payload.get("stop")

        stream_iter = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=True,
        )

        for chunk in stream_iter:
            await ws.send_text(json.dumps(chunk, default=str))
        await ws.send_text(json.dumps({"done": True}))
    except Exception as e:
        await ws.send_text(json.dumps({"error": str(e)}))
    finally:
        await ws.close()

@app.get("/")
async def root():
    return {"status": "ok", "model_path": MODEL_PATH}
