"""
main.py — FastAPI ครอบเครื่องยนต์ RAG (service.py) ให้เรียกจากภายนอกได้

รัน:  uvicorn main:app --host 0.0.0.0 --port 8000
docs: http://localhost:8000/docs

หมายเหตุสำคัญ (ข้อจำกัดของระบบนี้):
  - reranker รันบน GPU เครื่องเดียว → ควรรัน "worker เดียว" + คิว (อย่าใช้ --workers >1)
  - บูตครั้งแรกช้า (~นาที) เพราะอ่าน PDF + สร้าง BM25 ในหน่วยความจำ
  - endpoint/คีย์ อ่านจาก .env (ดู .env.example) — ไม่ hardcode
  - ป้องกันด้วย API key เบื้องต้น: ตั้ง env API_KEY แล้ว client ต้องส่ง header X-API-Key
    (ไม่ตั้ง = เปิดใช้ได้เลย เหมาะกับ dev — production ควรตั้ง)
"""
import json
import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

import rag
import service

_llm_cache: dict = {}


def get_llm(model: str | None):
    """สร้าง/แคช LLM client ต่อรุ่น (ค่าเริ่มต้น = rag.LLM_MODEL)"""
    name = model or rag.LLM_MODEL
    if name not in _llm_cache:
        rag.LLM_MODEL = name
        _llm_cache[name] = rag.build_llm()
    return _llm_cache[name]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # เปิด reranker เป็นค่าเริ่มต้น (สำคัญ — ตัวที่ดันมาตราที่ตรงคำถามจริงขึ้นมาก่อน)
    # ปิดได้ด้วย env RAG_RERANK=0
    rag.RERANK_ENABLED = os.environ.get("RAG_RERANK", "1") != "0"
    # โหลด/สร้างดัชนีครั้งเดียวตอนสตาร์ท (ครั้งแรกช้า — อ่าน PDF + build BM25)
    changed = rag.update_database()
    rag.build_vectorstore(force=changed)
    yield


app = FastAPI(title="Thai Law RAG API (ประมวลกฎหมายที่ดิน)", version="1.0", lifespan=lifespan)

# CORS — dev เปิดกว้าง; production จำกัด origin ตามจริง
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def require_api_key(x_api_key: str | None = Header(default=None)):
    """auth เบื้องต้น — บังคับเฉพาะเมื่อมี env API_KEY (ไม่ตั้ง = เปิด, เหมาะกับ dev)"""
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class ChatRequest(BaseModel):
    question: str
    history: list[dict] = []              # [{role: user|assistant, content: str}, ...]
    auto_group: bool = True              # True = LLM เลือกกลุ่มเอง
    groups: list[str] | None = None      # ใช้เมื่อ auto_group=False (กรองเอง)
    years: list[int] | None = None       # กรองปีเอกสาร (None = ทุกปี)
    model: str | None = None             # override LLM (None = ค่าเริ่มต้น)


@app.get("/health")
def health():
    return {"status": "ok", "chunks": len(rag._chunks), "model": rag.LLM_MODEL}


@app.get("/groups", dependencies=[Depends(require_api_key)])
def groups():
    return {"groups": service.list_groups()}


@app.get("/years", dependencies=[Depends(require_api_key)])
def years():
    return {"years": service.list_years()}


@app.get("/models", dependencies=[Depends(require_api_key)])
def models():
    return {"models": service.list_models()}


@app.post("/chat", dependencies=[Depends(require_api_key)])
def chat(req: ChatRequest):
    """ตอบคำถามแบบ streaming (SSE) — แต่ละบรรทัด 'data: {json}\\n\\n'
    event ตรงกับที่ service.answer_stream ส่งออก: stage / meta / token / reasoning / final"""
    llm = get_llm(req.model)
    all_groups = service.list_groups()

    def sse():
        try:
            for ev in service.answer_stream(
                llm, req.question, auto_group=req.auto_group, all_groups=all_groups,
                manual_groups=req.groups, year_filter=req.years, history=req.history,
                stream=True,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


class Feedback(BaseModel):
    rating: str                          # "up" | "down"
    question: str = ""
    answer: str = ""
    model: str = ""


@app.post("/feedback", dependencies=[Depends(require_api_key)])
def feedback(fb: Feedback):
    """เก็บ feedback ลง feedback.jsonl (บรรทัดละ 1 record)"""
    import datetime
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), **fb.model_dump()}
    path = os.path.join(os.path.dirname(__file__), "feedback.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}
