"""
main.py — FastAPI ครอบเครื่องยนต์ RAG (service.py) ให้เรียกจากภายนอกได้

รัน:  uvicorn main:app --host 0.0.0.0 --port 8000
docs: http://localhost:8000/docs        (ปิดได้ด้วย API_DOCS=0)

⚠️ ข้อจำกัดของระบบนี้ที่ต้องรู้ก่อนเอาไปให้คนอื่นใช้
  - reranker รันบน GPU เครื่องเดียว → ต้องรัน "worker เดียว" (อย่าใช้ --workers >1)
  - บูตครั้งแรกช้า (~นาที) เพราะอ่าน PDF + สร้าง BM25 ในหน่วยความจำ
  - endpoint/คีย์ อ่านจาก .env (ดู .env.example) — ไม่ hardcode
  - ฟิลด์ history (คำถามต่อเนื่อง) ยังไม่เคยวัดผล — ดูหมายเหตุที่ ChatRequest

── สัญญาของ API (อ่านก่อนแก้) ────────────────────────────────────────────────
ทุก endpoint อยู่ใต้ /v1 เพราะเมื่อมีคนใช้แล้ว รูปแบบข้อมูลจะกลายเป็นสัญญาที่แก้ตามใจไม่ได้
  เพิ่มฟิลด์ใหม่          = ปลอดภัย ของเดิมไม่พัง
  เปลี่ยนชื่อ/ลบ/เปลี่ยนชนิด = ของฝั่งผู้ใช้พังเงียบ ๆ → ต้องเปิด /v2 แทน แล้วคง /v1 ไว้
"""
import json
import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

import rag
import service

API_VERSION = "v1"

# ⚠️ ไม่ตั้ง API_KEY = เปิดให้ทุกคนเข้าฟรี ซึ่งเดิมเป็นค่าเริ่มต้นและลืมได้ง่ายมาก
#    ตอนนี้ต้องยอมรับความเสี่ยงเองด้วย ALLOW_NO_AUTH=1 ถึงจะสตาร์ทได้
_API_KEY = os.environ.get("API_KEY", "")
_ALLOW_NO_AUTH = os.environ.get("ALLOW_NO_AUTH") == "1"

# CORS — ไม่ใส่ = ไม่เปิดให้เว็บอื่นเรียก (ปลอดภัยกว่าเปิดกว้างเป็นค่าเริ่มต้น)
#   ตั้งเป็นรายการ origin คั่นด้วยจุลภาค เช่น  CORS_ORIGINS=https://a.example,https://b.example
_CORS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "").split(",") if o.strip()]

_llm_cache: dict = {}


def get_llm(model: "str | None"):
    """สร้าง/แคช LLM client ต่อรุ่น (ค่าเริ่มต้น = rag.LLM_MODEL)

    ⚠️ ห้ามเขียนทับ rag.LLM_MODEL — เดิมโค้ดตรงนี้ตั้ง rag.LLM_MODEL = name
       ทำให้ผู้ใช้คนหนึ่งส่ง model= มา แล้วค่าเริ่มต้นของ "ทุกคน" เปลี่ยนตามถาวร
       (บั๊กตระกูลเดียวกับ global ที่แก้ไปใน rag.retrieve)
    """
    name = model or rag.LLM_MODEL
    if name not in _llm_cache:
        _llm_cache[name] = rag.build_llm(name)
    return _llm_cache[name]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not _API_KEY and not _ALLOW_NO_AUTH:
        raise RuntimeError(
            "ยังไม่ได้ตั้ง env API_KEY — จะเปิด API โดยไม่มีการยืนยันตัวตนไม่ได้\n"
            "  ตั้งคีย์:            setx API_KEY \"<ความลับ>\"  แล้วเปิด terminal ใหม่\n"
            "  หรือถ้าตั้งใจเปิดโล่ง (เฉพาะเครื่องตัวเอง): ตั้ง ALLOW_NO_AUTH=1")
    if not _API_KEY:
        print("[!] เปิดโดยไม่มี API key (ALLOW_NO_AUTH=1) — อย่าใช้แบบนี้กับเครื่องที่คนอื่นเข้าถึงได้")
    if not _CORS:
        print("[i] ไม่ได้ตั้ง CORS_ORIGINS — เว็บจากโดเมนอื่นจะเรียกไม่ได้ (เรียกจาก server/CLI ได้ปกติ)")
    rag.RERANK_ENABLED = os.environ.get("RAG_RERANK", "1") != "0"
    changed = rag.update_database()
    rag.build_vectorstore(force=changed)

    # อุ่นเครื่องให้เสร็จ "ก่อน" บอกว่าพร้อม — ไม่งั้น /health ตอบ ok ตั้งแต่ยังไม่พร้อมจริง
    #   1. BM25 + ตัวตัดคำ: ถ้าไม่เรียก TOKENIZER_KIND จะยังเป็น "?" แล้ว /health
    #      รายงานสถานะเสื่อมไม่ได้เลยจนกว่าจะมีคำถามแรกเข้ามา
    #   2. reranker: โหลดตอนใช้ครั้งแรก วัดแล้วทำให้คำถามแรกช้า ~85 วินาที
    #      เทียบกับ ~8-20 วินาทีตอนอุ่นแล้ว — ผู้ใช้คนแรกไม่ควรเป็นคนจ่ายค่านี้
    rag._ensure_loaded()
    if rag.RERANK_ENABLED:
        print("[i] อุ่น reranker...")
        rag.rerank("ทดสอบ", [{"text": "ประมวลกฎหมายที่ดิน"}], 1)
    print(f"[i] พร้อมใช้งาน — ตัดคำด้วย {rag.TOKENIZER_KIND} · "
          f"rerank {'เปิด' if rag.RERANK_ENABLED else 'ปิด'}")
    yield


app = FastAPI(
    title="Thai Law RAG API (ประมวลกฎหมายที่ดิน)",
    version="1.0",
    lifespan=lifespan,
    docs_url="/docs" if os.environ.get("API_DOCS", "1") != "0" else None,
    redoc_url=None,
)

if _CORS:
    app.add_middleware(CORSMiddleware, allow_origins=_CORS,
                       allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"])


@app.exception_handler(Exception)
async def _unhandled(_req: Request, exc: Exception):
    """กันข้อความ error ดิบรั่วออกไป (อาจมี URL ของ endpoint/รายละเอียดภายในติดไปด้วย)
    ฝั่งเรายังเห็นของจริงใน log ตามปกติ"""
    print(f"[error] {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error",
                 "detail": "เกิดข้อผิดพลาดภายในระบบ ลองใหม่อีกครั้ง"})


def require_api_key(x_api_key: "str | None" = Header(default=None)):
    if _API_KEY and x_api_key != _API_KEY:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    # ⚠️ history ยังไม่เคยถูกวัดผล — ชุดคำถาม 120 ข้อเป็นคำถามเดี่ยวทั้งหมด
    #    ตัวเลข 99/100 ที่รายงานไว้จึงไม่ครอบคลุมการคุยต่อเนื่อง ใช้ได้แต่ยังไม่รับประกัน
    history: list[Turn] = Field(default_factory=list, max_length=20)
    auto_group: bool = True
    groups: "list[str] | None" = None
    years: "list[int] | None" = None
    model: "str | None" = None


router = APIRouter(prefix=f"/{API_VERSION}", dependencies=[Depends(require_api_key)])


@app.get("/health")
def health():
    """ไม่ต้องใช้คีย์ — ไว้ให้ตัวมอนิเตอร์เรียกเช็คว่าระบบยังอยู่

    tokenizer = "newmm" ปกติ · "bigram" แปลว่าไม่มี pythainlp แล้วการค้นด้วยคำหยาบลงมาก
    เอามาโชว์ตรงนี้เพราะเดิมรู้ได้จาก log ตอนเปิดระบบเท่านั้น ซึ่งเลื่อนหายไปแล้ว
    """
    degraded = rag.TOKENIZER_KIND == "bigram"
    return {"status": "degraded" if degraded else "ok", "version": API_VERSION,
            "chunks": len(rag._chunks), "model": rag.LLM_MODEL,
            "tokenizer": rag.TOKENIZER_KIND,
            "rerank": rag.RERANK_ENABLED,
            **({"warning": "ไม่มี pythainlp — ค้นด้วยคำทำงานแบบหยาบ (bigram)"} if degraded else {})}


@router.get("/groups")
def groups():
    return {"groups": service.list_groups()}


@router.get("/years")
def years():
    return {"years": service.list_years()}


@router.get("/models")
def models():
    return {"models": service.list_models()}


@router.post("/chat")
def chat(req: ChatRequest):
    """ตอบคำถามแบบ streaming (SSE) — แต่ละบรรทัด `data: {json}\\n\\n`

    ชนิดของ event ที่ส่งออก (ตรงกับ service.answer_stream):
      {"stage": str}    ความคืบหน้า เอาไว้โชว์ระหว่างรอ
      {"meta": {...}}   groups / n_sources / retrieval_mode / search_q
      {"token": str}    ข้อความคำตอบทีละชิ้น (ต่อกันเองฝั่ง client)
      {"final": {...}}  answer / chunks / groups_used / retrieval_mode / elapsed
      {"error": {...}}  เกิดข้อผิดพลาดกลางคัน — สตรีมจบเท่านี้

    ⚠️ retrieval_mode = "bm25_only" แปลว่าระบบค้นด้วยความหมายใช้ไม่ได้ชั่วคราว
       คำตอบยังใช้ได้แต่แม่นน้อยลง ควรแสดงให้ผู้ใช้เห็น (ตัว answer มีป้ายกำกับมาให้แล้ว)
    """
    if req.model and req.model not in service.list_models():
        raise HTTPException(status_code=400, detail="unknown model")
    llm = get_llm(req.model)
    all_groups = service.list_groups()

    def sse():
        try:
            for ev in service.answer_stream(
                llm, req.question, auto_group=req.auto_group, all_groups=all_groups,
                manual_groups=req.groups, year_filter=req.years,
                history=[t.model_dump() for t in req.history], stream=True,
            ):
                yield f"data: {json.dumps(ev, ensure_ascii=False)}\n\n"
        except Exception as e:
            # สตรีมเริ่มไปแล้วจึงเปลี่ยน HTTP status ไม่ได้ — ส่ง event error แทน
            # และไม่ส่งข้อความดิบออกไป (log ฝั่งเราเห็นของจริง)
            print(f"[error] /chat {type(e).__name__}: {e}")
            yield ("data: " + json.dumps(
                {"error": {"code": "stream_failed",
                           "detail": "ระบบตอบไม่สำเร็จกลางคัน ลองใหม่อีกครั้ง"}},
                ensure_ascii=False) + "\n\n")

    return StreamingResponse(sse(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


class Feedback(BaseModel):
    rating: Literal["up", "down"]
    question: str = Field(default="", max_length=2000)
    answer: str = Field(default="", max_length=20000)
    model: str = Field(default="", max_length=200)


@router.post("/feedback")
def feedback(fb: Feedback):
    """เก็บ feedback ลง feedback.jsonl (บรรทัดละ 1 record)"""
    import datetime
    rec = {"ts": datetime.datetime.now().isoformat(timespec="seconds"), **fb.model_dump()}
    path = os.path.join(os.path.dirname(__file__), "feedback.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"ok": True}


app.include_router(router)
