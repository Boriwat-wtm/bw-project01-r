"""
Thai Law RAG — ประมวลกฎหมายที่ดิน พ.ศ. ๒๔๙๗ (ราชกิจจานุเบกษา / สำนักงานคณะกรรมการกฤษฎีกา)

สถาปัตยกรรม: Direct Hybrid RAG บน PDF กฎหมายไทยใน data/
  1. โหลด PDF ด้วย PyMuPDF -> ตัด header/footer ซ้ำ -> chunk ตามขอบ "มาตรา/ข้อ"
     (หน้าสแกน = ImgPDF ตกไปเข้า OCR ภาษาไทยแบบ local อัตโนมัติ — ดู ocr.py)
  2. ฝังลง Chroma (semantic + metadata/กรอง group,ปี ในตัว) + ดัชนี BM25 (ตัดคำไทยด้วย newmm)
  3. ตอบคำถาม: retrieve แบบ hybrid (semantic + BM25 รวมด้วย RRF) -> rerank -> ส่ง context ให้ LLM

⚠️ เอกสารชุดนี้มีทั้งฉบับดั้งเดิม / ฉบับแก้ไขเพิ่มเติม / ฉบับรวมสะสมหลายเวอร์ชัน
   มาตราเดียวกันจึงซ้ำข้ามไฟล์ — retrieve() ยุบเหลือฉบับใหม่สุดให้ (ดู _dedupe_versions)

หมายเหตุ: ไม่ใช้ tool-calling agent — เพราะทุกคำถามต้อง retrieve เสมอ
การ retrieve ตรง ๆ เสถียรกว่า (เลี่ยงปัญหา model ไม่เรียก tool / ตอบว่าง / 400 error)
"""
import os
import re
import sys
import json
import time
import hashlib
from typing import Any, Callable, Optional

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# โหลดค่าลับ/เฉพาะเครื่อง (endpoint, API key) จากไฟล์ .env ที่ไม่เข้า git
# ทำให้ไม่ต้อง hardcode IP/คีย์ในโค้ด (ดู .env.example) — ไม่ override env ที่ตั้งไว้แล้ว
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ── config ─────────────────────────────────────────────────────────────────────
# ⚠️ endpoint/คีย์ ทั้งหมดอ่านจาก env (.env) — ห้าม hardcode IP/คีย์ (นี่คือ git repo)
#    ตั้งค่าจริงในไฟล์ .env (ดู .env.example) ไฟล์นั้นไม่เข้า git
# Embeddings: default = text-embedding-ada-002 ผ่าน endpoint เดียวกับ chat
# สลับกลับเป็น Ollama paraphrase-multilingual ได้ผ่าน env EMBED_MODEL
# ⚠️ เปลี่ยน EMBED_MODEL = มิติ vector เปลี่ยน → ต้อง rebuild Chroma collection ใหม่ (build_vectorstore force=True)
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-ada-002")

# Chat LLM ผ่าน OpenAI-compatible endpoint (ตั้ง LLM_BASE_URL + LLM_API_KEY ใน .env)
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "google/gemma-4-26b-a4b-it")  # เสถียรกว่า qwen

HERE = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(HERE, "chroma_db")          # Chroma persistent store (vectors + metadata)
COLLECTION_NAME = "thai_law"                            # ชื่อ collection ใน Chroma
HASH_FILE = os.path.join(HERE, "file_hashes.json")

# โยนเอกสารลงโฟลเดอร์นี้ (โฟลเดอร์ย่อย = group) รองรับหลายชนิดไฟล์
DATA_DIR = os.path.join(HERE, "data")
SUPPORTED_EXTS = {".pdf", ".txt", ".md", ".csv"}
DEFAULT_GROUP = "default"

# เอกสารอ้างอิงข้ามข้อกันสูง -> chunk ใหญ่ + overlap กว้าง กันบริบทขาด
# ⚠️ เปลี่ยน CHUNK_SIZE/OVERLAP = จำนวน/ขอบเขต chunk เปลี่ยน → ต้อง rebuild index (build_vectorstore force=True)
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))      # ตัวอักษรต่อ chunk (โดยประมาณ)
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "250")) # carry-over กันกฎถูกหั่นกลางข้อ
OCR_MIN_CHARS = int(os.environ.get("OCR_MIN_CHARS", "25"))  # หน้าที่ข้อความน้อยกว่านี้ = สแกน → OCR
TOP_K = 9              # chunk สุดท้ายที่ส่งให้ LLM (multi-query RRF-pooled → 9 ก้อน ~ครึ่งหน้า/ก้อน)

# ── Cross-encoder reranker (เปิดด้วย env RAG_RERANK=1 หรือ flag --rerank) ─────────
# pipeline: retrieve กว้าง (RRF) → เก็บ top-N → cross-encoder ให้คะแนน (query,chunk) → top-K
# แก้ปัญหา multi-query เจือจาง: RRF นับโหวต ส่วน reranker วัดความเกี่ยวข้องจริง
RERANK_ENABLED = os.environ.get("RAG_RERANK") == "1"
RERANK_MODEL   = os.environ.get("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")  # multilingual (ไทย+อังกฤษ)
RERANK_TOP_N   = int(os.environ.get("RERANK_TOP_N", "50"))  # candidate ที่ส่งให้ reranker ก่อนตัดเหลือ K (GPU รับไหว)

# globals (เซ็ตโดย build_vectorstore / _ensure_loaded)
_chroma_client = None          # chromadb.PersistentClient
_collection = None             # chromadb Collection (semantic + metadata, กรอง group ในตัว)
_embeddings: "OllamaEmbeddings | OpenAIEmbeddings | None" = None
_bm25: BM25Okapi | None = None
_chunks: list[dict] = []           # index-aligned กับ _bm25 corpus
_chunk_by_id: dict[str, dict] = {}


# ── helpers ──────────────────────────────────────────────────────────────────
def hash_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def load_hashes() -> dict:
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_hashes(hashes: dict) -> None:
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


# เลขไทย ๐-๙ -> อารบิก: ตัวบทพิมพ์ "มาตรา ๙" แต่คนถาม "มาตรา 9" — normalize ให้ตรงกัน
THAI_DIGITS = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

_th_tokenizer = None


def _thai_words(s: str) -> list[str]:
    """ตัดคำไทยด้วย pythainlp (newmm) — ไม่มี lib ก็ถอยไปใช้ bigram ตัวอักษร
    (หยาบกว่า แต่ยังค้นเจอ ดีกว่าปล่อยให้ BM25 ไม่มี token ไทยเลย)"""
    global _th_tokenizer
    if _th_tokenizer is None:
        try:
            from pythainlp.tokenize import word_tokenize
            _th_tokenizer = lambda t: word_tokenize(t, engine="newmm", keep_whitespace=False)
        except Exception:
            print("  [bm25] ไม่พบ pythainlp — ใช้ bigram ตัวอักษรแทน (ค้นไทยหยาบลง)")
            _th_tokenizer = lambda t: [t[i:i + 2] for i in range(len(t) - 1)] or [t]
    return _th_tokenizer(s)


def tokenize(text: str) -> list[str]:
    """ตัดคำสำหรับ BM25 — ไทยตัดคำจริง (newmm), อังกฤษ/ตัวเลขแยกกลุ่ม
    normalize เลขไทยก่อน → ถาม "มาตรา 9" ค้นเจอ "มาตรา ๙" """
    text = text.translate(THAI_DIGITS).lower()
    out: list[str] = []
    for m in re.finditer(r"[฀-๿]+|[a-z]+|[0-9]+", text):
        s = m.group()
        out.extend(_thai_words(s) if "฀" <= s[0] <= "๿" else [s])
    return [t for t in (t.strip() for t in out) if t]


# ── PDF -> chunks (article-aware) ─────────────────────────────────────────────
# ⚠️ ใช้ PyMuPDF (fitz) คัดข้อความ ไม่ใช่ pdfplumber — pdfplumber เรียงตัวอักษรตาม
#    x-coordinate ทำให้สระ/วรรณยุกต์ไทยที่ซ้อนบน-ล่างหลุดไปผิดตำแหน่ง:
#      pdfplumber -> "บรรดาที ดินทั งหลายอันเป นทรพั ยส์ นิ ของแผ่นดิน"   ✗
#      PyMuPDF    -> "บรรดาที่ดินทั้งหลายอันเป็นทรัพย์สินของแผ่นดิน"      ✓
#    ข้อความเพี้ยนพัง 3 ชั้นพร้อมกัน: embedding, BM25, และตัวบทที่ LLM คัดไปตอบ
def _clean_lines(text: str, header_re: "list | None" = None) -> list[str]:
    """ตัด header/footer ที่ซ้ำทุกหน้า (ต่อโดเมน) + บรรทัดว่างทิ้ง"""
    out = []
    for ln in text.splitlines():
        if header_re and any(p.search(ln) for p in header_re):
            continue
        if ln.strip():
            out.append(ln.rstrip())
    return out


# ── group จากชื่อไฟล์ (ชุดประมวลกฎหมายที่ดิน) ─────────────────────────────────
# ชื่อไฟล์รูปแบบ: <Code>_<Kind>-v<Ver>_<True|Img>PDF.pdf
#   LandCode2497_Main-v0_TruePDF.pdf      -> ฉบับดั้งเดิม พ.ศ. 2497
#   LandCode2497_Amend-v7_TruePDF.pdf     -> พ.ร.บ.แก้ไขเพิ่มเติม (ฉบับที่ ๗)
#   LandCode2497_Update-v7_TruePDF.pdf    -> ฉบับรวมสะสม หลังแก้ไขครั้งที่ ๗
#   LandCode2497_Update-vlast_TruePDF.pdf -> ฉบับรวมสะสมล่าสุด (กฎหมายที่ใช้ปัจจุบัน)
#   *_ImgPDF.pdf                          -> เวอร์ชันสแกนของเนื้อหาเดียวกัน (ต้อง OCR)
_FNAME_RE = re.compile(
    r"^(?P<code>[A-Za-z]+\d*)_(?P<kind>Main|Amend|Update)-v(?P<ver>\w+?)_(?P<src>True|Img)PDF$",
    re.I,
)
# ── กลุ่มเอกสาร ───────────────────────────────────────────────────────────────
# ⚠️ หัวใจของความถูกต้องทั้งระบบอยู่ตรงนี้:
#    "ฉบับรวมสะสมล่าสุด" (Update-vlast) = ตัวบทที่ใช้บังคับอยู่จริง ณ วันนี้
#    กฤษฎีกาไล่ปรับให้ทีละมาตราแล้ว — มาตราที่ไม่เคยแก้อยู่ครบเหมือนฉบับหลักคำต่อคำ
#    (เทียบทั้งฉบับแล้ว: เหมือนเป๊ะ 44 มาตรา / แก้แล้ว 57 / เพิ่มใหม่ 33 / ไม่ตกหล่นเลย)
#    ที่เหลือเป็น "หลักฐานอ้างอิงย้อนหลัง" ซึ่งมีตัวบทที่ถูกยกเลิกไปแล้วปนอยู่
#    → ค้นปนกันเมื่อไร = มีโอกาสตอบด้วยกฎหมายที่เลิกใช้แล้ว จึงต้องล็อกเป็น default
GROUP_IN_FORCE = "ฉบับใช้บังคับปัจจุบัน"      # Update-vlast — ค้นเป็นค่าเริ่มต้น
GROUP_HISTORY = "ฉบับย้อนหลังตามช่วงเวลา"     # Update-v1..v20 — ต้องระบุปีถึงจะปลด
GROUP_AMEND = "ประวัติการแก้ไข"               # Amend-v1..v15 — ถามว่าแก้อะไร/เมื่อไร
GROUP_ORIGINAL = "ฉบับดั้งเดิม พ.ศ. ๒๔๙๗"     # Main-v0 — ตัวบทตอนประกาศใช้ครั้งแรก

# กลุ่มที่ "ค้นได้เลยโดยไม่ต้องขอ" — นอกจากนี้ต้องมีสัญญาณชัดจากคำถาม
DEFAULT_SEARCH_GROUPS = {GROUP_IN_FORCE}


def parse_doc_name(path: str) -> dict:
    """แกะ metadata จากชื่อไฟล์ -> group / kind / version / is_scan / in_force
    in_force = ตัวบทนี้ยังใช้บังคับอยู่ไหม (มีเฉพาะ Update-vlast)
    ชื่อไฟล์ที่ไม่เข้าแพตเทิร์น -> group 'default' + in_force=True (ไม่รู้ก็ให้ค้นได้)"""
    stem = os.path.splitext(os.path.basename(path))[0]
    m = _FNAME_RE.match(stem)
    if not m:
        return {"group": DEFAULT_GROUP, "kind": "", "version": -1,
                "is_scan": False, "in_force": True}
    kind = m.group("kind").lower()
    ver = m.group("ver").lower()
    # 'vlast' -> 999 เพื่อให้เรียง/เทียบ "ใหม่สุด" ได้ด้วยตัวเลขตัวเดียว
    version = 999 if ver == "last" else int(ver) if ver.isdigit() else -1
    if kind == "update":
        in_force = version == 999
        group = GROUP_IN_FORCE if in_force else GROUP_HISTORY
    elif kind == "amend":
        group, in_force = GROUP_AMEND, False
    else:                                  # main — ตัวบท ๒๔๙๗ ดิบ (57 มาตราถูกแก้ไปแล้ว)
        group, in_force = GROUP_ORIGINAL, False
    return {"group": group, "kind": kind, "version": version,
            "is_scan": m.group("src").lower() == "img", "in_force": in_force}


# ── สกัด metadata ละเอียดจากตัวข้อความ ────────────────────────────────────────
# เก็บให้ละเอียดที่สุดเพราะทุกฟิลด์ = ช่องทางกรอง/จัดอันดับที่แม่นกว่าการเดาจาก embedding
# "มาตรา ๙", "มาตรา ๙/๑", "มาตรา ๘ ทวิ" — หัวข้อมาตราทุกแบบที่กฎหมายไทยใช้
# ⚠️ ต้องมี lookahead กันคำที่ขึ้นต้นเหมือนเลขลำดับ — "มาตรา ๙๗ ฉบับดั้งเดิม" ต้องอ่านว่า
#    มาตรา ๙๗ ไม่ใช่ "มาตรา ๙๗ ฉ" (ฉ = ลำดับที่ ๖) ที่ไปกินคำว่า "ฉบับ" เข้ามา
_ART_ORD = r"(?:ทวิ|ตรี|จัตวา|เบญจ|ฉ|สัตต|อัฏฐ|นว|ทศ)(?![฀-๏])"
_ART_RE = re.compile(rf"มาตรา\s*([๐-๙\d]+(?:/[๐-๙\d]+)?)\s*({_ART_ORD})?")
# หัวโครงสร้าง: หมวด ๑ / ภาค ๒ / ส่วนที่ ๓ / ลักษณะ ๑
_SEC_RE = re.compile(r"(หมวด|ภาค|ส่วนที่|ลักษณะ|บรรพ)\s*([๐-๙\d]+)")
# การอ้างถึงมาตราอื่น: "ตามมาตรา ๙๔", "แห่งมาตรา ๘", "ในมาตรา ๙๗ หรือมาตรา ๙๘"
_REF_RE = re.compile(r"(?:ตาม|แห่ง|ใน|ถึง|และ|หรือ|บทบัญญัติ)\s*มาตรา\s*([๐-๙\d]+(?:/[๐-๙\d]+)?)")


def _art_num(num: str, suffix: str = "") -> str:
    """'๙/๑' -> '9/1' | ('๘','ทวิ') -> '8 ทวิ' — รูปแบบเดียวกันทั้งฝั่ง index และฝั่งคำถาม
    (คนถามพิมพ์ 'มาตรา 9' ตัวบทพิมพ์ 'มาตรา ๙' ต้องเทียบกันติด)"""
    n = num.translate(THAI_DIGITS)
    return f"{n} {suffix}".strip() if suffix else n


def extract_article_meta(text: str) -> dict:
    """ดึงจาก chunk: มาตราทุกตัวที่ปรากฏ, เลขแบบอารบิก, มาตราที่อ้างถึง, หมวด/ภาค
    คืน string คั่นด้วย '|' (Chroma รับ metadata ได้แค่ str/int/float/bool ไม่รับ list)
    ห่อหัวท้ายด้วย '|' เพื่อให้ค้นแบบ exact ด้วย substring '|9|' ได้ ไม่ไปชนกับ '|91|'"""
    labels: list[str] = []
    nums: list[str] = []
    for m in _ART_RE.finditer(text or ""):
        num, suf = m.group(1), m.group(2) or ""
        lab = f"มาตรา {num} {suf}".strip()
        if lab not in labels:
            labels.append(lab)
            nums.append(_art_num(num, suf))
    refs: list[str] = []
    for m in _REF_RE.finditer(text or ""):
        r = _art_num(m.group(1))
        if r not in refs:
            refs.append(r)
    secs: list[str] = []
    for m in _SEC_RE.finditer(text or ""):
        s = f"{m.group(1)} {m.group(2).translate(THAI_DIGITS)}"
        if s not in secs:
            secs.append(s)
    def pack(xs: list[str]) -> str:
        return ("|" + "|".join(xs) + "|") if xs else ""
    return {
        "articles": pack(labels),        # '|มาตรา ๙|มาตรา ๙/๑|'
        "article_nums": pack(nums),      # '|9|9/1|'  ← ใช้ match กับเลขในคำถาม
        "refs": pack(refs),              # '|94|8|'   ← มาตราที่ตัวบทนี้อ้างถึง
        "section": secs[0] if secs else "",
        "n_articles": len(labels),
    }


def head_article_num(article_label: str) -> str:
    """ป้ายมาตราของ chunk -> เลขอารบิก ('มาตรา ๙/๑' -> '9/1'); '' ถ้าไม่ใช่ป้ายมาตรา
    (ป้ายอาจเป็น 'หมวด ๑' ซึ่งไม่ใช่มาตรา -> คืน '')"""
    m = _ART_RE.match(article_label or "")
    return _art_num(m.group(1), m.group(2) or "") if m else ""


def question_articles(question: str) -> list[str]:
    """เลขมาตราที่ผู้ใช้ระบุในคำถาม -> ['9', '9/1'] ([] = ไม่ได้ระบุ)
    รองรับทั้ง 'มาตรา ๙' และ 'มาตรา 9' (normalize เป็นอารบิกทั้งคู่)"""
    out: list[str] = []
    for m in _ART_RE.finditer(question or ""):
        n = _art_num(m.group(1), m.group(2) or "")
        if n not in out:
            out.append(n)
    return out


def _all_data_files() -> list[tuple[str, str]]:
    """คืน [(filepath, group)] ของไฟล์ที่รองรับใน data/
    โฟลเดอร์ย่อย = group; ไฟล์ที่วางตรง ๆ ใน data/ = group จากชื่อไฟล์ (ดู parse_doc_name)"""
    out: list[tuple[str, str]] = []
    if os.path.isdir(DATA_DIR):
        for entry in sorted(os.listdir(DATA_DIR)):
            full = os.path.join(DATA_DIR, entry)
            if os.path.isdir(full):
                for f in sorted(os.listdir(full)):
                    if os.path.splitext(f)[1].lower() in SUPPORTED_EXTS:
                        out.append((os.path.join(full, f), entry))
            elif os.path.splitext(entry)[1].lower() in SUPPORTED_EXTS:
                out.append((full, parse_doc_name(full)["group"]))
    return out


def _load_pdf(path: str, domain: str = "thai_law") -> list[dict]:
    """อ่าน PDF (PyMuPDF) -> records {text, article, page_start, page_end} (article-aware)
    ตัด chunk ตรงขอบ 'มาตรา/ข้อ' เพื่อไม่ให้ตัวบทมาตราเดียวถูกหั่นคนละก้อน
    หน้าที่คัดข้อความไม่ได้ (ImgPDF = สแกน) ตกไปเข้า OCR อัตโนมัติ"""
    import fitz
    prof = DOMAINS.get(domain, DOMAINS["thai_law"])
    major_re, section_re = prof["major_re"], prof["section_re"]

    recs: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    cur_article = ""
    page_start = 1

    def flush(page_end: int):
        nonlocal buf, buf_len, page_start
        text = "\n".join(buf).strip()
        if len(text) > 40:
            recs.append({"text": text, "article": cur_article,
                         "page_start": page_start, "page_end": page_end})
        # overlap: เก็บท้าย buffer ไว้ต่อ chunk ถัดไป
        tail, tlen = [], 0
        for ln in reversed(buf):
            if tlen + len(ln) > CHUNK_OVERLAP:
                break
            tail.insert(0, ln)
            tlen += len(ln)
        buf = tail
        buf_len = tlen
        page_start = page_end

    doc = fitz.open(path)
    try:
        total = doc.page_count
        for pno0 in range(total):
            pno = pno0 + 1
            raw = doc[pno0].get_text("text")
            # หน้าที่คัดข้อความแทบไม่ได้ = หน้าสแกน → OCR (local) มาเติม
            if len(raw.strip()) < OCR_MIN_CHARS:
                try:
                    import ocr
                    ocr_txt = ocr.page_text(path, pno0)
                    if len(ocr_txt.strip()) > len(raw.strip()):
                        raw = ocr_txt
                        print(f"  [OCR] {os.path.basename(path)} หน้า {pno}: "
                              f"อ่านได้ {len(raw.strip())} ตัวอักษร")
                except Exception as e:
                    print(f"  [OCR] หน้า {pno} ล้มเหลว: {e}")
            for ln in _clean_lines(raw, prof["header_re"]):
                m_major = major_re.match(ln)
                m_sec = section_re.match(ln)
                # เจอหัวข้อใหม่และ buffer ใหญ่พอ -> ตัด chunk ตรงขอบหัวข้อ
                if (m_major or m_sec) and buf_len >= CHUNK_SIZE * 0.6:
                    flush(pno)
                if m_major:
                    cur_article = prof["major_fmt"](m_major)
                elif m_sec:
                    cur_article = prof["section_fmt"](m_sec)
                buf.append(ln)
                buf_len += len(ln)
                if buf_len >= CHUNK_SIZE:
                    flush(pno)
            if buf_len >= CHUNK_SIZE:
                flush(pno)
        flush(total)
    finally:
        doc.close()

    return recs


def _chunk_text(text: str) -> list[dict]:
    """แบ่งข้อความล้วน (txt/md) เป็น records ตามขนาด + overlap (ไม่มี article/page)"""
    lines = [ln.rstrip() for ln in text.split("\n") if ln.strip()]
    recs: list[dict] = []
    buf: list[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        t = "\n".join(buf).strip()
        if len(t) > 40:
            recs.append({"text": t, "article": "", "page_start": 0, "page_end": 0})
        tail, tlen = [], 0
        for ln in reversed(buf):
            if tlen + len(ln) > CHUNK_OVERLAP:
                break
            tail.insert(0, ln)
            tlen += len(ln)
        buf = tail
        buf_len = tlen

    for ln in lines:
        buf.append(ln)
        buf_len += len(ln)
        if buf_len >= CHUNK_SIZE:
            flush()
    flush()
    return recs


def _load_csv(path: str) -> list[dict]:
    """อ่าน CSV -> records (แต่ละแถวเป็น 'col: value | ...' รวมหลายแถวต่อ chunk)
    หมายเหตุ: RAG เหมาะกับการ 'ค้น/อธิบาย' CSV ไม่เหมาะกับการ 'คำนวณ/รวมยอด'"""
    import pandas as pd
    df = pd.read_csv(path)
    recs: list[dict] = []
    buf: list[str] = []
    buf_len = 0
    start = 0

    def flush(end: int):
        nonlocal buf, buf_len, start
        t = "\n".join(buf).strip()
        if t:
            recs.append({"text": t, "article": f"rows {start}-{end}",
                         "page_start": start, "page_end": end})
        buf = []
        buf_len = 0
        start = end + 1

    last = -1
    for i, row in df.iterrows():
        last = int(i)
        block = " | ".join(
            f"{col}: {row[col]}" for col in df.columns
            if str(row[col]).strip() and str(row[col]).lower() != "nan")
        if not block:
            continue
        buf.append(block)
        buf_len += len(block)
        if buf_len >= CHUNK_SIZE:
            flush(last)
    flush(last)
    return recs


# ── year metadata (แยกตามปีเอกสาร — คนละมิติกับ group) ────────────────────────
# ปี: กฎหมายไทยใช้ พ.ศ. (25xx) เป็นหลัก + เผื่อ ค.ศ. (20xx) — int() ของ Python แปลงเลขไทยได้
_YEAR_RE = re.compile(r"(?<!\d)(25\d{2}|20\d{2})(?!\d)")


def extract_year(text: str, year_re=None) -> int:
    """ดึงปีตัวแรกที่พบ — 0 = ไม่พบ
    normalize เลขไทยก่อน เพราะตัวบทพิมพ์ "พ.ศ. ๒๕๒๐" ไม่ใช่ "พ.ศ. 2520" """
    m = (year_re or _YEAR_RE).search((text or "").translate(THAI_DIGITS))
    return int(m.group(1)) if m else 0


# กฤษฎีกาประทับปีที่พิมพ์เอกสารไว้ทุกไฟล์เท่ากันหมด → แยกฉบับไม่ได้ ต้องตัดทิ้ง
# ตอนหา as_of_year (ไม่งั้นทุกฉบับจะได้ปีเดียวกันหมด กรองปีก็ไร้ความหมาย)
PUBLISH_STAMP_YEAR = int(os.environ.get("PUBLISH_STAMP_YEAR", "2565"))


# ฉบับรวมสะสมแต่ละไฟล์ปิดท้ายด้วยรายการ "พ.ร.บ.แก้ไขเพิ่มเติมฯ (ฉบับที่ N) พ.ศ. YYYY"
# ที่รวมไว้ — ตัวสุดท้ายในรายการคือจุดเวลาของไฟล์นั้น แม่นกว่าเดาจาก "ปีสูงสุดในเอกสาร"
# มาก เพราะเอกสารทุกฉบับมีเชิงอรรถอ้างถึงฉบับแก้ในอนาคตปะปนอยู่
_AMEND_ENTRY_RE = re.compile(
    r"แก้ไขเพิ่มเติมประมวลกฎหมายที่ดิน\s*\(ฉบับที่\s*([๐-๙\d]+)\)\s*พ\.ศ\.\s*([๐-๙\d]+)")
# ฉบับ v1–v5 เก่ากว่ารายการแรกที่จับได้ (ฉบับที่ ๒ พ.ศ. ๒๕๒๑) จึงต้องมีเพดานกัน
# ไม่ให้ heuristic เดิมดันปีไปไกลเกินจริง
_EARLY_YEAR_CAP = 2520


def amend_level(text: str) -> tuple[int, int]:
    """ฉบับแก้ไขล่าสุดที่เอกสารนี้รวมไว้ -> (เลขฉบับ, พ.ศ.) | (0, 0) ถ้าไม่พบ"""
    got = [(int(a.translate(THAI_DIGITS)), int(b.translate(THAI_DIGITS)))
           for a, b in _AMEND_ENTRY_RE.findall(text or "")]
    return max(got) if got else (0, 0)


def doc_label(meta: dict, year: int) -> str:
    """ชื่ออ่านง่ายของเอกสาร ใช้อ้างอิงใน context ที่ส่งให้ LLM และในหน้า sources"""
    v, kind = meta.get("version", -1), meta.get("kind", "")
    if kind == "update" and v == 999:
        return "ประมวลกฎหมายที่ดิน (ฉบับใช้บังคับปัจจุบัน)"
    if kind == "update":
        return f"ประมวลกฎหมายที่ดิน (ฉบับ ณ พ.ศ. {year or '?'})"
    if kind == "amend":
        return f"พ.ร.บ.แก้ไขเพิ่มเติมประมวลกฎหมายที่ดิน (ฉบับที่ {v}) พ.ศ. {year or '?'}"
    if kind == "main":
        return "ประมวลกฎหมายที่ดิน พ.ศ. ๒๔๙๗ (ฉบับดั้งเดิม)"
    return "เอกสาร"


def detect_years(text: str) -> list[int]:
    """ดึงปีทั้งหมดที่เอ่ยถึงในคำถาม — [] = ไม่ระบุปี
    ⚠️ ต้อง normalize เลขไทยก่อน คนไทยพิมพ์ 'ปี ๒๕๔๕' ไม่ใช่ 'ปี 2545'
    (ปีที่ตรวจได้ = สัญญาณปลดล็อกค้นฉบับย้อนหลัง ถ้าพลาดตรงนี้ระบบจะไม่ยอมย้อนให้เลย)"""
    return sorted({int(y) for y in _YEAR_RE.findall((text or "").translate(THAI_DIGITS))})


def load_pdf_chunks() -> list[dict]:
    """สแกน data/ -> โหลดทุกไฟล์ (pdf/txt/md/csv) -> chunks พร้อม id/source/group/year
    (คงชื่อเดิมไว้เพื่อ compatibility กับ build_vectorstore / _ensure_loaded)"""
    all_chunks: list[dict] = []
    for path, group in _all_data_files():
        ext = os.path.splitext(path)[1].lower()
        domain = domain_of_group(group)          # โปรเจกต์นี้คืน thai_law เสมอ
        try:
            if ext == ".pdf":
                recs = _load_pdf(path, domain)
            elif ext in (".txt", ".md"):
                with open(path, encoding="utf-8", errors="ignore") as f:
                    recs = _chunk_text(f.read())
            elif ext == ".csv":
                recs = _load_csv(path)
            else:
                continue
        except Exception as e:
            print(f"  [!] อ่าน {path} ไม่สำเร็จ: {e}")
            continue
        src = os.path.basename(path)
        meta = parse_doc_name(path)
        yre = DOMAINS.get(domain, DOMAINS["thai_law"])["year_re"]
        # ปีประกาศใช้ อยู่ในตัวบทหน้าแรก ("ให้ไว้ ณ วันที่ ๕ กันยายน พ.ศ. ๒๕๒๐")
        # ไม่ใช่ในชื่อไฟล์ (2497 ในชื่อ = ปีของประมวลกฎหมาย ไม่ใช่ปีของฉบับแก้ไข)
        year = extract_year(recs[0]["text"] if recs else "", yre) or extract_year(src, yre)
        # ปีที่ตัวบทในไฟล์นี้สะท้อน — ปีสูงสุดที่เอ่ยถึงทั้งไฟล์ ตัดปีที่กฤษฎีกาประทับ
        # ตอนพิมพ์เอกสารออก (โผล่ทุกไฟล์เท่ากันหมด จึงแยกฉบับไม่ได้ ต้องตัดทิ้ง)
        as_of, amend_no = 0, 0
        if meta["kind"] == "update":
            full = " ".join(r["text"] for r in recs)
            amend_no, as_of = amend_level(full)
            if not as_of:      # ฉบับเก่ากว่ารายการแรกที่จับได้ -> ถอยไปใช้ปีสูงสุด แต่มีเพดาน
                ys = [y for y in detect_years(full) if y != PUBLISH_STAMP_YEAR]
                as_of = min(max(ys), _EARLY_YEAR_CAP) if ys else 0
        label = doc_label(meta, year or as_of)
        art_seq: dict[str, int] = {}   # นับ chunk ที่ n ของมาตราเดียวกัน (มาตรายาว = หลาย chunk)
        for i, r in enumerate(recs):
            art = r.get("article", "") or ""
            r["art_seq"] = art_seq[art] = art_seq.get(art, -1) + 1
            r["id"] = f"{src}::{i:04d}"
            r["source"] = src
            r["group"] = group
            r["year"] = year
            r["domain"] = domain
            r["text_key"] = text_key(r["text"])  # ลายนิ้วมือเนื้อหา (ใช้ยุบสำเนาซ้ำตอน retrieve)
            r["kind"] = meta["kind"]            # main / amend / update
            r["version"] = meta["version"]      # เลขฉบับ (vlast -> 999)
            r["is_scan"] = meta["is_scan"]      # มาจาก ImgPDF (OCR) หรือไม่
            r["in_force"] = meta["in_force"]    # ตัวบทที่ใช้บังคับอยู่จริงหรือไม่
            r["as_of_year"] = as_of             # ตัวบทนี้สะท้อนกฎหมาย ณ ปีไหน (0 = ไม่ทราบ)
            r["amend_no"] = amend_no            # รวมถึงฉบับแก้ไขที่เท่าไร (0 = ไม่ทราบ)
            r["doc_label"] = label              # ชื่ออ่านง่ายสำหรับอ้างอิงในคำตอบ
            r.update(extract_article_meta(r["text"]))   # articles/article_nums/refs/section
        all_chunks.extend(recs)
        tag = " [OCR]" if meta["is_scan"] else ""
        force = " ★ใช้บังคับ" if meta["in_force"] else ""
        print(f"  [{group}] {src}{tag}: {len(recs)} chunks "
              f"(พ.ศ. {year or as_of or '?'}){force}")
    return all_chunks


# ── vector store + BM25 ───────────────────────────────────────────────────────
def embed_tag() -> str:
    """tag สั้นจาก EMBED_MODEL สำหรับตั้งชื่อไฟล์/MLflow run (กันทับข้าม embedding)
    ada-002 -> 'ada' | paraphrase-multilingual -> 'para' | qwen/qwen3-8b -> 'q3-8b'"""
    m = EMBED_MODEL.lower()
    if "ada" in m:
        return "ada"
    if "paraphrase" in m:
        return "para"
    if "qwen3-8b" in m:
        return "q3-8b"
    if "3-large" in m:
        return "3-large"
    if "3-small" in m:
        return "3-small"
    return re.sub(r"[^a-z0-9]+", "-", m).strip("-")[:12] or "emb"


def _init_embeddings():
    global _embeddings
    if _embeddings is None:
        _m = EMBED_MODEL.lower()
        if (EMBED_MODEL.startswith("text-embedding") or "ada" in _m
                or _m.startswith("qwen/") or "/" in EMBED_MODEL):
            # ada-002, qwen/qwen3-8b ฯลฯ — OpenAI-compatible endpoint เดียวกับ chat
            _embeddings = OpenAIEmbeddings(base_url=LLM_BASE_URL, api_key=LLM_API_KEY, model=EMBED_MODEL)
        else:
            # paraphrase-multilingual ฯลฯ — ผ่าน Ollama
            _embeddings = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=EMBED_MODEL)


_reranker = None
_reranker_kind = None   # "jina" (AutoModel.rerank) หรือ "ce" (CrossEncoder.predict)


def _init_reranker():
    """โหลด reranker ครั้งเดียว (lazy) — เลือก GPU อัตโนมัติถ้ามี CUDA
    รองรับ 2 ตระกูล:
      - jina-reranker-v3 → AutoModel + trust_remote_code + .rerank() (สร้างบน Qwen3)
      - อื่นๆ (bge ฯลฯ) → sentence-transformers CrossEncoder + .predict()"""
    global _reranker, _reranker_kind
    if _reranker is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if "jina" in RERANK_MODEL.lower():
            from transformers import AutoModel
            try:                     # transformers ใหม่ใช้ dtype, เก่าใช้ torch_dtype
                m = AutoModel.from_pretrained(RERANK_MODEL, dtype="auto", trust_remote_code=True)
            except TypeError:
                m = AutoModel.from_pretrained(RERANK_MODEL, torch_dtype="auto", trust_remote_code=True)
            _reranker = m.to(device).eval()
            _reranker_kind = "jina"
        else:
            from sentence_transformers import CrossEncoder
            _reranker = CrossEncoder(RERANK_MODEL, device=device, max_length=512)
            _reranker_kind = "ce"
        if os.environ.get("RAG_DEBUG") == "1":
            print(f"    [rerank] โหลด {RERANK_MODEL} ({_reranker_kind}) บน {device.upper()}")
    return _reranker


def rerank(query: str, items: list[dict], k: int) -> list[dict]:
    """rerank: ให้คะแนนความเกี่ยวข้อง (query, item['text']) → คืน top-k
    items = list ของ dict ที่มีคีย์ 'text' (ใช้ได้ทั้ง chunk ของ rag และ detail dict ของ experiment)
    ถ้า reranker ใช้ไม่ได้ (ไม่ได้ลง/โหลดพัง) → fallback คืน k ตัวแรกตามลำดับเดิม"""
    if not items:
        return items
    try:
        model = _init_reranker()
        docs = [it["text"] for it in items]
        if _reranker_kind == "jina":
            # jina .rerank() คืน list เรียงแล้ว แต่ละตัวมี index ชี้กลับไปที่ docs
            res = model.rerank(query, docs, top_n=k)
            order = [(r["index"] if isinstance(r, dict) else r.index) for r in res]
            return [items[i] for i in order[:k]]
        scores = model.predict([(query, d) for d in docs])   # CrossEncoder
        order = sorted(range(len(items)), key=lambda i: -float(scores[i]))
        return [items[i] for i in order[:k]]
    except Exception as e:
        if os.environ.get("RAG_DEBUG") == "1":
            print(f"    [rerank] ใช้ไม่ได้ ({str(e)[:80]}) — fallback RRF")
        return items[:k]


def _build_bm25(chunks: list[dict]):
    global _bm25, _chunks, _chunk_by_id
    _chunks = chunks
    _chunk_by_id = {c["id"]: c for c in chunks}
    _bm25 = BM25Okapi([tokenize(c["text"]) for c in chunks])


def update_database() -> bool:
    """เช็คว่าไฟล์ใน data/ เปลี่ยนไหม — คืน True ถ้ามีไฟล์ใหม่/แก้ไข/หายไป
    (คงชื่อเดิมไว้เพื่อ compatibility กับ batch_test.py)"""
    files = _all_data_files()
    if not files:
        print(f"[!] ไม่พบเอกสารใน {DATA_DIR} (รองรับ: {', '.join(sorted(SUPPORTED_EXTS))})")
        return False
    cur = {os.path.relpath(p, HERE): hash_file(p) for p, _ in files}
    saved = load_hashes()
    if cur != saved:
        print(f"ตรวจพบไฟล์ใหม่/เปลี่ยนแปลง ({len(cur)} ไฟล์) — จะสร้างดัชนีใหม่")
        return True
    print(f"ไฟล์ไม่เปลี่ยนแปลง ({len(cur)} ไฟล์) ใช้ดัชนีเดิม")
    return False


def _init_chroma():
    """สร้าง/เปิด Chroma persistent client + collection (cosine space)"""
    global _chroma_client, _collection
    if _collection is not None:
        return
    _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def _chroma_metadata(c: dict) -> dict:
    """คัดเฉพาะ metadata ที่ Chroma รับได้ (str/int/float/bool) จาก chunk"""
    return {
        "group": c.get("group", DEFAULT_GROUP),
        "source": c.get("source", ""),
        "article": c.get("article", "") or "",
        "page_start": int(c.get("page_start", 0) or 0),
        "page_end": int(c.get("page_end", 0) or 0),
        "year": int(c.get("year", 0) or 0),        # ปีเอกสาร (0 = ไม่ระบุ)
        "kind": c.get("kind", "") or "",           # main / amend / update
        "version": int(c.get("version", -1)),      # เลขฉบับ (vlast -> 999)
        "is_scan": bool(c.get("is_scan", False)),  # มาจาก ImgPDF (OCR)
        # ── ฟิลด์ที่ใช้กรอง/จัดอันดับตอน retrieve ──
        "in_force": bool(c.get("in_force", False)),      # ใช้บังคับอยู่จริงไหม
        "as_of_year": int(c.get("as_of_year", 0) or 0),  # ตัวบท ณ ปีไหน
        "doc_label": c.get("doc_label", "") or "",       # ชื่อเอกสารอ่านง่าย
        "articles": c.get("articles", "") or "",         # '|มาตรา ๙|มาตรา ๙/๑|'
        "article_nums": c.get("article_nums", "") or "", # '|9|9/1|'
        "refs": c.get("refs", "") or "",                 # มาตราที่อ้างถึง
        "section": c.get("section", "") or "",           # หมวด/ภาค
        "n_articles": int(c.get("n_articles", 0) or 0),
    }


def build_vectorstore(force: bool = False):
    """สร้าง chunks จาก PDF -> embed -> Chroma collection (vectors+metadata) + BM25"""
    global _collection
    _init_embeddings()
    _init_chroma()
    assert _collection is not None and _embeddings is not None

    chunks = load_pdf_chunks()
    print(f"แบ่งเอกสารได้ {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    _build_bm25(chunks)

    if not force and len(chunks) > 0 and _collection.count() == len(chunks):
        print("ใช้ Chroma collection เดิม (ข้ามการ embed)\n")
        return

    # rebuild: ล้าง collection เดิมแล้วสร้างใหม่ (กัน id ค้าง / มิติ vector เปลี่ยน)
    _chroma_client.delete_collection(COLLECTION_NAME)
    _collection = _chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})

    print(f"กำลัง embed {len(chunks)} chunks ...")
    BATCH = 64
    for i in range(0, len(chunks), BATCH):
        batch = chunks[i:i + BATCH]
        vecs = _embeddings.embed_documents([c["text"] for c in batch])
        _collection.add(
            ids=[c["id"] for c in batch],
            embeddings=vecs,
            documents=[c["text"] for c in batch],
            metadatas=[_chroma_metadata(c) for c in batch],
        )
        print(f"  {min(i + BATCH, len(chunks))}/{len(chunks)}")

    print(f"Chroma collection บันทึกแล้ว ({_collection.count()} vectors)\n")
    _save_hashes()


def _save_hashes():
    """เซฟ hash ต่อไฟล์ (ไว้ตรวจรอบหน้าว่าต้อง rebuild ไหม)"""
    try:
        cur = {os.path.relpath(p, HERE): hash_file(p) for p, _ in _all_data_files()}
        save_hashes(cur)
    except Exception as e:
        print(f"[!] บันทึก file hashes ไม่สำเร็จ: {e}")


def _ensure_loaded():
    """โหลด globals ให้พร้อม retrieve แม้ build_vectorstore ยังไม่ถูกเรียกในรอบนี้"""
    _init_embeddings()
    _init_chroma()
    if _bm25 is None:               # BM25 อยู่ในหน่วยความจำ ต้อง build จาก chunks ทุกครั้งที่ import
        _build_bm25(load_pdf_chunks())


# ── retrieval (hybrid: semantic + BM25 via RRF) ───────────────────────────────
# ฉบับรวมสะสม (Update-v1..vlast) คือกฎหมายเล่มเดียวกันคนละช่วงเวลา — มาตราที่ไม่เคยถูก
# แก้จะปรากฏเหมือนกันเป๊ะทั้ง 21 ฉบับ ถ้าปล่อยไว้ ผลค้น top-9 จะเป็นมาตราเดียวซ้ำ 9 ฉบับ
# → ยุบเหลือ "ฉบับใหม่สุด" = กฎหมายที่ใช้อยู่จริง (ยังถามย้อนได้ด้วย filter ปี/กลุ่ม)
DEDUPE_VERSIONS = os.environ.get("RAG_DEDUPE", "1") != "0"

# chunk จาก OCR (ImgPDF) = เนื้อหาเดียวกับ TruePDF ที่มีคู่กันครบทุกไฟล์ แต่คุณภาพต่ำกว่ามาก
# EasyOCR อ่านเลขไทยพลาดบ่อย: "มาตรา ๓" -> "มาตรา ลก", "๑๙ กุมภาพันธ์" -> "๑ธี กุมภาพันธ์"
# ซึ่งเป็นตัวเลขที่ห้ามผิดที่สุดในงานกฎหมาย → กันไว้ท้ายแถว ใช้เป็น fallback เท่านั้น
# (จะติด top-k ก็ต่อเมื่อฉบับคัดข้อความได้ให้ผลไม่พอ) ปิดด้วย RAG_SCAN_DEMOTE=0
DEMOTE_SCANS = os.environ.get("RAG_SCAN_DEMOTE", "1") != "0"


def _demote_scans(chunks: list[dict]) -> list[dict]:
    """ดัน chunk ที่มาจาก OCR ไปท้ายลิสต์ คงลำดับเดิมภายในแต่ละฝั่ง"""
    if not DEMOTE_SCANS:
        return chunks
    clean = [c for c in chunks if not c.get("is_scan")]
    scans = [c for c in chunks if c.get("is_scan")]
    return clean + scans


# ถามเจาะเลขมาตรา = ผู้ใช้บอกมาแล้วว่าต้องการอะไร ไม่ควรให้ embedding มาเดาแทน
# เช่นถาม "มาตรา ๙" แล้วได้ "มาตรา ๑๐" มาอันดับ 1 เพราะเนื้อหาใกล้กัน — ไม่ใช่สิ่งที่ขอ
BOOST_EXACT_ARTICLE = os.environ.get("RAG_ART_BOOST", "1") != "0"


def _boost_exact_article(chunks: list[dict], wanted: list[str]) -> list[dict]:
    """ดัน chunk ที่ 'ขึ้นต้นด้วย' มาตราที่ถูกถามขึ้นหัวแถว คงลำดับเดิมภายในกลุ่ม
    3 ชั้น: ป้ายมาตราของ chunk ตรง > มาตรานั้นปรากฏใน chunk > ที่เหลือ
    (ป้าย = มาตราหลักของ chunk ก้อนนั้น จึงตรงความต้องการกว่าการแค่ 'มีเอ่ยถึง')"""
    if not (BOOST_EXACT_ARTICLE and wanted):
        return chunks
    want = set(wanted)
    primary, mentioned, rest = [], [], []
    for c in chunks:
        head = head_article_num(c.get("article", ""))
        nums = c.get("article_nums", "") or ""
        if head and head in want:
            primary.append(c)
        elif any(f"|{w}|" in nums for w in want):
            mentioned.append(c)
        else:
            rest.append(c)
    return primary + mentioned + rest


def text_key(text: str) -> str:
    """ลายนิ้วมือเนื้อหา — ตัดช่องว่างทิ้งก่อน hash
    (ตัวบทเดียวกันคนละฉบับ ขึ้นบรรทัด/เว้นวรรคไม่เท่ากัน แต่ตัวอักษรเหมือนกันเป๊ะ)"""
    return hashlib.md5(re.sub(r"\s+", "", text or "").encode("utf-8")).hexdigest()


def _dedupe_versions(chunks: list[dict]) -> list[dict]:
    """ยุบ chunk ที่ซ้ำข้ามฉบับ -> เก็บ version ใหม่สุด คงลำดับคะแนนเดิม
    ยุบ 2 ชั้น:
      1. เนื้อหาเหมือนกันเป๊ะ (text_key) — ปลอดภัยเสมอ ใช้กับทุกกลุ่ม
         ~29% ของดัชนีเป็นสำเนาแบบนี้ บางมาตราซ้ำถึง 20 ฉบับ
      2. มาตราเดียวกันในสายฉบับรวมสะสม (article, art_seq) — เนื้อหาต่างได้
         (= มาตรานั้นถูกแก้) เก็บฉบับใหม่สุด = กฎหมายที่ใช้อยู่จริง
    ฉบับดั้งเดิม/ฉบับแก้ไข ไม่โดนชั้นที่ 2 — เป็นคนละเอกสารจริง ไม่ใช่สำเนา"""
    best: dict[tuple, dict] = {}
    order: list[tuple] = []
    seen_text: dict[str, tuple] = {}      # text_key -> key ที่จองไว้แล้ว
    for c in chunks:
        tk = c.get("text_key") or text_key(c.get("text", ""))
        if tk in seen_text:               # ชั้น 1: เนื้อหาซ้ำเป๊ะ
            key = seen_text[tk]
        elif c.get("kind") == "update" and c.get("article"):
            key = (c.get("article"), c.get("art_seq", 0))   # ชั้น 2
            seen_text[tk] = key
        else:
            key = ("\x00keep", c.get("id"))
            seen_text[tk] = key
        if key not in best:
            best[key] = c
            order.append(key)
        elif int(c.get("version", -1)) > int(best[key].get("version", -1)):
            best[key] = c          # เจอฉบับใหม่กว่า -> แทนที่ คงอันดับเดิมไว้
    return [best[k] for k in order]


def version_at_year(year: int) -> int:
    """ฉบับรวมสะสมที่ 'ใช้บังคับอยู่ ณ ปี พ.ศ. ที่ถาม' -> เลข version (-1 = หาไม่ได้)
    ⚠️ ไม่ใช่ฉบับที่ as_of_year == year เป๊ะ ๆ — กฎหมายไม่ได้แก้ทุกปี
    ถามปี ๒๕๔๕ แต่ฉบับล่าสุดก่อนหน้านั้นคือ ๒๕๔๒ ก็ต้องได้ฉบับ ๒๕๔๒
    (ตัวบทปี ๒๕๔๒ ยังใช้บังคับอยู่ในปี ๒๕๔๕ จนกว่าจะมีการแก้ครั้งถัดไป)"""
    best_v, best_y = -1, -1
    for c in _chunks:
        if c.get("kind") != "update":
            continue
        y = int(c.get("as_of_year", 0) or 0)
        if 0 < y <= year and y > best_y:
            best_y, best_v = y, int(c.get("version", -1))
    return best_v


def retrieve(query: "str | list[str]", k: int = TOP_K,
             rerank_query: "Optional[str]" = None,
             groups: "Optional[list[str]]" = None,
             years: "Optional[list[int]]" = None,
             versions: "Optional[list[int]]" = None) -> list[dict]:
    """retrieve แบบ hybrid (semantic + BM25 รวมด้วย RRF)
    query เป็น str เดียว หรือ list[str] (multi-query RAG-Fusion):
      หลาย query → ค้นแยกกันทุกอัน แล้ว RRF รวม "ทุก ranked list" → top-k
      (chunk ที่หลาย query เจอตรงกัน ถูกบวกคะแนนซ้ำ → ลอยขึ้นบนสุดเอง)
    final K เท่าเดิมเสมอ ไม่ว่าจะกี่ query → token ตอนตอบไม่บาน
    ถ้า RERANK_ENABLED + ส่ง rerank_query มา: RRF เก็บ top-N → cross-encoder → top-k
    groups: จำกัดเฉพาะ chunk ในกลุ่มที่ระบุ (None = ทุกกลุ่ม — พฤติกรรมเดิม)
    years:  จำกัดเฉพาะ chunk ปีที่ระบุ (None = ทุกปี) — รวมกับ groups ได้ (โดเมน × ปี)"""
    _ensure_loaded()
    assert _collection is not None and _embeddings is not None and _bm25 is not None
    queries = [query] if isinstance(query, str) else list(query)
    pool = k * 4

    # กรองแบบหลายมิติ (group และ/หรือ year):
    #   semantic → ใช้ where ของ Chroma (กรองในดัชนี), BM25 → ใช้ allowed set (index-based)
    #   หลายเงื่อนไขรวมด้วย $and — allowed ฝั่ง BM25 = intersection ของทุกเงื่อนไข
    conds: list[dict] = []
    masks: list[set[int]] = []
    # ⚠️ ไม่ระบุกลุ่มมา = "ค้นตัวบทที่ใช้บังคับอยู่" ไม่ใช่ "ค้นทุกอย่าง"
    # ฉบับย้อนหลัง/ฉบับดั้งเดิม มีตัวบทที่ถูกยกเลิกไปแล้วปนอยู่ ถ้าปล่อยให้ค้นปนกัน
    # จะมีโอกาสตอบด้วยกฎหมายที่เลิกใช้แล้ว — ผู้เรียกต้องขอกลุ่มนั้นมาเองเท่านั้น
    if not groups:
        groups = sorted(DEFAULT_SEARCH_GROUPS)
    if groups:
        gset = set(groups)
        conds.append({"group": {"$in": list(gset)}})
        masks.append({i for i, c in enumerate(_chunks) if c.get("group") in gset})
    if years:
        yset = {int(y) for y in years}
        conds.append({"year": {"$in": list(yset)}})
        masks.append({i for i, c in enumerate(_chunks) if int(c.get("year", 0) or 0) in yset})
    if versions:
        vset = {int(v) for v in versions}
        conds.append({"version": {"$in": list(vset)}})
        masks.append({i for i, c in enumerate(_chunks) if int(c.get("version", -1)) in vset})

    where: "dict | None" = None
    allowed: "set[int] | None" = None
    if conds:
        allowed = set.intersection(*masks) if masks else set()
        if allowed:                # มี chunk ตรงเงื่อนไขจริง ค่อยกรอง (กัน empty result)
            where = conds[0] if len(conds) == 1 else {"$and": conds}
        else:
            allowed = None         # ไม่มี chunk ตรงเลย → ไม่กรอง (fallback กัน empty)
    n_res = max(1, min(pool, _collection.count()))

    fuse: dict[str, float] = {}
    for q in queries:
        # 1) semantic (Chroma cosine + กรอง group ด้วย where ในตัว)
        qv = _embeddings.embed_query(q)
        res = _collection.query(query_embeddings=[qv], n_results=n_res, where=where)
        sem_ids = (res["ids"][0] if res.get("ids") else [])[:pool]

        # 2) keyword (BM25)
        scores = _bm25.get_scores(tokenize(q))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        bm_ids = [_chunks[i]["id"] for i in order
                  if allowed is None or i in allowed][:pool]

        # 3) Reciprocal Rank Fusion — สะสมข้ามทุก query (semantic + bm25 = 2 list ต่อ query)
        for r, cid in enumerate(sem_ids):
            fuse[cid] = fuse.get(cid, 0.0) + 1.0 / (60 + r)
        for r, cid in enumerate(bm_ids):
            fuse[cid] = fuse.get(cid, 0.0) + 1.0 / (60 + r)

    ranked_ids = sorted(fuse, key=lambda c: -fuse[c])
    ranked = [_chunk_by_id[c] for c in ranked_ids if c in _chunk_by_id]
    if DEDUPE_VERSIONS:            # ยุบก่อนตัด k → ไม่เสีย slot ให้ฉบับเก่าที่เนื้อหาซ้ำ
        ranked = _dedupe_versions(ranked)
    ranked = _demote_scans(ranked)  # ดัน OCR ท้ายแถวก่อนตัด pool → reranker เห็นแต่ตัวบทที่สะอาด
    # เลขมาตราที่ผู้ใช้ระบุมาเอง — ดูจากคำถามจริง (rerank_query) ไม่ใช่ query ที่ LLM ขยาย
    wanted = question_articles(rerank_query or (query if isinstance(query, str) else ""))
    if RERANK_ENABLED and rerank_query:
        # ⚠️ ให้ reranker คืนทั้ง pool (ไม่ใช่แค่ k) แล้วค่อย demote/boost แล้วจึงตัด k
        #    ถ้าตัด k ก่อน มาตราที่ผู้ใช้ถามอาจไม่ติด k ตัวแรก → ไม่มีอะไรให้ boost
        n = max(RERANK_TOP_N, k)
        ranked = rerank(rerank_query, ranked[:n], n)
    return _boost_exact_article(_demote_scans(ranked), wanted)[:k]


def status_label(c: dict) -> str:
    """ป้ายบอกสถานะตัวบท — ให้ LLM รู้ว่าอันไหนเชื่อได้ อันไหนเป็นหลักฐานย้อนหลัง
    สำคัญมากเวลาผู้ใช้ขอค้นย้อนหลัง เพราะตัวบทเก่ากับใหม่จะมาอยู่ใน context เดียวกัน"""
    if c.get("in_force"):
        return "✅ ใช้บังคับปัจจุบัน"
    kind, yr = c.get("kind", ""), c.get("as_of_year") or c.get("year") or 0
    if kind == "update":
        return f"⚠️ ตัวบท ณ พ.ศ. {yr or '?'} — อาจถูกแก้ไขภายหลังแล้ว"
    if kind == "main":
        return "⚠️ ฉบับดั้งเดิม พ.ศ. ๒๔๙๗ — หลายมาตราถูกแก้ไขภายหลังแล้ว"
    if kind == "amend":
        return f"📌 ฉบับแก้ไขเพิ่มเติม พ.ศ. {yr or '?'} — ระบุว่าแก้มาตราใด"
    return ""


# ── เส้นทางที่ 2: เปรียบเทียบตัวบทข้ามฉบับ ────────────────────────────────────
# retrieve() ปกติ "ยุบเวอร์ชันซ้ำทิ้ง" เพื่อตอบว่ากฎหมายว่าอย่างไร ณ วันนี้
# แต่คำถาม "ต่างกันอย่างไร / ถูกแก้เมื่อไร" ต้องการสิ่งตรงข้าม คือเก็บทุกเวอร์ชันไว้เทียบ
# → เส้นทางนี้ไปหยิบตัวบทจาก metadata ตรง ๆ ข้าม RRF/rerank/dedupe ทั้งหมด
# ⚠️ ห้ามเทียบ "ข้อความของ chunk" ข้ามฉบับ — ขอบ chunk ของแต่ละฉบับไม่ตรงกัน
#    (เอกสารยาวไม่เท่ากัน จุดตัดจึงเลื่อน) ผลลัพธ์จะกลายเป็นเห็น "เปลี่ยนทุกฉบับ"
#    ทั้งที่ตัวบทเหมือนกันเป๊ะ → ต้องตัดเอา "ตัวบทมาตรานั้นจริง ๆ" จากเอกสารเต็มมาเทียบ
_ART_HEAD_RE = re.compile(rf"มาตรา\s+([๐-๙]+(?:/[๐-๙]+)?)\s*({_ART_ORD})?")
# header/footer ที่กฤษฎีกาประทับทุกหน้า — ถ้าไม่ตัดจะกลายเป็นความต่างปลอม
_DOC_NOISE_RE = re.compile(
    r"สำนักงานคณะกรรมการกฤษฎีกา|Office of the Council of State|\d\d/\d\d/\d\d\s+\d\d:\d\d")
# สิ่งที่ "ไม่ใช่ตัวบท" แต่เปลี่ยนไปทุกฉบับเพราะเอกสารยาวไม่เท่ากัน:
#   [58] = เลขเชิงอรรถ · | 32 = เลขหน้า
# ถ้าไม่ตัดออกก่อนเทียบ จะเห็นมาตราเดียวกัน "เปลี่ยน" ทุกฉบับทั้งที่ตัวบทเหมือนเดิมเป๊ะ
_LAYOUT_NOISE_RE = re.compile(r"\[\d+\]|\|\s*\d+")


def law_text_norm(body: str) -> str:
    """ตัวบทล้วน ๆ — ตัดเลขหน้า/เชิงอรรถ/ช่องว่างทิ้ง เหลือเฉพาะตัวอักษรที่เป็นเนื้อกฎหมาย"""
    return re.sub(r"\s+", "", _LAYOUT_NOISE_RE.sub(" ", body or ""))


def law_text_key(body: str) -> str:
    """ลายนิ้วมือของตัวบท (ใช้เทียบแบบเป๊ะ)"""
    return text_key(_LAYOUT_NOISE_RE.sub(" ", body or ""))


# เทียบเป๊ะเข้มเกินไป: PDF ฉบับต่างกันมีเศษสระลอยหลุดมาไม่เหมือนกัน (เช่น 'ทั้ที่กึ่')
# ทำให้มาตราที่ไม่ได้ถูกแก้ดูเหมือนเปลี่ยน (วัดได้ความเหมือน 99.4%)
# การแก้ไขกฎหมายจริงเปลี่ยนถ้อยคำเป็นเรื่องเป็นราว ความเหมือนจะต่ำกว่านี้มาก
LAW_SAME_RATIO = float(os.environ.get("LAW_SAME_RATIO", "0.99"))


def same_law_text(a: str, b: str) -> bool:
    """ตัวบทสองรุ่นนี้ถือว่า 'ไม่ถูกแก้' หรือไม่ — เทียบด้วยเกณฑ์ความเหมือน ไม่ใช่ตรงเป๊ะ"""
    if a == b:
        return True
    if not a or not b:
        return False
    import difflib
    return difflib.SequenceMatcher(None, a, b).ratio() >= LAW_SAME_RATIO

_article_cache: dict[str, dict[str, str]] = {}


def doc_articles(path: str) -> dict[str, str]:
    """อ่านเอกสารทั้งไฟล์ -> {เลขมาตราแบบอารบิก: ตัวบทของมาตรานั้น}
    ตัดตั้งแต่หัว 'มาตรา N' ถึงหัวมาตราถัดไป — เป็นหน่วยเดียวที่เทียบข้ามฉบับได้อย่างมีความหมาย"""
    if path in _article_cache:
        return _article_cache[path]
    import fitz
    doc = fitz.open(path)
    try:
        text = "".join(doc[i].get_text() for i in range(doc.page_count))
    finally:
        doc.close()
    text = _DOC_NOISE_RE.sub(" ", text)
    heads = list(_ART_HEAD_RE.finditer(text))
    out: dict[str, str] = {}
    for i, m in enumerate(heads):
        num = _art_num(m.group(1), m.group(2) or "")
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        body = " ".join(text[m.start():end].split())
        # เก็บครั้งแรกที่เจอ และเฉพาะที่ยาวพอจะเป็นตัวบทจริง (สั้น = การอ้างอิงถึงมาตรานั้น)
        if len(body) > 60 and num not in out:
            out[num] = body
    _article_cache[path] = out
    return out


def article_timeline(num: str) -> list[dict]:
    """ไล่ตัวบท 'มาตรา num' ข้ามทุกฉบับตามลำดับเวลา แล้วยุบช่วงที่ตัวบทไม่เปลี่ยน
    คืนเฉพาะ 'จุดที่เนื้อหาเปลี่ยนจริง' -> [{version, as_of_year, doc_label, text, in_force}]

    ใช้ได้ 2 อย่างด้วยกลไกเดียว:
      - เปรียบเทียบ  : เอาจุดเปลี่ยนมาวางเทียบกัน
      - ถูกแก้เมื่อไร : จำนวนจุดเปลี่ยน = จำนวนครั้งที่มาตรานี้ถูกแก้
    """
    _ensure_loaded()
    # เอกสารตัวแทนของแต่ละฉบับ (TruePDF เท่านั้น — ฉบับสแกนตัวเลขเพี้ยน เทียบไม่ได้)
    docs: dict[tuple, dict] = {}
    for c in _chunks:
        if c.get("kind") in ("main", "update") and not c.get("is_scan"):
            docs.setdefault((0 if c["kind"] == "main" else 1, int(c["version"])), c)

    out: list[dict] = []
    last_key = ""
    for order in sorted(docs):
        c = docs[order]
        body = doc_articles(os.path.join(DATA_DIR, c["source"])).get(num)
        if not body:
            continue
        tk = law_text_norm(body)          # เทียบเฉพาะตัวบท ไม่นับเลขหน้า/เชิงอรรถ
        if same_law_text(tk, last_key):   # ตัวบทเหมือนเดิม -> ไม่ใช่จุดเปลี่ยน ข้าม
            # แต่ถ้าฉบับที่ข้ามคือฉบับที่ใช้บังคับอยู่ ต้องเลื่อนป้าย "ใช้บังคับ" มาที่จุดเปลี่ยนล่าสุด
            # (ตัวบทเดียวกันยังมีผลถึงวันนี้ ไม่ใช่ว่าจุดเปลี่ยนนั้นเลิกใช้แล้ว)
            if c.get("in_force") and out:
                out[-1]["in_force"] = True
            continue
        last_key = tk
        out.append({
            "version": c["version"], "as_of_year": c.get("as_of_year", 0),
            "amend_no": c.get("amend_no", 0),
            "doc_label": c.get("doc_label", ""), "in_force": c.get("in_force", False),
            "article": f"มาตรา {num}", "kind": c.get("kind", ""),
            "source": c.get("source", ""), "page_start": 0, "text": body,
        })
    return out


def format_comparison(num: str, points: list[dict]) -> str:
    """จัด context เป็น 'คู่เทียบตามลำดับเวลา' — ต่างจาก format_context ที่เรียงตามคะแนน
    บอกชัดว่าอันไหนคือจุดเปลี่ยนที่เท่าไร และอันไหนคือฉบับที่ใช้บังคับอยู่"""
    if not points:
        return f"(ไม่พบตัวบทของมาตรา {num} ในเอกสารที่มี)"
    blocks = []
    for i, p in enumerate(points, 1):
        blocks.append(f"[รุ่นที่ {i}/{len(points)} | {timeline_when(p)} | "
                      f"{'✅ ใช้บังคับปัจจุบัน' if p['in_force'] else 'ตัวบทเดิม'}]\n{p['text']}")
    return "\n\n---\n\n".join(blocks)


def timeline_when(p: dict) -> str:
    """คำอธิบายจุดเวลาของรุ่นตัวบท — ระบุฉบับแก้ไขที่รวมไว้ถ้าทราบ"""
    if p.get("kind") == "main":
        return "ฉบับดั้งเดิม พ.ศ. ๒๔๙๗"
    if p.get("amend_no"):
        return f"หลังแก้ไขฉบับที่ {p['amend_no']} (พ.ศ. {p['as_of_year']})"
    return f"ราว พ.ศ. {p.get('as_of_year') or '?'}"


def format_context(chunks: list[dict]) -> str:
    """ประกอบ context ให้ LLM — หัวบล็อกบอก เอกสาร/สถานะ/มาตรา/หน้า
    ใส่สถานะทุกบล็อกเสมอ ไม่ใช่เฉพาะตอนปนกัน เพื่อไม่ให้โมเดลต้องเดาว่าอันที่ไม่มีป้ายคืออะไร"""
    blocks = []
    for c in chunks:
        parts = [c.get("doc_label") or c.get("source", "?")]
        st = status_label(c)
        if st:
            parts.append(st)
        if c.get("article"):
            parts.append(c["article"])
        if c.get("section"):
            parts.append(c["section"])
        if c.get("page_start"):
            parts.append(f"หน้า {c['page_start']}")
        blocks.append("[" + " | ".join(parts) + "]\n" + c["text"])
    return "\n\n---\n\n".join(blocks)


# ── LLM / answer ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "คุณคือผู้ช่วยตอบคำถามเกี่ยวกับกฎหมายไทย (จากราชกิจจานุเบกษา เช่น พระราชบัญญัติ กฎกระทรวง ประกาศ)\n\n"
    "กฎสำคัญ:\n"
    "- ตอบโดยอ้างอิงจาก 'เอกสารอ้างอิง' (context) ที่ให้มาเท่านั้น ห้ามใช้ความรู้ภายนอกหรือเดา\n"
    "- ระบุเลข 'มาตรา/ข้อ/หมวด' และหน้าที่อ้างอิงทุกครั้งที่เป็นไปได้ (เช่น 'ตามมาตรา ๕')\n"
    "- คัดข้อความตัวบท ตัวเลข วันที่ ให้ตรงเป๊ะจาก context — ⚠️ เลขมาตรา/ตัวเลข ห้ามแก้/ปัด/เดา\n"
    "- ถ้าถามการตีความ/ผลทางกฎหมาย: สรุปตัวบทที่เกี่ยวข้องก่อน แล้ววิเคราะห์ต่อจากตัวบทนั้น (ขึ้นต้น 'วิเคราะห์:')\n"
    "- ให้ตอบ 'ไม่พบข้อมูลนี้ในเอกสารที่ค้นเจอ' เฉพาะเมื่อ 'ตัวบท/ข้อเท็จจริงที่ถามโดยตรง' ไม่มีใน context จริง ๆ\n"
    "- ถ้าตัวบทอ้างถึงมาตราอื่น (cross-reference) ให้ระบุเลขมาตราที่อ้างถึงด้วย\n"
    "- หัวบล็อกบอกสถานะตัวบท ใช้ตัดสินว่าจะเชื่อบล็อกไหน — ห้ามคัดลอก '✅'/'⚠️' ลงในคำตอบ\n"
    "  · ถ้ามีทั้งบล็อก '✅ ใช้บังคับปัจจุบัน' และ '⚠️' ที่เนื้อหาขัดกัน ให้ตอบตามบล็อก ✅\n"
    "    แล้วบอกสั้น ๆ ว่าตัวบทเดิมต่างอย่างไร\n"
    "  · ถ้าตัวบทที่ใช้ตอบมาจากบล็อก '⚠️' (เช่นผู้ใช้ถามย้อนปี) ให้ตอบได้ตามปกติ\n"
    "    แต่ต้องระบุให้ชัดว่าเป็นตัวบท ณ ปีใด และอาจไม่ใช่ฉบับที่ใช้บังคับอยู่ในปัจจุบัน\n"
    "- ⚠️ บาง context มาจาก OCR อาจมีคำเพี้ยน — ตีความจากบริบทได้ แต่ห้ามแต่งเลขมาตรา/ตัวเลขขึ้นใหม่\n"
    "- ตอบเป็นภาษาไทยกระชับ ชัดเจน ตรงประเด็น"
)


# prompt เฉพาะเส้นทางเปรียบเทียบ — context ที่ได้เรียงตามเวลา ไม่ใช่ตามคะแนนความเกี่ยวข้อง
# โจทย์จึงต่างจากการตอบปกติ: ต้องชี้ว่า "อะไรเปลี่ยน เมื่อไร" ไม่ใช่ "กฎหมายว่าอย่างไร"
COMPARE_SYSTEM = (
    "คุณคือผู้ช่วยวิเคราะห์ความเปลี่ยนแปลงของตัวบทกฎหมายไทย\n\n"
    "context ที่ให้มาคือ 'ตัวบทมาตราเดียวกัน' หลายรุ่น เรียงตามเวลาจากเก่าไปใหม่ "
    "โดยตัดรุ่นที่ข้อความไม่เปลี่ยนออกแล้ว — แต่ละบล็อกจึงเป็นจุดที่มาตรานี้ถูกแก้จริง\n\n"
    "กฎสำคัญ:\n"
    "- ตอบจาก context เท่านั้น ห้ามเดา ห้ามใช้ความรู้ภายนอก\n"
    "- ชี้ให้ชัดว่า 'ถ้อยคำใดเปลี่ยนไปเป็นอะไร' โดยยกข้อความเดิมกับข้อความใหม่มาเทียบให้เห็น\n"
    "- ระบุว่าการเปลี่ยนแต่ละครั้งเกิดตอนไหน (ตามที่หัวบล็อกบอก)\n"
    "- ⚠️ คัดถ้อยคำ ตัวเลข เลขมาตรา ให้ตรงเป๊ะ ห้ามแก้/ปัด/เดา\n"
    "- สรุปท้ายว่า 'ตัวบทที่ใช้บังคับปัจจุบัน' คือรุ่นไหน และต่างจากฉบับดั้งเดิมอย่างไร\n"
    "- ถ้ามีบล็อกเดียว แปลว่ามาตรานี้ไม่เคยถูกแก้เลย ให้บอกตามนั้น\n"
    "- ตอบเป็นภาษาไทย กระชับ เป็นลำดับเวลา"
)


# คำสั่งให้ LLM ขยายคำถาม (Thai/English) -> คีย์เวิร์ดอังกฤษ + ศัพท์เทคนิคที่กฎใช้จริง
# แก้ปัญหา vocab mismatch (เช่น 'ที่ดินหลวง'↔'สาธารณสมบัติของแผ่นดิน', 'บุกรุก'↔'เข้าไปยึดถือครอบครอง')
EXPAND_SYSTEM = (
    "คุณเป็นตัวช่วยขยายคำค้นสำหรับค้นเอกสารกฎหมายไทย จากคำถาม (ไทย/อังกฤษ) "
    "ให้ output เป็นบรรทัดเดียวของคำค้น/ศัพท์กฎหมายไทย 6-15 คำ ที่น่าจะปรากฏในตัวบทจริง "
    "(คำพ้อง คำที่กฎหมายใช้ เลขมาตรา ชื่อพระราชบัญญัติ) ไม่ต้องอธิบาย ไม่ต้องมีเครื่องหมายคำพูด — คำค้นบรรทัดเดียว"
)


# Multi-query (RAG-Fusion): แตก 1 คำถาม -> 3 มุมค้นหา ทำใน LLM call เดียว
# แต่ละบรรทัด = คีย์เวิร์ดคนละมุม จะถูกเอาไปต่อท้ายคำถามเดิม (คงพิกัด/ตัวเลขเดิมไว้ทุกอัน)
EXPAND_MULTI_SYSTEM = (
    "คุณเป็นตัวช่วยขยายคำค้นเอกสารกฎหมายไทย จากคำถาม ให้ออกมา 3 บรรทัด แต่ละบรรทัดเป็นมุมค้นหาที่ต่างกัน:\n"
    "บรรทัด 1 — คำพ้อง/ศัพท์กฎหมายที่ตัวบทมักใช้จริง\n"
    "บรรทัด 2 — เลขมาตรา/ข้อ/หมวด ชื่อกฎหมาย ตัวเลข วันที่ที่เกี่ยวข้อง\n"
    "บรรทัด 3 — เรียบเรียงคำถามหลักใหม่สั้น ๆ\n"
    "ออกเฉพาะ 3 บรรทัด ไม่มีเลขลำดับ ไม่มีป้ายกำกับ ไม่มีคำอธิบาย"
)


# ══════════════════════════════════════════════════════════════════════════════
# CHUNKING PROFILE — กฎหมายไทย (โปรเจกต์นี้เป็นกฎหมายล้วน ไม่มี multi-domain)
# ══════════════════════════════════════════════════════════════════════════════
# หน่วยตัด chunk: "มาตรา ๕" / "ข้อ ๓" (เลขไทย ๐-๙ หรืออารบิก; เลขอาจอยู่คนละบรรทัดถ้ามาจาก OCR → optional)
_TH_MAJOR_RE = re.compile(r"^\s*(มาตรา|ข้อ)\s*([๐-๙\d]+(?:/[๐-๙\d]+)?)?\b")
# หัวโครงสร้าง (บริบท): หมวด / ภาค / ส่วนที่ / ลักษณะ / บรรพ
_TH_SECTION_RE = re.compile(r"^\s*(หมวด|ภาค|ส่วนที่|ลักษณะ|บรรพ)\s*([๐-๙\d]+)?")
# header/footer ราชกิจจานุเบกษา ที่ซ้ำทุกหน้า
_TH_HEADER_RE = [
    re.compile(r"ราชกิจจานุเบกษา"),
    re.compile(r"^\s*หน้า\s+[๐-๙\d]"),
    re.compile(r"^\s*เล่ม\s+[๐-๙\d]"),
    re.compile(r"ตอนที่\s*[๐-๙\d]"),
]


def _th_label(m) -> str:
    """'มาตรา ๕' / 'หมวด ๑' (ถ้ามีเลข) หรือแค่ 'มาตรา' (เผื่อ OCR แยกเลขคนละบรรทัด)"""
    return f"{m.group(1)} {m.group(2)}".strip() if m.lastindex and m.group(2) else m.group(1)


# โปรไฟล์เดียว = กฎหมายไทย (ยังคงชื่อ DOMAINS/domain ไว้เพื่อไม่ต้องแก้ pipeline ที่อ้างถึง)
DOMAINS = {
    "thai_law": {
        "header_re": _TH_HEADER_RE,
        "major_re": _TH_MAJOR_RE,
        "section_re": _TH_SECTION_RE,
        "major_fmt": _th_label,
        "section_fmt": _th_label,
        "year_re": _YEAR_RE,                 # 25xx (พ.ศ.) / 20xx
        "system_prompt": SYSTEM_PROMPT,       # ไทย (ดูด้านบน)
        "expand_system": EXPAND_SYSTEM,
        "expand_multi_system": EXPAND_MULTI_SYSTEM,
        "user_intro": "เอกสารอ้างอิง (กฎหมายไทย จากราชกิจจานุเบกษา):",
    },
}


def domain_of_group(group: str) -> str:
    """โปรเจกต์นี้เป็นกฎหมายล้วน → คืน 'thai_law' เสมอ"""
    return "thai_law"


def build_llm():
    """คืน ChatOpenAI ชี้ไป endpoint ใหม่ (OpenAI-compatible) — direct RAG"""
    if not LLM_API_KEY:
        raise RuntimeError(
            "ยังไม่ได้ตั้ง env var LLM_API_KEY\n"
            "ตั้งถาวร (PowerShell):  setx LLM_API_KEY \"<token>\"  แล้วเปิด terminal ใหม่"
        )
    return ChatOpenAI(
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY,
        model=LLM_MODEL,
        temperature=0,
        max_tokens=8000,   # เผื่อโมเดล thinking คิดยาว — ให้พื้นที่พอออก 'คำตอบจริง' หลังคิดเสร็จ
        timeout=int(os.environ.get("LLM_TIMEOUT", "600")),  # กันแขวนถาวร (ตั้งผ่าน env ได้)
        max_retries=2,     # openai client ลองซ้ำเองถ้า timeout/5xx
    )


# ── timing ของ run_agent ครั้งล่าสุด (อ่านได้หลังเรียก เหมือน profile_rag) ────────
_last_timing = {"expand_s": 0.0, "retrieve_s": 0.0, "answer_s": 0.0, "total_s": 0.0}


# ── token usage tracker (สะสมจากทุก LLM call ผ่าน endpoint ใหม่) ─────────────────
_token_usage = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}


def track_usage(resp: Any) -> None:
    """ดึง token usage จาก response ของ ChatOpenAI แล้วบวกสะสมลง _token_usage
    (รองรับทั้ง usage_metadata แบบใหม่ และ response_metadata.token_usage แบบเก่า)"""
    try:
        um = getattr(resp, "usage_metadata", None)
        if um:
            _token_usage["prompt"] += int(um.get("input_tokens", 0) or 0)
            _token_usage["completion"] += int(um.get("output_tokens", 0) or 0)
            _token_usage["total"] += int(um.get("total_tokens", 0) or 0)
        else:
            tu = (getattr(resp, "response_metadata", {}) or {}).get("token_usage", {}) or {}
            _token_usage["prompt"] += int(tu.get("prompt_tokens", 0) or 0)
            _token_usage["completion"] += int(tu.get("completion_tokens", 0) or 0)
            _token_usage["total"] += int(tu.get("total_tokens", 0) or 0)
        _token_usage["calls"] += 1
    except Exception:
        pass


def reset_usage() -> None:
    for k in _token_usage:
        _token_usage[k] = 0


# ── invoke + retry (กัน endpoint ส่ง response หล่นกลางคัน) ────────────────────────
# endpoint บางครั้งส่งข้อความกลับมาไม่ครบ (ตัวอักษรช่วงต้น/กลางหล่นหาย) ทำให้
#   - คำตอบขาด เช่น "Minimum.2 หน้า 42)"  → ตอบไม่รู้เรื่อง
#   - JSON ของ judge หัวขาด เช่น {".2.1 ..."}  → parse ไม่ได้ → score=0 ทั้งที่ตอบถูก
# วิธีแก้: invoke แล้วเช็คด้วย ok_fn ถ้าผิดปกติให้ลองใหม่ (default 2 ครั้ง)
ANSWER_MIN_LEN = 25   # คำตอบ RAG ที่สมบูรณ์มักยาวกว่านี้ (มีอ้าง Article/หน้า) — สั้นกว่านี้ = น่าจะหล่น

# streaming: รับ token ทีละตัว → โชว์ heartbeat 'กำลังคิด/กำลังเขียน' + จับ reasoning กัน blank
# เปิดด้วย env RAG_STREAM=1 (ปิด default — ไม่กระทบสคริปต์อื่น)
STREAM_ENABLED = os.environ.get("RAG_STREAM") == "1"


def looks_truncated(text: str) -> bool:
    """เดาว่า response ถูกตัด/หล่นกลางคัน — สั้นผิดปกติ"""
    return len((text or "").strip()) < ANSWER_MIN_LEN


def _stream_invoke(llm: Any, messages: "list[Any]", label: str = "") -> Any:
    """เรียกโมเดลแบบ stream: ปริ้น heartbeat live ('🤔 กำลังคิด' → '✍️ กำลังเขียน')
    ให้ user เห็นว่าไม่ค้าง + จับ reasoning_content ไว้ ถ้า content ว่างจะเอามาใช้แทน (กัน blank)
    คืน AIMessage (มี .content, usage ถูก track แล้ว)"""
    from langchain_core.messages import AIMessage
    t0 = time.perf_counter()
    reasoning_parts: list[str] = []
    content_parts: list[str] = []
    full = None
    n_tok = 0
    last_print = -1.0
    stage = "think"
    try:
        stream = llm.stream(messages, stream_usage=True)   # ขอ token usage ท้าย stream
    except TypeError:
        stream = llm.stream(messages)
    for chunk in stream:
        full = chunk if full is None else full + chunk
        ak = getattr(chunk, "additional_kwargs", {}) or {}
        rc = ak.get("reasoning_content") or ak.get("reasoning") or ""
        c = str(getattr(chunk, "content", "") or "")
        if rc:
            reasoning_parts.append(str(rc)); n_tok += 1
        if c:
            content_parts.append(c); n_tok += 1
            if stage == "think":
                stage = "write"
                sys.stdout.write("\n")   # ขึ้นบรรทัดใหม่ตอนสลับ คิด→เขียน
        now = time.perf_counter() - t0
        if now - last_print >= 0.5:      # throttle การปริ้นไม่ให้ถี่เกิน
            last_print = now
            if stage == "think":
                sys.stdout.write(f"\r       🤔 กำลังคิด... ({n_tok} token, {now:.0f}s)   ")
            else:
                sys.stdout.write(f"\r       ✍️  กำลังเขียนคำตอบ... ({now:.0f}s)   ")
            sys.stdout.flush()
    total = time.perf_counter() - t0
    sys.stdout.write(f"\r       ✓ เสร็จ ({total:.0f}s)                         \n")
    sys.stdout.flush()
    if full is not None:
        track_usage(full)   # track ครั้งเดียวจาก chunk รวม (invoke_retry จะไม่ track ซ้ำ)
    content = "".join(content_parts).strip()
    reasoning = "".join(reasoning_parts).strip()
    if not content and reasoning:
        content = reasoning   # fallback: content ว่าง → เอาส่วนที่คิดมาใช้ (กัน blank=0)
    try:
        return AIMessage(content=content, additional_kwargs={"reasoning_content": reasoning},
                         response_metadata=getattr(full, "response_metadata", {}) or {},
                         usage_metadata=getattr(full, "usage_metadata", None))
    except Exception:
        return AIMessage(content=content)   # เผื่อ langchain เวอร์ชันเก่าไม่รับ usage_metadata


def _invoke_once(llm: Any, messages: "list[Any]", label: str = "") -> Any:
    """เรียกโมเดล 1 ครั้ง + track usage — สลับ stream/ปกติ ตาม STREAM_ENABLED"""
    if STREAM_ENABLED:
        return _stream_invoke(llm, messages, label)   # track ข้างในแล้ว
    resp = llm.invoke(messages)
    track_usage(resp)
    return resp


def invoke_retry(llm: Any, messages: "list[Any]", ok_fn: "Optional[Callable[[str], bool]]" = None,
                 max_retries: int = 2, label: str = "") -> Any:
    """llm.invoke + track_usage + retry ถ้า ok_fn(content) เป็น False (output หล่น/ผิดปกติ)
    คืน response ตัวสุดท้ายเสมอ (ดีสุดที่ได้) — ถ้า ok_fn=None จะไม่ retry"""
    last = None
    for attempt in range(max_retries + 1):
        resp = _invoke_once(llm, messages, label)
        last = resp
        content = str(getattr(resp, "content", "") or "")
        if ok_fn is None or ok_fn(content):
            return resp
        if os.environ.get("RAG_DEBUG") == "1":
            print(f"    [retry {label}] attempt {attempt+1}: output ผิดปกติ "
                  f"({len(content)} ตัว) — ลองใหม่")
    return last


def expand_query(llm: Any, question: str) -> str:
    """ขยายคำถามด้วย LLM เป็นคีย์เวิร์ดอังกฤษ+ศัพท์เทคนิค แล้วต่อท้ายคำถามเดิม
    (คงคำถามเดิมไว้เพื่อรักษาโค้ด/พิกัดที่พิมพ์มาตรง ๆ เช่น Y=535)
    หมายเหตุ: single-query เดิม — ยังใช้โดย speed_test.py / experiment_buckets.py"""
    try:
        r = llm.invoke([SystemMessage(content=EXPAND_SYSTEM), HumanMessage(content=question)])
        track_usage(r)
        kw = " ".join(str(r.content).split())
        if kw:
            return f"{question} {kw}"
    except Exception:
        pass
    return question


def expand_queries(llm: Any, question: str, n: int = 3, domain: str = "thai_law") -> list[str]:
    """Multi-query (RAG-Fusion): คืน n+1 query = [คำถามต้นฉบับ] + n มุมที่ generate
      query[0] = คำถามต้นฉบับ (raw) — สัญญาณ semantic สะอาด ไม่ถูก keyword เบือน
      query[1..n] = คำถามเดิม + คีย์เวิร์ดคนละมุม (ศัพท์กฎหมายไทยที่ตัวบทใช้จริง)
    ทุกอันมีคำถามเดิม -> พิกัด/ตัวเลขไม่เพี้ยน; RRF จะรวมทุกสัญญาณ
    คืนอย่างน้อย [คำถามต้นฉบับ] เสมอ ถ้า LLM พัง"""
    queries = [question]   # query[0] = คำถามต้นฉบับ (raw)
    expand_multi = DOMAINS.get(domain, DOMAINS["thai_law"])["expand_multi_system"]
    try:
        r = llm.invoke([SystemMessage(content=expand_multi), HumanMessage(content=question)])
        track_usage(r)
        for raw in str(r.content).splitlines():
            # ตัด bullet/เลขนำหน้าเผื่อ model ใส่มา เช่น "1) ", "- "
            kw = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", " ".join(raw.split()))
            if kw:
                queries.append(f"{question} {kw}")
            if len(queries) >= n + 1:
                break
    except Exception:
        pass
    return queries


def run_agent(llm: Any, question: str) -> str:
    """Direct RAG: ขยายคำถาม -> retrieve (hybrid) -> ส่ง context ให้ LLM ตอบ
    (พารามิเตอร์ชื่อเดิมเพื่อ compatibility กับ batch_test.py)"""
    debug = os.environ.get("RAG_DEBUG") == "1"
    t0 = time.perf_counter()
    search_qs = expand_queries(llm, question)     # LLM #1 (แตก 1 คำถาม -> หลายมุม)
    t1 = time.perf_counter()
    if debug:
        for i, sq in enumerate(search_qs):
            print(f"    [debug] query[{i}]: {sq[:160]}")
    chunks = retrieve(search_qs, rerank_query=question)  # multi-query + RRF (+ rerank ถ้าเปิด)
    t2 = time.perf_counter()
    if debug:
        locs = ", ".join(f"{c['article']}(p{c['page_start']})" for c in chunks)
        print(f"    [debug] retrieved {len(chunks)} chunks: {locs}")
    context = format_context(chunks)
    user = (
        f"เอกสารอ้างอิง (กฎหมายไทย จากราชกิจจานุเบกษา):\n\n{context}\n\n"
        f"================\n"
        f"คำถาม: {question}\n\n"
        f"ตอบโดยอ้างอิงเฉพาะเอกสารด้านบน พร้อมระบุเลขมาตรา/ข้อ/หน้า"
    )
    resp = invoke_retry(llm, [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user)],
                        ok_fn=lambda c: not looks_truncated(c), label="answer")  # LLM #2
    t3 = time.perf_counter()
    _last_timing.update(expand_s=t1 - t0, retrieve_s=t2 - t1,
                        answer_s=t3 - t2, total_s=t3 - t0)
    return str(resp.content)


# ── main (interactive) ────────────────────────────────────────────────────────
def main():
    print("=== Thai Law RAG (ราชกิจจานุเบกษา) ===")
    print(f"LLM   : {LLM_MODEL}")
    print(f"Embed : {EMBED_MODEL}")
    print(f"Server: {OLLAMA_BASE_URL}\n")

    changed = update_database()
    build_vectorstore(force=changed)

    llm = build_llm()
    print("พร้อมแล้ว! พิมพ์คำถาม (หรือ 'quit' เพื่อออก)\n")

    while True:
        try:
            q = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not q:
            continue
        if q.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break
        try:
            print(f"\nAI: {run_agent(llm, q)}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
