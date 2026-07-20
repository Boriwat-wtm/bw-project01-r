# Thai Law RAG — ประมวลกฎหมายที่ดิน พ.ศ. ๒๔๙๗

ระบบถาม-ตอบกฎหมายไทยแบบ Hybrid RAG บนเอกสารจากสำนักงานคณะกรรมการกฤษฎีกา
ตอบพร้อมอ้างอิงเลขมาตราและหน้าเสมอ — ไม่ตอบจากความรู้นอกเอกสาร

## ติดตั้ง

```powershell
python -m venv .venv
.venv\Scripts\activate

# torch ต้องเป็น CUDA build ถ้ามี GPU (reranker + OCR เร็วขึ้นหลายเท่า)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt pythainlp
```

คัดลอก `.env.example` เป็น `.env` แล้วเติม endpoint + API key จริง:

```
LLM_BASE_URL=http://your-llm-endpoint/v1
LLM_API_KEY=sk-xxxxxxxx
```

## รัน

| คำสั่ง | ได้อะไร |
|---|---|
| `streamlit run app.py` | หน้าเว็บแชท (ประวัติแชท, แหล่งอ้างอิง, 👍👎) |
| `uvicorn main:app --port 8000` | REST/SSE API — docs ที่ `/docs` |
| `python rag.py` | โหมด CLI ถาม-ตอบในเทอร์มินัล |

⚠️ รอบแรกช้า (~สิบนาที) เพราะต้องอ่าน PDF ทุกไฟล์ + OCR หน้าสแกน + สร้าง embedding
รอบถัดไปอ่านจาก `chroma_db/` กับ `ocr_cache/` ทันที

## เอกสารใน `data/`

ชื่อไฟล์เป็นตัวกำหนด group/metadata เอง (ไม่ต้องแยกโฟลเดอร์) — ดู `rag.parse_doc_name()`

```
LandCode2497_<Kind>-v<Ver>_<True|Img>PDF.pdf
```

| Kind | group ที่ได้ | ค้นเมื่อไร |
|---|---|---|
| `Update-vlast` | ฉบับใช้บังคับปัจจุบัน | 🟢 **ค่าเริ่มต้น** — ทุกคำถามที่ไม่ระบุปี |
| `Update-v1..v20` | ฉบับย้อนหลังตามช่วงเวลา | 🔒 เมื่อคำถามระบุ พ.ศ. |
| `Amend-v1..v15` | ประวัติการแก้ไข | 🔒 เมื่อถามว่าแก้อะไร/เมื่อไร |
| `Main-v0` | ฉบับดั้งเดิม พ.ศ. ๒๔๙๗ | 🔒 เมื่อถามตัวบทดั้งเดิม |

`TruePDF` = คัดข้อความได้ · `ImgPDF` = สแกน → เข้า OCR ภาษาไทยอัตโนมัติ (local, ดู `ocr.py`)

### ⚖️ ทำไม default ต้องเป็น `vlast` ไฟล์เดียว

กฤษฎีกาไล่ปรับตัวบทให้ทีละมาตราแล้วในฉบับรวมสะสม เทียบทั้งฉบับกับ `Main-v0` ได้ว่า:

```
เหมือนเป๊ะกับฉบับหลัก (ไม่เคยถูกแก้)   44 มาตรา
เนื้อหาต่าง (ถูกแก้ไปแล้ว)             57 มาตรา
มาตราที่เพิ่มใหม่ (๘ ทวิ, ๙/๑ ฯลฯ)     33 มาตรา
ตกหล่น                                  0 มาตรา
```

`vlast` จึงครอบคลุมทั้ง *"มาตราที่ไม่เคยแก้ → ใช้ของหลัก"* และ *"มาตราที่แก้แล้ว → ใช้ของใหม่"* อยู่ในไฟล์เดียว

ไฟล์ที่เหลือมี **ตัวบทที่ถูกยกเลิกไปแล้วปนอยู่** (เช่น `มาตรา ๙๗` ใน `Main-v0` เป็นเวอร์ชันก่อนแก้)
ถ้าค้นปนกันเมื่อไร = มีโอกาสตอบด้วยกฎหมายที่เลิกใช้แล้ว → `retrieve()` จึงล็อกไว้ที่กลุ่มที่ใช้บังคับเสมอ
การปลดล็อกตัดสิน**ด้วยกฎในโค้ด** (`service.pick_groups`) ไม่ใช่ให้ LLM เดา

นอกจากนี้ยังยุบสำเนาซ้ำ (`_dedupe_versions` — 29% ของดัชนี) และดัน chunk จาก OCR ท้ายแถว (`_demote_scans`)

## สถาปัตยกรรม

```
PDF ──PyMuPDF──> chunk ตามขอบ "มาตรา/ข้อ" ──> Chroma (semantic)
  └─ หน้าสแกน ──EasyOCR──┘                 └─> BM25 (ตัดคำไทย newmm)

ถาม ──> rewrite คำถามต่อเนื่อง ──> เลือกกลุ่ม (LLM router) ──> ขยายคำค้น 3 มุม
     ──> hybrid retrieve (RRF) ──> ยุบเวอร์ชันซ้ำ ──> rerank ──> LLM ตอบ
```

| ไฟล์ | หน้าที่ |
|---|---|
| `rag.py` | เครื่องยนต์: โหลด/chunk/embed/retrieve/rerank + prompt |
| `service.py` | pipeline ระดับแอป (routing, multi-turn) — ไม่ผูก UI |
| `app.py` | Streamlit UI |
| `main.py` | FastAPI |
| `ocr.py` | OCR ไทยแบบ local (EasyOCR + PyMuPDF) |

### จุดที่ต้องระวัง

- **ห้ามใช้ pdfplumber คัดข้อความไทย** — มันเรียงตาม x-coordinate ทำให้สระ/วรรณยุกต์
  หลุดตำแหน่ง (`"ทรพั ยส์ นิ"` แทน `"ทรัพย์สิน"`) พังทั้ง embedding, BM25 และตัวบทที่ LLM คัดไปตอบ
- **เลขไทย ๐-๙** ถูก normalize เป็นอารบิกตอน tokenize → ถาม "มาตรา 9" เจอ "มาตรา ๙"
- เปลี่ยน `CHUNK_SIZE` / `EMBED_MODEL` ต้อง rebuild: `python -c "import rag; rag.build_vectorstore(force=True)"`
- reranker ใช้ GPU ตัวเดียว → FastAPI ต้องรัน **worker เดียว** (อย่าใช้ `--workers >1`)

## ตัวแปร env ที่ใช้บ่อย

| ตัวแปร | ค่าเริ่มต้น | ผล |
|---|---|---|
| `RAG_RERANK` | `1` (API) / `0` (CLI) | เปิด cross-encoder reranker |
| `RAG_DEDUPE` | `1` | ยุบมาตราซ้ำข้ามฉบับรวมสะสม |
| `RAG_ART_BOOST` | `1` | ถามเจาะ "มาตรา ๙" → ดันมาตรานั้นขึ้นอันดับ 1 |
| `RAG_SCAN_DEMOTE` | `1` | ดัน chunk จาก OCR ท้ายแถว |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1500` / `250` | ขนาด chunk (เปลี่ยนแล้วต้อง rebuild) |
| `OCR_DPI` | `200` | ความละเอียดตอน render หน้าสแกนเข้า OCR |
| `API_KEY` | — | ตั้งแล้ว FastAPI จะบังคับ header `X-API-Key` |
