# Thai Law RAG — อิมเมจสำหรับรัน API (ประมวลกฎหมายที่ดิน)
#
# ตั้งใจให้ "พึ่งพาตัวเองได้" — ดัชนี เอกสาร และโมเดล reranker อยู่ในอิมเมจครบ
# เครื่องปลายทางจึงไม่ต้องต่อ HuggingFace และไม่ต้อง embed ใหม่ตอนบูต
# สิ่งเดียวที่ต้องต่อออกไปคือ endpoint ของ LLM/embedding ตามที่ตั้งใน .env
#
# build : docker compose build
# รัน   : docker compose up -d
FROM python:3.12-slim

# libgl1/libglib2 = ของที่ opencv (ใต้ easyocr) ต้องใช้ — ติดไว้เผื่อกรณีต้อง OCR จริง
# curl = ใช้ทำ healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

# สร้างผู้ใช้ตั้งแต่ต้น เพื่อให้ไฟล์ก้อนใหญ่ "เกิดมาเป็นของ app" เลย
# ⚠️ ห้ามใช้ chown -R กับโฟลเดอร์ที่มีไฟล์ใหญ่ทีหลัง — overlayfs จะคัดลอกไฟล์ทั้งก้อน
#    ขึ้นชั้นใหม่เพราะ metadata เปลี่ยน วัดแล้วโมเดล 2.3 GB ถูกทำสำเนาซ้ำ = เสียเปล่า 2.41 GB
RUN useradd -m -u 10001 app

WORKDIR /app

# ── ชั้น dependency แยกจากชั้นโค้ด: แก้โค้ดแล้วไม่ต้องลง pip ใหม่ ──
COPY requirements.txt .
# torch ไม่ได้ตรึงใน requirements.txt เพราะ build ผูกกับ CUDA ของแต่ละเครื่อง
# ค่าปริยายคือตัว CPU (เล็กกว่าราว 2 GB) — docker-compose.gpu.yml ส่ง arg นี้มาทับเป็น cu124
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
# ⚠️ ต้องลง torchvision พร้อมกันและต้องมี -c ล็อกไว้ตอนลง requirements.txt
#    easyocr ต้องใช้ torchvision ถ้าไม่ได้ลงไว้ก่อน pip จะไปหยิบจาก PyPI แล้ว
#    "ลาก torch ตัว CUDA มาทับ" ตัว CPU ที่เพิ่งลงไป -- วัดแล้วได้ torch 2.13.0
#    พร้อม nvidia-* 2.7 GB + triton 691 MB ทั้งที่อิมเมจนี้ไม่ได้ใช้ GPU
#    และเวอร์ชันก็ไม่ตรงกับชุดที่วัดผล 98/100 ไว้ด้วย
RUN pip install --no-cache-dir --index-url "$TORCH_INDEX_URL" \
        torch==2.6.0 torchvision==0.21.0 \
    && printf 'torch==2.6.0\ntorchvision==0.21.0\n' > /tmp/torch-lock.txt \
    && pip install --no-cache-dir -c /tmp/torch-lock.txt -r requirements.txt \
    && rm /tmp/torch-lock.txt \
    && TORCH_INDEX_URL="$TORCH_INDEX_URL" python - <<'PY'
import os
import torch
import torchvision
want_cpu = os.environ["TORCH_INDEX_URL"].rstrip("/").endswith("cpu")
cuda = torch.version.cuda
assert torch.__version__.startswith("2.6.0"), f"torch ถูกอัปทับเป็น {torch.__version__}"
if want_cpu:
    assert not cuda, f"ขอตัว CPU แต่ได้ตัว CUDA ({cuda}) — จะพก nvidia-* มาหลาย GB"
else:
    assert cuda, "ขอตัว CUDA แต่ได้ตัว CPU — reranker จะช้ามาก"
print(f"torch {torch.__version__} · torchvision {torchvision.__version__} · "
      f"cuda={cuda or 'ไม่มี (CPU)'}")
PY

# ── โหลด reranker เข้าอิมเมจตั้งแต่ตอน build ──
# ไม่งั้นคำถามแรกจะช้า ~85 วินาที และเครื่องที่ออกเน็ตไม่ได้จะรันไม่ได้เลย
# โหลดในนาม app ตั้งแต่แรก ไฟล์ 2.3 GB จะได้ไม่ต้องถูก chown ทีหลัง
ENV HF_HOME=/opt/hf
RUN mkdir -p /opt/hf && chown app:app /opt/hf && chown app:app /app
USER app
RUN python -c "from sentence_transformers import CrossEncoder; \
CrossEncoder('BAAI/bge-reranker-v2-m3')" \
    && find /opt/hf -name '*.h5' -delete

# ── โค้ด + ข้อมูล + ดัชนี ──
# ocr_cache ต้องมาด้วย ไม่งั้นหน้าสแกน 148 หน้าจะถูก OCR ใหม่ตอนบูต
# และถ้าเครื่องนั้นไม่มีโมเดล EasyOCR จะได้ข้อความว่าง -> chunk เลื่อน -> ดัชนีเพี้ยน
# --chown ตอนคัดลอก = ไม่ต้องมีชั้น chown แยกอีก (chroma ต้องเขียนไฟล์ได้)
COPY --chown=app:app . .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAG_RERANK=1

EXPOSE 8000

# start-period เผื่อไว้ยาว — ตอนบูตต้องอ่าน PDF 53 ไฟล์ สร้าง BM25 และอุ่น reranker
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
