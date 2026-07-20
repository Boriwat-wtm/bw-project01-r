"""
ocr.py — OCR ภาษาไทยแบบ local (EasyOCR) สำหรับ PDF สแกน/รูปภาพ

เรียกจาก rag._load_pdf เฉพาะหน้าที่ "คัดข้อความไม่ได้" (= หน้าสแกน)
- local 100%: โมเดล EasyOCR อยู่ในเครื่อง (~/.EasyOCR), ประมวลผลบน CPU/GPU ของเครื่องนี้
  ไม่ส่งรูป/ข้อความออกเน็ต (download_enabled=False)
- cache ผล OCR ลง ocr_cache/<hash>.txt → OCR แต่ละหน้าครั้งเดียว (OCR แพง/ช้า)
- lazy import: TruePDF (คัดข้อความได้อยู่แล้ว) จะไม่โหลด easyocr/torch เลย
"""
import hashlib
import os

HERE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(HERE, "ocr_cache")
OCR_DPI = int(os.environ.get("OCR_DPI", "200"))       # render หน้า PDF -> รูป ที่ DPI นี้
OCR_LANGS = ["th", "en"]                              # ไทย + อังกฤษ (กฎหมายมักมีอังกฤษปน)

_reader = None


def _get_reader():
    """โหลด EasyOCR ครั้งเดียว (local, ห้ามต่อเน็ต) — ใช้ GPU ถ้ามี"""
    global _reader
    if _reader is None:
        import easyocr
        import torch
        gpu = bool(torch.cuda.is_available())
        _reader = easyocr.Reader(OCR_LANGS, gpu=gpu,
                                 download_enabled=False, verbose=False)
        print(f"  [OCR] โหลด EasyOCR (local, gpu={gpu})")
    return _reader


def _cache_path(path: str, page_index: int) -> str:
    key = f"{os.path.abspath(path)}::{os.path.getmtime(path)}::{page_index}::{OCR_DPI}"
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return os.path.join(CACHE_DIR, f"{h}.txt")


def page_text(path: str, page_index: int) -> str:
    """OCR หน้า page_index (เริ่มจาก 0) ของ PDF -> ข้อความ (cache ไว้ ทำครั้งเดียว)"""
    cf = _cache_path(path, page_index)
    if os.path.exists(cf):
        with open(cf, encoding="utf-8") as f:
            return f.read()

    import fitz  # PyMuPDF — render หน้า PDF เป็นรูป (local, ไม่พึ่ง binary ภายนอก)
    doc = fitz.open(path)
    try:
        pix = doc[page_index].get_pixmap(dpi=OCR_DPI)
        png = pix.tobytes("png")
    finally:
        doc.close()

    reader = _get_reader()
    lines = reader.readtext(png, detail=0)     # detail=0 -> คืน list[str] เรียงตามลำดับอ่าน
    text = "\n".join(lines)

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cf, "w", encoding="utf-8") as f:
        f.write(text)
    return text
