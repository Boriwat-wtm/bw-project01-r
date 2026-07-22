"""
ocr_bench.py — วัดคุณภาพ OCR โดยใช้ "ไฟล์คู่" ที่มีอยู่แล้วเป็นเฉลย

    python eval/ocr_bench.py                          # วัดตัวปัจจุบัน (EasyOCR)
    python eval/ocr_bench.py --engine company         # วัดตัวบริษัท
    python eval/ocr_bench.py --engine company --mlflow --tag company-ocr

═══ ทำไมถึงวัดได้โดยไม่ต้องนั่งพิมพ์เฉลยเอง ═══
data/ มีเอกสารเดียวกันสองรูปแบบ:
    LandCode2497_Amend-v3_ImgPDF.pdf    สแกนจากราชกิจจานุเบกษา  -> ต้อง OCR
    LandCode2497_Amend-v3_TruePDF.pdf   ตัวอักษรจริงจากกฤษฎีกา  -> ใช้เป็นเฉลย
คู่แบบนี้มี 16 คู่ ในจำนวนนี้ 9 คู่ที่ฝั่ง Img ไม่มี text layer เลย = ต้อง OCR จริง

⚠️ ข้อจำกัดที่ต้องรู้ก่อนอ่านผล
สองไฟล์ไม่ได้เหมือนกัน 100% — ฉบับราชกิจจานุเบกษามีหัวหนังสือ/ลายเซ็น/หมายเหตุ
ที่ฉบับกฤษฎีกาไม่มี (จำนวนหน้าจึงไม่เท่ากัน เช่น Main-v0 = 67 หน้า vs 36 หน้า)
ดังนั้น char_sim จะไม่มีวันได้ 1.00 แม้ OCR จะสมบูรณ์แบบ
  -> ใช้ตัวเลขนี้ "เทียบระหว่างเครื่อง OCR ด้วยกัน" เท่านั้น อย่าอ่านเป็นคะแนนดิบ
  -> ตัวที่อ่านเป็นคะแนนดิบได้คือ article_recall และ graph_phrase (ดูด้านล่าง)

═══ วัด 4 อย่าง เรียงตามความสำคัญต่อ "ระบบนี้" ไม่ใช่ตาม OCR ทั่วไป ═══
1. article_recall  เลข "มาตรา N" ที่เฉลยมี OCR อ่านเจอกี่ %
                   -> ต่ำ = chunk ถูกติดป้ายมาตราผิด = ค้นยังไงก็ไม่เจอ
2. graph_phrase    วลีที่ amendment graph ใช้ต่อสายการแก้ไข อยู่ครบไหม
                   ("ให้ยกเลิกความใน", "แก้ไขเพิ่มเติมโดย", "เพิ่มความต่อไปนี้เป็น")
                   -> ต่ำ = สายการแก้ไขขาด = ตอบ "ฉบับก่อนหน้าของก่อนหน้า" ไม่ได้
3. digit_acc       ลำดับเลขไทยตรงกับเฉลยแค่ไหน
                   -> ในกฎหมายเลขผิดตัวเดียวความหมายเปลี่ยน (ห้ามโอน ๕ ปี vs ๑๐ ปี)
4. char_sim        ความเหมือนระดับตัวอักษร (สระ/วรรณยุกต์ไทยหลุดไหม)

ผลออกเป็นตาราง + ไฟล์ JSON และส่งขึ้น MLflow ได้ (--mlflow)
"""
import argparse
import difflib
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import fitz          # noqa: E402  PyMuPDF
import rag           # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# วลีที่ rag.amendment_graph() ใช้แกะสายการแก้ไข — OCR ทำวลีพวกนี้พังเมื่อไร กราฟขาดทันที
GRAPH_PHRASES = ["ให้ยกเลิกความใน", "แก้ไขเพิ่มเติมโดย", "เพิ่มความต่อไปนี้เป็น",
                 "ให้ใช้ความต่อไปนี้แทน", "ประมวลกฎหมายที่ดิน"]

_ART_RE = re.compile(r"มาตรา\s*([๐-๙]+)")
_DIGIT_RE = re.compile(r"[๐-๙]+")


# ══════════════════════════════════════════════════════════════════════════════
# เครื่อง OCR — เพิ่มตัวใหม่ที่นี่ที่เดียว ส่วนที่เหลือของไฟล์ไม่ต้องแก้
# ══════════════════════════════════════════════════════════════════════════════
# เครื่อง OCR แต่ละตัวรับ (path ของ PDF, จำนวนหน้า) แล้วคืน "ข้อความทั้งเอกสาร"
# ใช้ระดับเอกสารเพราะบางเจ้ารับทั้งไฟล์ ไม่ได้รับทีละหน้า (เช่นตัวของบริษัท)
NO_CACHE = False       # ตั้งด้วย --no-cache

# prompt สั่ง OCR — ใช้กับเครื่องที่เป็น LLM อ่านภาพ (ไม่ใช่ OCR แบบเดิม)
#
# ⚠️ จงใจ "ไม่" ใส่คำสั่งให้แก้คำผิดตามบริบท ซึ่ง prompt ต้นฉบับของบริษัทมี
#    ("Correct Thai misspellings and tone marks based on context")
#    เพราะกับตัวบทกฎหมาย การให้โมเดลเดาแทนเราคือความเสี่ยง: มันอาจเปลี่ยน
#    "ห้ามโอน ๕ ปี" เป็น "๑๐ ปี" เพราะบริบทดูเข้าท่ากว่า แล้วออกมาอ่านลื่นจน
#    ไม่มีอะไรส่อว่าผิด — ผิดแบบเงียบอันตรายกว่าผิดแบบเห็นชัด
#    อยากลองแบบเดิมให้ตั้ง env OCR_PROMPT ทับ แล้ววัดเทียบดูว่า digit_acc ต่างไหม
STRICT_PROMPT = """Perform verbatim OCR of this Thai legal document. Output ONLY raw text.
1. If the page is blank or contains only artifacts, output ONLY: "EMPTY".
2. Follow strict top-to-bottom reading order. Preserve line breaks and indentation.
3. Transcribe EXACTLY what is printed, character by character.
4. Do NOT correct spelling, tone marks, or numbers. Do NOT guess unclear characters
   from context — if a character is unreadable, output "?" instead.
5. Thai numerals must be transcribed as Thai numerals, digit by digit.
6. Ignore logos, seals, stamps, and page headers/footers."""


def engine_easyocr(pdf_path: str, n_pages: int) -> str:
    """ตัวที่ใช้อยู่ตอนนี้ — local 100% ไม่ส่งข้อมูลออกเน็ต (มี cache ใน ocr_cache/)

    ⚠️ cache ทำให้ตัวเลข "วิ/หน้า" เป็น 0 — ตอนเทียบความเร็วกับเครื่องอื่นต้องใช้ --no-cache
    ไม่งั้นเท่ากับเอา "เวลาอ่านไฟล์" ไปแข่งกับ "เวลา OCR จริง" ซึ่งไม่ยุติธรรม"""
    import ocr
    out = []
    for i in range(n_pages):
        if not NO_CACHE:
            out.append(ocr.page_text(pdf_path, i))
            continue
        doc = fitz.open(pdf_path)
        try:
            png = doc[i].get_pixmap(dpi=ocr.OCR_DPI).tobytes("png")
        finally:
            doc.close()
        out.append("\n".join(ocr._get_reader().readtext(png, detail=0)))
    return "\n".join(out)


# ── OCR ของบริษัท (external service) — เป็นงานแบบคิว 3 จังหวะ ─────────────────
#   ① POST /v3/ai-process-file            อัปโหลดทั้งไฟล์ -> ได้ job_id
#   ② GET  .../{job_id}/status            ถามซ้ำจนกว่าจะ completed
#   ③ GET  .../{job_id}/result            ดึงข้อความออกมา
# ต่างจาก EasyOCR ที่ยืนรอหน้าเตา — อันนี้รับบัตรคิวแล้วค่อยมารับของ
COMPANY_CACHE = os.path.join(os.path.dirname(DATA_DIR), "ocr_cache_company")


def _company_cfg() -> dict:
    """อ่านค่าจาก .env — รองรับ auth 2 แบบ เพราะคู่มือใช้ Bearer key แต่บางคนได้มา
    เป็น username/password ให้ใส่แบบไหนก็ได้ที่มี (Bearer มาก่อนถ้ามีทั้งคู่)"""
    base = os.environ.get("OCR_BASE_URL")
    if not base:
        raise SystemExit("ยังไม่ได้ตั้ง OCR_BASE_URL ใน .env (ดู .env.example)")

    # service ประกาศไว้ใน openapi.json ว่า securitySchemes = HTTPBearer เท่านั้น
    # ส่งแบบอื่น (Basic ฯลฯ) FastAPI จะตอบ 403 ทันที -> ต้องเป็น token เท่านั้น
    key = os.environ.get("OCR_API_KEY")
    if not key:
        extra = ""
        if os.environ.get("OCR_USERNAME") or os.environ.get("OCR_PASSWORD"):
            extra = ("\n    ⚠️ เจอ OCR_USERNAME/OCR_PASSWORD ใน .env แต่ service นี้ใช้ไม่ได้ —\n"
                     "       มันรับเฉพาะ Bearer token (ตรวจจาก openapi.json แล้ว)\n"
                     "       ต้องขอ API key จากทีมที่ดูแล service มาใส่แทน")
        raise SystemExit("ยังไม่ได้ตั้ง OCR_API_KEY ใน .env (ดู .env.example)" + extra)
    auth = {"Authorization": f"Bearer {key}"}

    return {
        "base": base.rstrip("/"),
        "headers": auth,
        # service ภายในใช้ self-signed cert -> ปกติต้องปิด verify
        # ตั้ง OCR_VERIFY_SSL=1 ได้ถ้าวันหนึ่งเปลี่ยนไปใช้ cert จริง
        "verify": os.environ.get("OCR_VERIFY_SSL") == "1",
        "model": os.environ.get("OCR_MODEL_NAME", "qwen/qwen3.5-27b"),
        "pre_engine": os.environ.get("OCR_PRE_ENGINE", "tesseract"),
        "prompt": os.environ.get("OCR_PROMPT", STRICT_PROMPT),
        "poll": float(os.environ.get("OCR_POLL_SEC", "3")),
        "max_wait": float(os.environ.get("OCR_MAX_WAIT_SEC", "7200")),
    }


def _company_pages(results: dict) -> list[str]:
    """ดึงข้อความรายหน้าจาก results.pages[] — เผื่อคีย์ไว้หลายแบบกันสเปกเปลี่ยน"""
    out = []
    for pg in results.get("pages") or []:
        if isinstance(pg, str):
            out.append(pg)
            continue
        txt = ((pg.get("ai_processing") or {}).get("content")
               or pg.get("content") or pg.get("text") or pg.get("markdown") or "")
        out.append(txt)
    return out


def engine_company(pdf_path: str, n_pages: int) -> str:
    """ส่ง PDF ทั้งไฟล์เข้าคิว OCR ของบริษัท แล้วรอผล

    ตั้ง disable_structure=true เพราะเราต้องการแค่ "ข้อความ" — ส่วนสกัด JSON
    ของ service นั้นออกแบบมาสำหรับเอกสารอีกประเภท ไม่เกี่ยวกับตัวบทกฎหมายของเรา
    (และการปิดไว้ทำให้เร็วขึ้นกับตัดตัวแปรออกไปหนึ่งตัวตอนเทียบผล)"""
    import requests
    import urllib3

    cfg = _company_cfg()
    if not cfg["verify"]:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    # cache แยกจาก EasyOCR — งานนี้ช้าและมีค่าใช้จ่าย ทำซ้ำฟรีดีกว่า
    key = f"{os.path.abspath(pdf_path)}::{os.path.getmtime(pdf_path)}::{cfg['model']}::{cfg['prompt'][:40]}"
    cf = os.path.join(COMPANY_CACHE, __import__("hashlib").md5(key.encode()).hexdigest() + ".txt")
    if not NO_CACHE and os.path.exists(cf):
        with open(cf, encoding="utf-8") as f:
            return f.read()

    # ① อัปโหลดเข้าคิว
    with open(pdf_path, "rb") as f:
        r = requests.post(
            f"{cfg['base']}/v3/ai-process-file",
            headers=cfg["headers"],
            files=[("file", (os.path.basename(pdf_path), f, "application/pdf"))],
            data={"pages": "",                       # ว่าง = ทุกหน้า
                  "prompt": cfg["prompt"],
                  "ocr_engine": cfg["pre_engine"],
                  "model": cfg["model"],
                  "extraction_mode": "one_per_page",  # อยากได้ผลแยกรายหน้า
                  "use_thinking": "false",
                  "disable_structure": "true"},       # เอา OCR อย่างเดียว
            verify=cfg["verify"], timeout=(10, 600),
        )
    r.raise_for_status()
    job = r.json()
    job_id = job.get("job_id") or job.get("id")
    if not job_id:
        raise RuntimeError(f"อัปโหลดแล้วไม่ได้ job_id — คีย์ที่ได้: {list(job)[:8]}")

    # ② ถามสถานะจนกว่าจะเสร็จ
    t0 = time.time()
    while True:
        st = requests.get(f"{cfg['base']}/v3/ai-process-file/{job_id}/status",
                          headers=cfg["headers"], verify=cfg["verify"],
                          timeout=(10, 60)).json()
        status = str(st.get("status", "")).lower()
        if status in ("completed", "success", "succeeded"):
            break
        if status in ("failed", "error"):
            raise RuntimeError(f"งาน OCR ล้มเหลว: {st.get('error') or st}")
        if time.time() - t0 > cfg["max_wait"]:
            raise TimeoutError(f"รอเกิน {cfg['max_wait']:.0f} วินาที (status ล่าสุด={status})")
        time.sleep(cfg["poll"])

    # ③ ดึงผล — เอารายหน้าก่อน ถ้าไม่มีค่อยใช้ก้อนรวม
    res = requests.get(f"{cfg['base']}/v3/ai-process-file/{job_id}/result",
                       headers=cfg["headers"], verify=cfg["verify"],
                       timeout=(10, 600)).json()
    results = res.get("results") or res
    pages = _company_pages(results)
    text = "\n".join(pages) if any(pages) else (results.get("combined_markdown") or "")
    if not text.strip():
        raise RuntimeError(f"ไม่ได้ข้อความกลับมา — คีย์ใน results: {list(results)[:8]}")

    os.makedirs(COMPANY_CACHE, exist_ok=True)
    with open(cf, "w", encoding="utf-8") as f:
        f.write(text)
    return text


ENGINES = {"easyocr": engine_easyocr, "company": engine_company}


# ══════════════════════════════════════════════════════════════════════════════
def find_pairs() -> list[dict]:
    """คู่ ImgPDF/TruePDF ทั้งหมด พร้อมบอกว่าคู่ไหน "ต้อง OCR จริง"
    (ฝั่ง Img บางไฟล์มี text layer ติดมาอยู่แล้ว — rag.py จะไม่เรียก OCR เลย ดู OCR_MIN_CHARS)"""
    pairs = []
    for fn in sorted(os.listdir(DATA_DIR)):
        if not fn.endswith("_ImgPDF.pdf"):
            continue
        tru = os.path.join(DATA_DIR, fn.replace("_ImgPDF", "_TruePDF"))
        if not os.path.exists(tru):
            continue
        img = os.path.join(DATA_DIR, fn)
        d = fitz.open(img)
        try:
            embedded = sum(len(d[i].get_text().strip()) for i in range(d.page_count))
            pages = d.page_count
        finally:
            d.close()
        pairs.append({"name": fn.replace("LandCode2497_", "").replace("_ImgPDF.pdf", ""),
                      "img": img, "true": tru, "pages": pages,
                      "needs_ocr": embedded < rag.OCR_MIN_CHARS * pages})
    return pairs


def read_true(path: str) -> str:
    """ตัวบทจากไฟล์ TruePDF — ตัด header กฤษฎีกาที่ประทับทุกหน้าออก (สแกนไม่มีอันนี้)"""
    d = fitz.open(path)
    try:
        raw = "\n".join(d[i].get_text() for i in range(d.page_count))
    finally:
        d.close()
    return rag._DOC_NOISE_RE.sub(" ", raw)


def score(ocr_text: str, true_text: str) -> dict:
    """เทียบ OCR กับเฉลย — คืน 4 ตัวชี้วัด (ดูคำอธิบายหัวไฟล์)"""
    o_norm, t_norm = rag.law_text_norm(ocr_text), rag.law_text_norm(true_text)

    # 1) เลขมาตรา — วัดแบบ set เพราะเราสนใจ "หาเจอไหม" ไม่ใช่ "เจอกี่ครั้ง"
    t_arts, o_arts = set(_ART_RE.findall(true_text)), set(_ART_RE.findall(ocr_text))
    art_recall = len(t_arts & o_arts) / len(t_arts) if t_arts else None

    # 2) วลีที่ amendment graph ต้องใช้ — นับเฉพาะวลีที่เฉลย "มี" จริงเท่านั้น
    want = [p for p in GRAPH_PHRASES if p.replace(" ", "") in t_norm]
    got = [p for p in want if p.replace(" ", "") in o_norm]
    phrase_recall = len(got) / len(want) if want else None

    # 3) เลขไทย — เทียบเป็น "ลำดับ" ไม่ใช่ set เพราะตำแหน่งมีความหมาย (ปี/จำนวนวัน/ลำดับมาตรา)
    t_dig, o_dig = _DIGIT_RE.findall(true_text), _DIGIT_RE.findall(ocr_text)
    digit_acc = (difflib.SequenceMatcher(None, t_dig, o_dig).ratio() if t_dig else None)

    # 4) ตัวอักษร — จับสระ/วรรณยุกต์ที่หลุด (ปัญหาคลาสสิกของ OCR ไทย)
    char_sim = difflib.SequenceMatcher(None, t_norm, o_norm).ratio() if t_norm else None

    return {"article_recall": art_recall, "graph_phrase": phrase_recall,
            "digit_acc": digit_acc, "char_sim": char_sim,
            "articles_missed": sorted(t_arts - o_arts), "n_articles": len(t_arts),
            "phrases_missed": [p for p in want if p not in got]}


def run(engine: str, pairs: list[dict], all_pairs: bool) -> list[dict]:
    fn = ENGINES[engine]
    todo = pairs if all_pairs else [p for p in pairs if p["needs_ocr"]]
    results = []
    for i, p in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {p['name']} ({p['pages']} หน้า) ...", end=" ", flush=True)
        t0 = time.perf_counter()
        try:
            text = fn(p["img"], p["pages"])
            err = None
        except Exception as e:                      # เครื่องนี้พังก็ให้ตัวอื่นวัดต่อได้
            text, err = "", f"{type(e).__name__}: {e}"
        elapsed = time.perf_counter() - t0
        r = {"name": p["name"], "pages": p["pages"], "sec": round(elapsed, 1),
             "sec_per_page": round(elapsed / max(1, p["pages"]), 1), "error": err}
        r.update(score(text, read_true(p["true"])) if not err else {})
        results.append(r)
        print(f"{r.get('article_recall', 0) or 0:.0%} มาตรา · {elapsed:.0f}s"
              if not err else f"❌ {err[:60]}")
    return results


def pct(v) -> str:
    return "  —  " if v is None else f"{v:6.1%}"


def summarise(engine: str, results: list[dict]) -> dict:
    ok = [r for r in results if not r.get("error")]
    print("\n" + "═" * 78)
    print(f"เครื่อง OCR: {engine}")
    print("═" * 78)
    print(f"{'เอกสาร':<14}{'มาตรา':>8}{'วลีกราฟ':>10}{'เลขไทย':>9}{'ตัวอักษร':>10}"
          f"{'วิ/หน้า':>9}  มาตราที่หาย")
    print("─" * 78)
    for r in results:
        if r.get("error"):
            print(f"{r['name']:<14}  ❌ {r['error'][:50]}")
            continue
        missed = ",".join(r["articles_missed"][:6]) or "-"
        print(f"{r['name']:<14}{pct(r['article_recall']):>8}{pct(r['graph_phrase']):>10}"
              f"{pct(r['digit_acc']):>9}{pct(r['char_sim']):>10}"
              f"{r['sec_per_page']:>9.1f}  {missed}")

    avg = {k: (sum(r[k] for r in ok if r.get(k) is not None)
               / max(1, sum(1 for r in ok if r.get(k) is not None)))
           for k in ("article_recall", "graph_phrase", "digit_acc", "char_sim")}
    tot_pages = sum(r["pages"] for r in ok)
    tot_sec = sum(r["sec"] for r in ok)
    print("─" * 78)
    print(f"{'เฉลี่ย':<14}{pct(avg['article_recall']):>8}{pct(avg['graph_phrase']):>10}"
          f"{pct(avg['digit_acc']):>9}{pct(avg['char_sim']):>10}"
          f"{tot_sec / max(1, tot_pages):>9.1f}")
    print(f"\nรวม {len(ok)} เอกสาร · {tot_pages} หน้า · {tot_sec:.0f} วินาที"
          f" · ล้มเหลว {len(results) - len(ok)}")
    print("\n⚠️ char_sim เทียบข้ามเครื่องเท่านั้น — ฉบับราชกิจจานุเบกษามีหัวหนังสือ/ลายเซ็น"
          "\n   ที่ฉบับกฤษฎีกาไม่มี จึงไม่มีทางได้ 1.00 แม้ OCR จะถูกหมด")
    avg.update({"pages": tot_pages, "sec": round(tot_sec, 1),
                "sec_per_page": round(tot_sec / max(1, tot_pages), 2),
                "n_docs": len(ok), "n_failed": len(results) - len(ok)})
    return avg


def log_mlflow(engine: str, avg: dict, out_path: str, tag: str) -> str:
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "thai-law-rag-eval"))
    with mlflow.start_run(run_name=f"ocr-{tag or engine}") as run:
        mlflow.log_params({"kind": "ocr_bench", "engine": engine,
                           "ocr_dpi": os.environ.get("OCR_DPI", "200"),
                           "n_docs": avg["n_docs"], "pages": avg["pages"]})
        mlflow.log_metrics({k: v for k, v in avg.items() if isinstance(v, (int, float))})
        mlflow.log_artifact(out_path)
        return run.info.run_id


def main() -> int:
    ap = argparse.ArgumentParser(description="วัดคุณภาพ OCR ด้วยคู่ ImgPDF/TruePDF")
    ap.add_argument("--engine", default="easyocr", choices=sorted(ENGINES))
    ap.add_argument("--all", action="store_true",
                    help="วัดทุกคู่ รวมคู่ที่ฝั่ง Img มี text layer อยู่แล้ว (ปกติข้าม)")
    ap.add_argument("--only", nargs="*", help="เจาะเฉพาะเอกสาร เช่น --only Amend-v3 Main-v0")
    ap.add_argument("--no-cache", action="store_true",
                    help="ไม่ใช้ ocr_cache — จำเป็นตอนวัดความเร็วจริงเพื่อเทียบข้ามเครื่อง")
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    global NO_CACHE
    NO_CACHE = a.no_cache

    pairs = find_pairs()
    if a.only:
        pairs = [p for p in pairs if p["name"] in set(a.only)]
    n_ocr = sum(1 for p in pairs if p["needs_ocr"])
    print(f"พบคู่เทียบ {len(pairs)} คู่ — ต้อง OCR จริง {n_ocr} คู่"
          f"{'' if a.all else ' (วัดเฉพาะกลุ่มนี้)'}\n")

    results = run(a.engine, pairs, a.all)
    avg = summarise(a.engine, results)

    # ⚠️ ต้องมีเวลากำกับในชื่อไฟล์ (แบบเดียวกับ run_eval.py) — ตอนแรกใช้ชื่อตายตัวแล้วรอบ
    # ที่รันทีหลังเขียนทับผลรอบเต็มหายไปเลย ผลการวัดต้องสะสมได้ ไม่ใช่เหลือแค่รอบล่าสุด
    stamp = time.strftime("%Y%m%d_%H%M%S")
    scope = "" if not a.only else "_partial"
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f"ocr_bench_{a.engine}{('_' + a.tag) if a.tag else ''}{scope}_{stamp}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"engine": a.engine, "summary": avg, "results": results},
                  f, ensure_ascii=False, indent=2)
    print(f"\nบันทึกผลดิบ: {out}")

    if a.mlflow:
        print(f"MLflow run: {log_mlflow(a.engine, avg, out, a.tag)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
