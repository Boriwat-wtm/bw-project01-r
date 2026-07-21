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
NO_CACHE = False       # ตั้งด้วย --no-cache


def engine_easyocr(pdf_path: str, page_index: int) -> str:
    """ตัวที่ใช้อยู่ตอนนี้ — local 100% ไม่ส่งข้อมูลออกเน็ต (มี cache ใน ocr_cache/)

    ⚠️ cache ทำให้ตัวเลข "วิ/หน้า" เป็น 0 — ตอนเทียบความเร็วกับเครื่องอื่นต้องใช้ --no-cache
    ไม่งั้นเท่ากับเอา "เวลาอ่านไฟล์" ไปแข่งกับ "เวลา OCR จริง" ซึ่งไม่ยุติธรรม"""
    import ocr
    if not NO_CACHE:
        return ocr.page_text(pdf_path, page_index)
    doc = fitz.open(pdf_path)
    try:
        png = doc[page_index].get_pixmap(dpi=ocr.OCR_DPI).tobytes("png")
    finally:
        doc.close()
    return "\n".join(ocr._get_reader().readtext(png, detail=0))


def engine_company(pdf_path: str, page_index: int) -> str:
    """OCR ของบริษัท — TODO: เติม 3 จุดตามสเปกจริง แล้วรันเทียบได้เลย

    อ่าน endpoint/key จาก env (อย่า hardcode — .env ถูก gitignore ไว้แล้ว):
        OCR_API_URL   เช่น http://<internal-host>:<port>/v1/ocr
        OCR_API_KEY   (ถ้าต้องใช้)
    """
    url = os.environ.get("OCR_API_URL")
    if not url:
        raise SystemExit("ยังไม่ได้ตั้ง OCR_API_URL — ใส่ใน .env ก่อน (ดู .env.example)")

    # render หน้า PDF เป็น PNG ด้วย DPI เดียวกับตัวเดิม เพื่อให้เทียบกันอย่างยุติธรรม
    dpi = int(os.environ.get("OCR_DPI", "200"))
    doc = fitz.open(pdf_path)
    try:
        png = doc[page_index].get_pixmap(dpi=dpi).tobytes("png")
    finally:
        doc.close()

    # ── TODO 1: ประกอบ request ให้ตรงสเปกของบริษัท ──────────────────────────
    import urllib.request
    boundary = "----ocrbench"
    body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
            f"filename=\"page.png\"\r\nContent-Type: image/png\r\n\r\n").encode() \
        + png + f"\r\n--{boundary}--\r\n".encode()
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    key = os.environ.get("OCR_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"        # TODO 2: หรือ x-api-key แล้วแต่สเปก

    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))

    # ── TODO 3: ดึงข้อความออกจาก response ให้ตรงรูปแบบจริง ────────────────────
    for k in ("text", "result", "content", "data"):
        if isinstance(payload.get(k), str):
            return payload[k]
    raise SystemExit(f"อ่าน response ไม่ออก — คีย์ที่มี: {list(payload)[:8]}")


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
            text = "\n".join(fn(p["img"], n) for n in range(p["pages"]))
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
