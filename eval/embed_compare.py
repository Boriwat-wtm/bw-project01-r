"""
embed_compare.py — ทดลองเปลี่ยน embedding แล้ววัดว่าดีขึ้นไหม โดยไม่แตะดัชนีหลัก

    python eval/embed_compare.py                       # เทียบทุกตัวที่ตั้งไว้
    python eval/embed_compare.py --only ada-002 te3-large
    python eval/embed_compare.py --mlflow              # ส่งผลขึ้น MLflow ด้วย

═══ วัดอย่างไร ═══
วัดที่ "retrieval" ตรง ๆ (run_eval.py --retrieval) ไม่เรียก LLM เพราะ:
    - embedding มีผลต่อ "การค้นเจอ" ไม่ใช่ "การเรียบเรียงคำตอบ"
    - retrieval เป็น deterministic -> รันครั้งเดียวพอ ไม่ต้องเฉลี่ยหลายรอบ
    - เร็ว ฟรี ไม่สุ่ม -> เทียบ embedding กันได้สะอาด

═══ ทำไมไม่ทับดัชนีเดิม ═══
embedding แต่ละตัวมิติเวกเตอร์ไม่เท่ากัน (ada=1536, 3-large=3072, gemma=768)
จึงสร้างดัชนีแยกโฟลเดอร์ (chroma_db_<ชื่อ>) ผ่าน env CHROMA_DIR
ดัชนีหลัก chroma_db (ada) ไม่ถูกแตะเลย — ทดลองเสร็จลบโฟลเดอร์ทดลองทิ้งได้

⚠️ ข้อจำกัดที่ต้องรู้ก่อนเชื่อผล
ชุดคำถาม 30 ข้อเน้น "เลขมาตรา" ซึ่ง BM25 (keyword) จับได้อยู่แล้วโดยไม่พึ่ง embedding
ส่วนที่ embedding ที่ดีกว่าจะช่วยจริงคือ "คำถามที่พิมพ์คนละคำกับตัวบท" ซึ่งชุดนี้วัดน้อย
-> ถ้าผลออกมาพอ ๆ กัน ไม่ได้แปลว่า embedding ใหม่ไม่ดี แค่ชุดทดสอบนี้ไม่ไวพอจะเห็น
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ตัวที่จะเทียบ — ตรวจแล้วว่าพร้อมใช้จริงในเครื่องนี้ (ก.ค. 2569)
#   ada-002        : ตัวปัจจุบัน ใช้ดัชนีหลักเดิม (dir=None = chroma_db)
#   te3-large      : OpenAI รุ่นใหม่แทน ada โดยตรง — อยู่บน endpoint เดียวกันแล้ว
#   embgemma       : Google EmbeddingGemma — มีใน Ollama แล้ว (local ฟรี)
# อยากเพิ่ม bge-m3 ให้ `ollama pull bge-m3` ก่อน แล้วเติมบรรทัดใหม่
CONFIGS: "dict[str, dict]" = {
    "ada-002":   {"model": "text-embedding-ada-002", "dir": None},
    "te3-large": {"model": "text-embedding-3-large", "dir": "chroma_db_te3large"},
    "embgemma":  {"model": "embeddinggemma",         "dir": "chroma_db_embgemma"},
}


def _env_for(cfg: dict) -> dict:
    """env ที่ทำให้ subprocess ใช้ embedding + ที่เก็บดัชนีตามที่กำหนด"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8", EMBED_MODEL=cfg["model"])
    if cfg["dir"]:
        env["CHROMA_DIR"] = os.path.join(ROOT, cfg["dir"])
    else:
        env.pop("CHROMA_DIR", None)          # ใช้ chroma_db เดิม
    return env


def _run(cmd: list, env: dict) -> "tuple[int, list[str]]":
    """รัน subprocess แบบสตรีมออกจอไปด้วย เก็บบรรทัดไว้ให้ parse ทีหลัง"""
    lines: list[str] = []
    p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                         errors="replace", bufsize=1)
    for line in p.stdout:                     # type: ignore[union-attr]
        line = line.rstrip("\n")
        lines.append(line)
        print("   │ " + line[:110])
    p.wait()
    return p.returncode, lines


def build_index(name: str, cfg: dict) -> bool:
    """สร้างดัชนีของ embedding นี้ (ข้ามถ้ามีครบแล้ว — build_vectorstore เช็ค count ให้)"""
    print(f"\n[{name}] สร้าง/ตรวจดัชนี ({cfg['model']}) ...")
    code, _ = _run(
        [sys.executable, "-c",
         "import rag; rag.update_database(); rag.build_vectorstore(force=False)"],
        _env_for(cfg))
    if code != 0:
        print(f"[{name}] ❌ สร้างดัชนีไม่สำเร็จ (exit {code}) — ข้ามตัวนี้")
    return code == 0


def eval_retrieval(name: str, cfg: dict, use_mlflow: bool) -> "dict | None":
    """รัน retrieval eval แล้วอ่านผลจากไฟล์ results ที่มันเขียน"""
    print(f"[{name}] วัด retrieval ...")
    cmd = [sys.executable, os.path.join("eval", "run_eval.py"), "--retrieval"]
    if use_mlflow:
        cmd += ["--mlflow", "--tag", f"embed-{name}"]
    code, lines = _run(cmd, _env_for(cfg))
    if code != 0:
        print(f"[{name}] ❌ eval ไม่สำเร็จ (exit {code})")
        return None

    # run_eval พิมพ์ "ผลดิบ: <relpath>" — อ่านไฟล์นั้นเอาเลขที่แม่นยำ ไม่ parse ตัวเลขจากจอ
    path = None
    for ln in lines:
        m = re.search(r"ผลดิบ:\s*(.+\.json)", ln)
        if m:
            path = os.path.join(ROOT, m.group(1).strip())
    if not path or not os.path.exists(path):
        print(f"[{name}] ❌ หาไฟล์ผลไม่เจอ")
        return None

    data = json.load(open(path, encoding="utf-8"))
    res = data["results"]
    n = len(res)
    by_lv = {lv: [r for r in res if r["level"] == lv] for lv in ("easy", "medium", "hard")}
    return {
        "name": name, "model": cfg["model"],
        "passed": sum(r["passed"] for r in res), "total": n,
        "fact_ratio": sum(r["ratio"] for r in res) / n,
        "by_level": {lv: sum(r["passed"] for r in v) for lv, v in by_lv.items() if v},
        "failed_ids": [r["id"] for r in res if not r["passed"]],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="เทียบ embedding หลายตัวด้วย retrieval eval")
    ap.add_argument("--only", nargs="*", help=f"เลือกเฉพาะบางตัว: {', '.join(CONFIGS)}")
    ap.add_argument("--mlflow", action="store_true", help="ส่งผลขึ้น MLflow")
    a = ap.parse_args()

    names = a.only or list(CONFIGS)
    bad = [x for x in names if x not in CONFIGS]
    if bad:
        raise SystemExit(f"ไม่รู้จัก: {bad} — มีให้เลือก: {list(CONFIGS)}")

    print("═" * 78)
    print("ทดลองเทียบ embedding — วัดที่ retrieval (ไม่เรียก LLM)")
    print(f"ตัวที่เทียบ: {', '.join(names)}")
    print("═" * 78)

    t0 = time.time()
    rows = []
    for name in names:
        cfg = CONFIGS[name]
        if not build_index(name, cfg):
            continue
        r = eval_retrieval(name, cfg, a.mlflow)
        if r:
            rows.append(r)

    # ── ตารางสรุป ────────────────────────────────────────────────────────────
    print("\n" + "═" * 78)
    print("สรุปผล — ยิ่งผ่านมาก ยิ่งค้นเจอข้อเท็จจริงที่ต้องการมาก")
    print("═" * 78)
    print(f"{'embedding':<12}{'โมเดล':<26}{'ผ่าน':>7}{'ข้อเท็จจริง':>12}"
          f"{'ง':>4}{'ก':>4}{'ย':>4}")
    print("─" * 78)
    base = next((r for r in rows if r["name"] == "ada-002"), None)
    for r in rows:
        lv = r["by_level"]
        delta = ""
        if base and r["name"] != "ada-002":
            d = r["passed"] - base["passed"]
            delta = f"  ({'+' if d >= 0 else ''}{d} เทียบ ada)"
        print(f"{r['name']:<12}{r['model']:<26}{r['passed']:>4}/{r['total']}"
              f"{r['fact_ratio']*100:>10.0f}%"
              f"{lv.get('easy',0):>4}{lv.get('medium',0):>4}{lv.get('hard',0):>4}{delta}")
    print("─" * 78)
    print("(ง=ง่าย ก=กลาง ย=ยาก · เต็มอย่างละ 10)")

    if base:
        for r in rows:
            if r["name"] != "ada-002":
                only_new = set(base["failed_ids"]) - set(r["failed_ids"])
                only_old = set(r["failed_ids"]) - set(base["failed_ids"])
                if only_new or only_old:
                    print(f"\n{r['name']} เทียบ ada:")
                    if only_new:
                        print(f"   ✅ ตัวใหม่ค้นเจอเพิ่ม: {', '.join(sorted(only_new))}")
                    if only_old:
                        print(f"   ⚠️ ตัวใหม่กลับค้นไม่เจอ: {', '.join(sorted(only_old))}")

    print(f"\nรวมเวลา {time.time()-t0:.0f}s")
    print("\n⚠️ ชุดคำถามนี้เน้นเลขมาตรา (BM25 จับได้โดยไม่พึ่ง embedding) — ถ้าผลพอ ๆ กัน")
    print("   ไม่ได้แปลว่า embedding ใหม่ไม่ดี ดูคำอธิบายหัวไฟล์ embed_compare.py")
    if base and len(rows) > 1:
        best = max(rows, key=lambda r: (r["passed"], r["fact_ratio"]))
        print(f"\nดีสุดในชุดนี้: {best['name']} ({best['passed']}/{best['total']})")
        print("ถ้าจะใช้จริง -> ยืนยันด้วย full eval (มี LLM + reranker) ก่อนเปลี่ยน .env EMBED_MODEL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
