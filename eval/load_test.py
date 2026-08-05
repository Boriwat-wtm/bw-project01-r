"""
load_test.py — ยิง API จริงเพื่อดูว่า "รับหลายคนพร้อมกัน" กับ "รันยาว ๆ" ไหวไหม

    # เปิดเซิร์ฟเวอร์ไว้ก่อน (คนละหน้าต่าง)
    uvicorn main:app --port 8000

    python eval/load_test.py                    # ครบทุกชุด
    python eval/load_test.py --only isolation   # เฉพาะชุดที่สนใจ
    python eval/load_test.py --sustain 30       # ยิงต่อเนื่องกี่ครั้ง

วัด 3 อย่าง เรียงตามความสำคัญ

1. คำตอบปนกันไหม (isolation) — สำคัญที่สุด
   โปรเจกต์นี้เคยเจอบั๊ก "ตัวแปรใช้ร่วมกันข้ามคำขอ" มาแล้ว 2 ครั้ง
   (rag._last_mode และ rag.LLM_MODEL) ทั้งคู่โผล่เฉพาะตอนมีหลายคนใช้พร้อมกัน
   ยิงคำถามที่มีคำตอบเป็น "ตัวเลขคนละตัว" พร้อมกัน แล้วเช็คว่าแต่ละคำตอบ
   ตรงกับคำถามของตัวเอง และไม่มีเลขของคนอื่นหลุดมา

2. ยิงพร้อมกันแล้วช้าลงแค่ไหน
   reranker ใช้ GPU ตัวเดียว จึงคาดว่าจะเข้าคิว ไม่ใช่เร็วขึ้น
   ตัวเลขนี้เอาไปบอกผู้เรียกได้ว่าควรทำคิวฝั่งตัวเองไหม

3. ยิงต่อเนื่องนาน ๆ แล้วเสื่อมไหม
   ดูว่าเวลาตอบค่อย ๆ ยืดขึ้นไหม และมีคำขอไหนพังกลางทางไหม
   (หน่วยความจำวัดแยกด้วย PowerShell: Get-Process -Id <pid>)
"""
import argparse
import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), ".env"))
except Exception:
    pass

BASE = os.environ.get("LOAD_TEST_BASE", "http://127.0.0.1:8000")
KEY = os.environ.get("API_KEY", "")

# คำถามที่ "คำตอบเป็นตัวเลขคนละตัว" — ปนกันเมื่อไรจับได้ทันที
PROBES = [
    {"id": "P1", "q": "โฉนดที่ดินต้องทำกี่ฉบับ และเก็บไว้ที่ใดบ้าง",
     "mine": ["สองฉบับ", "2ฉบับ"], "others": ["ห้าสิบไร่", "สามสิบวัน"]},
    {"id": "P2", "q": "วัดวาอารามจะได้มาซึ่งที่ดิน ได้มาไม่เกินกี่ไร่",
     "mine": ["50ไร่", "ห้าสิบไร่"], "others": ["สองฉบับ", "สามสิบวัน"]},
    {"id": "P3", "q": "ตามมาตรา ๘๑ การจดทะเบียนสิทธิที่ได้มาโดยทางมรดก ต้องปิดประกาศกี่วัน",
     "mine": ["30วัน", "สามสิบวัน"], "others": ["ห้าสิบไร่", "สองฉบับ"]},
    {"id": "P4", "q": "ประมวลกฎหมายที่ดินใช้บังคับเป็นกฎหมายตั้งแต่วันใด",
     "mine": ["2497"], "others": ["ห้าสิบไร่", "สองฉบับ"]},
    {"id": "P5", "q": "การประชุมของคณะกรรมการจัดที่ดิน ต้องมีกรรมการมาประชุมเท่าใดจึงเป็นองค์ประชุม",
     "mine": ["เกินกว่ากึ่งหนึ่ง", "เกินกึ่งหนึ่ง"], "others": ["ห้าสิบไร่", "สองฉบับ"]},
]


def norm(s: str) -> str:
    import re
    import rag
    return re.sub(r"[\s,*_`]+", "", (s or "").translate(rag.THAI_DIGITS))


def hit(alts, text_norm: str) -> bool:
    return any(norm(a) in text_norm for a in alts)


def ask(q: str, timeout: float = 300) -> dict:
    """ยิง 1 คำถาม -> {answer, elapsed, mode, error}"""
    t0 = time.perf_counter()
    answer, mode, err = "", "", ""
    try:
        with httpx.stream("POST", f"{BASE}/v1/chat",
                          headers={"X-API-Key": KEY},
                          json={"question": q}, timeout=timeout) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                ev = json.loads(line[6:])
                if "final" in ev:
                    answer = ev["final"]["answer"]
                    mode = ev["final"].get("retrieval_mode", "")
                elif "error" in ev:
                    err = str(ev["error"])
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    return {"answer": answer, "elapsed": time.perf_counter() - t0,
            "mode": mode, "error": err}


def grade(probe: dict, answer: str) -> dict:
    a = norm(answer)
    return {"mine_ok": hit(probe["mine"], a),
            "leaked": [x for x in probe["others"] if hit([x], a)]}


# ── 1. คำตอบปนกันไหม ──────────────────────────────────────────────────────────
def test_isolation() -> int:
    print("\n" + "=" * 74)
    print("1. ยิงพร้อมกัน 5 คำถาม — คำตอบปนกันไหม")
    print("=" * 74)
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(PROBES)) as ex:
        results = list(ex.map(lambda p: ask(p["q"]), PROBES))
    wall = time.perf_counter() - t0

    bad = 0
    for p, r in zip(PROBES, results):
        g = grade(p, r["answer"])
        ok = g["mine_ok"] and not g["leaked"] and not r["error"]
        bad += not ok
        print(f"  {'✅' if ok else '❌'} {p['id']}  {r['elapsed']:5.1f}s  {p['q'][:44]}")
        if r["error"]:
            print(f"        error: {r['error'][:90]}")
        if not g["mine_ok"]:
            print(f"        ⚠️ ไม่มีคำตอบของตัวเอง (ต้องมี {p['mine']})")
        if g["leaked"]:
            print(f"        🔴 คำตอบของคำถามอื่นหลุดมา: {g['leaked']}")

    print(f"\n  เวลารวมเมื่อยิงพร้อมกัน (wall): {wall:.1f}s")
    print(f"  ผลรวมเวลาแต่ละคำขอ            : {sum(r['elapsed'] for r in results):.1f}s")
    print(f"  {'✅ ไม่มีคำตอบปนกัน' if not bad else f'❌ มีปัญหา {bad} ข้อ'}")
    return bad


# ── 2. ยิงพร้อมกันช้าลงแค่ไหน ────────────────────────────────────────────────
def test_concurrency() -> int:
    print("\n" + "=" * 74)
    print("2. ทีละคำขอ vs ยิงพร้อมกัน — ช้าลงแค่ไหน")
    print("=" * 74)
    qs = [p["q"] for p in PROBES[:3]]

    t0 = time.perf_counter()
    serial = [ask(q) for q in qs]
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(qs)) as ex:
        conc = list(ex.map(ask, qs))
    t_conc = time.perf_counter() - t0

    print(f"  ทีละคำขอ 3 ข้อ   : {t_serial:6.1f}s  "
          f"(เฉลี่ย {statistics.mean(r['elapsed'] for r in serial):.1f}s/ข้อ)")
    print(f"  ยิงพร้อมกัน 3 ข้อ : {t_conc:6.1f}s  "
          f"(เฉลี่ย {statistics.mean(r['elapsed'] for r in conc):.1f}s/ข้อ)")
    if t_conc < t_serial:
        print(f"  → เร็วขึ้น {t_serial / t_conc:.2f} เท่า (ทำงานคู่ขนานได้บ้าง)")
    else:
        print(f"  → ช้าลง {t_conc / t_serial:.2f} เท่า (เข้าคิวกัน)")
    print(f"  คำขอที่พัง: {sum(1 for r in serial + conc if r['error'])}")
    print("\n  หมายเหตุ: ต่อให้เวลารวมพอ ๆ กัน 'คนใช้แต่ละคน' จะรู้สึกช้าลง"
          "\n  เพราะคำขอของตัวเองใช้เวลานานขึ้นตามจำนวนคนที่ยิงพร้อมกัน")
    return sum(1 for r in serial + conc if r["error"])


# ── 3. ยิงต่อเนื่องแล้วเสื่อมไหม ─────────────────────────────────────────────
def test_sustain(n: int) -> int:
    print("\n" + "=" * 74)
    print(f"3. ยิงต่อเนื่อง {n} ครั้ง — เวลาตอบยืดขึ้นไหม / มีพังไหม")
    print("=" * 74)
    times, errs = [], 0
    for i in range(n):
        p = PROBES[i % len(PROBES)]
        r = ask(p["q"])
        times.append(r["elapsed"])
        if r["error"]:
            errs += 1
            print(f"  ❌ ครั้งที่ {i+1}: {r['error'][:80]}")
        elif not grade(p, r["answer"])["mine_ok"]:
            errs += 1
            print(f"  ❌ ครั้งที่ {i+1}: ตอบไม่ตรงคำถาม ({p['id']})")
        if (i + 1) % 5 == 0:
            print(f"  ครั้งที่ {i+1:2}/{n}  เฉลี่ย 5 ครั้งล่าสุด {statistics.mean(times[-5:]):.1f}s")

    half = len(times) // 2
    first, last = statistics.mean(times[:half]), statistics.mean(times[half:])
    print(f"\n  ครึ่งแรก  {first:5.1f}s/ข้อ")
    print(f"  ครึ่งหลัง {last:5.1f}s/ข้อ")
    drift = (last - first) / first * 100 if first else 0
    print(f"  เปลี่ยนไป {drift:+.0f}%  "
          f"{'(ยอมรับได้)' if abs(drift) < 25 else '⚠️ (เสื่อมชัดเจน)'}")
    print(f"  คำขอที่พัง: {errs}/{n}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["isolation", "concurrency", "sustain"])
    ap.add_argument("--sustain", type=int, default=15, help="ยิงต่อเนื่องกี่ครั้ง")
    args = ap.parse_args()

    if not KEY:
        print("⚠️ ไม่พบ API_KEY — ตั้งใน .env ก่อน")
        return 1
    try:
        h = httpx.get(f"{BASE}/health", timeout=10).json()
        print(f"เซิร์ฟเวอร์: {BASE}  ({h.get('chunks')} chunks · {h.get('model')})")
    except Exception as e:
        print(f"⚠️ ต่อเซิร์ฟเวอร์ไม่ได้ที่ {BASE} — เปิด uvicorn ก่อน ({type(e).__name__})")
        return 1

    bad = 0
    if args.only in (None, "isolation"):
        bad += test_isolation()
    if args.only in (None, "concurrency"):
        bad += test_concurrency()
    if args.only in (None, "sustain"):
        bad += test_sustain(args.sustain)

    print("\n" + "─" * 74)
    print("✅ ผ่านทั้งหมด" if not bad else f"❌ มีปัญหารวม {bad} จุด")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
