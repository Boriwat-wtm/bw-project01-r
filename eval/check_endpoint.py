# -*- coding: utf-8 -*-
"""เช็คว่า endpoint พร้อมใช้งานไหม — รันก่อนเดโม่/ก่อนรัน eval ทุกครั้ง

    python eval/check_endpoint.py

เช็ค 2 อย่างแยกกัน เพราะใช้คนละโมเดล คนละโควตา (ถึงจะอยู่ endpoint เดียวกัน):
    1. embedding  — ใช้ตอน "ค้น" ถ้าตัวนี้ล่ม เส้นค้นปกติใช้ไม่ได้ทั้งเส้น
    2. chat       — ใช้ตอน "ตอบ" ถ้าตัวนี้ล่ม ตอบไม่ได้เลย

⚠️ ไม่พิมพ์ค่า API key / URL ออกมา — แสดงแค่ชื่อโมเดลกับผลว่าใช้ได้ไหม
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import rag        # noqa: E402


def line(s: str = "") -> None:
    print(s)


def check(name: str, model: str, fn) -> bool:
    """รันฟังก์ชันทดสอบ 1 อย่าง แล้วรายงานผล — คืน True ถ้าใช้ได้"""
    line(f"── {name}  ({model})")
    t0 = time.perf_counter()
    try:
        detail = fn()
        line(f"   ✅ ใช้ได้  ({time.perf_counter() - t0:.1f}s)  {detail}")
        return True
    except Exception as e:
        kind = type(e).__name__
        msg = " ".join(str(e).split())
        line(f"   ❌ ใช้ไม่ได้  ({kind})")
        line(f"      {msg[:300]}")
        if "429" in msg or "quota" in msg.lower() or "RateLimit" in kind:
            line("      ⚠️ โควตาหมด/ถูกจำกัดอัตรา — เป็นเรื่องบัญชีหรือเครดิต ไม่ใช่โค้ดพัง")
            line("         ต้องแจ้งทีมที่ดูแล endpoint")
        elif "401" in msg or "403" in msg:
            line("      ⚠️ คีย์ไม่ถูกต้อง/ไม่มีสิทธิ์ — เช็ค LLM_API_KEY ใน .env")
        elif "Connection" in kind or "Timeout" in kind:
            line("      ⚠️ ต่อไม่ติด — เช็คว่าอยู่ในวงเน็ตบริษัท/VPN หรือยัง")
        return False


def main() -> None:
    line("=" * 62)
    line("เช็คความพร้อมของ endpoint")
    line("=" * 62)
    # ไม่พิมพ์ URL/คีย์ — บอกแค่ว่าตั้งค่าไว้หรือยัง
    line(f"LLM_BASE_URL : {'ตั้งค่าแล้ว' if rag.LLM_BASE_URL else '❌ ยังไม่ได้ตั้ง'}")
    line(f"LLM_API_KEY  : {'ตั้งค่าแล้ว' if rag.LLM_API_KEY else '❌ ยังไม่ได้ตั้ง'}"
         f"  (ยาว {len(rag.LLM_API_KEY)} ตัวอักษร)")
    line()

    def _embed():
        rag._init_embeddings()
        v = rag._embeddings.embed_query("ทดสอบระบบ")
        return f"เวกเตอร์ {len(v)} มิติ"

    def _chat():
        llm = rag.build_llm()
        r = llm.invoke("ตอบสั้น ๆ ว่า OK")
        return f"ตอบกลับ: {str(r.content)[:40]}"

    ok_embed = check("1. embedding (ใช้ตอนค้น)", rag.EMBED_MODEL, _embed)
    line()
    ok_chat = check("2. chat (ใช้ตอนตอบ)", rag.LLM_MODEL, _chat)

    line()
    line("=" * 62)
    if ok_embed and ok_chat:
        line("✅ พร้อมใช้งานเต็มระบบ — ถามได้ทุกแบบ และรัน eval ได้")
    elif ok_chat and not ok_embed:
        line("⚠️ ใช้ได้บางส่วน")
        line("   ❌ คำถามที่ต้องค้น (เช่น 'ใบจองคืออะไร') — ใช้ไม่ได้")
        line("   ✅ คำถามที่เข้าเส้นกราฟ (เช่น 'มาตรา ๙ ถูกแก้กี่ครั้ง') — ยังใช้ได้")
        line("      เพราะเส้นกราฟอ่านจาก metadata + PDF ตรง ๆ ไม่ต้องใช้ embedding")
        line("   ❌ รัน eval — ทำไม่ได้")
    elif ok_embed and not ok_chat:
        line("⚠️ ค้นได้แต่ตอบไม่ได้ — ระบบใช้งานจริงไม่ได้")
    else:
        line("❌ ใช้ไม่ได้ทั้งหมด — เช็คเน็ต/VPN หรือค่าใน .env ก่อน")
    line("=" * 62)
    sys.exit(0 if (ok_embed and ok_chat) else 1)


if __name__ == "__main__":
    main()
