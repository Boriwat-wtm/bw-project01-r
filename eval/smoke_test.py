"""
smoke_test.py — ชุดตรวจเร็ว ไม่เรียก LLM รันเสร็จในไม่กี่วินาที

    python eval/smoke_test.py

ต่างจาก run_eval.py ตรงที่:
    run_eval.py    วัด "คุณภาพคำตอบ" ต้องยิง LLM 30 ครั้ง ใช้เวลา ~12 นาที มีค่าใช้จ่าย
    smoke_test.py  ตรวจ "กติกาที่ต้องไม่มีวันพัง" ด้วยโค้ดล้วน ฟรีและเร็ว

ควรรันทุกครั้งก่อน commit — บักที่เจอในโปรเจกต์นี้ 3 ตัวจาก 5 ตัวเป็นชนิดที่ชุดนี้จับได้
(metadata ค้างของเก่า / เลขไทยไม่ถูก normalize / ปีในคำถามถูกตีความผิด)

ออก exit code 1 ถ้ามีข้อไหนตก เอาไปต่อ CI ได้เลย
"""
import inspect
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import rag        # noqa: E402
import service    # noqa: E402
# ⚠️ ต้อง alias — ไฟล์นี้มีฟังก์ชันชื่อ main() อยู่แล้ว ถ้า import ตรง ๆ
#    def main() ข้างล่างจะทับชื่อโมดูลทิ้ง แล้ว main.health พังตอนรัน
import main as api  # noqa: E402  (แค่ import โมดูล ไม่ได้สตาร์ทเซิร์ฟเวอร์)

_results: list[tuple[bool, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((bool(ok), name, detail))


def section(title: str) -> None:
    _results.append((None, title, ""))


# ══════════════════════════════════════════════════════════════════════════════
def test_index_integrity():
    """ดัชนีครบและ metadata ไม่ค้างของเก่า — บักนี้เคยเกิดจริงหลังเปลี่ยนสูตร as_of_year"""
    section("ความสมบูรณ์ของดัชนี")
    rag._init_chroma()
    rag._ensure_loaded()
    n_mem, n_db = len(rag._chunks), rag._collection.count()
    check("จำนวน chunk ในดัชนีตรงกับที่อ่านจาก PDF", n_mem == n_db, f"{n_mem} vs {n_db}")

    sample = rag._chunks[::37]
    got = rag._collection.get(ids=[c["id"] for c in sample], include=["metadatas"])
    db = dict(zip(got["ids"], got["metadatas"]))
    fields = ["group", "version", "in_force", "as_of_year", "amend_no", "article_nums"]
    bad = [(c["id"], f) for c in sample if c["id"] in db
           for f in fields if str(db[c["id"]].get(f)) != str(c.get(f))]
    check("metadata ใน Chroma ตรงกับที่โค้ดปัจจุบันคำนวณได้",
          not bad, f"ไม่ตรง {len(bad)} ช่อง เช่น {bad[:2]}" if bad else "")

    in_force = [c for c in rag._chunks if c.get("in_force")]
    check("มีกลุ่มตัวบทที่ใช้บังคับอยู่จริง", len(in_force) > 0, f"{len(in_force)} chunks")
    check("ตัวบทที่ใช้บังคับมาจากฉบับรวมล่าสุดเท่านั้น",
          all(c["version"] == 999 for c in in_force))


def test_thai_text():
    """ข้อความไทยต้องไม่เพี้ยน + เลขไทยต้องถูก normalize"""
    section("การอ่านภาษาไทย")
    body = rag.doc_articles("data/LandCode2497_Update-vlast_TruePDF.pdf").get("9", "")
    check("คัดข้อความไทยได้ถูกต้อง (สระ/วรรณยุกต์ไม่หลุด)",
          "ที่ดิน" in body and "ที ดิน" not in body, body[:60])
    check("ตัดคำไทยสำหรับ BM25 ได้",
          "ที่ดิน" in rag.tokenize("ห้ามบุกรุกที่ดินของรัฐ"))
    check("แปลงเลขไทยเป็นอารบิกตอน tokenize", "9" in rag.tokenize("มาตรา ๙"))
    check("ถาม 'มาตรา 9' ต้องหมายถึงมาตราเดียวกับ 'มาตรา ๙'",
          rag.question_articles("มาตรา 9")[0] == rag.question_articles("มาตรา ๙")[0])
    check("อ่านลำดับ ทวิ/ตรี ได้", rag.question_articles("มาตรา ๘ ทวิ") == ["8 ทวิ"])
    check("ไม่กิน 'ฉ' จากคำว่า 'ฉบับ' มาเป็นเลขลำดับ",
          rag.question_articles("มาตรา ๙๗ ฉบับดั้งเดิม") == ["97"])


def test_question_intent():
    """อ่านเจตนาคำถามให้ถูก — พลาดตรงนี้ = เสิร์ฟกฎหมายผิดเวอร์ชัน"""
    section("การเข้าใจคำถาม")
    y = rag.classify_years("ที่ดินที่ออกใบจองหลังวันที่ ๑๔ ธันวาคม พ.ศ. ๒๕๑๕ ห้ามโอนกี่ปี")
    check("ปีที่เป็นเงื่อนไขในตัวบท ต้องไม่ถูกอ่านว่าขอย้อนเวลา", not y["asof"], str(y))
    y = rag.classify_years("พ.ร.บ. (ฉบับที่ ๑๕) พ.ศ. ๒๕๖๒ บังคับใช้เมื่อใด")
    check("ปีที่เป็นชื่อเอกสาร ต้องไม่ถูกอ่านว่าขอย้อนเวลา", not y["asof"], str(y))
    y = rag.classify_years("มาตรา ๒๐ ณ วันที่ ๑ มกราคม พ.ศ. ๒๕๖๒ ว่าอย่างไร")
    check("คำขอย้อนเวลาจริง ต้องถูกจับได้", y["asof"] == [2562], str(y))

    check("อ่านการไล่เลขฉบับได้ครบ",
          rag.question_amendments("เทียบฉบับที่ ๑๓, ๑๔ และ ๑๕") == [13, 14, 15])
    check("alias เลขมาตราที่ถูกเปลี่ยน (๙ ทวิ -> ๙/๑)",
          "9/1" in rag.question_articles("มาตรา ๙ ทวิ ยังมีอยู่ไหม"))

    allg = service.list_groups()
    g, _, v = service.pick_groups("บุกรุกที่ดินของรัฐมีโทษอย่างไร", allg)
    check("คำถามทั่วไปต้องล็อกที่ตัวบทที่ใช้บังคับเท่านั้น",
          g == [rag.GROUP_IN_FORCE] and not v, f"{g} {v}")
    check("คำถามเปรียบเทียบต้องเข้าเส้นทาง chain",
          service.detect_compare("มาตรา ๖๑ ถูกแก้กี่ครั้ง") == "61")
    check("คำถามธรรมดาต้องไม่เข้าเส้นทาง chain",
          not service.detect_compare("มาตรา ๙ ห้ามทำอะไร"))
    # "ถูกยกเลิก" เป็นเหตุการณ์วงจรชีวิตมาตราเหมือน "ถูกแก้" — ต้องเข้าเส้น chain
    # (เคยพลาดจริง: เส้นค้นปกติเจอสถานะ (ยกเลิก) แต่ค้นไม่ถึงบรรทัด "ยกเลิกโดย ปว.๔๙")
    check("คำถาม 'ถูกยกเลิกโดยใคร' ต้องเข้าเส้นทาง chain",
          service.detect_compare("มาตรา ๓๕ ถูกยกเลิกโดยกฎหมายใด") == "35")
    check("ยกเลิก + ไม่ระบุมาตรา ต้องไม่เข้าเส้นทาง chain",
          not service.detect_compare("ฉบับที่ ๑๒ พ.ศ. ๒๕๕๑ ยกเลิกหมวดใด"))
    check("'ยกเลิกความในมาตรา' (ฝั่งฉบับแก้ไข) ต้องไม่เข้าเส้นทาง chain",
          not service.detect_compare("ฉบับที่ ๔ ยกเลิกความในมาตรา ๖๑ จริงหรือไม่"))
    # "ฉบับ" ในกฎหมายไทยมีสองความหมาย: จำนวน พ.ร.บ. ที่มาแก้ กับ จำนวนใบสำเนา
    # เดิม _COMPARE_HINT จับ "กี่ฉบับ" ลอย ๆ ทำให้คำถามเรื่องจำนวนใบไหลไปตอบประวัติมาตรา
    check("'กี่ฉบับ' ที่หมายถึงจำนวนใบ ต้องไม่เข้าเส้นทาง chain",
          not service.detect_compare("โฉนดที่ดินตามมาตรา ๕๗ ต้องทำกี่ฉบับ"))
    check("'กฎหมายกี่ฉบับ' ที่หมายถึงจำนวน พ.ร.บ. ต้องเข้าเส้นทาง chain",
          service.detect_compare("มาตรา ๖๙ ทวิ ถูกแก้ไขโดยกฎหมายกี่ฉบับ และฉบับใดบ้าง") == "69 ทวิ")


def test_multiturn_routing():
    """คำถามต่อเนื่อง: การเขียนคำถามใหม่ต้องไม่มีสิทธิ์เปลี่ยน 'เส้นทาง' ของระบบ

    เจอจากชุด eval/multiturn.json ข้อ M4 — ผู้ใช้เปลี่ยนเรื่องกลางบทสนทนา
    แต่ rewrite_followup ลากเลขมาตราจากเทิร์นก่อนมาใส่ ทำให้ระบบเปลี่ยนไป
    เดินเส้นเปรียบเทียบแล้วตอบประวัติมาตราเก่าแทนคำถามที่ถามจริง (ผิดทั้งข้อ)
    """
    section("คำถามต่อเนื่อง (multi-turn)")
    # ── สิทธิ์ยืมเลขมาตราจากบทสนทนา ──
    check("คำถามที่เขียน 'มาตรานี้' ยืมเลขมาตราจากประวัติได้",
          service._may_borrow_article("มาตรานี้ถูกแก้ไขโดยกฎหมายฉบับใด"))
    check("คำถามที่มีเลขมาตราของตัวเอง ไม่ถือว่ายืมใคร",
          service._may_borrow_article("มาตรา ๓๑ ถูกแก้ไขกี่ครั้ง"))
    check("คำถามที่เปลี่ยนเรื่อง (ไม่มีทั้งเลขมาตราและคำอ้างถึง) ยืมไม่ได้",
          not service._may_borrow_article("โฉนดที่ดินต้องทำกี่ฉบับ และเก็บไว้ที่ใดบ้าง"))

    # ── เติมเลขมาตราให้ 'มาตรานี้' ด้วยโค้ด ไม่ต้องรอ LLM ──
    hist = [{"role": "user", "content": "ตามมาตรา ๑๑๑ ผู้ฝ่าฝืนมาตรา ๘๖ มีโทษเท่าใด"},
            {"role": "assistant", "content": "ปรับไม่เกินสองหมื่นบาท จำคุกไม่เกินสองปี"}]
    q = "มาตรานี้ถูกแก้ไขโดยกฎหมายฉบับใด"
    got = service.carry_article(q, q, hist)
    check("'มาตรานี้' ต้องถูกแทนด้วยเลขมาตราล่าสุดในบทสนทนา",
          "111" in got, got)
    check("คำถามที่มีเลขมาตราอยู่แล้ว ต้องไม่ถูกแตะ",
          service.carry_article("มาตรา ๙ ห้ามทำอะไร", "มาตรา ๙ ห้ามทำอะไร", hist)
          == "มาตรา ๙ ห้ามทำอะไร")
    # "นี้" ที่อ้างถึงคำนามอื่น ห้ามถูกแทนด้วยเลขมาตรา — จะเพี้ยนหนักกว่าเดิม
    check("'โฉนดนี้' ต้องไม่ถูกเติมเลขมาตรา",
          service.carry_article("โฉนดนี้ออกเมื่อใด", "โฉนดนี้ออกเมื่อใด", hist)
          == "โฉนดนี้ออกเมื่อใด")
    check("ไม่มีประวัติ ต้องคืนคำถามเดิม",
          service.carry_article(q, q, []) == q)

    # ── เส้นทางเปรียบเทียบต้องรายงานคำถามที่เขียนใหม่ (เดิมส่ง "" ตายตัว) ──
    src = inspect.getsource(service._answer_compare)
    check("เส้นทางเปรียบเทียบต้องบอกผู้เรียกได้ว่าคำถามถูกเขียนใหม่",
          '"search_q": reported_q' in src or "'search_q': reported_q" in src)


def test_degrade_gracefully():
    """ระบบเสื่อมได้ แต่ต้อง "เสื่อมแบบมีเสียง" — ห้ามเงียบ

    บั๊กที่ใช้เวลาไล่จับนานที่สุดในโปรเจกต์นี้ทุกตัวมีหน้าตาเดียวกัน:
    คุณภาพตกลงจริง แต่ไม่มีอะไรบอกใครเลย
    """
    section("เสื่อมแล้วต้องมีเสียง")

    # ── 1. ไม่มี pythainlp -> BM25 ตัดคำหยาบ ต้องรู้ได้จากข้างนอก ──
    check("มีตัวบอกว่าใช้ตัวตัดคำอะไรอยู่",
          rag.TOKENIZER_KIND in ("newmm", "bigram"), rag.TOKENIZER_KIND)
    check("เครื่องนี้ใช้ newmm อยู่จริง (ถ้าเป็น bigram แปลว่าลืมลง pythainlp)",
          rag.TOKENIZER_KIND == "newmm", rag.TOKENIZER_KIND)
    check("/health ต้องรายงานตัวตัดคำออกไปด้วย",
          "TOKENIZER_KIND" in inspect.getsource(api.health))

    # ── 2. LLM ล่ม -> ต้องคืนตัวบทที่ค้นเจอ ไม่ใช่ error เปล่า ──
    chunks = rag.retrieve("โฉนดที่ดินต้องทำกี่ฉบับ", k=4)
    fb = service.fallback_answer(chunks, ConnectionError("endpoint ล่ม"))
    check("LLM ล่ม -> คำตอบสำรองต้องมีตัวบทจริงติดมาด้วย",
          len(fb) > 300 and "มาตรา" in fb, f"{len(fb)} ตัวอักษร")
    check("คำตอบสำรองต้องบอกชัดว่าไม่ได้เรียบเรียง",
          "ยังไม่ได้เรียบเรียง" in fb)
    check("คำตอบสำรองต้องบอกชนิดของข้อผิดพลาด",
          "ConnectionError" in fb)
    check("ไม่มีตัวบทเลยก็ต้องไม่พัง",
          "ไม่พบ" in service.fallback_answer([], ConnectionError("x")))
    # error ตัวจริงจาก endpoint ต้องเข้าเงื่อนไขนี้ ไม่ใช่แค่ ConnectionError ที่เราจำลอง
    # (เคยพลาดมาแล้วตอนทำ BM25 fallback — เทสผ่านทั้งที่ของจริงยังพัง)
    try:
        import openai
        for name in ("APIConnectionError", "APITimeoutError", "RateLimitError",
                     "InternalServerError"):
            cls = getattr(openai, name, None)
            if cls:
                check(f"openai.{name} ต้องถูกดักเป็น 'LLM ล่ม'",
                      issubclass(cls, service.LLM_DOWN))
    except ImportError:
        check("import openai ได้", False, "ไม่พบ openai")

    # ── 3. ส่ง chunks ไปให้ _stream_answer จริงทั้งสองเส้นทาง ──
    for fn, label in ((service.answer_stream, "เส้นทางปกติ"),
                      (service._answer_compare, "เส้นทางเปรียบเทียบ")):
        check(f"{label} ต้องส่งตัวบทสำรองให้ _stream_answer",
              "chunks=" in inspect.getsource(fn))


def test_amendment_graph():
    """สายการแก้ไขต้องตรงกับที่ตัวบทเขียนไว้ (เฉลยจากไฟล์ QA ของลูกค้า)"""
    section("สายการแก้ไข (amendment graph)")
    expect = {
        "61":     [2515, 2528, 2543, 2551],   # ปว.๓๓๔ -> ฉ.๔ -> ฉ.๙ -> ฉ.๑๑
        "69 ทวิ": [2515, 2520, 2528],          # ปว.๓๓๔ -> ฉ.๑ (ไม่มีเลขในชื่อ) -> ฉ.๔
        "81":     [2515, 2528, 2543, 2556],
        "104":    [2497, 2534, 2543, 2562],
    }
    for art, years in expect.items():
        got = [c["year"] for c in rag.article_chain(art)]
        check(f"สายของมาตรา {art}", got == years, f"ได้ {got} ควรเป็น {years}")
    check("ต้นสายที่อยู่นอกชุดข้อมูลถูกกำกับไว้ (กัน hallucinate)",
          rag.article_chain("61")[0]["in_corpus"] is False)
    check("มาตราที่ไม่เคยถูกแก้ ต้องไม่มีสายปลอม", rag.article_chain("999") == [])

    # ── กันบั๊กที่เคยเกิดจริงกลับมา (regression) ──
    # ตัวคั่นใน _PREV_REV เคยเป็น [^พ] ทำให้ ปว. ที่ลงวันที่เดือนซึ่งมีตัว "พ" อยู่ในชื่อ
    # แกะไม่ออก -> สายขาดเงียบ ๆ 10 มาตรา  · เดือนที่พัง: กุมภาพันธ์/พฤษภาคม/พฤศจิกายน
    for mth in ["มกราคม", "กุมภาพันธ์", "พฤษภาคม", "พฤศจิกายน", "ธันวาคม"]:
        p = rag._parse_prev(f"ซึ่งแก้ไขเพิ่มเติมโดยประกาศของคณะปฏิวัติ ฉบับที่ ๙๖ "
                            f"ลงวันที่ ๒๙ {mth} พ.ศ. ๒๕๑๕")
        check(f"แกะชื่อ ปว. ได้แม้ลงวันที่เดือน '{mth}'",
              p is not None and p["no"] == 96 and p["year"] == 2515, f"ได้ {p}")
    # เอกสารจริงเขียน "ประกาศคณะปฏิวัติ" (ไม่มี 'ของ') และ "พุทธศักราช" (ไม่ใช่ 'พ.ศ.') ด้วย
    p = rag._parse_prev("ซึ่งแก้ไขเพิ่มเติมโดยประกาศคณะปฏิวัติ ฉบับที่ ๓๓๔ "
                        "ลงวันที่ ๑๓ ธันวาคม พุทธศักราช ๒๕๑๕")
    check("แกะชื่อ ปว. ได้แม้เขียนว่า 'ประกาศคณะปฏิวัติ ... พุทธศักราช'",
          p is not None and p["no"] == 334, f"ได้ {p}")
    # PDF ตัดบรรทัดกลางชื่อกฎหมาย -> normalize แล้วเหลือช่องว่างคั่น ต้องยังแกะออก
    p = rag._parse_prev("ซึ่งแก้ไขเพิ่มเติมโดยพระราชบัญญัติแก้ไขเพิ่มเติมประมวล "
                        "กฎหมายที่ดิน (ฉบับที่ ๙) พ.ศ. ๒๕๔๓")
    check("แกะชื่อ พ.ร.บ. ได้แม้มีช่องว่างแทรกกลางชื่อกฎหมาย",
          p is not None and p["no"] == 9 and p["year"] == 2543, f"ได้ {p}")
    # ท้าย clause ต้องไม่ล้นไปหยิบชื่อฉบับของ clause ถัดไป (ม.๗ เคยได้ ปว.๙๖ ของ ม.๒๐)
    check("ไม่ดูดชื่อฉบับของ clause ถัดไปมาใส่ (ม.๗ ถูกยกเลิกเฉย ๆ ไม่มีฉบับก่อนหน้า)",
          rag.article_chain("7")[0]["year"] == 2497,
          str(rag.article_chain("7")[0]))
    # และ clause ที่ถูกกลืนต้องไม่หายจากกราฟ (ม.๒๐ ในฉบับที่ ๓ เคยหายไปทั้งมาตรา)
    check("clause ที่ตามหลัง clause อื่นต้องไม่หายจากกราฟ (ม.๒๐)",
          [c["year"] for c in rag.article_chain("20")] == [2515, 2526, 2562],
          str([c["year"] for c in rag.article_chain("20")]))
    # ไม่ควรเหลือรายการที่แกะชื่อฉบับก่อนหน้าไม่ออกอีก — ถ้าโผล่มาแปลว่ามีสำนวนใหม่ที่ยังไม่รองรับ
    unknown = sorted({a for a, lst in rag.amendment_graph().items()
                      for e in lst if e.get("over") and e["over"]["kind"] == "อื่น"}, key=len)
    check("ไม่มีมาตราไหนที่แกะชื่อฉบับก่อนหน้าไม่ออก", not unknown, f"ยังเหลือ: {unknown}")


def test_entity_index():
    """ดัชนีตัวละคร — ตอบคำถามแบบ 'รวบรวมให้ครบ' ที่การค้น top-k ทำไม่ได้"""
    section("ดัชนีตัวละคร (entity index)")
    idx = rag.entity_index()
    check("สร้างดัชนีได้และมีครบทุก entity ที่ประกาศไว้",
          len(idx) >= len(rag.ENTITY_PATTERNS) - 1, f"ได้ {len(idx)}/{len(rag.ENTITY_PATTERNS)}")
    # ค่าคงที่ — ถ้าเลขนี้ขยับแปลว่าตัวบทหรือ pattern เปลี่ยน ต้องรู้ตัว
    check("อธิบดี อยู่ใน 13 มาตรา", len(idx.get("อธิบดี", [])) == 13,
          f"ได้ {len(idx.get('อธิบดี', []))}")
    check("มาตราที่ต้องมีอยู่ในรายการของอธิบดี (๘ ตรี/๒๗/๕๐/๖๑)",
          {"8 ตรี", "27", "50", "61"} <= set(idx.get("อธิบดี", [])),
          str(idx.get("อธิบดี")))
    # ⚠️ "รัฐมนตรี" ต้องไม่กินคำว่า "คณะรัฐมนตรี" มาเป็นของตัวเอง
    check("แยก 'รัฐมนตรี' ออกจาก 'คณะรัฐมนตรี' ได้",
          set(idx.get("คณะรัฐมนตรี", [])) != set(idx.get("รัฐมนตรี", []))
          and "112" not in idx.get("รัฐมนตรี", []),
          f"รัฐมนตรี={idx.get('รัฐมนตรี')} คณะรัฐมนตรี={idx.get('คณะรัฐมนตรี')}")
    # รันซ้ำต้องได้ผลเดิมเป๊ะ — นี่คือเหตุผลทั้งหมดที่ใช้โค้ดนับแทน LLM สกัด
    rag._entity_cache = None
    check("รันซ้ำได้ผลเดิมทุกตัวอักษร (deterministic)", rag.entity_index() == idx)

    check("นับจำนวนครั้งได้ ไม่ใช่แค่เจอ/ไม่เจอ",
          rag.count_entities("ให้อธิบดีหรือผู้ซึ่งอธิบดีมอบหมาย").get("อธิบดี") == 2,
          str(rag.count_entities("ให้อธิบดีหรือผู้ซึ่งอธิบดีมอบหมาย")))
    check("จับ entity จากข้อความได้", "ใบจอง" in rag.question_entities("ใบจองโอนได้ไหม"))

    # ⚠️ ต้องดึงตัวบทจากไฟล์ ไม่ใช่จาก _chunks — ก้อนค้นหั่นตามความยาว ไม่ได้หั่นตามมาตรา
    #    มาตราสั้น ๆ (๕๑ ๕๔ ๙๑ ๙๘) จึงไม่มีก้อนเป็นของตัวเอง ถ้าดึงผิดทางจะได้ศูนย์
    body = rag.entity_articles("อธิบดี")
    missing = [a for a in ("8 ตรี", "27", "50", "51", "54", "91", "98")
               if f"[มาตรา {a}]" not in body]
    check("ดึงตัวบทได้ครบแม้มาตราที่ไม่มีก้อนค้นเป็นของตัวเอง", not missing,
          f"ขาด: {missing}")


def test_retrieval():
    """กติกาการค้นที่ห้ามพัง"""
    section("การค้นเอกสาร")
    allg = service.list_groups()

    q = "มาตรา ๙ ห้ามทำอะไรในที่ดินของรัฐ"
    g, y, v = service.pick_groups(q, allg)
    cs = rag.retrieve(q, k=5, rerank_query=q, groups=g, years=y or None, versions=v or None)
    check("ถามเจาะมาตรา -> มาตรานั้นต้องติดอันดับ 1",
          rag.head_article_num(cs[0].get("article", "")) == "9",
          f"ได้ {cs[0].get('article')}")
    check("คำถามทั่วไปต้องไม่มีตัวบทที่ถูกยกเลิกหลุดมา",
          all(c.get("in_force") for c in cs),
          str([c["doc_label"] for c in cs if not c.get("in_force")][:2]))
    check("chunk จาก OCR ต้องไม่ติดอันดับต้น ๆ", not any(c.get("is_scan") for c in cs[:3]))

    check("ถามมาตราที่ไม่มีจริง ต้องไม่คืนอะไรมั่ว", rag.article_timeline("999") == [])

    # ── embedding ล่ม ต้องไม่ทำให้ค้นไม่ได้เลย (BM25 คำนวณในเครื่อง ไม่ต้องต่อเน็ต) ──
    # เคยเกิดจริง: โควตา embedding หมด แล้วเส้นค้นตายทั้งเส้นทั้งที่ BM25 ยังใช้ได้
    #
    # ⚠️ ข้อนี้ดักความพลาดที่เคยเกิดจริงตอนเขียนฟีเจอร์นี้: ตอนแรกดัก error ด้วย
    #    (ConnectionError, TimeoutError, OSError) แล้วเทสผ่าน เพราะเทสจำลองด้วย
    #    ConnectionError — แต่ error จริงจาก endpoint สืบทอดจาก openai.OpenAIError
    #    ไม่ใช่ ConnectionError เลย ผลคือของจริงจะไม่ถูกดัก แล้วระบบพังเหมือนเดิม
    #    เทสข้างล่างเช็คกับคลาส error ตัวจริง ไม่ใช่ตัวที่เราจำลองขึ้นเอง
    try:
        import openai
        real_errs = ["APIConnectionError", "APITimeoutError",
                     "RateLimitError", "AuthenticationError"]
        uncaught = [n for n in real_errs
                    if not issubclass(getattr(openai, n), rag.SEMANTIC_DOWN)]
        check("error จริงจาก endpoint ต้องถูกดักครบทุกตัว", not uncaught,
              f"ดักไม่ได้: {uncaught}")
    except ImportError:
        pass

    class _DeadEmbed:
        def embed_query(self, _q):
            raise ConnectionError("simulated")

    _orig_embed = rag._embeddings
    rag._embeddings = _DeadEmbed()
    try:
        stages: list = []
        st: dict = {}
        cs = rag.retrieve("ที่ดินที่ออกใบจองห้ามโอนภายในกี่ปี",
                          groups=[rag.GROUP_IN_FORCE], on_stage=stages.append, stats=st)
        check("embedding ล่ม -> ยังค้นได้ด้วย BM25", len(cs) > 0, f"ได้ {len(cs)} ก้อน")
        check("embedding ล่ม -> ยังเจอมาตราที่มีคำตอบ",
              any(rag.head_article_num(c.get("article", "")) in ("31", "33") for c in cs),
              str([c.get("article") for c in cs[:5]]))
        check("embedding ล่ม -> รายงานโหมด bm25_only กลับมา",
              st.get("mode") == "bm25_only", str(st))
        # ⚠️ ข้อนี้สำคัญที่สุด — ลดคุณภาพแบบเงียบ ๆ คือบั๊กที่อยู่ได้นานที่สุด
        check("embedding ล่ม -> ต้องแจ้งเตือนออกมา ไม่เงียบ",
              any("⚠️" in s for s in stages), f"stages={stages[-2:]}")
    finally:
        rag._embeddings = _orig_embed
    st2: dict = {}
    rag.retrieve("มาตรา ๙", groups=[rag.GROUP_IN_FORCE], stats=st2)
    check("endpoint กลับมาปกติ -> โหมดกลับเป็น hybrid", st2.get("mode") == "hybrid", str(st2))
    # ⚠️ โหมดต้องเป็นของ "แต่ละคำถาม" ไม่ใช่ตัวแปรกลาง — ไม่งั้นคำถามที่ปกติจะไปทับ
    #    ค่าของคำถามที่ล่ม แล้วคนนั้นจะไม่ได้รับคำเตือน (FastAPI รับหลายคำถามพร้อมกันได้)
    check("โหมดแยกกันต่อคำถาม ไม่ปนกัน",
          st.get("mode") == "bm25_only" and st2.get("mode") == "hybrid",
          f"{st} / {st2}")
    brief = rag.amendment_brief(4)
    check("สรุป พ.ร.บ.แก้ไข ดึงได้ครบ (วันมีผลบังคับ + มาตราที่แก้)",
          "วันมีผลบังคับ" in brief and "มาตรา 31" in brief)


def test_no_superseded_leak():
    """กติกาที่สำคัญที่สุด: คำถามปกติต้องไม่ได้ตัวบทที่ถูกยกเลิกไปแล้ว"""
    section("กันตัวบทที่เลิกใช้แล้วหลุดเข้าคำตอบ")
    allg = service.list_groups()
    for q in ["คนต่างด้าวถือครองที่ดินได้แค่ไหน",
              "โฉนดที่ดินออกได้เมื่อใด",
              "ค่าธรรมเนียมจดทะเบียนคิดอย่างไร"]:
        g, y, v = service.pick_groups(q, allg)
        cs = rag.retrieve(q, k=8, rerank_query=q, groups=g, years=y or None,
                          versions=v or None)
        leak = [c["doc_label"] for c in cs if not c.get("in_force")]
        check(f"'{q[:32]}'", not leak, f"หลุดมา {len(leak)}: {leak[:2]}")


def test_portable_index():
    """ดัชนีที่ commit ไว้ต้องใช้ได้เมื่อ clone ไปเครื่องอื่น (โดยเฉพาะ Linux)

    บักจริงที่เจอตอนเตรียม deploy: file_hashes.json เซฟคีย์เป็น 'data\\x.pdf' แบบ Windows
    พอรันบน Linux ได้ 'data/x.pdf' -> ไม่ตรงทั้ง 53 ไฟล์ -> main.py สั่ง
    build_vectorstore(force=True) -> ลบ collection แล้ว embed ใหม่ทั้งชุดตั้งแต่บูตแรก
    """
    section("ดัชนีย้ายเครื่องได้")
    saved = rag.load_hashes()
    check("file_hashes.json มีครบทุกไฟล์ใน data/",
          len(saved) == len(rag._all_data_files()),
          f"{len(saved)} vs {len(rag._all_data_files())}")
    check("คีย์ใน file_hashes.json ใช้ '/' ไม่ใช่ '\\' (ย้าย OS แล้วยังตรง)",
          bool(saved) and not any("\\" in k for k in saved),
          f"เจอ backslash: {[k for k in saved if chr(92) in k][:2]}")
    check("_rel_key คืนค่าเป็น '/' เสมอ",
          "\\" not in rag._rel_key(os.path.join(rag.DATA_DIR, "x.pdf")))
    check("บูตแล้วไม่สั่ง rebuild (ไม่ embed ใหม่โดยไม่จำเป็น)", not rag.update_database())

    # ของที่ต้องมีจริงบนเครื่องปลายทาง ไม่งั้นบูตไม่ขึ้น
    check("มีโฟลเดอร์ data/ พร้อม PDF", len(rag._all_data_files()) > 0)
    check("มี chroma_db ที่มีเวกเตอร์อยู่จริง", rag._collection.count() > 0,
          f"{rag._collection.count()} เวกเตอร์")


# ══════════════════════════════════════════════════════════════════════════════
def main() -> int:
    for fn in (test_index_integrity, test_thai_text, test_question_intent,
               test_multiturn_routing, test_degrade_gracefully,
               test_amendment_graph, test_entity_index,
               test_retrieval, test_no_superseded_leak, test_portable_index):
        try:
            fn()
        except Exception as e:                       # ให้ชุดที่เหลือรันต่อได้
            check(f"{fn.__name__} ทำงานไม่จบ", False, f"{type(e).__name__}: {e}")

    print()
    npass = nfail = 0
    for ok, name, detail in _results:
        if ok is None:
            print(f"\n── {name} " + "─" * max(0, 56 - len(name)))
            continue
        npass, nfail = npass + bool(ok), nfail + (not ok)
        print(f"  {'✅' if ok else '❌'} {name}" + (f"\n       {detail}" if detail and not ok else ""))
    print("\n" + "─" * 62)
    print(f"ผ่าน {npass} · ตก {nfail}")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main())
