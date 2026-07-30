# -*- coding: utf-8 -*-
"""ทดลองเทียบ 3 วิธีตอบคำถาม "ต้อง hop จริง" — ไม่ได้ต่อเข้าแอปจริง

    A) baseline_multiquery_graph  = ระบบปัจจุบัน (service.answer_stream)
       คำถาม -> แตก 4 มุมพร้อมกัน -> ค้นรอบเดียว -> rerank -> ตอบ
       hop ที่ทำได้ = hop ที่คำนวณล่วงหน้าไว้ในกราฟการแก้ไข (article_chain/timeline)

    B) iterative_multihop_llm     = ให้ LLM คิดคำค้นรอบสองเอง
       ค้นรอบ 1 -> LLM อ่านผลแล้วบอก "ยังขาดอะไร ค้นอะไรต่อ" -> ค้นรอบ 2 -> ตอบ
       ⚠️ เสี่ยง: คืนอำนาจตัดสินใจให้โมเดล = ผลไม่นิ่ง (รอบแรกที่ลอง เลือก entity ผิด)

    C) entityhop_code             = ให้ "โค้ด" สกัด entity แล้วค้นรอบสองอัตโนมัติ
       ค้นรอบ 1 -> regex ดึงตำแหน่ง/องค์กร/ชื่อเอกสารจากตัวบทที่ค้นได้ -> ค้นรอบ 2 -> ตอบ
       ตรงหลักการโปรเจกต์: "ข้อเท็จจริงให้โค้ดคำนวณ การเรียบเรียงให้ LLM"

ทุกวิธีให้คะแนนด้วย run_eval.score_item ตัวเดียวกับ eval หลัก -> เทียบกันได้ตรง ๆ
เปิด RAG_TRACE=1 จะได้ trace คนละต้นต่อวิธี ชื่อตามด้านบน (เทียบใน MLflow ได้เลย)

    python experiments/multihop_iterative.py --id H2H1              # 3 วิธี 1 ข้อ
    python experiments/multihop_iterative.py --id H2H1 --mode B     # เฉพาะวิธี B
    python experiments/multihop_iterative.py --all                  # ทั้งชุด 8 ข้อ
"""
import argparse
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

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

import rag        # noqa: E402
import service    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(os.path.dirname(HERE), "eval")

# ── prompt ตัดสินใจว่า "ต้องค้นต่อไหม" ────────────────────────────────────────
# จงใจบังคับให้ตอบสั้นและมีรูปแบบตายตัว — ถ้าปล่อยอิสระ โมเดลจะบรรยายยาว
# แล้วเราแยกไม่ออกว่าตกลงมันขออะไร (บทเรียนเดียวกับ expand_queries)
FOLLOWUP_SYSTEM = """คุณช่วยวางแผนการค้นเอกสารกฎหมายไทย
ผู้ใช้ถามคำถามหนึ่ง และเราค้นเอกสารรอบแรกมาให้แล้ว
งานของคุณ: ดูว่าเอกสารที่ค้นมา "พอตอบคำถามครบหรือยัง"

ถ้าพอแล้ว ตอบคำเดียวว่า: ENOUGH
ถ้ายังไม่พอ ให้ตอบเป็นคำค้นใหม่ 1-2 บรรทัด (บรรทัดละ 1 คำค้น) โดย:
  - ต้องใช้ "ชื่อ/คำ" ที่เพิ่งรู้จากเอกสารรอบแรก (เช่น ชื่อตำแหน่ง ชื่อเอกสาร ชื่อองค์กร เลขฉบับ)
  - เป็นคำค้นที่ตามหาข้อมูล "ส่วนที่ยังขาด" ไม่ใช่ถามซ้ำของเดิม
  - ห้ามอธิบาย ห้ามใส่หมายเลขนำหน้า ตอบเฉพาะคำค้น"""


def plan_followup(llm, question: str, context: str, max_ctx: int = 6000) -> list[str]:
    """ให้ LLM อ่าน context รอบแรกแล้วบอกว่าต้องค้นอะไรต่อ — คืน [] ถ้าพอแล้ว"""
    user = (f"คำถาม: {question}\n\n"
            f"เอกสารที่ค้นมารอบแรก:\n{context[:max_ctx]}\n\n"
            f"พอตอบครบหรือยัง? ถ้าไม่พอ ต้องค้นอะไรต่อ")
    try:
        r = llm.invoke([SystemMessage(content=FOLLOWUP_SYSTEM), HumanMessage(content=user)])
        rag.track_usage(r)
    except Exception as e:
        print(f"    [!] plan_followup พัง: {type(e).__name__} — ถือว่าพอแล้ว")
        return []
    out = str(r.content or "").strip()
    if not out or "ENOUGH" in out.upper():
        return []
    qs = []
    for line in out.splitlines():
        q = re.sub(r"^\s*(?:\d+[.)]|[-*])\s*", "", " ".join(line.split()))
        if q and "ENOUGH" not in q.upper() and len(q) > 3:
            qs.append(q)
    return qs[:2]


# ── วิธี C: ให้ "โค้ด" สกัด entity จากตัวบทรอบแรก แทนที่จะให้ LLM คิดคำค้นเอง ──────
# ทำไมต้องเป็น regex ไม่ใช่ LLM: รอบแรกที่ลองวิธี B โมเดลขอค้น "เจ้าพนักงานที่ดิน"
# ทั้งที่ตัวบทเขียนว่า "อธิบดี" — คนละตำแหน่งกันในกฎหมาย ผลคือ hop รอบสองไปผิดทาง
# การดึงคำจากตัวบทตรง ๆ ไม่มีทางเพี้ยนแบบนั้น (ตรวจย้อนได้ว่าคำนี้มาจากมาตราไหน)
ACTOR_RE = re.compile(
    r"(อธิบดี|รัฐมนตรีว่าการกระทรวงมหาดไทย|รัฐมนตรี|คณะรัฐมนตรี|"
    r"เจ้าพนักงานที่ดินจังหวัด|เจ้าพนักงานที่ดิน|พนักงานเจ้าหน้าที่|"
    r"ผู้ว่าราชการจังหวัด|นายอำเภอ|คณะอนุกรรมการ|คณะกรรมการ)")
DOCUMENT_RE = re.compile(
    r"(ใบจอง|โฉนดที่ดิน|หนังสือรับรองการทำประโยชน์|ใบไต่สวน|"
    r"หนังสือสำคัญสำหรับที่หลวง|ใบแทน|ตราจอง)")


def extract_hop_entities(question: str, chunks: list[dict], top: int = 2) -> list[str]:
    """หา entity ที่ 'เพิ่งรู้จากรอบแรก' — คือโผล่ในตัวบทที่ค้นได้ แต่ไม่ได้อยู่ในคำถามเดิม

    เงื่อนไข "ไม่อยู่ในคำถามเดิม" สำคัญมาก: ถ้าผู้ใช้พิมพ์ 'อธิบดี' มาเองอยู่แล้ว
    การค้นซ้ำด้วยคำเดิมไม่ได้ข้อมูลใหม่ — hop ที่มีค่าคือ hop ไปยังสิ่งที่เพิ่งรู้"""
    from collections import Counter
    text = " ".join(c.get("text", "") for c in chunks)
    counts: "Counter[str]" = Counter()
    for rx in (ACTOR_RE, DOCUMENT_RE):
        for m in rx.finditer(text):
            e = m.group(1)
            if e not in question:          # ต้องเป็นของใหม่จริง ๆ
                counts[e] += 1
    return [e for e, _ in counts.most_common(top)]


def answer_entityhop(llm, question: str, groups: list, verbose: bool = True) -> dict:
    """วิธี C: ค้นรอบ 1 -> โค้ดสกัด entity -> ค้นรอบ 2 ด้วย entity นั้น -> ตอบ
    ไม่มี LLM call เพิ่มสำหรับการวางแผน (ต่างจากวิธี B) จึงเร็วกว่าและนิ่งกว่า"""
    t0 = time.perf_counter()
    g, y, v = service.pick_groups(question, groups)
    qs1 = rag.expand_queries(llm, question)
    chunks = rag.retrieve(qs1, rerank_query=question, groups=g,
                          years=y or None, versions=v or None)
    seen = {c["id"] for c in chunks}
    all_chunks = list(chunks)
    hop_log = [{"hop": 1, "queries": qs1, "n_new": len(chunks),
                "articles": [c.get("article", "") for c in chunks[:6]]}]
    if verbose:
        arts = ", ".join(c.get("article", "") for c in chunks[:4])
        print(f"    hop 1: ค้น {len(qs1)} คำ -> {len(chunks)} ก้อน ({arts})")

    ents = extract_hop_entities(question, chunks)
    if ents:
        # คำค้นรอบสอง: "เจาะ entity ล้วน" — จงใจไม่ผสมคำถามเดิมเข้าไป
        # รอบแรกลองแบบ "entity + คำถามเดิมทั้งประโยค" แล้วพัง: คำถามเดิมยาวกว่ามาก
        # จนกลบสัญญาณของ entity ผลคือดึงก้อนเดิมกลับมา (ซ้ำของที่มีแล้ว) ได้ก้อนใหม่ 1 ก้อน
        # ของที่ยังขาดคือ "entity นี้ไปโผล่ที่ไหนอีก" ซึ่งต้องค้นด้วยตัว entity ล้วน ๆ
        qs2 = []
        for e in ents:
            qs2 += [f"อำนาจหน้าที่ของ{e}", f"ให้{e}มีอำนาจ"]
        if verbose:
            print(f"    โค้ดสกัดได้: {ents} -> ค้นต่อ {len(qs2)} คำ")
        # rerank ด้วย entity เป็นหลัก (ไม่ใช่คำถามเดิม) — รอบนี้กำลังตามหา "entity ไปโผล่ที่ไหนอีก"
        more = rag.retrieve(qs2, rerank_query=f"อำนาจหน้าที่ของ{ents[0]}", groups=g,
                            years=y or None, versions=v or None)
        fresh = [c for c in more if c["id"] not in seen]
        for c in fresh:
            seen.add(c["id"])
        all_chunks.extend(fresh)
        hop_log.append({"hop": 2, "queries": qs2, "entities": ents, "n_new": len(fresh),
                        "articles": [c.get("article", "") for c in fresh[:6]]})
        if verbose:
            arts = ", ".join(c.get("article", "") for c in fresh[:4])
            print(f"    hop 2: ก้อนใหม่ {len(fresh)} ({arts or '-'})")
    elif verbose:
        print("    โค้ดไม่เจอ entity ใหม่ -> ไม่ต้อง hop ต่อ")

    context = rag.format_context(all_chunks)
    msgs = [SystemMessage(content=rag.DOMAINS["thai_law"]["system_prompt"]),
            HumanMessage(content=service.build_user_prompt(question, context))]
    resp = rag.invoke_retry(llm, msgs, ok_fn=lambda c: not rag.looks_truncated(c),
                            label="answer-entityhop")
    return {"answer": str(resp.content), "hops": hop_log, "n_chunks": len(all_chunks),
            "elapsed": round(time.perf_counter() - t0, 1)}


def answer_baseline(llm, question: str, groups: list, verbose: bool = True) -> dict:
    """วิธี A: ระบบปัจจุบันตรง ๆ (service.answer_stream) — ไม่แตะอะไรเลย"""
    t0 = time.perf_counter()
    answer, chunks = "", []
    for ev in service.answer_stream(llm, question, all_groups=groups, stream=False):
        if "final" in ev:
            answer = ev["final"]["answer"]
            chunks = ev["final"]["chunks"]
    if verbose:
        arts = ", ".join(c.get("article", "") for c in chunks[:4])
        print(f"    ค้นรอบเดียว -> {len(chunks)} ก้อน ({arts})")
    return {"answer": answer, "hops": [{"hop": 1, "n_new": len(chunks)}],
            "n_chunks": len(chunks), "elapsed": round(time.perf_counter() - t0, 1)}


def answer_iterative(llm, question: str, groups: list, max_hops: int = 2,
                     verbose: bool = True) -> dict:
    """ตอบแบบ iterative: ค้น -> ให้ LLM ดูว่าขาดอะไร -> ค้นต่อ -> ตอบ
    คืน dict สรุปสิ่งที่เกิดขึ้นทุกรอบ (ไว้เทียบกับ baseline)"""
    t0 = time.perf_counter()
    g, y, v = service.pick_groups(question, groups)
    seen_ids: set[str] = set()
    all_chunks: list[dict] = []
    hop_log: list[dict] = []
    queries = [question]

    seen_queries: set[str] = set()
    stop_reason = f"ชนเพดาน {max_hops} รอบ"

    for hop in range(1, max_hops + 1):
        # รอบแรกใช้ multi-query เหมือนระบบจริง รอบถัดไปใช้คำค้นที่ LLM คิดเอง
        qs = rag.expand_queries(llm, question) if hop == 1 else queries
        chunks = rag.retrieve(qs, rerank_query=queries[0] if hop == 1 else " ".join(qs),
                              groups=g, years=y or None, versions=v or None)
        fresh = [c for c in chunks if c["id"] not in seen_ids]
        for c in fresh:
            seen_ids.add(c["id"])
        all_chunks.extend(fresh)
        arts = [c.get("article", "") for c in fresh[:6] if c.get("article")]
        hop_log.append({"hop": hop, "queries": qs, "n_new": len(fresh), "articles": arts})
        if verbose:
            print(f"    hop {hop}: ค้น {len(qs)} คำ -> ก้อนใหม่ {len(fresh)} "
                  f"({', '.join(arts[:4]) or '-'})")

        # ── เบรกที่ 1 (โค้ดตัดสิน): รอบนี้ไม่ได้ของใหม่เลย = ค้นต่อไปก็ไม่ได้อะไร ──
        # นี่คือ "ข้อเท็จจริง" ที่นับได้ ไม่ใช่เรื่องที่ต้องถาม LLM ว่าพอหรือยัง
        # จากการทดลอง: LLM ไม่เคยบอก ENOUGH เลยแม้รอบ 3/4/5 จะได้ของใหม่ 0 ก้อน
        if hop > 1 and not fresh:
            stop_reason = f"หยุดที่รอบ {hop}: ไม่ได้ก้อนใหม่ (โค้ดตัดสิน)"
            if verbose:
                print(f"    ⛔ {stop_reason}")
            break
        if hop >= max_hops:
            break

        queries = plan_followup(llm, question, rag.format_context(all_chunks))
        # ── เบรกที่ 2 (โค้ดตัดสิน): LLM ขอค้นคำเดิมที่เคยค้นไปแล้ว = วนที่เดิม ──
        queries = [q for q in queries if q not in seen_queries]
        seen_queries.update(queries)
        if not queries:
            stop_reason = (f"หยุดที่รอบ {hop}: "
                           + ("LLM บอกพอแล้ว" if not seen_queries else "คำค้นซ้ำของเดิม"))
            if verbose:
                print(f"    ⛔ {stop_reason}")
            break
        if verbose:
            print(f"    LLM ขอค้นต่อ: {queries}")

    context = rag.format_context(all_chunks)
    sys_prompt = rag.DOMAINS["thai_law"]["system_prompt"]
    msgs = [SystemMessage(content=sys_prompt),
            HumanMessage(content=service.build_user_prompt(question, context))]
    resp = rag.invoke_retry(llm, msgs, ok_fn=lambda c: not rag.looks_truncated(c),
                            label="answer-iterative")
    return {"answer": str(resp.content), "hops": hop_log, "n_chunks": len(all_chunks),
            "stop_reason": stop_reason, "elapsed": round(time.perf_counter() - t0, 1)}


# ── ให้คะแนนด้วยไม้บรรทัดเดียวกับ run_eval.py (เทียบกันได้ตรง ๆ) ──────────────
def score(item: dict, answer: str) -> dict:
    sys.path.insert(0, EVAL_DIR)
    import run_eval           # noqa: E402  (ใช้ norm/score_item ตัวเดียวกัน)
    return run_eval.score_item(item, answer)


# ── วิธี E: hop ตาม "ลิงก์" ที่ตัวบทเขียนไว้เอง (ฟิลด์ refs) ────────────────────
# กฎหมายไทยเขียนทางเดินไว้ในตัวมันเอง: "...ให้ปฏิบัติตามมาตรา ๖๑...", "ภายใต้บังคับมาตรา ๙"
# ตอน build ดัชนี เราเก็บลิงก์พวกนี้ไว้แล้วในฟิลด์ refs ('|9|61|') แต่ไม่เคยเอามาเดิน
# hop แบบนี้ "นิ่ง 100%" เพราะลิงก์ไม่เปลี่ยน — input เดิม -> ผลเดิมเสมอ ไม่มี LLM เกี่ยว
# และจบแน่นอน เพราะลิงก์ในเอกสารมีจำนวนจำกัด ไม่ใช่จินตนาการไม่รู้จบ
def collect_refs(chunks: list[dict], max_refs: int = 4) -> list[str]:
    """รวมเลขมาตราที่ chunk ชุดนี้ 'อ้างถึง' แต่ยังไม่มีตัวบทอยู่ในมือ
    เรียงตามจำนวนครั้งที่ถูกอ้าง — ยิ่งหลาย chunk อ้างถึง ยิ่งน่าจะสำคัญกับคำถาม"""
    from collections import Counter
    have = {rag.head_article_num(c.get("article", "")) for c in chunks}
    have.discard("")
    cnt: "Counter[str]" = Counter()
    for c in chunks:
        for r in (c.get("refs") or "").strip("|").split("|"):
            if r and r not in have:
                cnt[r] += 1
    return [r for r, _ in cnt.most_common(max_refs)]


def chunks_for_article(num: str, in_force_only: bool = True) -> list[dict]:
    """ดึง chunk ที่ 'เนื้อหาหลัก' เป็นมาตรานั้นจริง ๆ (ดูฟิลด์ article ไม่ใช่แค่เอ่ยถึง)
    ดึงตรงจากดัชนี ไม่ผ่านการค้น -> ไม่มีทางพลาดและไม่มีทางได้ของมั่ว"""
    num = rag.ARTICLE_ALIASES.get(num, num)
    out = [c for c in rag._chunks
           if rag.head_article_num(c.get("article", "")) == num
           and (c.get("in_force") if in_force_only else True)]
    return out[:2]                      # กัน chunk ยาว ๆ ที่ถูกหั่นหลายก้อน


def answer_refshop(llm, question: str, groups: list, verbose: bool = True) -> dict:
    """วิธี E: ค้นรอบ 1 -> อ่านลิงก์ refs จากก้อนที่ได้ -> ดึงมาตราปลายทางตรงจากดัชนี -> ตอบ
    ไม่เรียก LLM วางแผน ไม่ค้นรอบสอง = นิ่งและเร็ว"""
    t0 = time.perf_counter()
    g, y, v = service.pick_groups(question, groups)
    qs1 = rag.expand_queries(llm, question)
    chunks = rag.retrieve(qs1, rerank_query=question, groups=g,
                          years=y or None, versions=v or None)
    seen = {c["id"] for c in chunks}
    all_chunks = list(chunks)
    hop_log = [{"hop": 1, "queries": qs1, "n_new": len(chunks),
                "articles": [c.get("article", "") for c in chunks[:6]]}]
    if verbose:
        print(f"    hop 1: ค้น {len(qs1)} คำ -> {len(chunks)} ก้อน "
              f"({', '.join(c.get('article', '') for c in chunks[:4])})")

    refs = collect_refs(chunks)
    if refs:
        fresh = []
        for r in refs:
            for c in chunks_for_article(r):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    fresh.append(c)
        all_chunks.extend(fresh)
        hop_log.append({"hop": 2, "refs": refs, "n_new": len(fresh),
                        "articles": [c.get("article", "") for c in fresh[:6]]})
        if verbose:
            print(f"    ลิงก์ที่ตัวบทเขียนไว้: มาตรา {', '.join(refs)}")
            print(f"    hop 2: ดึงตรงจากดัชนี -> ก้อนใหม่ {len(fresh)} "
                  f"({', '.join(c.get('article', '') for c in fresh[:4]) or '-'})")
    elif verbose:
        print("    ตัวบทที่ค้นได้ไม่มีลิงก์ไปมาตราอื่น -> ไม่ต้อง hop")

    context = rag.format_context(all_chunks)
    msgs = [SystemMessage(content=rag.DOMAINS["thai_law"]["system_prompt"]),
            HumanMessage(content=service.build_user_prompt(question, context))]
    resp = rag.invoke_retry(llm, msgs, ok_fn=lambda c: not rag.looks_truncated(c),
                            label="answer-refshop")
    return {"answer": str(resp.content), "hops": hop_log, "n_chunks": len(all_chunks),
            "stop_reason": f"เดินลิงก์ครบ {len(refs)} เส้น (โค้ดกำหนด)",
            "elapsed": round(time.perf_counter() - t0, 1)}


# ── วิธี F: "ชั้นเติมของ" — ค้นปกติเป็นพื้นเสมอ แล้วเติมข้อเท็จจริงที่คำนวณได้ทับ ──
# แนวคิด: ไม่ต้องรู้ล่วงหน้าว่า user จะถามแบบไหน เพราะไม่ได้ "เลือกทางใดทางหนึ่ง"
# แต่เช็คทุกกฎแล้วเติมเฉพาะที่เข้าเงื่อนไข — กฎยิงพลาดก็แค่มี context เกินนิดหน่อย
# การค้นปกติยังทำงานอยู่เสมอ ต่างจากเส้นกราฟปัจจุบันที่ "เข้าแล้วข้ามการค้นทั้งหมด"
# ทุกชั้นเป็น deterministic (regex + กราฟ + metadata) ไม่มี LLM ตัดสินใจ -> รันซ้ำได้ผลเดิม
def answer_layered(llm, question: str, groups: list, verbose: bool = True) -> dict:
    t0 = time.perf_counter()
    g, y, v = service.pick_groups(question, groups)
    layers: list[str] = []          # ข้อเท็จจริงที่โค้ดคำนวณได้ เอาไปวางหัว context

    # ── ชั้นพื้น: ค้น hybrid ปกติ (ทำเสมอ ไม่มีวันข้าม) ──
    qs = rag.expand_queries(llm, question)
    chunks = rag.retrieve(qs, rerank_query=question, groups=g,
                          years=y or None, versions=v or None)
    seen = {c["id"] for c in chunks}
    hop_log = [{"hop": 1, "layer": "ค้น hybrid", "n_new": len(chunks),
                "articles": [c.get("article", "") for c in chunks[:6]]}]
    if verbose:
        print(f"    [พื้น] ค้น -> {len(chunks)} ก้อน "
              f"({', '.join(c.get('article', '') for c in chunks[:4])})")

    # ── ชั้น 1: กราฟการแก้ไข (ถ้าคำถามเข้าเงื่อนไข) ──
    art = service.detect_compare(question)
    if art:
        chain = rag.article_chain(art)
        points = rag.article_timeline(art)
        if chain:
            layers.append(rag.format_chain(art, chain))
        if points:
            layers.append(f"ตัวบท 'มาตรา {art}' ทุกรุ่นที่เนื้อหาเปลี่ยนจริง "
                          f"(พบ {len(points)} รุ่น):\n\n{rag.format_comparison(art, points)}")
        hop_log.append({"hop": 2, "layer": "กราฟการแก้ไข", "article": art,
                        "n_chain": len(chain), "n_points": len(points)})
        if verbose:
            print(f"    [+กราฟ] มาตรา {art}: สาย {len(chain)} ทอด · ตัวบท {len(points)} รุ่น")

    # ── ชั้น 2: ลิงก์ refs ที่ตัวบทเขียนไว้เอง ──
    refs = collect_refs(chunks)
    if refs:
        fresh = []
        for r in refs:
            for c in chunks_for_article(r):
                if c["id"] not in seen:
                    seen.add(c["id"])
                    fresh.append(c)
        chunks = chunks + fresh
        hop_log.append({"hop": 3, "layer": "ลิงก์ refs", "refs": refs, "n_new": len(fresh),
                        "articles": [c.get("article", "") for c in fresh[:6]]})
        if verbose:
            print(f"    [+refs] ลิงก์ มาตรา {', '.join(refs)} -> ก้อนใหม่ {len(fresh)}")

    # ── ชั้น 3: สรุปฉบับแก้ไข (ของเดิมที่ระบบมีอยู่แล้ว) ──
    amend_nos = rag.question_amendments(question)
    if len(amend_nos) >= 2:
        ov = rag.amendment_overlap(amend_nos)
        if ov:
            layers.append(ov)
    for no in amend_nos:
        brief = rag.amendment_brief(no)
        if brief:
            layers.append(brief)
    if amend_nos and verbose:
        print(f"    [+ฉบับ] สรุปฉบับที่ {amend_nos}")

    context = rag.format_context(chunks)
    if layers:
        context = "\n\n".join(layers) + "\n\n" + "=" * 16 + "\n\n" + context
    msgs = [SystemMessage(content=rag.DOMAINS["thai_law"]["system_prompt"]),
            HumanMessage(content=service.build_user_prompt(question, context))]
    resp = rag.invoke_retry(llm, msgs, ok_fn=lambda c: not rag.looks_truncated(c),
                            label="answer-layered")
    return {"answer": str(resp.content), "hops": hop_log, "n_chunks": len(chunks),
            "stop_reason": f"เติม {len(layers)} ชั้น (กฎในโค้ดทั้งหมด)",
            "elapsed": round(time.perf_counter() - t0, 1)}


# ── วิธี G: ชั้นเติมของ + ดัชนีตัวละคร (entity index) ─────────────────────────
# ปัญหาที่ยังเหลือหลังทำ F: คำถามแบบ "รวบรวมให้ครบ" ยังตอบไม่ได้ (H2H1 ตก 0/3 ทุกวิธี)
# เพราะ "อธิบดี" อยู่ใน 13 มาตรา แต่ค้นได้ทีละ 12 ก้อน — ไม่ใช่ปัญหาการ hop แต่เป็น
# ข้อจำกัดของ top-k เอง (เรียงตามความคล้าย ไม่ใช่ความครบ) hop กี่รอบก็ไม่ครบ
#
# G เติมอีก 2 ชั้นบน F:
#   ชั้น entity : เอา "รายชื่อมาตราทั้งหมดที่พูดถึง X" (โค้ดนับ) วางหัว context
#   ชั้น pull   : ดึงตัวบทของมาตราในรายการนั้นที่ยังไม่มีในมือ เข้ามาให้ LLM อ่านจริง
#
# ⚠️ entity ต้องหาจาก "ก้อนที่ค้นได้" ไม่ใช่จากคำถาม — คำถามจริงมักบรรยายตัวละคร
#    แทนที่จะเรียกชื่อ ("ผู้ที่มีอำนาจสั่งเพิกถอนโฉนด..." ไม่มีคำว่าอธิบดีสักตัว)
#    ขั้นนี้ยังนิ่ง 100% เพราะเป็น regex บนก้อนที่ได้มา ไม่มี LLM ตัดสินใจ

# ยิงเฉพาะคำถามที่ "ต้องการความครบ" — คำถามเจาะจุดเดียวไม่ต้องเติม จะได้ไม่บวม context
_SCOPE_HINT = re.compile(r"อะไรบ้าง|ใดบ้าง|อื่นใดอีก|อื่นอีก|อีกบ้าง|ทั้งหมด|"
                         r"มีกี่|กี่มาตรา|ครบถ้วน|ทุกมาตรา|รวบรวม")


_POWER_HINT = re.compile(r"อำนาจ|หน้าที่|สั่ง|อนุญาต|พิจารณา|แต่งตั้ง|กำหนด|ผู้ใด")


def pick_entities(question: str, chunks: list[dict],
                  top_n: int = 4, max_ents: int = 2) -> list[str]:
    """เลือก entity ที่ควรเติมรายชื่อมาตราให้ — เป็นกฎในโค้ดล้วน ไม่มี LLM ตัดสิน

    ⚠️ ห้ามดูแค่คำถาม: คำถามจริงมักบรรยายตัวละครแทนที่จะเรียกชื่อ
       "ผู้ที่มีอำนาจสั่งเพิกถอนโฉนดที่ดิน..." -> จับได้แต่ "โฉนดที่ดิน" ซึ่งเป็นกรรม
       ไม่ใช่ประธาน ต้องดูตัวบทที่ค้นได้ด้วยถึงจะรู้ว่าประธานคือ "อธิบดี"

    ⚠️ และห้ามนับแค่ 'เจอ/ไม่เจอ': มาตราเดียวเอ่ยถึงหลายตัว ต้องนับจำนวนครั้ง
       ตัวที่เป็นประธานของมาตราจะถูกเอ่ยซ้ำมากกว่าตัวประกอบ
    """
    from collections import Counter
    cnt: "Counter[str]" = Counter()
    for c in chunks[:top_n]:                       # ดูเฉพาะก้อนอันดับต้น — ก้อนท้ายมักหลุดประเด็น
        for name, n in rag.count_entities(c.get("text", "")).items():
            cnt[name] += n
    for name in rag.question_entities(question):   # ที่คำถามเรียกชื่อมาตรง ๆ ให้แต้มพิเศษ
        cnt[name] += 2
    cand = [n for n, _ in cnt.most_common()]
    # คำถามแนว "มีอำนาจ/หน้าที่อะไรบ้าง" -> ต้องเป็นผู้กระทำ ไม่ใช่ชื่อเอกสาร
    if _POWER_HINT.search(question or ""):
        actors = [n for n in cand if n in rag.ENTITY_ACTORS]
        if actors:
            cand = actors
    return cand[:max_ents]


def answer_entity_index(llm, question: str, groups: list, verbose: bool = True) -> dict:
    """วิธี G: F + ดัชนีตัวละคร — ทุกชั้นเป็นโค้ดล้วน ไม่มี LLM ตัดสินใจว่าจะไปไหนต่อ"""
    t0 = time.perf_counter()
    g, y, v = service.pick_groups(question, groups)
    layers: list[str] = []

    # ── ชั้นพื้น: ค้น hybrid ปกติ (ทำเสมอ) ──
    qs = rag.expand_queries(llm, question)
    chunks = rag.retrieve(qs, rerank_query=question, groups=g,
                          years=y or None, versions=v or None)
    seen = {c["id"] for c in chunks}
    hop_log = [{"hop": 1, "layer": "ค้น hybrid", "n_new": len(chunks),
                "articles": [c.get("article", "") for c in chunks[:6]]}]
    if verbose:
        print(f"    [พื้น] ค้น -> {len(chunks)} ก้อน "
              f"({', '.join(c.get('article', '') for c in chunks[:4])})")

    # ── ชั้น 1: กราฟการแก้ไข ──
    art = service.detect_compare(question)
    if art:
        chain, points = rag.article_chain(art), rag.article_timeline(art)
        if chain:
            layers.append(rag.format_chain(art, chain))
        if points:
            layers.append(f"ตัวบท 'มาตรา {art}' ทุกรุ่นที่เนื้อหาเปลี่ยนจริง "
                          f"(พบ {len(points)} รุ่น):\n\n{rag.format_comparison(art, points)}")
        hop_log.append({"hop": 2, "layer": "กราฟการแก้ไข", "article": art})
        if verbose:
            print(f"    [+กราฟ] มาตรา {art}: สาย {len(chain)} ทอด · ตัวบท {len(points)} รุ่น")

    # ── ชั้น 2: ดัชนีตัวละคร (ของใหม่) ──
    ents, pulled_arts = [], []
    if _SCOPE_HINT.search(question):
        ents = pick_entities(question, chunks)
        for name in ents:
            brief = rag.entity_brief(name)          # รายชื่อมาตรา (บอกว่ามีอะไรบ้าง)
            if brief:
                layers.append(brief)
            body = rag.entity_articles(name)        # ตัวบทจริงของมาตราเหล่านั้น
            if body:
                layers.append(body)
                pulled_arts += rag.entity_index().get(name, [])
        hop_log.append({"hop": 3, "layer": "ดัชนีตัวละคร", "entities": ents,
                        "n_new": len(pulled_arts), "articles": pulled_arts})
        if verbose:
            print(f"    [+ตัวละคร] {', '.join(ents) or '-'} "
                  f"-> เติมตัวบท {len(pulled_arts)} มาตรา ({', '.join(pulled_arts[:8])}...)")
    elif verbose:
        print("    [+ตัวละคร] คำถามไม่ได้ถามแบบ 'ครบถ้วน' -> ไม่เติม")

    # ── ชั้น 3: ลิงก์ refs ──
    refs = collect_refs(chunks)
    if refs:
        fresh = [c for r in refs for c in chunks_for_article(r) if c["id"] not in seen]
        for c in fresh:
            seen.add(c["id"])
        chunks = chunks + fresh
        hop_log.append({"hop": 4, "layer": "ลิงก์ refs", "refs": refs, "n_new": len(fresh)})
        if verbose:
            print(f"    [+refs] ลิงก์ มาตรา {', '.join(refs)} -> ก้อนใหม่ {len(fresh)}")

    # ── ชั้น 4: สรุปฉบับแก้ไข ──
    amend_nos = rag.question_amendments(question)
    if len(amend_nos) >= 2:
        ov = rag.amendment_overlap(amend_nos)
        if ov:
            layers.append(ov)
    for no in amend_nos:
        brief = rag.amendment_brief(no)
        if brief:
            layers.append(brief)

    context = rag.format_context(chunks)
    if layers:
        context = "\n\n".join(layers) + "\n\n" + "=" * 16 + "\n\n" + context
    msgs = [SystemMessage(content=rag.DOMAINS["thai_law"]["system_prompt"]),
            HumanMessage(content=service.build_user_prompt(question, context))]
    resp = rag.invoke_retry(llm, msgs, ok_fn=lambda c: not rag.looks_truncated(c),
                            label="answer-entity-index")
    return {"answer": str(resp.content), "hops": hop_log, "n_chunks": len(chunks),
            "stop_reason": f"เติม {len(layers)} ชั้น · ตัวละคร {ents or '-'} (กฎในโค้ดทั้งหมด)",
            "elapsed": round(time.perf_counter() - t0, 1)}


# ── บังคับปิดเส้นกราฟ เพื่อวัด "ค้นอย่างเดียว" ────────────────────────────────
class no_graph:
    """ปิด detect_compare ชั่วคราว -> answer_stream ตกไปเส้นค้นปกติเสมอ
    ใช้แยกวัดว่า 'ถ้าไม่มีกราฟช่วย ระบบค้นอย่างเดียวทำได้แค่ไหน'"""

    def __enter__(self):
        self._orig = service.detect_compare
        service.detect_compare = lambda q: ""
        return self

    def __exit__(self, *exc):
        service.detect_compare = self._orig
        return False


def answer_search_only(llm, question: str, groups: list, verbose: bool = True) -> dict:
    """แบบ A: ค้นปกติล้วน (ปิดกราฟ) — trace เดิมของระบบก่อนมีกราฟ"""
    with no_graph():
        r = answer_baseline(llm, question, groups, verbose=verbose)
    return r


def answer_graph(llm, question: str, groups: list, verbose: bool = True) -> dict:
    """แบบ C: ปล่อยให้เส้นกราฟทำงานตามปกติ (detect_compare -> _answer_compare)
    ถ้าคำถามไม่เข้าเงื่อนไขกราฟ จะตกไปเส้นค้นปกติเอง (บอกไว้ในผลลัพธ์)"""
    routed = bool(service.detect_compare(question))
    if verbose:
        print(f"    เส้นกราฟ: {'ทำงาน (มาตรา ' + service.detect_compare(question) + ')' if routed else 'ไม่เข้าเงื่อนไข -> ตกไปเส้นค้นปกติ'}")
    r = answer_baseline(llm, question, groups, verbose=verbose)
    r["graph_routed"] = routed
    return r


# ── ทะเบียนวิธี — ชื่อในนี้คือชื่อ trace ที่จะเห็นใน MLflow ────────────────────
MAX_HOPS = 5          # เพดานกัน LLM วนไม่จบ (ปกติมันบอก ENOUGH เองก่อนถึง)

MODES = {
    "A": ("A_search_only", "ค้นปกติล้วน (ปิดกราฟ) = trace เดิม",
          lambda llm, q, g: answer_search_only(llm, q, g)),
    "B": ("B_multihop_iterative", f"Multi-hop: LLM ค้นต่อเองจนพอ (เพดาน {MAX_HOPS} รอบ)",
          lambda llm, q, g: answer_iterative(llm, q, g, max_hops=MAX_HOPS)),
    "C": ("C_graph_chain", "กราฟ: article_chain/timeline ดึงตรงจากกราฟการแก้ไข",
          lambda llm, q, g: answer_graph(llm, q, g)),
    # วิธีเสริมจากการทดลองก่อนหน้า (ไม่ได้อยู่ในชุดเทียบหลัก) — ให้โค้ดสกัด entity
    "D": ("D_entityhop_code", "โค้ดสกัด entity แล้วค้นรอบสองอัตโนมัติ",
          lambda llm, q, g: answer_entityhop(llm, q, g)),
    "E": ("E_refs_hop", "hop ตามลิงก์ที่ตัวบทเขียนไว้เอง (ฟิลด์ refs) — นิ่ง 100%",
          lambda llm, q, g: answer_refshop(llm, q, g)),
    "F": ("F_layered_best", "ชั้นเติมของ: ค้นเสมอ + กราฟ + refs + สรุปฉบับ (ที่แนะนำ)",
          lambda llm, q, g: answer_layered(llm, q, g)),
    "G": ("G_entity_index", "F + ดัชนีตัวละคร (โค้ดนับว่า X อยู่มาตราใดบ้าง) — แก้คำถามแบบ 'รวบรวมให้ครบ'",
          lambda llm, q, g: answer_entity_index(llm, q, g)),
}
DEFAULT_MODES = ["A", "B", "C"]


def run_traced(mode: str, llm, question: str, groups: list) -> dict:
    """รัน 1 วิธี ภายใต้ span ราก 1 อัน ชื่อตาม MODES — จะได้ trace ต้นเดียวต่อวิธี
    เทียบกันใน MLflow ได้ตรง ๆ (ไม่เปิด trace ก็รันได้ปกติ)"""
    name, _desc, fn = MODES[mode]
    if not rag.TRACE_ENABLED:
        return fn(llm, question, groups)
    import mlflow
    with mlflow.start_span(name=name) as span:
        span.set_inputs({"question": question, "mode": mode, "method": name})
        r = fn(llm, question, groups)
        span.set_outputs({"answer": r["answer"], "n_chunks": r["n_chunks"],
                          "n_hops": len(r["hops"]), "elapsed_s": r["elapsed"]})
        return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="multihop.json")
    ap.add_argument("--id", nargs="*", help="ทำเฉพาะข้อที่ระบุ")
    ap.add_argument("--all", action="store_true", help="ทำทั้งชุด")
    ap.add_argument("--mode", nargs="*", choices=list(MODES),
                    help="เลือกวิธี (ไม่ระบุ = ทำครบ A B C)")
    ap.add_argument("--q", help="ถามคำถามสด ๆ แทนการอ่านจากชุดคำถาม (ไม่มีเฉลย)")
    ap.add_argument("--repeat", type=int, default=1,
                    help="รันซ้ำกี่รอบต่อข้อ — ใช้วัด 'ความนิ่ง' (ผลต่างระหว่างรอบ)")
    args = ap.parse_args()

    if args.q:                     # ถามสด — ไม่มีเฉลย ให้คะแนนไม่ได้ ดูเส้นทางอย่างเดียว
        items = [{"id": "ADHOC", "level": "-", "q": args.q, "gold": "(ไม่มีเฉลย)",
                  "must": []}]
    else:
        gt = json.load(open(os.path.join(EVAL_DIR, args.file), encoding="utf-8"))
        items = gt["items"]
        if args.id:
            want = {x.upper() for x in args.id}
            items = [i for i in items if i["id"] in want]
        elif not args.all:
            items = items[:1]
    modes = args.mode or DEFAULT_MODES

    rag._ensure_loaded()
    llm = rag.build_llm()
    groups = service.list_groups()
    print(f"\n=== เทียบ {len(modes)} วิธี × {len(items)} ข้อ "
          f"(trace: {'เปิด' if rag.TRACE_ENABLED else 'ปิด'}) ===")
    for m in modes:
        print(f"   {m}. {MODES[m][0]:<26} {MODES[m][1]}")

    results = []
    for it in items:
        print(f"\n[{it['id']}] {it['q'][:72]}")
        for m in modes:
            name = MODES[m][0]
            print(f"  ── {m}: {name}")
            for rd in range(1, args.repeat + 1):
                r = run_traced(m, llm, it["q"], groups)
                sc = score(it, r["answer"])
                results.append({**{k: it[k] for k in ("id", "level")}, **sc,
                                "mode": m, "method": name, "round": rd,
                                "elapsed": r["elapsed"], "n_chunks": r["n_chunks"],
                                "n_hops": len(r["hops"]), "hops": r["hops"],
                                "q": it["q"], "gold": it["gold"], "answer": r["answer"]})
                mark = "✅" if sc["passed"] else "❌"
                pre = f"    รอบ {rd}: " if args.repeat > 1 else "    "
                print(f"{pre}{mark} {sc['ratio']*100:.0f}%  {r['elapsed']}s  "
                      f"{len(r['hops'])} รอบค้น  {r['n_chunks']} ก้อน"
                      + (f"  [{r['stop_reason']}]" if r.get("stop_reason") else "")
                      + ("" if sc["passed"] else f"  ขาด: {', '.join(sc['miss'])}"))

    # ── ตารางสรุป: เน้น "ความนิ่ง" เพราะนั่นคือสิ่งที่การรันซ้ำต้องการวัด ──
    import statistics as st
    print("\n" + "═" * 96)
    print(f"{'วิธี':<22}{'ข้อ':<7}{'ผ่าน':<8}{'เวลา (ต่ำ-สูง)':<20}{'ก้อน (ต่ำ-สูง)':<18}{'นิ่ง?'}")
    print("─" * 96)
    for m in modes:
        for iid in dict.fromkeys(r["id"] for r in results):
            rs = [r for r in results if r["mode"] == m and r["id"] == iid]
            if not rs:
                continue
            n = len(rs)
            times = [r["elapsed"] for r in rs]
            chks = [r["n_chunks"] for r in rs]
            passes = [r["passed"] for r in rs]
            # นิ่ง = ทุกรอบผ่าน/ตกเหมือนกัน และใช้ก้อนเท่ากันทุกรอบ
            steady = len(set(passes)) == 1 and len(set(chks)) == 1
            trange = (f"{min(times):.0f}s" if len(set(times)) == 1
                      else f"{min(times):.0f}-{max(times):.0f}s")
            crange = (f"{chks[0]}" if len(set(chks)) == 1
                      else f"{min(chks)}-{max(chks)}")
            print(f"{MODES[m][0]:<22}{iid:<7}{sum(passes)}/{n:<6}"
                  f"{trange:<20}{crange:<18}"
                  f"{'✅ นิ่ง' if steady else '❌ แกว่ง'}")
    print("═" * 96)
    for m in modes:
        rs = [r for r in results if r["mode"] == m]
        chks = [r["n_chunks"] for r in rs]
        sd = st.stdev(chks) if len(chks) > 1 else 0.0
        print(f"  {MODES[m][0]:<22} รวม {sum(r['passed'] for r in rs)}/{len(rs)}  "
              f"เวลาเฉลี่ย {sum(r['elapsed'] for r in rs)/len(rs):.1f}s  "
              f"ก้อนเฉลี่ย {sum(chks)/len(chks):.1f} (ส่วนเบี่ยงเบน ±{sd:.1f})")

    out = os.path.join(HERE, f"hop_compare_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"modes": {m: MODES[m][0] for m in modes}, "repeat": args.repeat,
               "results": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nผลดิบ: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
