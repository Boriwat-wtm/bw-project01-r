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
        # คำค้นรอบสอง: เอา entity ที่เพิ่งรู้ ผสมกับเจตนาของคำถามเดิม
        qs2 = [f"{e} {question}" for e in ents] + [f"อำนาจหน้าที่ของ{e}" for e in ents[:1]]
        if verbose:
            print(f"    โค้ดสกัดได้: {ents} -> ค้นต่อ {len(qs2)} คำ")
        more = rag.retrieve(qs2, rerank_query=f"{' '.join(ents)} {question}", groups=g,
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

        if hop >= max_hops:
            break
        queries = plan_followup(llm, question, rag.format_context(all_chunks))
        if not queries:
            if verbose:
                print("    LLM บอกว่าพอแล้ว -> หยุด")
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
            "elapsed": round(time.perf_counter() - t0, 1)}


# ── ให้คะแนนด้วยไม้บรรทัดเดียวกับ run_eval.py (เทียบกันได้ตรง ๆ) ──────────────
def score(item: dict, answer: str) -> dict:
    sys.path.insert(0, EVAL_DIR)
    import run_eval           # noqa: E402  (ใช้ norm/score_item ตัวเดียวกัน)
    return run_eval.score_item(item, answer)


# ── ทะเบียนวิธี — ชื่อในนี้คือชื่อ trace ที่จะเห็นใน MLflow ────────────────────
MODES = {
    "A": ("baseline_multiquery_graph", "ระบบปัจจุบัน: แตก 4 มุม ค้นรอบเดียว + กราฟ",
          lambda llm, q, g: answer_baseline(llm, q, g)),
    "B": ("iterative_multihop_llm", "LLM คิดคำค้นรอบสองเอง (2 รอบ)",
          lambda llm, q, g: answer_iterative(llm, q, g, max_hops=2)),
    "C": ("entityhop_code", "โค้ดสกัด entity แล้วค้นรอบสองอัตโนมัติ",
          lambda llm, q, g: answer_entityhop(llm, q, g)),
}


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
    args = ap.parse_args()

    gt = json.load(open(os.path.join(EVAL_DIR, args.file), encoding="utf-8"))
    items = gt["items"]
    if args.id:
        want = {x.upper() for x in args.id}
        items = [i for i in items if i["id"] in want]
    elif not args.all:
        items = items[:1]
    modes = args.mode or list(MODES)

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
            r = run_traced(m, llm, it["q"], groups)
            sc = score(it, r["answer"])
            results.append({**{k: it[k] for k in ("id", "level")}, **sc,
                            "mode": m, "method": name, "elapsed": r["elapsed"],
                            "n_chunks": r["n_chunks"], "n_hops": len(r["hops"]),
                            "hops": r["hops"], "q": it["q"], "gold": it["gold"],
                            "answer": r["answer"]})
            mark = "✅" if sc["passed"] else "❌"
            print(f"    {mark} {sc['ratio']*100:.0f}%  {r['elapsed']}s  "
                  f"{len(r['hops'])} รอบ  {r['n_chunks']} ก้อน"
                  + ("" if sc["passed"] else f"  ขาด: {', '.join(sc['miss'])}"))

    print("\n" + "─" * 74)
    print(f"{'วิธี':<28} {'ผ่าน':<8} {'เวลาเฉลี่ย':<12} {'ก้อนเฉลี่ย'}")
    for m in modes:
        rs = [r for r in results if r["mode"] == m]
        n = len(rs)
        print(f"{MODES[m][0]:<28} {sum(r['passed'] for r in rs)}/{n:<6} "
              f"{sum(r['elapsed'] for r in rs)/n:>6.1f}s      "
              f"{sum(r['n_chunks'] for r in rs)/n:>5.0f}")

    out = os.path.join(HERE, f"hop_compare_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"modes": {m: MODES[m][0] for m in modes}, "results": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nผลดิบ: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
