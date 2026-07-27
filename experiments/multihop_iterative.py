# -*- coding: utf-8 -*-
"""ต้นแบบ iterative multi-hop retrieval — ใช้ "ทดลองเทียบ" กับระบบปัจจุบันเท่านั้น

ระบบปัจจุบัน (service.answer_stream):
    คำถาม -> แตก 4 มุม (พร้อมกัน) -> ค้นรอบเดียว -> rerank -> ตอบ
    hop ที่ทำได้ = hop ที่คำนวณล่วงหน้าไว้ในกราฟการแก้ไข (article_chain/timeline)

ต้นแบบนี้ (iterative):
    คำถาม -> ค้นรอบ 1 -> ให้ LLM อ่านผลแล้วตัดสินใจว่า "ยังขาดอะไร ต้องค้นอะไรต่อ"
          -> ค้นรอบ 2 (ด้วยคำค้นที่ LLM คิดเอง) -> รวม context ทุกรอบ -> ตอบ

⚠️ ไม่ได้ต่อเข้าแอปจริง และตั้งใจไม่ต่อ จนกว่าตัวเลขจะบอกว่าคุ้ม
   เพราะการให้ LLM ตัดสินใจว่าจะค้นอะไรต่อ = คืนอำนาจตัดสินใจกลับไปให้โมเดล
   ซึ่งสวนทางกับหลักการของโปรเจกต์ ("ข้อเท็จจริงให้โค้ดคำนวณ") และทำให้ผลไม่นิ่ง

    python experiments/multihop_iterative.py --id H2H1        # ทดลองข้อเดียว
    python experiments/multihop_iterative.py --all --hops 2   # ทั้งชุด
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="multihop.json")
    ap.add_argument("--id", nargs="*", help="ทำเฉพาะข้อที่ระบุ")
    ap.add_argument("--all", action="store_true", help="ทำทั้งชุด")
    ap.add_argument("--hops", type=int, default=2, help="ค้นสูงสุดกี่รอบ (default 2)")
    args = ap.parse_args()

    gt = json.load(open(os.path.join(EVAL_DIR, args.file), encoding="utf-8"))
    items = gt["items"]
    if args.id:
        want = {x.upper() for x in args.id}
        items = [i for i in items if i["id"] in want]
    elif not args.all:
        items = items[:1]

    rag._ensure_loaded()
    llm = rag.build_llm()
    groups = service.list_groups()
    print(f"\n=== iterative multi-hop (สูงสุด {args.hops} รอบ) — {len(items)} ข้อ ===\n")

    results = []
    for it in items:
        print(f"[{it['id']}] {it['q'][:70]}")
        r = answer_iterative(llm, it["q"], groups, max_hops=args.hops)
        sc = score(it, r["answer"])
        results.append({**{k: it[k] for k in ("id", "level")}, **sc,
                        "elapsed": r["elapsed"], "n_chunks": r["n_chunks"],
                        "n_hops": len(r["hops"]), "hops": r["hops"],
                        "q": it["q"], "gold": it["gold"], "answer": r["answer"]})
        mark = "✅" if sc["passed"] else "❌"
        print(f"    {mark} {sc['ratio']*100:.0f}%  {r['elapsed']}s  "
              f"{len(r['hops'])} รอบ  {r['n_chunks']} ก้อน"
              + ("" if sc["passed"] else f"  ขาด: {', '.join(sc['miss'])}"))
        print(f"    ตอบ: {r['answer'][:150]}".replace("\n", " ") + "\n")

    n = len(results)
    npass = sum(r["passed"] for r in results)
    print("─" * 70)
    print(f"iterative multi-hop: ผ่าน {npass}/{n}  "
          f"เฉลี่ย {sum(r['elapsed'] for r in results)/n:.1f}s/ข้อ  "
          f"เฉลี่ย {sum(r['n_chunks'] for r in results)/n:.0f} ก้อน/ข้อ")

    out = os.path.join(HERE, f"multihop_iterative_{time.strftime('%Y%m%d_%H%M%S')}.json")
    json.dump({"mode": f"iterative (max {args.hops} hops)", "passed": npass,
               "total": n, "results": results},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"ผลดิบ: {os.path.relpath(out)}")


if __name__ == "__main__":
    main()
