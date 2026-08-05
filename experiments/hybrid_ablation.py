# -*- coding: utf-8 -*-
"""วัดเองว่า "ค้นสองสายรวมกัน" ดีกว่าใช้สายเดียวจริงไหม — บนข้อมูลของเรา ไม่ใช่เชื่อตามที่อ่านมา

    python experiments/hybrid_ablation.py
    ABL_RERANK=0 python experiments/hybrid_ablation.py    # ปิด rerank ดูเฉพาะชั้น RRF
    ABL_RAW=1    python experiments/hybrid_ablation.py    # ใช้คำถามดิบ (แบบเดิม ไว้เทียบ)
    ABL_FILE=ground_truth_v3.json python experiments/hybrid_ablation.py

── ทำไมต้องเขียนใหม่ (5 ส.ค. 2569) ───────────────────────────────────────────
รุ่นแรกวัดด้วย "คำถามดิบ" และทำสำเนาลอจิกมาไม่ครบ ผลที่ได้ (semantic 22 / BM25 24 /
hybrid 21) จึงเอาไปสรุปแทนระบบจริงไม่ได้เลย ต่างกัน 3 จุดใหญ่:

    1. ระบบจริงแตกคำถามเป็น 4 มุมก่อนค้น = รวม 8 ลิสต์อันดับ ไม่ใช่ 2
       กลไกการเบียดกันใน RRF ต่างไปคนละเรื่อง
    2. สำเนาเดิม "ขาดขั้นดันอันดับ" ทั้งสองตัว (_boost_amend, _boost_exact_article)
       ซึ่งเป็นกลไกหลักที่ทำให้ถามเจาะมาตราแล้วเจอ
    3. สำเนาเดิมส่ง "คำถามที่ใช้ค้น" ให้ reranker แต่ระบบจริงส่ง "คำถามของผู้ใช้"

รุ่นนี้ทำสำเนาให้ตรงกับ rag.retrieve ทุกขั้น แล้วมี verify_copy() ตรวจให้ด้วยว่า
โหมด hybrid ของสำเนา = ผลของ rag.retrieve จริง ๆ — ถ้าวันหลังใครแก้ retrieve
แล้วลืมแก้ที่นี่ ตัวตรวจจะฟ้องทันที ไม่ปล่อยให้วัดผิดเงียบ ๆ

เหตุผลที่ยังต้องทำสำเนาแทนการเรียก rag.retrieve ตรง ๆ: retrieve ไม่มีสวิตช์
"ปิดสายใดสายหนึ่ง" และไม่ควรมี เพราะเป็นความซับซ้อนที่มีไว้ทดลองล้วน ๆ

⚠️ expand_queries เรียก LLM จึงไม่นิ่ง 100% — สคริปต์นี้ขยายคำถาม "ครั้งเดียว"
   แล้วใช้ชุดเดียวกันกับทั้ง 3 โหมด เพื่อให้เทียบกันอย่างยุติธรรมภายในรอบเดียว
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "eval"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import rag
import run_eval
import service

RERANK = os.environ.get("ABL_RERANK", "1") == "1"
RAW = os.environ.get("ABL_RAW") == "1"      # 1 = คำถามดิบ (แบบรุ่นแรก ไว้เทียบ)
K = rag.TOP_K


def retrieve_mode(queries: list, rerank_query: str, mode: str,
                  groups: list, years=None, versions=None) -> list:
    """สำเนาของ rag.retrieve ที่ปิดสายใดสายหนึ่งได้ — mode: sem | bm25 | hybrid

    ทุกขั้นต้องตรงกับ rag.retrieve (ดู verify_copy) ยกเว้นการเลือกว่าจะเปิดสายไหน
    """
    pool = K * 4
    conds, masks = [], []
    if not groups:
        groups = sorted(rag.DEFAULT_SEARCH_GROUPS)
    gset = set(groups)
    conds.append({"group": {"$in": list(gset)}})
    masks.append({i for i, c in enumerate(rag._chunks) if c.get("group") in gset})
    if years:
        yset = {int(y) for y in years}
        conds.append({"year": {"$in": list(yset)}})
        masks.append({i for i, c in enumerate(rag._chunks)
                      if int(c.get("year", 0) or 0) in yset})
    if versions:
        vset = {int(v) for v in versions}
        conds.append({"version": {"$in": list(vset)}})
        masks.append({i for i, c in enumerate(rag._chunks)
                      if int(c.get("version", -1)) in vset})

    where, allowed = None, set.intersection(*masks) if masks else set()
    if allowed:
        where = conds[0] if len(conds) == 1 else {"$and": conds}
    else:
        allowed = None
    n_res = max(1, min(pool, rag._collection.count()))

    fuse: dict = {}
    for q in queries:
        if mode in ("sem", "hybrid"):
            qv = rag._embeddings.embed_query(q)
            res = rag._collection.query(query_embeddings=[qv], n_results=n_res, where=where)
            for r, cid in enumerate((res["ids"][0] if res.get("ids") else [])[:pool]):
                fuse[cid] = fuse.get(cid, 0.0) + 1.0 / (60 + r)
        if mode in ("bm25", "hybrid"):
            sc = rag._bm25.get_scores(rag.tokenize(q))
            order = sorted(range(len(sc)), key=lambda i: -sc[i])
            ids = [rag._chunks[i]["id"] for i in order
                   if allowed is None or i in allowed][:pool]
            for r, cid in enumerate(ids):
                fuse[cid] = fuse.get(cid, 0.0) + 1.0 / (60 + r)

    ranked = [rag._chunk_by_id[c] for c in sorted(fuse, key=lambda c: -fuse[c])
              if c in rag._chunk_by_id]
    if rag.DEDUPE_VERSIONS:
        ranked = rag._dedupe_versions(ranked)
    ranked = rag._demote_scans(ranked)
    wanted = rag.question_articles(rerank_query)
    if RERANK and rag.RERANK_ENABLED:
        n = max(rag.RERANK_TOP_N, K)
        ranked = rag.rerank(rerank_query, ranked[:n], n)
    ranked = rag._demote_scans(ranked)
    ranked = rag._boost_amend(ranked, rag.question_amendments(rerank_query))
    return rag._boost_exact_article(ranked, wanted)[:K]


def verify_copy(items: list, allg: list) -> bool:
    """โหมด hybrid ของสำเนา ต้องให้ผลเหมือน rag.retrieve จริง — ไม่งั้นวัดผิดตั้งแต่ต้น"""
    print("ตรวจว่าสำเนาตรงกับ rag.retrieve จริงไหม...", end=" ", flush=True)
    bad = []
    for it in items[:5]:
        q = it["q"]
        g, y, v = service.pick_groups(q, allg)
        mine = retrieve_mode([q], q, "hybrid", g, y or None, v or None)
        real = rag.retrieve(q, k=K, rerank_query=q, groups=g or None,
                            years=y or None, versions=v or None)
        if [c["id"] for c in mine] != [c["id"] for c in real]:
            bad.append(it["id"])
    if bad:
        print(f"❌ ไม่ตรง {len(bad)} ข้อ: {bad}")
        print("   -> rag.retrieve เปลี่ยนสูตรไปแล้ว ต้องมาแก้ retrieve_mode ให้ตรงก่อน")
        print("      ผลที่วัดจากสำเนาที่ไม่ตรง เอาไปสรุปแทนระบบจริงไม่ได้")
        return False
    print("✅ ตรงทุกข้อที่สุ่มตรวจ")
    return True


def main() -> None:
    path = os.path.join(ROOT, "eval", os.environ.get("ABL_FILE", "ground_truth.json"))
    d = json.load(open(path, encoding="utf-8"))
    items = d["items"] if isinstance(d, dict) else d
    allg = service.list_groups()

    rag._ensure_loaded()
    print(f"ไฟล์: {os.path.basename(path)} ({len(items)} ข้อ) | "
          f"rerank: {'on' if RERANK and rag.RERANK_ENABLED else 'off'} | "
          f"คำถาม: {'ดิบ' if RAW else 'ขยาย 4 มุม (ตรงกับระบบจริง)'}")
    if not verify_copy(items, allg):
        sys.exit(1)

    # ขยายคำถามครั้งเดียว ใช้ชุดเดียวกันทั้ง 3 โหมด = เทียบกันยุติธรรม
    plan = []
    if RAW:
        for it in items:
            g, y, v = service.pick_groups(it["q"], allg)
            plan.append((it, [it["q"]], it["q"], g, y, v))
    else:
        llm = rag.build_llm()
        print(f"ขยายคำถาม {len(items)} ข้อ...", end=" ", flush=True)
        t0 = time.time()
        for it in items:
            g, y, v = service.pick_groups(it["q"], allg)
            qs = rag.expand_queries(llm, it["q"], domain="thai_law")
            plan.append((it, qs, it["q"], g, y, v))
        print(f"เสร็จ ({time.time()-t0:.0f}s · เฉลี่ย "
              f"{sum(len(p[1]) for p in plan)/len(plan):.1f} มุม/ข้อ)")
    print()

    res = {}
    for mode in ("sem", "bm25", "hybrid"):
        hit, fails = 0, []
        for it, qs, rq, g, y, v in plan:
            chunks = retrieve_mode(qs, rq, mode, g, y or None, v or None)
            text = "\n".join(c["text"] for c in chunks)
            if run_eval.score_item(it, text)["passed"]:
                hit += 1
            else:
                fails.append(it["id"])
        res[mode] = (hit, fails)
        print(f"{mode:8s}  {hit}/{len(items)}   ตก: {', '.join(fails) or '-'}")

    sem_f, bm_f, hy_f = (set(res[m][1]) for m in ("sem", "bm25", "hybrid"))
    print()
    print("semantic ตก แต่ BM25 ได้      :", sorted(sem_f - bm_f) or "-")
    print("BM25 ตก แต่ semantic ได้      :", sorted(bm_f - sem_f) or "-")
    print("ตกทั้งคู่                     :", sorted(sem_f & bm_f) or "-")
    print("hybrid ตก ทั้งที่มีสายเดียวได้ :", sorted(hy_f - (sem_f & bm_f)) or "-")
    print("hybrid ได้ ทั้งที่ตกทั้งคู่     :", sorted((sem_f & bm_f) - hy_f) or "-")

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = os.path.join(HERE, f"ablation_{stamp}.json")
    json.dump({"file": os.path.basename(path), "rerank": RERANK, "raw": RAW,
               "results": {m: {"passed": res[m][0], "failed": res[m][1]}
                           for m in res},
               "expanded": {it["id"]: qs for it, qs, *_ in plan}},
              open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nผลดิบ: {os.path.basename(out)}")


if __name__ == "__main__":
    main()
