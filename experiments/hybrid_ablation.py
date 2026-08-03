# -*- coding: utf-8 -*-
"""วัดเองว่า "ค้นสองสายรวมกัน" ดีกว่าใช้สายเดียวจริงไหม — บนข้อมูลของเรา ไม่ใช่เชื่อตามที่อ่านมา

ทำสำเนาลอจิกรวมอันดับของ rag.retrieve มาไว้ที่นี่ เพื่อ "ปิดสายใดสายหนึ่ง" ได้
โดยไม่ต้องแตะโค้ดจริง — rag.retrieve ไม่มีสวิตช์แบบนี้ และไม่ควรมี เพราะเป็นของสำหรับ
ทดลองล้วน ๆ (แลกมาด้วยการที่ถ้า retrieve เปลี่ยนสูตร ต้องมาแก้ที่นี่ด้วย)

⚠️ ใช้ "คำถามดิบ" ไม่ผ่าน expand_queries จึงรันซ้ำได้ผลเดิม แต่ก็แปลว่า
   **ไม่ตรงกับระบบจริง** ซึ่งแตกคำถามเป็น 4 มุม = รวม 8 รายการอันดับ ไม่ใช่ 2
   กลไกการเบียดกันใน RRF ต่างไปมาก อย่าเอาผลจากที่นี่ไปสรุปแทนระบบจริง

ผลที่วัดได้ (31 ก.ค. 2569 · ground_truth.json 30 ข้อ · เปิด rerank):
    semantic  22/30      BM25  24/30      hybrid  21/30
มี 5 ข้อที่สายใดสายหนึ่งหาเจอ แต่พอรวมกันแล้วหาไม่เจอ — สูตร RRF ให้ "ปานกลาง
ทั้งสองสาย" ชนะ "ยอดเยี่ยมสายเดียว" (1/(60+9)*2 > 1/(60+0)) จึงเบียดของถูกตกได้

    python experiments/hybrid_ablation.py
    ABL_RERANK=0 python experiments/hybrid_ablation.py   # ปิด rerank ดูเฉพาะ RRF
    ABL_FILE=ground_truth_v3.json python experiments/hybrid_ablation.py
"""
import json
import os
import sys

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

rag._ensure_loaded()
POOL, K = 50, rag.TOP_K
RERANK = os.environ.get("ABL_RERANK", "1") == "1"


def retrieve_mode(q: str, mode: str, groups: list) -> list:
    """mode: sem | bm25 | hybrid -> คืน chunk top-K"""
    gset = set(groups)
    where = {"group": {"$in": list(gset)}}
    allowed = {i for i, c in enumerate(rag._chunks) if c.get("group") in gset}
    fuse: dict = {}
    if mode in ("sem", "hybrid"):
        qv = rag._embeddings.embed_query(q)
        res = rag._collection.query(query_embeddings=[qv], n_results=POOL, where=where)
        for r, cid in enumerate((res["ids"][0] if res.get("ids") else [])[:POOL]):
            fuse[cid] = fuse.get(cid, 0.0) + 1.0 / (60 + r)
    if mode in ("bm25", "hybrid"):
        sc = rag._bm25.get_scores(rag.tokenize(q))
        order = sorted(range(len(sc)), key=lambda i: -sc[i])
        ids = [rag._chunks[i]["id"] for i in order if i in allowed][:POOL]
        for r, cid in enumerate(ids):
            fuse[cid] = fuse.get(cid, 0.0) + 1.0 / (60 + r)
    ranked = [rag._chunk_by_id[c] for c in sorted(fuse, key=lambda c: -fuse[c])
              if c in rag._chunk_by_id]
    ranked = rag._dedupe_versions(ranked)
    ranked = rag._demote_scans(ranked)
    if RERANK:
        ranked = rag.rerank(q, ranked[:POOL], K)
    return ranked[:K]


def main() -> None:
    path = os.path.join(ROOT, "eval", os.environ.get("ABL_FILE", "ground_truth.json"))
    d = json.load(open(path, encoding="utf-8"))
    items = d["items"] if isinstance(d, dict) else d
    allg = service.list_groups()

    print("rerank:", "on" if RERANK else "off", "| pool", POOL, "-> top", K)
    print()
    res = {}
    for mode in ("sem", "bm25", "hybrid"):
        hit, fails = 0, []
        for it in items:
            g, _y, _v = service.pick_groups(it["q"], allg)
            chunks = retrieve_mode(it["q"], mode, g or [rag.GROUP_IN_FORCE])
            text = "\n".join(c["text"] for c in chunks)
            if run_eval.score_item(it, text)["passed"]:
                hit += 1
            else:
                fails.append(it["id"])
        res[mode] = (hit, fails)
        print(f"{mode:8s}  {hit}/{len(items)}   ตก: {', '.join(fails)}")

    sem_f = set(res["sem"][1])
    bm_f = set(res["bm25"][1])
    hy_f = set(res["hybrid"][1])
    print()
    print("semantic ตก แต่ BM25 ได้ :", sorted(sem_f - bm_f) or "-")
    print("BM25 ตก แต่ semantic ได้ :", sorted(bm_f - sem_f) or "-")
    print("ตกทั้งคู่                 :", sorted(sem_f & bm_f) or "-")
    print("hybrid ตก ทั้งที่มีสายเดียวได้:", sorted(hy_f - (sem_f & bm_f)) or "-")


if __name__ == "__main__":
    main()
