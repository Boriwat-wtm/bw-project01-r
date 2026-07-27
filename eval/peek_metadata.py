# -*- coding: utf-8 -*-
"""ดู metadata ที่เก็บใน Chroma — เครื่องมือส่องดูเฉย ๆ ไม่แก้อะไร

    python eval/peek_metadata.py                 # สรุปภาพรวม + ตัวอย่าง 1 ก้อน
    python eval/peek_metadata.py 9               # ทุกก้อนที่มี "มาตรา ๙"
    python eval/peek_metadata.py 61 --in-force   # เฉพาะฉบับใช้บังคับปัจจุบัน
    python eval/peek_metadata.py --id LandCode2497_Update-vlast_TruePDF.pdf::0050
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass
import rag        # noqa: E402


def show(meta: dict, text: str = "") -> None:
    print(json.dumps(meta, ensure_ascii=False, indent=2))
    if text:
        print(f"   ── ข้อความ (100 ตัวแรก): {text[:100].strip()}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("article", nargs="?", help="เลขมาตรา เช่น 9 หรือ 9/1 (เว้นว่าง = สรุปภาพรวม)")
    ap.add_argument("--id", help="ดู chunk เดียวตาม id ตรง ๆ")
    ap.add_argument("--in-force", action="store_true", help="เฉพาะฉบับใช้บังคับปัจจุบัน")
    ap.add_argument("--fields", action="store_true", help="แสดงรายชื่อ 18 ฟิลด์แล้วออก")
    args = ap.parse_args()

    if args.fields:
        keys = list(rag._chroma_metadata({}).keys())
        print(f"metadata เก็บ {len(keys)} ฟิลด์ (นิยามที่ rag.py _chroma_metadata):")
        for i, k in enumerate(keys, 1):
            print(f"  {i:>2}. {k}")
        return

    rag._init_chroma()
    rag._ensure_loaded()
    print(f"ดัชนี: {rag._collection.count():,} chunk  ·  ที่เก็บ: {rag.CHROMA_DIR}\n")

    if args.id:
        got = rag._collection.get(ids=[args.id], include=["metadatas", "documents"])
        if not got["ids"]:
            print("ไม่พบ id นี้")
            return
        show(got["metadatas"][0], got["documents"][0])
        return

    if not args.article:                       # ไม่ระบุมาตรา = สรุปภาพรวม + 1 ตัวอย่าง
        from collections import Counter
        groups = Counter(c.get("group", "") for c in rag._chunks)
        print("แบ่งตามกลุ่มเอกสาร:")
        for g, n in groups.most_common():
            print(f"  {n:>4}  {g}")
        print("\nตัวอย่าง metadata 1 ก้อน (ฉบับใช้บังคับปัจจุบัน):")
        sample = next(c for c in rag._chunks if c.get("in_force"))
        got = rag._collection.get(ids=[sample["id"]], include=["metadatas", "documents"])
        show(got["metadatas"][0], got["documents"][0])
        return

    # ระบุมาตรา = หาทุกก้อนที่มีมาตรานั้น (ดูจาก article_nums ที่เก็บทุกมาตรา)
    want = f"|{args.article}|"
    hits = [c for c in rag._chunks if want in c.get("article_nums", "")]
    if args.in_force:
        hits = [c for c in hits if c.get("in_force")]
    print(f"เจอ {len(hits)} ก้อนที่มีมาตรา {args.article}"
          + (" (เฉพาะฉบับใช้บังคับ)" if args.in_force else "") + ":\n")
    ids = [c["id"] for c in hits]
    got = rag._collection.get(ids=ids, include=["metadatas"]) if ids else {"ids": [], "metadatas": []}
    for cid, meta in zip(got["ids"], got["metadatas"]):
        print(f"# {cid}")
        show(meta)


if __name__ == "__main__":
    main()
