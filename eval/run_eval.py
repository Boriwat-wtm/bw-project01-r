"""
run_eval.py — รันชุดวัดผล 30 ข้อแล้วให้คะแนนอัตโนมัติ

    python eval/run_eval.py                 # รันทั้ง 30 ข้อ
    python eval/run_eval.py --level easy    # เฉพาะระดับ
    python eval/run_eval.py --id E1 H4      # เฉพาะข้อที่ระบุ
    python eval/run_eval.py --retrieval     # วัดแค่ retrieval ไม่เรียก LLM (เร็ว/ฟรี)

วิธีให้คะแนน — key-fact matching:
  เฉลยแต่ละข้อถอดเป็น "ข้อเท็จจริงที่ต้องมี" (must) = ตัวเลข ปี พ.ศ. เลขมาตรา วลีชี้ขาด
  แล้วเช็คว่าคำตอบมีครบไหม โดย normalize เลขไทย->อารบิก + ตัดช่องว่าง/จุลภาคก่อนเทียบ
  ผ่าน = มี must ครบทุกตัว และไม่มี must_not เลย

  ⚠️ วัด "ข้อเท็จจริงถูกไหม" ไม่ได้วัด "อธิบายดีไหม" — ข้อที่ตกควรอ่านคำตอบจริงประกอบ
  ผลดิบถูกเก็บลง eval/results_<timestamp>.json ทุกครั้งเพื่อเทียบข้ามรอบได้
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

import rag        # noqa: E402
import service    # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def norm(s: str) -> str:
    """ตัดสิ่งที่ไม่ควรทำให้เทียบพลาด: เลขไทย, ช่องว่าง, จุลภาค, markdown bold
    '๒๐,๐๐๐ บาท' และ '20000บาท' ต้องเทียบติดกัน"""
    s = (s or "").translate(rag.THAI_DIGITS)
    return re.sub(r"[\s,*_`​]+", "", s)


def has_fact(fact, answer_norm: str) -> bool:
    """fact เป็น str = ต้องมีตรง ๆ | เป็น list = มีตัวใดตัวหนึ่งก็พอ

    ⚠️ ต้องรองรับหลายรูปแบบ เพราะกฎหมายไทยเขียนตัวเลขเป็น "ตัวหนังสือ"
    ตัวบทเขียน "เมื่อพ้นกำหนดหนึ่งร้อยแปดสิบวัน" แต่คนถาม/LLM ตอบว่า "180 วัน"
    ถ้าเทียบแค่รูปเดียวจะตัดสินว่าผิดทั้งที่ถูก"""
    alts = fact if isinstance(fact, list) else [fact]
    return any(norm(a) in answer_norm for a in alts)


def fact_label(fact) -> str:
    return fact[0] if isinstance(fact, list) else fact


def score_item(item: dict, answer: str) -> dict:
    """ให้คะแนน 1 ข้อ -> {passed, hit, miss, bad, ratio}"""
    a = norm(answer)
    must = item.get("must", [])
    hit = [fact_label(k) for k in must if has_fact(k, a)]
    miss = [fact_label(k) for k in must if not has_fact(k, a)]
    bad = [fact_label(k) for k in item.get("must_not", []) if has_fact(k, a)]
    return {"passed": not miss and not bad, "hit": hit, "miss": miss, "bad": bad,
            "ratio": len(hit) / len(must) if must else 1.0}


def run_one(llm, item: dict, groups: list, retrieval_only: bool) -> dict:
    """รัน 1 ข้อ -> ผลพร้อมคะแนน"""
    t0 = time.perf_counter()
    q = item["q"]
    if retrieval_only:
        # ไม่เรียก LLM ตอบ — เอาตัวบทที่ค้นได้มาต่อกันแล้ววัดว่ามีข้อเท็จจริงที่ต้องการไหม
        # (แยกให้ชัดว่า "หาไม่เจอ" กับ "หาเจอแต่ตอบพลาด" คนละปัญหากัน)
        g, y, v = service.pick_groups(q, groups)
        chunks = rag.retrieve(q, k=rag.TOP_K, rerank_query=q, groups=g,
                              years=y or None, versions=v or None)
        answer = "\n".join(c["text"] for c in chunks)
        used, n_src = g, len(chunks)
    else:
        answer, used, n_src = "", [], 0
        for ev in service.answer_stream(llm, q, all_groups=groups, stream=False):
            if "final" in ev:
                answer = ev["final"]["answer"]
                used = ev["final"]["groups_used"]
                n_src = len(ev["final"]["chunks"])
    sc = score_item(item, answer)
    return {**{k: item[k] for k in ("id", "level", "trap")}, **sc,
            "q": q, "gold": item["gold"], "answer": answer,
            "groups": used, "n_sources": n_src,
            "elapsed": round(time.perf_counter() - t0, 1)}


def git_rev() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def log_mlflow(results: list, mode: str, out_path: str, run_idx: int, tag: str) -> str:
    """ส่งผลขึ้น MLflow — อ่าน MLFLOW_TRACKING_URI จาก env
    ไม่ตั้ง = เก็บลง mlruns/ ในเครื่อง (ตั้งทีหลังแล้วส่งขึ้น server ได้โดยไม่ต้องแก้โค้ด)"""
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    # แยก experiment ของโปรเจกต์นี้ไว้ต่างหาก — เปลี่ยนได้ด้วย env MLFLOW_EXPERIMENT
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "thai-law-rag-eval"))

    n = len(results)
    lv = lambda x: [r for r in results if r["level"] == x]          # noqa: E731
    with mlflow.start_run(run_name=f"{tag}-r{run_idx}") as run:
        mlflow.log_params({
            "mode": mode, "n_items": n, "git_commit": git_rev(),
            "llm_model": rag.LLM_MODEL, "embed_model": rag.EMBED_MODEL,
            "rerank_model": rag.RERANK_MODEL, "rerank_enabled": rag.RERANK_ENABLED,
            "top_k": rag.TOP_K, "rerank_top_n": rag.RERANK_TOP_N,
            "chunk_size": rag.CHUNK_SIZE, "chunk_overlap": rag.CHUNK_OVERLAP,
            "dedupe": rag.DEDUPE_VERSIONS, "art_boost": rag.BOOST_EXACT_ARTICLE,
            "scan_demote": rag.DEMOTE_SCANS, "law_same_ratio": rag.LAW_SAME_RATIO,
        })
        mlflow.log_metrics({
            "passed": sum(r["passed"] for r in results),
            "pass_rate": sum(r["passed"] for r in results) / n,
            "fact_ratio": sum(r["ratio"] for r in results) / n,
            "elapsed_total_s": sum(r["elapsed"] for r in results),
            "elapsed_avg_s": sum(r["elapsed"] for r in results) / n,
            **{f"pass_{k}": sum(r["passed"] for r in lv(k)) for k in
               ("easy", "medium", "hard", "veryhard") if lv(k)},
        })
        # ผ่าน/ตก รายข้อ เก็บเป็น metric เพื่อ plot เทียบข้ามรอบได้ว่าข้อไหนแกว่ง
        for r in results:
            mlflow.log_metric(f"item_{r['id']}", float(r["passed"]))
        mlflow.set_tag("failed_ids", ",".join(r["id"] for r in results if not r["passed"]))
        mlflow.log_artifact(out_path)
        return run.info.run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["easy", "medium", "hard", "veryhard"])
    ap.add_argument("--id", nargs="*")
    ap.add_argument("--retrieval", action="store_true",
                    help="วัดเฉพาะ retrieval ไม่เรียก LLM")
    ap.add_argument("--repeat", type=int, default=1,
                    help="รันซ้ำกี่รอบ — LLM ให้ผลไม่คงที่ 100% ควรรัน 3 รอบแล้วดูค่าเฉลี่ย")
    ap.add_argument("--mlflow", action="store_true", help="ส่งผลขึ้น MLflow")
    ap.add_argument("--tag", default="eval", help="ชื่อกำกับ run ใน MLflow")
    # ชุดคำถามอื่นที่ใช้ schema เดียวกัน เช่น adversarial.json (ชุดคำถามหลอก —
    # วัดว่าระบบ "รู้ตัวว่าไม่รู้" มั้ย ต้องปฏิเสธเมื่อคำตอบไม่อยู่ในเอกสาร)
    ap.add_argument("--file", default="ground_truth.json",
                    help="ไฟล์ชุดคำถาม (ใน eval/) — default: ground_truth.json")
    args = ap.parse_args()

    gt = json.load(open(os.path.join(HERE, args.file), encoding="utf-8"))
    items = gt["items"]
    if args.level:
        items = [i for i in items if i["level"] == args.level]
    if args.id:
        want = {x.upper() for x in args.id}
        items = [i for i in items if i["id"] in want]

    rag._ensure_loaded()
    groups = service.list_groups()
    llm = None if args.retrieval else rag.build_llm()
    mode = "retrieval เท่านั้น" if args.retrieval else f"เต็มระบบ ({rag.LLM_MODEL})"
    print(f"\nรัน {len(items)} ข้อ — โหมด: {mode}\n" + "─" * 78)

    rounds = []
    for rd in range(1, args.repeat + 1):
        if args.repeat > 1:
            print(f"\n### รอบที่ {rd}/{args.repeat} " + "#" * 50)
        results = []
        for it in items:
            r = run_one(llm, it, groups, args.retrieval)
            results.append(r)
            mark = "✅" if r["passed"] else "❌"
            detail = "" if r["passed"] else f"  ขาด: {', '.join(r['miss'])}" + \
                (f"  ห้ามมี: {', '.join(r['bad'])}" if r["bad"] else "")
            print(f"{mark} {r['id']:<4} {r['ratio']*100:>3.0f}%  {r['elapsed']:>5.1f}s  "
                  f"{r['q'][:46]}{detail}")

        print("─" * 78)
        n = len(results)
        npass = sum(r["passed"] for r in results)
        print(f"ผ่าน {npass}/{n} ({npass/n*100:.0f}%)  |  "
              f"ข้อเท็จจริงที่จับได้ {sum(r['ratio'] for r in results)/n*100:.0f}%  |  "
              f"รวม {sum(r['elapsed'] for r in results):.0f}s")
        for lv in ("easy", "medium", "hard", "veryhard"):
            sub = [r for r in results if r["level"] == lv]
            if sub:
                print(f"   {lv:<7} {sum(r['passed'] for r in sub)}/{len(sub)}")

        failed = [r for r in results if not r["passed"] and r["trap"]]
        if failed:
            print("\ntrap ที่ยังไม่ผ่าน:")
            for r in failed:
                print(f"   {r['id']} — {r['trap']}")

        out = os.path.join(HERE, f"results_{time.strftime('%Y%m%d_%H%M%S')}"
                                 f"{'_retrieval' if args.retrieval else ''}.json")
        json.dump({"mode": mode, "passed": npass, "total": n, "results": results},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nผลดิบ: {os.path.relpath(out)}")
        rounds.append(results)

        if args.mlflow:
            try:
                rid = log_mlflow(results, mode, out, rd, args.tag)
                print(f"MLflow run: {rid}")
            except Exception as e:
                print(f"[!] ส่ง MLflow ไม่สำเร็จ: {type(e).__name__}: {str(e)[:160]}")

    if args.repeat > 1:
        summarise(rounds)


def summarise(rounds: list) -> None:
    """สรุปหลายรอบ — LLM ให้ผลไม่คงที่ ต้องดูค่าเฉลี่ยและ 'ข้อที่แกว่ง' ไม่ใช่เลขรอบเดียว"""
    import statistics as st
    scores = [sum(r["passed"] for r in rs) for rs in rounds]
    n = len(rounds[0])
    print("\n" + "═" * 78)
    print(f"สรุป {len(rounds)} รอบ — ผ่าน {'/'.join(map(str, scores))} จาก {n}")
    print(f"  เฉลี่ย {st.mean(scores):.1f}/{n} ({st.mean(scores)/n*100:.0f}%)"
          + (f"  ส่วนเบี่ยงเบน ±{st.stdev(scores):.1f}" if len(scores) > 1 else ""))
    # ข้อที่ผลไม่เหมือนกันทุกรอบ = จุดที่ยังไม่เสถียร ควรดูก่อนเชื่อตัวเลข
    per: dict[str, list] = {}
    for rs in rounds:
        for r in rs:
            per.setdefault(r["id"], []).append(r["passed"])
    flaky = {i: v for i, v in per.items() if len(set(v)) > 1}
    always_fail = [i for i, v in per.items() if not any(v)]
    print(f"  ผ่านทุกรอบ {sum(1 for v in per.values() if all(v))} ข้อ"
          f" · ตกทุกรอบ {len(always_fail)} ข้อ · แกว่ง {len(flaky)} ข้อ")
    if flaky:
        print("  ข้อที่แกว่ง: " + ", ".join(
            f"{i}({''.join('✓' if x else '✗' for x in v)})" for i, v in flaky.items()))
    if always_fail:
        print("  ตกทุกรอบ: " + ", ".join(always_fail))


if __name__ == "__main__":
    main()
