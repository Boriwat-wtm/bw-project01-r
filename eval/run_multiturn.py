"""
run_multiturn.py — วัดคำถามต่อเนื่องหลายเทิร์น (multi-turn)

    python eval/run_multiturn.py                # รันทั้ง 6 บทสนทนา
    python eval/run_multiturn.py --id M3 M4     # เฉพาะบางบท
    python eval/run_multiturn.py --repeat 3     # วัดความนิ่ง
    python eval/run_multiturn.py --mlflow       # ส่งผลขึ้น MLflow

ทำไมต้องมีชุดนี้แยก
  ชุดวัดผลเดิม 100 ข้อเป็น "คำถามเดี่ยว" ทั้งหมด แต่โค้ดจริงมี rewrite_followup
  ทำงานทุกครั้งที่มีประวัติแชท และ FastAPI เปิดฟิลด์ history ให้คนนอกส่งเข้ามาได้
  → เส้นทางนี้ถูกส่งมอบโดยไม่เคยวัดผลเลย

วัด 2 ชั้น (ต่างจาก run_eval.py ที่วัดชั้นเดียว)
  ชั้นที่ 1  คำถามที่เขียนใหม่ (search_q) ถูกไหม — rewrite_has / rewrite_lacks / rewrite_same
  ชั้นที่ 2  คำตอบสุดท้ายถูกไหม — must / must_not (เกณฑ์เดียวกับ run_eval.py)

  ต้องแยกให้ออก เพราะถ้าเขียนคำถามใหม่ผิดตั้งแต่ต้น ระบบจะไปค้นผิดตัวบท
  แล้วไม่มีทางตอบถูกเลย — คนละปัญหากับ "ค้นเจอแล้วแต่เรียบเรียงพลาด"
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import rag        # noqa: E402
import service    # noqa: E402
from run_eval import git_rev, has_fact, norm, score_item   # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def check_rewrite(turn: dict, question: str, search_q: str) -> dict:
    """ตรวจชั้นที่ 1 — ระบบเขียนคำถามต่อเนื่องใหม่ได้ถูกไหม

    หมายเหตุ: service ส่ง search_q = "" กลับมาเมื่อ "ไม่ได้แก้" (เท่ากับคำถามเดิม)
    ต้องแปลงกลับก่อนเทียบ ไม่งั้นจะตัดสินว่าไม่มีคำที่ต้องการทั้งที่คำถามเดิมมีอยู่แล้ว
    """
    actual = search_q or question
    a = norm(actual)
    notes, ok = [], True

    if turn.get("rewrite_same"):
        if actual.strip() != question.strip():
            ok = False
            notes.append("ต้องคืนคำถามเดิมแต่ถูกแก้")

    for k in turn.get("rewrite_has", []):
        if not has_fact(k, a):
            ok = False
            notes.append(f"ขาด '{k}'")

    for k in turn.get("rewrite_lacks", []):
        if has_fact(k, a):
            ok = False
            notes.append(f"ลากของเก่ามา '{k}'")

    return {"ok": ok, "notes": notes, "search_q": actual,
            "changed": actual.strip() != question.strip()}


def run_conversation(llm, item: dict, groups: list) -> dict:
    """รัน 1 บทสนทนา — ป้อนประวัติสะสมทีละเทิร์นเหมือนผู้ใช้จริง"""
    history: list = []
    turns_out = []
    t_conv = time.perf_counter()

    for i, turn in enumerate(item["turns"], 1):
        q = turn["q"]
        t0 = time.perf_counter()
        answer, search_q, n_src, route = "", "", 0, "ปกติ"
        for ev in service.answer_stream(llm, q, all_groups=groups,
                                        history=history, stream=False):
            if "meta" in ev:
                search_q = ev["meta"].get("search_q", "")
                # เส้นทางที่เดินจริงสำคัญพอ ๆ กับคำตอบ — คำถามต่อเนื่องที่เขียนใหม่ผิด
                # ทำให้ระบบ "เปลี่ยนเส้นทาง" ได้ ซึ่งพังหนักกว่าตอบไม่ครบ
                if ev["meta"].get("compare_article"):
                    route = f"เทียบข้ามฉบับ ม.{ev['meta']['compare_article']}"
            elif "final" in ev:
                answer = ev["final"]["answer"]
                n_src = len(ev["final"]["chunks"])

        # เทิร์นแรกไม่มีประวัติ จึงไม่มีอะไรให้เขียนใหม่ — ข้ามการตรวจชั้นที่ 1
        rw = ({"ok": True, "notes": [], "search_q": q, "changed": False} if i == 1
              else check_rewrite(turn, q, search_q))
        sc = score_item(turn, answer)

        turns_out.append({
            "n": i, "q": q, "passed": sc["passed"] and rw["ok"],
            "answer_ok": sc["passed"], "rewrite_ok": rw["ok"],
            "miss": sc["miss"], "bad": sc["bad"], "rewrite_notes": rw["notes"],
            "search_q": rw["search_q"], "rewrite_changed": rw["changed"],
            "route": route, "answer": answer, "n_sources": n_src,
            "elapsed": round(time.perf_counter() - t0, 1),
        })

        history.append({"role": "user", "content": q})
        history.append({"role": "assistant", "content": answer})

    return {"id": item["id"], "kind": item["kind"], "about": item["about"],
            "turns": turns_out,
            "passed": all(t["passed"] for t in turns_out),
            "elapsed": round(time.perf_counter() - t_conv, 1)}


def log_mlflow(results: list, out_path: str, run_idx: int) -> str:
    import mlflow
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "thai-law-rag-eval"))

    turns = [t for r in results for t in r["turns"]]
    follow = [t for t in turns if t["n"] > 1]          # เฉพาะเทิร์นที่มีประวัติจริง
    with mlflow.start_run(run_name=f"multiturn-r{run_idx}") as run:
        mlflow.log_params({
            "mode": "multiturn", "n_conversations": len(results),
            "n_turns": len(turns), "git_commit": git_rev(),
            "llm_model": rag.LLM_MODEL, "embed_model": rag.EMBED_MODEL,
            "history_turns": service.HISTORY_TURNS,
        })
        mlflow.log_metrics({
            "conversations_passed": sum(r["passed"] for r in results),
            "conversation_pass_rate": sum(r["passed"] for r in results) / len(results),
            "turns_passed": sum(t["passed"] for t in turns),
            "turn_pass_rate": sum(t["passed"] for t in turns) / len(turns),
            "rewrite_pass_rate": sum(t["rewrite_ok"] for t in follow) / max(1, len(follow)),
            "answer_pass_rate": sum(t["answer_ok"] for t in turns) / len(turns),
            "elapsed_avg_s": sum(t["elapsed"] for t in turns) / len(turns),
        })
        for r in results:
            mlflow.log_metric(f"item_{r['id']}", float(r["passed"]))
        mlflow.set_tag("failed_ids", ",".join(r["id"] for r in results if not r["passed"]))
        mlflow.log_artifact(out_path)
        return run.info.run_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", nargs="*")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--mlflow", action="store_true")
    ap.add_argument("--file", default="multiturn.json")
    args = ap.parse_args()

    data = json.load(open(os.path.join(HERE, args.file), encoding="utf-8"))
    items = data["items"]
    if args.id:
        items = [x for x in items if x["id"] in args.id]

    rag.update_database()
    rag.build_vectorstore()
    llm = rag.build_llm()
    groups = service.list_groups()

    for run_idx in range(1, args.repeat + 1):
        head = f"multi-turn · {len(items)} บทสนทนา"
        print(f"\n{'=' * 72}\n{head}" + (f"  (รอบ {run_idx}/{args.repeat})"
                                         if args.repeat > 1 else "") + f"\n{'=' * 72}")
        results = []
        for item in items:
            r = run_conversation(llm, item, groups)
            results.append(r)
            mark = "✅" if r["passed"] else "❌"
            print(f"\n{mark} {r['id']} · {r['kind']} — {r['about']}")
            for t in r["turns"]:
                tm = "✅" if t["passed"] else "❌"
                print(f"   {tm} เทิร์น {t['n']}: {t['q']}")
                if t["rewrite_changed"]:
                    print(f"        เขียนใหม่เป็น: {t['search_q']}")
                if t["route"] != "ปกติ":
                    print(f"        เส้นทาง: {t['route']}  ({t['n_sources']} ตัวบท)")
                if not t["rewrite_ok"]:
                    print(f"        ⚠️ เขียนคำถามใหม่: {' · '.join(t['rewrite_notes'])}")
                if t["miss"]:
                    print(f"        ⚠️ คำตอบขาด: {' · '.join(t['miss'])}")
                if t["bad"]:
                    print(f"        ⚠️ คำตอบมีสิ่งที่ห้ามมี: {' · '.join(t['bad'])}")

        turns = [t for r in results for t in r["turns"]]
        follow = [t for t in turns if t["n"] > 1]
        print(f"\n{'-' * 72}")
        print(f"บทสนทนาผ่าน   {sum(r['passed'] for r in results)}/{len(results)}")
        print(f"เทิร์นผ่าน      {sum(t['passed'] for t in turns)}/{len(turns)}")
        print(f"  เขียนคำถามใหม่ถูก  {sum(t['rewrite_ok'] for t in follow)}/{len(follow)}"
              f"   (เฉพาะเทิร์นที่มีประวัติ)")
        print(f"  คำตอบถูก          {sum(t['answer_ok'] for t in turns)}/{len(turns)}")
        print(f"เวลาเฉลี่ยต่อเทิร์น {sum(t['elapsed'] for t in turns) / len(turns):.1f} วินาที")

        stamp = time.strftime("%Y%m%d_%H%M%S")
        out = os.path.join(HERE, f"multiturn_{stamp}.json")
        json.dump({"results": results, "n_turns": len(turns)},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"\nผลดิบ: {os.path.basename(out)}")
        if args.mlflow:
            print(f"MLflow run: {log_mlflow(results, out, run_idx)}")


if __name__ == "__main__":
    main()
