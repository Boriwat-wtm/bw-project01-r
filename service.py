"""
service.py — เครื่องยนต์ RAG ระดับแอปพลิเคชัน (ไม่ผูกกับ UI ใดๆ)

รวม logic ที่เคยปนอยู่ใน app.py (Streamlit) ออกมาให้เป็นกลาง:
  - group routing, multi-turn query rewriting, prompt building
  - answer_stream(): pipeline ทั้งเส้นเป็น "generator" ที่ yield เหตุการณ์ออกมา
    → คนเรียก (Streamlit / FastAPI / CLI) ตัดสินใจเองว่าจะเอาเหตุการณ์ไปทำอะไร

เหตุการณ์ที่ answer_stream yield:
  {"stage": str}                                  ขั้นตอนปัจจุบัน (ให้ UI โชว์ progress)
  {"meta":  {"groups", "search_q", "n_sources"}}  ข้อมูลหลังค้นเสร็จ
  {"token": str}                                  token คำตอบ (streaming)
  {"reasoning": str}                              token ส่วน reasoning (โมเดลคิด)
  {"final": {"answer","chunks","reasoning","groups_used","elapsed","search_q"}}
"""
import json
import re
import time
import urllib.request
from typing import Iterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import rag

# ── config ────────────────────────────────────────────────────────────────────
HISTORY_TURNS = 3          # จำนวนคู่ถาม-ตอบล่าสุดที่ส่งเข้าโมเดล (มากไป = เปลือง+ไขว้เขว)
HISTORY_ANSWER_MAX = 700   # ตัดคำตอบเก่าให้สั้น กัน token บาน

# คำอธิบายกลุ่มเอกสาร — ใช้ให้ LLM router เลือกกลุ่มที่ถูกต้อง
# กลุ่มมาจาก rag.parse_doc_name() (แกะจากชื่อไฟล์) ไม่ใช่ชื่อโฟลเดอร์
# กลุ่มไหนไม่มีในนี้ router จะใช้ชื่อกลุ่มแทน (ยังทำงานได้ แต่เลือกแม่นน้อยลง)
GROUP_DESC = {
    rag.GROUP_IN_FORCE: (
        "ประมวลกฎหมายที่ดิน ฉบับที่ใช้บังคับอยู่ปัจจุบัน (รวมการแก้ไขทุกฉบับแล้ว) — "
        "ตอบคำถามทั่วไปว่า 'กฎหมายว่าอย่างไร' เช่น สิทธิในที่ดิน โฉนด น.ส.๓ "
        "การออกหนังสือแสดงสิทธิ ค่าธรรมเนียม คนต่างด้าว บทกำหนดโทษ"
    ),
    rag.GROUP_AMEND: (
        "พ.ร.บ.แก้ไขเพิ่มเติมประมวลกฎหมายที่ดิน (ฉบับที่ ๑–๑๕) แต่ละฉบับแยกกัน — "
        "ใช้เมื่อถามว่ามาตราไหนถูกแก้เมื่อไร ฉบับที่เท่าไร ปี พ.ศ. ใด หรือถามเหตุผลการแก้ไข"
    ),
    rag.GROUP_HISTORY: (
        "ประมวลกฎหมายที่ดินฉบับรวม ณ ช่วงเวลาต่าง ๆ ในอดีต — "
        "ใช้เฉพาะเมื่อถามว่า 'ตอนปี พ.ศ. ... กฎหมายว่าอย่างไร'"
    ),
    rag.GROUP_ORIGINAL: (
        "ประมวลกฎหมายที่ดิน พ.ศ. ๒๔๙๗ ฉบับดั้งเดิมตอนประกาศใช้ครั้งแรก — "
        "ใช้เมื่อถามตัวบทดั้งเดิมโดยตรง หรือขอเทียบกับของปัจจุบัน"
    ),
}

# คำที่บ่งว่าผู้ใช้ถาม "ประวัติการแก้ไข" ไม่ใช่ "ตัวบทปัจจุบัน"
_HISTORY_HINT = re.compile(
    r"แก้ไข|แก้เมื่อ|ฉบับที่|เดิม|ดั้งเดิม|ก่อนหน้า|เคยเป็น|ย้อนหลัง|ประวัติ|"
    r"เปลี่ยนแปลง|ต่างจาก|เทียบ|ยกเลิก|เพิ่มเติม"
)

# คำที่บ่งว่าต้องการ "ไล่ดูตัวบทข้ามฉบับ" ไม่ใช่ "ตัวบท ณ จุดเวลาใดจุดหนึ่ง"
_COMPARE_HINT = re.compile(
    r"ต่างกัน|ต่างจาก|แตกต่าง|เปรียบเทียบ|เทียบ|ก่อนและหลัง|ก่อนกับหลัง|"
    r"ถูกแก้|เคยแก้|แก้ไขกี่|กี่ครั้ง|แก้เมื่อไร|แก้ไขเมื่อไร|ประวัติการแก้|"
    r"เปลี่ยนไปอย่างไร|เปลี่ยนแปลงอย่างไร|"
    # คำถามแนวไล่สาย (chain) — "ก่อนหน้าของก่อนหน้า", "ฉบับใดบ้าง", "ที่มาอย่างไร"
    r"ก่อนหน้า|กี่ฉบับ|ฉบับใดบ้าง|ฉบับไหนบ้าง|ลำดับเวลา|ที่มาอย่างไร|มีที่มา|ยังมีอยู่|"
    # ถามว่าส่วนต่าง ๆ ของมาตราเดียวกันมาจากฉบับใด (provenance ระดับวรรค)
    r"คนละฉบับ|มาจากกฎหมาย|มาจากฉบับ|แก้ไขโดยฉบับใด|"
    # ถามว่า "บทบัญญัตินี้มีมาตั้งแต่เมื่อไร" — คำตอบอยู่ใน amendment graph โดยตรง
    # ถ้าไม่ส่งมาเส้นทางนี้ ระบบต้องหวังให้ปีบังเอิญติดมากับตัวบทที่ค้นเจอ ซึ่งไม่แน่นอน
    r"เพิ่มเข้ามาเมื่อ|เพิ่มเมื่อ|มีตั้งแต่เมื่อ|เริ่มมีเมื่อ|บัญญัติเมื่อ|เพิ่มโดยฉบับ"
)


def detect_compare(question: str) -> str:
    """คำถามนี้ต้องการ 'ไล่ตัวบทข้ามฉบับ' ไหม -> คืนเลขมาตราที่ถาม ('' = ไม่ใช่)

    ต้องมีครบสองอย่าง: คำที่บ่งการเปรียบเทียบ/ประวัติ + เลขมาตราที่เจาะจง
    เพราะเส้นทางนี้ดึงตัวบทตามเลขมาตราตรง ๆ ถ้าไม่รู้ว่ามาตราไหนก็ทำงานไม่ได้
    (ถามลอย ๆ ว่า 'กฎหมายเปลี่ยนไปอย่างไรบ้าง' จะตกไปใช้เส้นทางค้นปกติ)"""
    if not _COMPARE_HINT.search(question or ""):
        return ""
    arts = rag.question_articles(question)
    return arts[0] if arts else ""


def pick_groups(question: str, all_groups: list[str]) -> tuple[list[str], list[int], list[int]]:
    """ตัดสิน 'ควรค้นกลุ่มไหน' ด้วยกฎในโค้ด ไม่ใช่ให้ LLM เดา
    เหตุผล: 'กฎหมายฉบับไหนใช้บังคับอยู่' เป็นข้อเท็จจริง ไม่ใช่เรื่องตีความ
    ถ้าปล่อยให้ router พลาด = ตอบด้วยตัวบทที่ถูกยกเลิกไปแล้ว ซึ่งรับไม่ได้

    default        -> ฉบับใช้บังคับปัจจุบันเท่านั้น
    ระบุ พ.ศ.      -> ฉบับย้อนหลังที่ใช้บังคับ ณ ปีนั้น (+ ประวัติการแก้ไข)
    ถามประวัติแก้ไข -> + ประวัติการแก้ไข + ฉบับดั้งเดิม
    คืน (groups, years, versions)"""
    groups = [g for g in all_groups if g == rag.GROUP_IN_FORCE] or list(all_groups)
    versions: list[int] = []
    # ⚠️ ย้อนเวลาเฉพาะปีที่เป็น "คำขอย้อนเวลา" จริง ๆ เท่านั้น
    #    ปีที่เป็นชื่อเอกสาร ("ฉบับที่ ๑๕ พ.ศ. ๒๕๖๒") หรือเงื่อนไขในตัวบท
    #    ("ออกใบจองหลังวันที่ ... ๒๕๑๕") ห้ามเอามาล็อกเวอร์ชัน ไม่งั้นจะตอบด้วยกฎหมายเก่า
    kinds = rag.classify_years(question)
    asof = [y for y in kinds["asof"] if y != rag.PUBLISH_STAMP_YEAR]
    if asof:
        # แปลงปี -> ฉบับที่ใช้บังคับ ณ ปีนั้น (ไม่ใช่ฉบับที่ตีพิมพ์ปีนั้นเป๊ะ ๆ)
        # กรองด้วย version แทน year — กฎหมายไม่ได้แก้ทุกปี กรองด้วยปีตรง ๆ จะได้ศูนย์ผล
        versions = [v for v in {rag.version_at_year(y) for y in asof} if v >= 0]
        if versions:
            groups = [g for g in all_groups if g == rag.GROUP_HISTORY] or groups
        groups += [g for g in all_groups if g == rag.GROUP_AMEND]
    # อ้างถึง "ฉบับที่ N" = คำตอบน่าจะอยู่ในตัว พ.ร.บ. แก้ไขฉบับนั้น (ดันอันดับใน retrieve)
    if rag.question_amendments(question):
        groups += [g for g in all_groups if g == rag.GROUP_AMEND]
    if _HISTORY_HINT.search(question or ""):    # ถามเรื่องการแก้ไข -> ปลดประวัติ
        groups += [g for g in all_groups if g in (rag.GROUP_AMEND, rag.GROUP_ORIGINAL)]
    return list(dict.fromkeys(groups)), [], versions


# ── ข้อมูลดัชนี (อ่านหลัง rag โหลด index แล้ว) ────────────────────────────────
def list_groups() -> list[str]:
    """รายชื่อกลุ่มเอกสารทั้งหมดในดัชนี"""
    return sorted({c.get("group", "") for c in rag._chunks if c.get("group")})


def list_years() -> list[int]:
    """ปีเอกสารทั้งหมดในดัชนี (>0 เท่านั้น; 0 = ไม่ระบุปี)"""
    return sorted({int(c.get("year", 0) or 0) for c in rag._chunks if c.get("year")})


def list_models() -> list[str]:
    """ดึงรายชื่อ model จาก endpoint (/v1/models) — กรอง embedding ออก"""
    try:
        req = urllib.request.Request(
            f"{rag.LLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {rag.LLM_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        ids = sorted(m["id"] for m in data.get("data", [])
                     if m.get("id") and "embed" not in m["id"].lower())
        if ids:
            return ids
    except Exception:
        pass
    return [rag.LLM_MODEL]


# ── group auto-routing ────────────────────────────────────────────────────────
def route_groups(llm, question: str, groups: list[str]) -> list[str]:
    """ให้ LLM เลือกว่าคำถามควรค้นในกลุ่มเอกสารไหน (ตอบเป็นเลข → map กลับเป็นชื่อกลุ่ม)
    fallback: ทุกกลุ่ม ถ้า LLM ตอบเพี้ยน/พัง"""
    lines = "\n".join(f"{i+1}. {g} — {GROUP_DESC.get(g, g)}" for i, g in enumerate(groups))
    sys_prompt = (
        "You are a routing classifier for a Thai law document search system. Given a user "
        "question (Thai or English), decide which SINGLE document group best answers it.\n\n"
        f"Groups:\n{lines}\n\n"
        "Default to the group holding the CURRENT consolidated law — most questions ask what "
        "the law says today, not its history.\n"
        'Reply with ONLY ONE number — the single most relevant group (e.g. "1"). '
        "Return TWO numbers only if the question clearly needs both the current text AND its "
        "amendment history. Strongly prefer exactly ONE. No words, no explanation."
    )
    try:
        r = llm.invoke([SystemMessage(content=sys_prompt), HumanMessage(content=question)])
        rag.track_usage(r)
        nums = re.findall(r"\d+", str(r.content))
        picked = [groups[int(n) - 1] for n in nums if 1 <= int(n) <= len(groups)]
        picked = list(dict.fromkeys(picked))     # de-dup คงลำดับ
        if picked:
            return picked
    except Exception:
        pass
    return list(groups)


# ── multi-turn (ถามต่อเนื่องได้) ──────────────────────────────────────────────
def recent_turns(history: list, n: int = HISTORY_TURNS) -> list:
    """คู่ถาม-ตอบล่าสุด n คู่ (ตัดคำตอบยาวๆ ให้สั้นลง)"""
    out = []
    for m in history:
        role = m.get("role")
        content = str(m.get("content", "")).strip()
        if not content or role not in ("user", "assistant"):
            continue
        if role == "assistant" and len(content) > HISTORY_ANSWER_MAX:
            content = content[:HISTORY_ANSWER_MAX] + " …"
        out.append({"role": role, "content": content})
    return out[-(n * 2):]


def rewrite_followup(llm, question: str, history: list) -> str:
    """แปลงคำถามต่อเนื่องให้ "สมบูรณ์ในตัว" ก่อนเอาไปค้น/เลือกกลุ่ม
    เช่น "แล้วโทษล่ะ" + ประวัติ → "โทษของการบุกรุกที่ดินของรัฐตามมาตรา ๙ คืออะไร"
    ไม่มีประวัติ หรือคำถามสมบูรณ์อยู่แล้ว → คืนคำถามเดิม"""
    turns = recent_turns(history, 2)
    if not turns:
        return question
    convo = "\n".join(("ผู้ใช้: " if t["role"] == "user" else "ผู้ช่วย: ") + t["content"]
                      for t in turns)
    sys_prompt = (
        "Rewrite the user's latest question into a STANDALONE question that can be "
        "understood without the conversation — resolve pronouns and implied subjects "
        "from the history (e.g. 'แล้วโทษล่ะ' -> 'โทษของการเข้าไปยึดถือครอบครองที่ดินของรัฐคืออะไร').\n"
        "- Keep any มาตรา/ข้อ number mentioned earlier if the new question refers to it.\n"
        "Rules:\n"
        "- If the question is ALREADY standalone, return it UNCHANGED.\n"
        "- Keep the user's original language.\n"
        "- Reply with ONLY the question. No quotes, no explanation."
    )
    try:
        r = llm.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=f"บทสนทนาก่อนหน้า:\n{convo}\n\nคำถามล่าสุด: {question}"),
        ])
        rag.track_usage(r)
        out = " ".join(str(r.content).split()).strip().strip('"')
        # กันโมเดลเพี้ยน (ตอบยาวเป็นย่อหน้า/ตอบว่าง) → ใช้ของเดิมปลอดภัยกว่า
        if out and len(out) <= max(240, len(question) * 4):
            return out
    except Exception:
        pass
    return question


def build_user_prompt(question: str, context: str, domain: str = "thai_law") -> str:
    prof = rag.DOMAINS.get(domain, rag.DOMAINS["thai_law"])
    cite = "เลขมาตรา/ข้อ/หน้า" if domain == "thai_law" else "เลข Article/หน้า"
    return (
        f"{prof['user_intro']}\n\n{context}\n\n"
        f"================\n"
        f"คำถาม: {question}\n\n"
        f"ตอบโดยอ้างอิงเฉพาะเอกสารด้านบน พร้อมระบุ{cite}"
    )


def _retrieve_live(**kw) -> Iterator[dict]:
    """เรียก rag.retrieve() แล้ว yield {"stage": ...} ออกมา "ตามเวลาจริง"

    ⚠️ retrieve ไม่ใช่ generator ถ้าเก็บ stage ใส่ list แล้ว yield ทีหลัง ผู้ใช้จะเห็น
    ข้อความค้างอยู่ ๖๐ วินาทีแล้วโผล่พรวดเดียวตอนจบ ซึ่งไม่ต่างจากไม่มีเลย
    จึงรันในเธรดแยกแล้วส่ง stage ผ่านคิว — ไม่ต้องรื้อ retrieve ให้เป็น generator
    ซึ่งจะกระทบผู้เรียกทุกที่ (eval, FastAPI, CLI)

    ผลลัพธ์ส่งกลับผ่าน StopIteration.value → ผู้เรียกใช้ `chunks = yield from ...`"""
    import queue
    import threading
    q: "queue.Queue" = queue.Queue()
    box: dict = {}

    def work():
        try:
            box["out"] = rag.retrieve(on_stage=q.put, **kw)
        except BaseException as e:      # เก็บไว้โยนต่อในเธรดหลัก จะได้ traceback ตามปกติ
            box["err"] = e
        finally:
            q.put(None)

    t = threading.Thread(target=work, daemon=True)
    t.start()
    while True:
        s = q.get()
        if s is None:
            break
        yield {"stage": s}
    t.join()
    if "err" in box:
        raise box["err"]
    return box.get("out", [])


def _stream_answer(llm, messages, stream: bool):
    """ยิง messages เข้า LLM แล้ว yield token — ใช้ร่วมกันทั้งสองเส้นทาง
    คืน (answer, reasoning) ผ่าน StopIteration value ของ generator"""
    answer, reasoning = "", ""
    if stream:
        full = None
        try:
            gen = llm.stream(messages, stream_usage=True)
        except TypeError:
            gen = llm.stream(messages)
        for chunk in gen:
            full = chunk if full is None else full + chunk
            ak = getattr(chunk, "additional_kwargs", {}) or {}
            rc = ak.get("reasoning_content") or ak.get("reasoning") or ""
            c = str(getattr(chunk, "content", "") or "")
            if rc and not answer:
                reasoning += str(rc)
                yield {"reasoning": str(rc)}
            if c:
                answer += c
                yield {"token": c}
        if not answer and reasoning:      # thinking model ที่ส่งมาแต่ reasoning
            answer = reasoning
        if full is not None:
            rag.track_usage(full)
    else:
        resp = rag.invoke_retry(llm, messages,
                                ok_fn=lambda c: not rag.looks_truncated(c), label="answer")
        answer = str(resp.content)
        ak = getattr(resp, "additional_kwargs", {}) or {}
        reasoning = str(ak.get("reasoning_content") or ak.get("reasoning") or "")
        yield {"token": answer}
    return answer, reasoning


def _answer_compare(llm, question: str, num: str, history: list,
                    stream: bool, t0: float) -> Iterator[dict]:
    """เส้นทางเปรียบเทียบ: ไล่ตัวบทมาตรา num ข้ามทุกฉบับ -> ให้ LLM ชี้ว่าอะไรเปลี่ยนเมื่อไร
    ไม่ใช้ embedding/BM25/rerank เลย — ดึงจาก metadata ตรง ๆ จึงไม่มีทางพลาดมาตรา"""
    yield {"stage": f"ไล่ตัวบทมาตรา {num} ข้ามทุกฉบับ"}
    points = rag.article_timeline(num)
    # สายการแก้ไขจาก amendment graph — ระบุ "ใครแก้ เมื่อไร ทับงานของใคร" แบบชี้ขาด
    # ต่างจาก timeline ที่บอกแค่ "ตัวบทเปลี่ยนตอนไหน" — สองอันนี้เสริมกัน
    chain = rag.article_chain(num)
    yield {"meta": {"groups": [rag.GROUP_IN_FORCE, rag.GROUP_HISTORY],
                    "n_sources": len(points), "search_q": "",
                    "compare_article": num, "n_changes": max(0, len(points) - 1),
                    "chain_len": max(0, len(chain) - 1)}}

    yield {"stage": "เขียนคำตอบ"}
    n = len(points)
    parts = []
    # มาตรานี้ถูก "เปลี่ยนเลข" ตอนแก้ไขไหม — ต้องบอก LLM ตรง ๆ ไม่ใช่แค่เอาไปขยายคำค้น
    # ไม่งั้นถามว่า "ม.๙ ทวิ ยังมีอยู่ไหม" จะตอบไม่ได้ว่ามันกลายเป็น ม.๙/๑
    alias = rag.ARTICLE_ALIASES.get(num)
    if alias:
        parts.append(f"⚠️ มาตรา {num} ถูกยกเลิกแล้วบัญญัติใหม่ในชื่อ 'มาตรา {alias}' "
                     f"(เปลี่ยนเลขมาตรา ไม่ใช่ถูกยกเลิกทิ้ง) — ตัวบทปัจจุบันอยู่ที่มาตรา {alias}")
        alias_chain = rag.article_chain(alias)
        if alias_chain:
            parts.append(rag.format_chain(alias, alias_chain))
        alias_pts = rag.article_timeline(alias)
        if alias_pts:
            parts.append(f"ตัวบทของมาตรา {alias} (ฉบับปัจจุบัน):\n\n"
                         f"{rag.format_comparison(alias, alias_pts[-1:])}")
    if chain:
        parts.append(rag.format_chain(num, chain))
    parts.append(f"ตัวบท 'มาตรา {num}' ทุกรุ่นที่เนื้อหาเปลี่ยนจริง เรียงจากเก่าไปใหม่ "
                 f"(พบ {n} รุ่น):\n\n{rag.format_comparison(num, points)}")
    messages = [
        SystemMessage(content=rag.COMPARE_SYSTEM),
        HumanMessage(content="\n\n".join(parts) +
                             f"\n\n================\nคำถาม: {question}"),
    ]
    answer, reasoning = yield from _stream_answer(llm, messages, stream)
    yield {"final": {"answer": answer, "chunks": points, "reasoning": reasoning,
                     "groups_used": [rag.GROUP_IN_FORCE, rag.GROUP_HISTORY],
                     "elapsed": time.perf_counter() - t0, "search_q": ""}}


# ── core pipeline (generator — ไม่ผูก UI) ─────────────────────────────────────
def answer_stream(llm, question: str, *, auto_group: bool = True,
                  all_groups: Optional[list[str]] = None,
                  manual_groups: Optional[list[str]] = None,
                  year_filter: Optional[list[int]] = None,
                  history: Optional[list] = None,
                  stream: bool = True) -> Iterator[dict]:
    """RAG pipeline: routing → multi-turn rewrite → hybrid retrieve+rerank → LLM ตอบ
    yield เหตุการณ์ทีละขั้น (ดู docstring หัวไฟล์) — คนเรียกเอาไปแสดง/ส่ง SSE เอง"""
    t0 = time.perf_counter()
    history = history or []
    all_groups = all_groups or []

    # 0) คำถามต่อเนื่อง → เขียนใหม่ให้สมบูรณ์ก่อน (ใช้ค้น+เลือกกลุ่ม; ตอนตอบใช้คำถามเดิม+ประวัติ)
    search_q = question
    if history:
        yield {"stage": "เข้าใจคำถามต่อเนื่อง"}
        search_q = rewrite_followup(llm, question, history)

    # 1) แยกเส้นทาง: คำถามเปรียบเทียบ/ประวัติของ "มาตราหนึ่ง ๆ" ใช้กลไกคนละแบบ
    #    เส้นทางปกติยุบเวอร์ชันซ้ำทิ้งเพื่อตอบว่ากฎหมายว่าอย่างไร ซึ่งทำลายข้อมูลที่การ
    #    เปรียบเทียบต้องใช้พอดี → ที่นี่จึงไปดึงตัวบทตามเลขมาตราตรง ๆ ข้าม RRF ทั้งหมด
    cmp_article = detect_compare(search_q)
    if cmp_article:
        yield from _answer_compare(llm, question, cmp_article, history, stream, t0)
        return

    # 2) เลือกกลุ่มเอกสาร — ใช้กฎในโค้ด ไม่ใช่ LLM (ดูเหตุผลใน pick_groups)
    #    เลือกเองจาก UI (manual) ชนะเสมอ — ผู้ใช้รู้ว่าตัวเองต้องการอะไร
    if not auto_group and manual_groups:
        groups_used, auto_years, versions = manual_groups, [], []
    else:
        yield {"stage": "เลือกกลุ่มเอกสาร"}
        groups_used, auto_years, versions = pick_groups(search_q, all_groups)
    year_filter = year_filter or auto_years or None
    filter_groups = groups_used or None
    # โดเมน → เลือก prompt/expand/intro (โปรเจกต์นี้มีโปรไฟล์เดียว: thai_law)
    domain = rag.domain_of_group(groups_used[0]) if groups_used else "thai_law"

    # 2) ขยายคำถาม (multi-query) + 3) retrieve (hybrid RRF + filter + rerank)
    #    ซอย stage ให้ละเอียด — ช่วงนี้กินเวลาหลายสิบวินาที ถ้าเงียบผู้ใช้จะคิดว่าค้าง
    yield {"stage": "แตกคำถามเป็นหลายมุมค้นหา"}
    search_qs = rag.expand_queries(llm, search_q, domain=domain)

    chunks = yield from _retrieve_live(
        query=search_qs, rerank_query=search_q, groups=filter_groups,
        years=year_filter, versions=versions or None)
    # ── เติม "ข้อมูลที่คำนวณจากเอกสารด้วยโค้ด" ไว้หัว context ────────────────
    # ไม่ใช่การสั่ง LLM ผ่าน prompt (ซึ่งกระทบทุกคำถามและโมเดลอาจไม่ทำตาม)
    # แต่เป็นการรับประกันว่าข้อเท็จจริงชี้ขาดอยู่ใน context แน่นอน — และตรวจย้อนได้ว่ามาจากไหน
    # ยิงเฉพาะเมื่อเข้าเงื่อนไข คำถามอื่นจึงไม่เห็นความเปลี่ยนแปลงใด ๆ
    facts = []
    amend_nos = rag.question_amendments(search_q)
    if len(amend_nos) >= 2:                            # ถามถึงหลายฉบับ -> เทียบให้เลย
        ov = rag.amendment_overlap(amend_nos)
        if ov:
            facts.append(ov)
    for no in amend_nos:                               # ถามถึง "ฉบับที่ N"
        brief = rag.amendment_brief(no)
        if brief:
            facts.append(brief)
    if rag._ROLE_CUE.search(search_q):                 # ถามถึงองค์ประกอบคณะกรรมการ
        roles = rag.format_roles(chunks)
        if roles:
            facts.append(roles)
    context = rag.format_context(chunks)
    if facts:
        context = "\n\n".join(facts) + "\n\n" + "=" * 16 + "\n\n" + context
    yield {"meta": {"groups": groups_used, "n_sources": len(chunks),
                    "search_q": search_q if search_q != question else ""}}

    # 4) ประกอบข้อความ: system (ตามโดเมน) + ประวัติล่าสุด + (เอกสาร + คำถามปัจจุบัน)
    sys_prompt = rag.DOMAINS.get(domain, rag.DOMAINS["thai_law"])["system_prompt"]
    messages: list = [SystemMessage(content=sys_prompt)]
    for t in recent_turns(history):
        messages.append(HumanMessage(content=t["content"]) if t["role"] == "user"
                        else AIMessage(content=t["content"]))
    messages.append(HumanMessage(content=build_user_prompt(question, context, domain)))

    # 5) ตอบ (stream = yield ทีละ token) — ใช้ตัวเดียวกับเส้นทางเปรียบเทียบ
    yield {"stage": "เขียนคำตอบ"}
    answer, reasoning = yield from _stream_answer(llm, messages, stream)

    yield {"final": {"answer": answer, "chunks": chunks, "reasoning": reasoning,
                     "groups_used": groups_used, "elapsed": time.perf_counter() - t0,
                     "search_q": search_q if search_q != question else ""}}
