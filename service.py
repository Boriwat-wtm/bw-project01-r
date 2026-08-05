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

# คำที่บ่งว่าคำถาม "อ้างถึงสิ่งที่พูดไปแล้ว" ในบทสนทนา — ใช้ตัดสินว่าคำถามนี้
# มีสิทธิ์ยืมเลขมาตราจากเทิร์นก่อนมาใช้ไหม (ดู _may_borrow_article)
_ANAPHORA_CUE = re.compile(r"นี้|นั้น|ดังกล่าว|ข้างต้น|อันนี้|ตัวนี้")

# ผู้ใช้เขียน "มาตรานี้/มาตรานั้น" ตรง ๆ — แคบกว่า _ANAPHORA_CUE เพราะระบุชัดว่า
# กำลังอ้างถึง "มาตรา" ไม่ใช่คำนามอื่น ("โฉนดนี้" ห้ามเข้าเงื่อนไขนี้)
_ARTICLE_ANAPHORA = re.compile(r"(มาตรา|ข้อ)\s*(นี้|นั้น|ดังกล่าว|ข้างต้น)")

# คำที่บ่งว่าต้องการ "ไล่ดูตัวบทข้ามฉบับ" ไม่ใช่ "ตัวบท ณ จุดเวลาใดจุดหนึ่ง"
_COMPARE_HINT = re.compile(
    r"ต่างกัน|ต่างจาก|แตกต่าง|เปรียบเทียบ|เทียบ|ก่อนและหลัง|ก่อนกับหลัง|"
    r"ถูกแก้|เคยแก้|แก้ไขกี่|กี่ครั้ง|แก้เมื่อไร|แก้ไขเมื่อไร|ประวัติการแก้|"
    r"เปลี่ยนไปอย่างไร|เปลี่ยนแปลงอย่างไร|"
    # คำถามแนวไล่สาย (chain) — "ก่อนหน้าของก่อนหน้า", "ฉบับใดบ้าง", "ที่มาอย่างไร"
    # ⚠️ "กี่ฉบับ" ลอย ๆ ใช้ไม่ได้ — "ฉบับ" ในกฎหมายไทยมีสองความหมายคนละเรื่อง
    #    "ถูกแก้โดยกฎหมายกี่ฉบับ" = จำนวน พ.ร.บ. ที่มาแก้  (ต้องเข้าเส้นเทียบ)
    #    "โฉนดต้องทำกี่ฉบับ"      = จำนวนใบสำเนา          (ห้ามเข้า — คนละเรื่องเลย)
    #    เดิมจับทั้งสองแบบ ทำให้ "โฉนดตามมาตรา ๕๗ ทำกี่ฉบับ" ไหลไปตอบประวัติ ม.๕๗
    r"ก่อนหน้า|กฎหมายกี่ฉบับ|แก้ไขกี่ฉบับ|แก้มากี่ฉบับ|กี่ฉบับแล้ว|"
    r"ฉบับใดบ้าง|ฉบับไหนบ้าง|ลำดับเวลา|ที่มาอย่างไร|มีที่มา|ยังมีอยู่|"
    # ถามว่าส่วนต่าง ๆ ของมาตราเดียวกันมาจากฉบับใด (provenance ระดับวรรค)
    r"คนละฉบับ|มาจากกฎหมาย|มาจากฉบับ|แก้ไขโดยฉบับใด|"
    # ถามว่า "บทบัญญัตินี้มีมาตั้งแต่เมื่อไร" — คำตอบอยู่ใน amendment graph โดยตรง
    # ถ้าไม่ส่งมาเส้นทางนี้ ระบบต้องหวังให้ปีบังเอิญติดมากับตัวบทที่ค้นเจอ ซึ่งไม่แน่นอน
    r"เพิ่มเข้ามาเมื่อ|เพิ่มเมื่อ|มีตั้งแต่เมื่อ|เริ่มมีเมื่อ|บัญญัติเมื่อ|เพิ่มโดยฉบับ|"
    # วงจรชีวิตอีกด้าน: การ "ถูกยกเลิก" — มาตราที่ถูกยกเลิกมีบรรทัด "ยกเลิกโดย..." อยู่ในตัวบทเสมอ
    # เส้นทางนี้ดึงตัวบทตามเลขมาตราตรง ๆ จึงตอบชี้ขาดได้ว่าใครยกเลิก เมื่อไร (ใช้ได้ทุกมาตรา
    # ที่เคยถูกยกเลิก เช่น ๓๔-๔๙, ๑๐๑, ๑๐๕ ทวิ) — เส้นค้นปกติเคยพลาด: เจอสถานะ (ยกเลิก)
    # แต่ค้นไม่ถึงบรรทัดที่บอกว่าใครเป็นผู้ยกเลิก · จงใจไม่รวม "ยกเลิกความใน/ยกเลิกหมวด"
    # เพราะนั่นคือคำถามฝั่ง "ฉบับที่ N ทำอะไร" ซึ่งเส้นปกติ + amendment_brief ตอบได้ดีอยู่แล้ว
    r"ถูกยกเลิก|ยกเลิกโดย|ยกเลิกไปแล้ว|ยกเลิกเมื่อ|ยกเลิกหรือไม่|ยกเลิกหรือยัง"
)


@rag.traced("TOOL")
def detect_compare(question: str) -> str:
    """คำถามนี้ต้องการ 'ไล่ตัวบทข้ามฉบับ' ไหม -> คืนเลขมาตราที่ถาม ('' = ไม่ใช่)

    ต้องมีครบสองอย่าง: คำที่บ่งการเปรียบเทียบ/ประวัติ + เลขมาตราที่เจาะจง
    เพราะเส้นทางนี้ดึงตัวบทตามเลขมาตราตรง ๆ ถ้าไม่รู้ว่ามาตราไหนก็ทำงานไม่ได้
    (ถามลอย ๆ ว่า 'กฎหมายเปลี่ยนไปอย่างไรบ้าง' จะตกไปใช้เส้นทางค้นปกติ)"""
    if not _COMPARE_HINT.search(question or ""):
        return ""
    arts = rag.question_articles(question)
    return arts[0] if arts else ""


def carry_article(question: str, search_q: str, history: list) -> str:
    """เติมเลขมาตราจากบทสนทนาให้คำถามที่เขียนว่า "มาตรานี้" แต่ไม่มีเลข

    ทำไมต้องมี: rewrite_followup เป็นการเรียก LLM ซึ่งบางครั้งคืนคำถามเดิมมาดื้อ ๆ
    (วัดได้จากชุด multiturn ข้อ M1) พอไม่มีเลขมาตรา ระบบก็ไปค้นด้วยคำว่า
    "มาตรานี้ถูกแก้ไขโดยกฎหมายฉบับใด" ซึ่งไม่ชี้ไปที่ตัวบทไหนเลย — ได้ chunk เดียว

    "มาตราไหนที่พูดถึงล่าสุด" เป็นข้อเท็จจริงที่โค้ดหาเองได้จากประวัติ
    ไม่ต้องรอให้ LLM ยอมทำ (หลักเดิมของโปรเจกต์: ข้อเท็จจริงให้โค้ดคำนวณ)

    เงื่อนไขแคบไว้ก่อน — ต้องเขียน "มาตรานี้/ข้อนั้น" ตรง ๆ และคำถามต้องไม่มี
    เลขมาตราของตัวเองอยู่แล้ว (แค่ "โฉนดนี้" จึงไม่เข้าเงื่อนไข เพราะ "นี้"
    ตัวนั้นอ้างถึงโฉนด ไม่ใช่มาตรา — เติมเลขมาตราเข้าไปจะเพี้ยนหนักกว่าเดิม)
    """
    if not history or rag.question_articles(search_q):
        return search_q
    if not _ARTICLE_ANAPHORA.search(question):
        return search_q
    for turn in reversed(recent_turns(history)):        # ใหม่สุดก่อน
        arts = rag.question_articles(turn.get("content", ""))
        if arts:
            return _ARTICLE_ANAPHORA.sub(f"มาตรา {arts[0]}", search_q, count=1)
    return search_q


def _may_borrow_article(question: str) -> bool:
    """คำถาม (ตัวที่ผู้ใช้พิมพ์จริง) มีสิทธิ์ "ยืมเลขมาตรา" จากบทสนทนาก่อนหน้าไหม

    มีเลขมาตราของตัวเองอยู่แล้ว  -> ใช่ (ไม่ได้ยืมใคร)
    มีคำอ้างถึง เช่น "มาตรานี้"   -> ใช่ (ตั้งใจอ้างของเดิมจริง ๆ)
    ไม่มีทั้งสองอย่าง             -> ไม่ใช่ — คำถามสมบูรณ์ในตัวแล้ว ไม่ได้พูดถึงมาตราไหน
    """
    return bool(rag.question_articles(question) or _ANAPHORA_CUE.search(question or ""))


@rag.traced("TOOL")
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


@rag.traced("LLM")
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
        "Rules:\n"
        "- If the question ALREADY makes sense on its own, return it UNCHANGED. "
        "Users change topic mid-conversation; a new topic is NOT a follow-up.\n"
        "- NEVER add a มาตรา/ข้อ number from the history unless the question "
        "actually points back to it (e.g. 'มาตรานี้', 'มาตราดังกล่าว'). "
        "Adding one that the user did not mean sends the search to the wrong law.\n"
        "- Keep any มาตรา/ข้อ number mentioned earlier ONLY if the new question refers to it.\n"
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
    import contextvars
    import queue
    import threading
    q: "queue.Queue" = queue.Queue()
    box: dict = {}
    # trace context (RAG_TRACE=1) เป็นของประจำเธรด — ก๊อปไปให้เธรดลูกด้วย
    # ไม่งั้น span ของ retrieve จะหลุดเป็น trace แยก ไม่เกาะใต้ answer_stream
    # (ตอนปิด trace: ctx.run ก็แค่รันฟังก์ชันตามปกติ ไม่มีผลอะไร)
    ctx = contextvars.copy_context()

    def work():
        try:
            box["out"] = ctx.run(lambda: rag.retrieve(on_stage=q.put, **kw))
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


# error ที่แปลว่า "endpoint คุยไม่ได้" — ตระกูลเดียวกับที่ retrieve ใช้ดัก embedding ล่ม
# เพราะ chat กับ embedding ยิงไป endpoint เดียวกัน ล่มพร้อมกันเป็นเรื่องปกติ
LLM_DOWN = rag.SEMANTIC_DOWN


def fallback_answer(chunks: list, err: Exception, n: int = 6) -> str:
    """LLM เขียนคำตอบไม่ได้ -> คืน "ตัวบทที่ค้นเจอ" ให้ผู้ใช้อ่านเอง

    เดิมกรณีนี้ผู้ใช้ได้ error เปล่า ๆ ทั้งที่ระบบค้นตัวบทเจอแล้ว —
    งานที่ทำไปแล้ว 90% ถูกทิ้งเพราะขั้นสุดท้ายพัง ทั้งที่ของที่ค้นได้มีค่าในตัวเอง
    ทนายอ่านตัวบทเองได้ ขอแค่ชี้ให้ถูกมาตรา
    """
    head = ("> ⚠️ **ระบบเขียนคำตอบไม่สำเร็จ** "
            f"({type(err).__name__}) — แต่ค้นตัวบทที่เกี่ยวข้องเจอแล้ว\n"
            "> ข้างล่างคือตัวบทดิบเรียงตามความเกี่ยวข้อง **ยังไม่ได้เรียบเรียงและไม่ได้ตรวจ**\n"
            "> ถ้าต้องการคำตอบที่เรียบเรียงแล้ว กรุณาถามใหม่อีกครั้ง\n")
    if not chunks:
        return head + "\n(ไม่พบตัวบทที่เกี่ยวข้องด้วย)"
    parts = [head]
    for c in chunks[:n]:
        art = c.get("article") or "-"
        page = c.get("page_start") or "-"
        label = c.get("doc_label") or c.get("source") or ""
        flag = "" if c.get("in_force") else "  ⚠️ ไม่ใช่ฉบับที่ใช้บังคับปัจจุบัน"
        text = " ".join(str(c.get("text", "")).split())[:600]
        parts.append(f"\n**{art}** · หน้า {page} · {label}{flag}\n\n{text} …")
    return "\n".join(parts)


def _stream_answer(llm, messages, stream: bool, chunks: "Optional[list]" = None):
    """ยิง messages เข้า LLM แล้ว yield token — ใช้ร่วมกันทั้งสองเส้นทาง
    คืน (answer, reasoning) ผ่าน StopIteration value ของ generator

    chunks: ส่งมาด้วยแล้ว ถ้า LLM ล่มจะคืนตัวบทเหล่านี้แทน error เปล่า
            (None = ไม่มีของสำรอง ปล่อย exception ขึ้นไปตามเดิม)
    """
    answer, reasoning = "", ""
    try:
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
            # จดลง trace: สาย stream เรียก llm.stream ตรง ๆ ไม่ผ่าน invoke_retry
            # เลยไม่มี span LLM ของตัวเอง — จด prompt+คำตอบเต็มไว้ตรงนี้แทน
            rag.trace_note("llm_stream_answer",
                           inputs={"messages": [str(getattr(m, "content", "")) for m in messages]},
                           outputs={"answer": answer, "reasoning": reasoning})
        else:
            resp = rag.invoke_retry(llm, messages,
                                    ok_fn=lambda c: not rag.looks_truncated(c), label="answer")
            answer = str(resp.content)
            ak = getattr(resp, "additional_kwargs", {}) or {}
            reasoning = str(ak.get("reasoning_content") or ak.get("reasoning") or "")
            yield {"token": answer}
    except LLM_DOWN as e:
        # ไม่มีของสำรองให้คืน = พฤติกรรมเดิม ปล่อย error ขึ้นไปให้ผู้เรียกจัดการ
        if chunks is None:
            raise
        rag.trace_note("llm_failed", outputs={"error": type(e).__name__,
                                              "partial_chars": len(answer)})
        fb = fallback_answer(chunks, e)
        if answer:      # สตรีมไปแล้วบางส่วนก่อนล่ม ต้องบอกว่าท่อนบนถูกตัดกลางคัน
            fb = ("\n\n> ⚠️ **คำตอบด้านบนถูกตัดกลางคัน** ระบบเขียนต่อไม่ได้\n\n" + fb)
        yield {"stage": "⚠️ เขียนคำตอบไม่สำเร็จ — คืนตัวบทที่ค้นเจอแทน"}
        yield {"token": fb}
        answer, reasoning = (answer + fb if answer else fb), ""
    return answer, reasoning


def _answer_compare(llm, question: str, num: str, history: list,
                    stream: bool, t0: float, search_q: str = "") -> Iterator[dict]:
    """เส้นทางเปรียบเทียบ: ไล่ตัวบทมาตรา num ข้ามทุกฉบับ -> ให้ LLM ชี้ว่าอะไรเปลี่ยนเมื่อไร
    ไม่ใช้ embedding/BM25/rerank เลย — ดึงจาก metadata ตรง ๆ จึงไม่มีทางพลาดมาตรา

    search_q = คำถามหลังเขียนใหม่ (ถ้าเป็นคำถามต่อเนื่อง) — รายงานออกไปให้ผู้เรียกเห็น
    เดิมตรงนี้ส่ง "" ตายตัว ทำให้เส้นทางนี้ไม่เคยบอกเลยว่าคำถามถูกเขียนใหม่หรือยัง
    ผู้เรียก API จึงตรวจไม่ได้ว่าระบบเข้าใจคำถามต่อเนื่องตรงกับที่ตั้งใจไหม"""
    yield {"stage": f"ไล่ตัวบทมาตรา {num} ข้ามทุกฉบับ"}
    points = rag.article_timeline(num)
    # สายการแก้ไขจาก amendment graph — ระบุ "ใครแก้ เมื่อไร ทับงานของใคร" แบบชี้ขาด
    # ต่างจาก timeline ที่บอกแค่ "ตัวบทเปลี่ยนตอนไหน" — สองอันนี้เสริมกัน
    chain = rag.article_chain(num)
    reported_q = search_q if search_q and search_q != question else ""
    yield {"meta": {"groups": [rag.GROUP_IN_FORCE, rag.GROUP_HISTORY],
                    "n_sources": len(points), "search_q": reported_q,
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
    answer, reasoning = yield from _stream_answer(llm, messages, stream, chunks=points)
    yield {"final": {"answer": answer, "chunks": points, "reasoning": reasoning,
                     "groups_used": [rag.GROUP_IN_FORCE, rag.GROUP_HISTORY],
                     "elapsed": time.perf_counter() - t0, "search_q": reported_q}}


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
        # กันกรณี LLM ไม่ยอมแทน "มาตรานี้" ให้ — เติมเองจากประวัติด้วยโค้ด
        search_q = carry_article(question, search_q, history)

    # 1) แยกเส้นทาง: คำถามเปรียบเทียบ/ประวัติของ "มาตราหนึ่ง ๆ" ใช้กลไกคนละแบบ
    #    เส้นทางปกติยุบเวอร์ชันซ้ำทิ้งเพื่อตอบว่ากฎหมายว่าอย่างไร ซึ่งทำลายข้อมูลที่การ
    #    เปรียบเทียบต้องใช้พอดี → ที่นี่จึงไปดึงตัวบทตามเลขมาตราตรง ๆ ข้าม RRF ทั้งหมด
    cmp_article = detect_compare(search_q)
    # ⚠️ การเขียนคำถามต่อเนื่องใหม่ต้องไม่มีสิทธิ์ "เปลี่ยนเส้นทาง" ของระบบ
    #    เจอจากชุด multiturn (M4): ผู้ใช้เปลี่ยนเรื่องเป็น "โฉนดต้องทำกี่ฉบับ"
    #    แต่ rewrite ลาก "มาตรา ๓๑" จากเทิร์นก่อนมาใส่ พอมีเลขมาตราโผล่ + คำว่า
    #    "กี่ฉบับ" ที่อยู่ใน _COMPARE_HINT อยู่แล้ว -> ไหลไปเส้นเปรียบเทียบ
    #    แล้วตอบประวัติ ม.๓๑ แทนคำถามที่ถามจริง (ผิดทั้งข้อ ไม่ใช่แค่ไม่ครบ)
    #    กติกา: ยืมเลขมาตราจากบทสนทนาได้ ก็ต่อเมื่อคำถามเดิมอ้างถึงมันจริง
    if cmp_article and search_q != question and not _may_borrow_article(question):
        cmp_article = ""
    if cmp_article:
        yield from _answer_compare(llm, question, cmp_article, history, stream, t0,
                                   search_q=search_q)
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

    # rstats รับโหมดที่ retrieve ใช้จริงกลับมา — ต้องเป็น dict ต่อคำถาม ไม่ใช่ตัวแปรกลาง
    # เพราะ retrieve รันในเธรดแยกและรับหลายคำถามพร้อมกันได้ (ดูคอมเมนต์ใน rag.retrieve)
    rstats: "dict[str, str]" = {}
    chunks = yield from _retrieve_live(
        query=search_qs, rerank_query=search_q, groups=filter_groups,
        years=year_filter, versions=versions or None, stats=rstats)
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
    # ⚠️ ถ้า retrieve ตกไปโหมดสำรอง (embedding ล่ม เหลือแต่ BM25) ต้องบอกให้รู้ทุกทาง
    #    ทั้ง event ที่ UI อ่าน และตัวคำตอบเอง — คุณภาพลดลงแบบเงียบ ๆ คือสิ่งที่อันตรายที่สุด
    retrieval_mode = rstats.get("mode", "hybrid")
    degraded = retrieval_mode == "bm25_only"
    if degraded:
        yield {"stage": "⚠️ ค้นด้วยคำอย่างเดียว (ระบบค้นด้วยความหมายใช้ไม่ได้ชั่วคราว)"}
    yield {"meta": {"groups": groups_used, "n_sources": len(chunks),
                    "retrieval_mode": retrieval_mode,
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
    # ส่ง chunks ไปด้วย: LLM ล่มแล้วยังคืนตัวบทที่ค้นเจอได้ ดีกว่าทิ้งงานที่ทำมาแล้วทั้งหมด
    answer, reasoning = yield from _stream_answer(llm, messages, stream, chunks=chunks)

    # ติดป้ายบนตัวคำตอบด้วย — ไม่ใช่แค่ event เพราะคำตอบอาจถูกก๊อปไปใช้ต่อโดยไม่มี event ติดไป
    if degraded:
        answer = ("> ⚠️ **คำตอบนี้ค้นด้วยคำสำคัญอย่างเดียว** เพราะระบบค้นด้วยความหมาย"
                  "ใช้ไม่ได้ชั่วคราว อาจหาตัวบทที่ใช้คำต่างจากคำถามไม่เจอ "
                  "— ควรตรวจสอบซ้ำหรือถามใหม่ภายหลัง\n\n" + answer)

    yield {"final": {"answer": answer, "chunks": chunks, "reasoning": reasoning,
                     "groups_used": groups_used, "elapsed": time.perf_counter() - t0,
                     "retrieval_mode": retrieval_mode,
                     "search_q": search_q if search_q != question else ""}}


# ── MLflow Tracing: ครอบ answer_stream เป็น span ราก (เปิดด้วย RAG_TRACE=1 — ดู rag.py) ──
# answer_stream เป็น generator: mlflow.trace รองรับโดยเก็บ "ทุกค่าที่ yield" เป็น output
# ซึ่งตอน streaming = token ทีละตัวเป็นร้อย ๆ event → ใช้ output_reducer คัดทิ้ง
# เหลือเฉพาะ event สรุป (stage/meta/final) — เนื้อคำตอบเต็มมีใน final อยู่แล้ว
if rag.TRACE_ENABLED:
    import mlflow

    def _reduce_stream_events(outputs: list) -> dict:
        keep = [o for o in outputs
                if isinstance(o, dict) and not (o.keys() & {"token", "reasoning"})]
        n_tok = sum(1 for o in outputs if isinstance(o, dict) and "token" in o)
        return {"n_token_events": n_tok, "events": keep}

    try:
        answer_stream = mlflow.trace(answer_stream, name="answer_stream",
                                     span_type="CHAIN",
                                     output_reducer=_reduce_stream_events)
    except TypeError:   # mlflow รุ่นที่ไม่มี output_reducer — เก็บทุก event (บวมแต่ใช้ได้)
        answer_stream = mlflow.trace(answer_stream, name="answer_stream",
                                     span_type="CHAIN")
