"""
Streamlit chatbot UI สำหรับ Thai Law RAG — ประมวลกฎหมายที่ดิน พ.ศ. ๒๔๙๗

รัน:  streamlit run app.py
(reuse ฟังก์ชันใน rag.py ทั้งหมด — แตะ rag.retrieve แค่เพิ่มพารามิเตอร์ groups)

ก่อนรัน ต้องตั้ง env var LLM_API_KEY:
    setx LLM_API_KEY "<token>"   แล้วเปิด terminal ใหม่

ฟีเจอร์:
  - เลือก LLM model จาก endpoint (dropdown)
  - เลือกกลุ่มเอกสารอัตโนมัติ (auto routing) หรือเลือกเองใน sidebar
  - streaming, แหล่งอ้างอิง, reasoning
  - 👍/👎 ให้คะแนน → log ลง feedback.jsonl
  - export บทสนทนาเป็น .md / คัดลอกคำตอบ
"""
import html
import json
import os
import re
import time
import urllib.request
from datetime import datetime

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import rag
import service   # เครื่องยนต์ RAG (ไม่ผูก UI) — ใช้ร่วมกับ FastAPI ได้

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Thai Law RAG",
    page_icon="⚖️",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── design system (Gridgeist — "technical register") ──────────────────────────
# thesis: เครื่องมืออ่านกฎที่แม่นยำ — พื้นผิวเงียบบนคอลัมน์อ่าน, metadata มาตรา/หน้า
#         แบบ mono เป็นลายเซ็นทางสายตา, แดง F1 ทำหน้าที่เดียว = บอก active state
# tokens: spacing 4/8/12/16/24/32px · radius 5-6px (controls) / 14px (bubble)
#         hairline 1px · ไม่มี shadow/gradient · mono = ข้อมูลเทคนิคเท่านั้น
_MONO = ("ui-monospace, 'SF Mono', 'Cascadia Mono', 'JetBrains Mono', "
         "Menlo, Consolas, monospace")

_PALETTE = {
    "dark":  {"bg": "#0E1117", "panel": "#141821", "bubble": "#1D222B",
              "text": "#E6E8EC", "sub": "#8B929E", "border": "#252A34",
              "accent": "#E10600"},     # accent แดง — ใช้เฉพาะ active state
    "light": {"bg": "#FFFFFF", "panel": "#F7F8FA", "bubble": "#F0F2F5",
              "text": "#15181E", "sub": "#5F6773", "border": "#E3E6EB",
              "accent": "#D40500"},
}


def build_theme_css(dark: bool) -> str:
    p = _PALETTE["dark" if dark else "light"]
    return f"""
    <style>
      /* ── surface ─────────────────────────────────────────────────────────── */
      .stApp {{ background: {p['bg']}; color: {p['text']}; }}
      [data-testid="stHeader"] {{ background: {p['bg']}; }}
      [data-testid="stBottomBlockContainer"], [data-testid="stBottom"] > div {{
          background: {p['bg']}; }}
      [data-testid="stSidebar"] {{ background: {p['panel']};
          border-right: 1px solid {p['border']}; }}
      /* คอลัมน์อ่าน — measure ~70ch (stBottom เป็น sticky ใน flow แล้ว จองที่ของตัวเอง
         padding ล่างแค่พอหายใจ ไม่ต้องเผื่อความสูงกล่องพิมพ์) */
      .block-container {{ padding-top: 2rem; padding-bottom: 1.5rem; max-width: 720px; }}
      .stApp p, .stApp li, .stApp label {{ color: {p['text']}; }}
      .stApp h1, .stApp h2, .stApp h3 {{ color: {p['text']}; font-weight: 600;
          letter-spacing: -.012em; }}

      /* ── inputs ──────────────────────────────────────────────────────────── */
      [data-baseweb="select"] > div, [data-baseweb="input"] > div,
      .stTextInput input, textarea, [data-testid="stChatInput"] textarea {{
          background: {p['panel']} !important; color: {p['text']} !important;
          border-color: {p['border']} !important; }}
      /* กล่องพิมพ์ = กล่องเดี่ยว (model selector อยู่ใน "ตั้งค่า" ที่ sidebar) */
      [data-testid="stChatInput"] {{ background: {p['panel']};
          border: 1px solid {p['border']}; border-radius: 10px; box-shadow: none; }}
      [data-testid="stChatInput"]:focus-within {{ border-color: {p['sub']}; }}

      /* ── welcome (empty state) — ไม่มี hero/badge/logo ────────────────────── */
      .welcome {{ padding: 2.5rem 0 .25rem; }}
      .welcome h2 {{ font-size: 1.375rem; margin: 0 0 .375rem; }}
      .welcome p {{ color: {p['sub']}; font-size: .9rem; margin: 0; max-width: 46ch; }}
      /* หัวหมวด = label เป็นมิตร (มีอีโมจิได้) + เส้นบางบอกโครง */
      .ex-cat {{ font-size: .8rem; font-weight: 550; color: {p['sub']};
          margin: 1.5rem 0 .5rem; padding-bottom: .375rem;
          border-bottom: 1px solid {p['border']}; }}

      /* ── chat ────────────────────────────────────────────────────────────── */
      [data-testid="stChatMessage"] {{ background: transparent; border-radius: 0;
          padding: .375rem 0; gap: .625rem; }}
      [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li {{ line-height: 1.7; }}
      /* user: ซ่อน avatar + บับเบิลชิดขวา */
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > :first-child {{
          display: none; }}
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) > :last-child {{
          background: {p['bubble']}; border-radius: 14px; padding: .5rem 1rem;
          max-width: 80%; width: fit-content; margin-left: auto; }}
      [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) [data-testid="stChatMessageContent"] > div > *:last-child {{
          margin-bottom: 0; }}
      /* assistant: ข้อความเปล่า */
      [data-testid="stChatMessageAvatarAssistant"] {{ background: {p['panel']};
          border: 1px solid {p['border']}; color: {p['sub']};
          width: 26px; height: 26px; font-size: .7rem; }}

      /* ── metadata (mono) = ลายเซ็นของโปรดักต์ ─────────────────────────────── */
      .meta {{ font-family: {_MONO}; font-size: .7rem; color: {p['sub']};
          letter-spacing: .02em; margin: .25rem 0 .5rem; padding-top: .375rem;
          border-top: 1px solid {p['border']}; }}
      /* แถวอ้างอิง: ไฟล์ | มาตรา | หน้า — เส้นบางจัดแนว ไม่ตีกรอบ */
      .src-row {{ display: flex; gap: 1rem; font-family: {_MONO}; font-size: .72rem;
          color: {p['sub']}; padding: .3rem 0; border-bottom: 1px solid {p['border']}; }}
      .src-row .g {{ flex: 0 0 9rem; color: {p['text']}; overflow: hidden;
          text-overflow: ellipsis; white-space: nowrap; }}
      .src-row .a {{ flex: 1; overflow: hidden; text-overflow: ellipsis;
          white-space: nowrap; }}
      .src-row .p {{ flex: 0 0 3.25rem; text-align: right; }}

      /* ── controls — เงียบ ไม่มี pill / ไม่มีแดงบน hover ───────────────────── */
      .stButton button {{ border-radius: 6px; border: 1px solid {p['border']};
          background: transparent; color: {p['text']}; font-size: .85rem;
          font-weight: 450; transition: background .12s ease, border-color .12s ease; }}
      .stButton button:hover {{ background: {p['bubble']}; border-color: {p['sub']};
          color: {p['text']}; }}

      /* ── sidebar = ลิสต์ประวัติแชท ────────────────────────────────────────── */
      .sb-brand {{ font-size: .88rem; font-weight: 600; color: {p['text']};
          letter-spacing: -.005em; padding-bottom: .625rem; margin-bottom: .75rem;
          border-bottom: 1px solid {p['border']}; }}
      [data-testid="stSidebar"] .stButton button {{ border-radius: 5px; text-align: left;
          justify-content: flex-start; font-size: .82rem; font-weight: 450;
          white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
          border-color: transparent; background: transparent; padding-left: .55rem; }}
      [data-testid="stSidebar"] .stButton button:hover {{ background: {p['bubble']}; }}
      /* ปุ่มแชทใหม่ = ปุ่มจริง (มีขอบ) แยกจากแถวลิสต์ */
      [data-testid="stSidebar"] .st-key-newchat button {{ border: 1px solid {p['border']};
          font-weight: 500; margin-bottom: .25rem; }}
      /* แชทที่เปิดอยู่ = เส้นแดงซ้าย (accent ทำหน้าที่เดียว: บอก state) */
      [data-testid="stSidebar"] .stButton button[kind="primary"] {{
          background: {p['bubble']} !important; color: {p['text']} !important;
          border: 1px solid {p['border']} !important;
          border-left: 2px solid {p['accent']} !important;
          border-radius: 5px !important; text-align: left;
          justify-content: flex-start; font-weight: 500; }}

      @media (prefers-reduced-motion: reduce) {{
        * {{ transition: none !important; animation: none !important; }}
      }}
    </style>
    """


# ธีมสว่าง/มืด: ให้ Streamlit เป็นเจ้าของ (ตั้งไว้ที่ .streamlit/config.toml → theme.light/dark)
# เพราะ widget ของมันเอง (dropdown ที่กางออก, tooltip, เมนู) CSS เราเอื้อมไม่ถึง
# ที่นี่แค่ "อ่าน" ว่าตอนนี้ธีมไหน แล้วทำ CSS ของคลาสเราเองให้สีตรงกัน
try:
    _dark = st.context.theme.type != "light"
except Exception:      # เผื่อรอบแรกที่เบราว์เซอร์ยังไม่รายงานธีม
    _dark = True
st.markdown(build_theme_css(_dark), unsafe_allow_html=True)

FEEDBACK_FILE = os.path.join(os.path.dirname(__file__), "feedback.jsonl")
HISTORY_FILE = os.path.join(os.path.dirname(__file__), "chat_history.json")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _new_cid() -> str:
    return "c" + os.urandom(8).hex()   # สุ่ม กันชนกัน (นาฬิกา Windows หยาบ ~15ms)


def _title_of(messages: list) -> str:
    """ตั้งชื่อแชทจากคำถามแรกของผู้ใช้ (ตัดให้สั้น)"""
    for m in messages:
        if m.get("role") == "user":
            t = " ".join(str(m.get("content", "")).split())
            return (t[:38] + "…") if len(t) > 38 else (t or "แชทใหม่")
    return "แชทใหม่"


def load_store() -> dict:
    """โหลด "หลายแชท" จากไฟล์ — รองรับ format เก่า (list เดียว) ด้วย"""
    try:
        with open(HISTORY_FILE, encoding="utf-8") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = None
    if isinstance(data, dict) and isinstance(data.get("conversations"), list):
        return {"conversations": data["conversations"], "current": data.get("current")}
    if isinstance(data, list) and data:            # อัปเกรดจาก format เก่า → 1 แชท
        c = {"id": _new_cid(), "title": _title_of(data), "messages": data, "updated": _now()}
        return {"conversations": [c], "current": c["id"]}
    return {"conversations": [], "current": None}


def current_conv():
    """แชทที่กำลังเปิดอยู่ (fallback = อันแรก / None ถ้าไม่มีเลย)"""
    for c in st.session_state.conversations:
        if c["id"] == st.session_state.current:
            return c
    return st.session_state.conversations[0] if st.session_state.conversations else None


def bind_current() -> None:
    """ชี้ st.session_state.messages ไปที่ list ของแชทปัจจุบัน (object เดียวกัน → append แล้วอัปเดตตาม)"""
    conv = current_conv()
    st.session_state.current = conv["id"] if conv else None
    st.session_state.messages = conv["messages"] if conv else []


def start_new_chat() -> None:
    """เปิดแชทใหม่ (ถ้าแชทปัจจุบันว่างอยู่แล้ว ใช้อันเดิม ไม่สร้างซ้ำ)"""
    cur = current_conv()
    if cur is not None and not cur["messages"]:
        st.session_state.current = cur["id"]
    else:
        c = {"id": _new_cid(), "title": "แชทใหม่", "messages": [], "updated": _now()}
        st.session_state.conversations.insert(0, c)
        st.session_state.current = c["id"]
    bind_current()


def delete_chat(cid: str) -> None:
    st.session_state.conversations = [c for c in st.session_state.conversations if c["id"] != cid]
    if not st.session_state.conversations:
        start_new_chat()
    elif st.session_state.current == cid:
        st.session_state.current = st.session_state.conversations[0]["id"]
    bind_current()
    save_store()


def save_store() -> None:
    """อัปเดต title/updated ของแชทปัจจุบัน แล้วเซฟทั้งหมดลงไฟล์ (กันหายตอน refresh)"""
    conv = current_conv()
    if conv is not None:
        conv["messages"] = st.session_state.messages
        conv["title"] = _title_of(st.session_state.messages)
        conv["updated"] = _now()
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump({"conversations": st.session_state.conversations,
                       "current": st.session_state.current}, f, ensure_ascii=False)
    except Exception as e:
        print(f"[!] เซฟประวัติแชทไม่สำเร็จ: {e}")

# ตัวอย่างคำถาม — ตรงกับประมวลกฎหมายที่ดิน พ.ศ. ๒๔๙๗ ที่โหลดอยู่ใน data/
EXAMPLE_GROUPS = {
    "⚖️ ตัวบท": [
        "มาตรา ๙ ห้ามทำอะไรในที่ดินของรัฐบ้าง",
        "ที่ดินสาธารณสมบัติของแผ่นดินถอนสภาพได้ในกรณีใดบ้าง",
        "ใครมีอำนาจหน้าที่ดูแลรักษาที่ดินของรัฐ",
    ],
    "📜 เอกสารสิทธิ": [
        "โฉนดที่ดินกับหนังสือรับรองการทำประโยชน์ต่างกันอย่างไร",
        "การออกโฉนดที่ดินมีหลักเกณฑ์อย่างไร",
    ],
    "⚠️ โทษ / ค่าธรรมเนียม": [
        "บุกรุกที่ดินของรัฐมีโทษอย่างไร",
        "ค่าธรรมเนียมการจดทะเบียนสิทธิและนิติกรรมเท่าไหร่",
    ],
    "🕐 ประวัติการแก้ไข": [
        "ประมวลกฎหมายที่ดินถูกแก้ไขมาแล้วกี่ฉบับ",
        "ฉบับที่ ๑๕ พ.ศ. ๒๕๖๒ แก้ไขเรื่องอะไร",
    ],
}

# คำอธิบายกลุ่มเอกสาร — ใช้ให้ LLM router เลือกกลุ่มที่ถูกต้อง (กลุ่มที่ไม่รู้จักใช้ชื่อกลุ่มแทน)
# GROUP_DESC ย้ายไป service.py แล้ว (ใช้ร่วมกับ FastAPI)


# ── cached backend ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)   # ใช้ loading screen ของเราเองแทน (ดูตอนเรียก)
def init_index():
    """สร้าง/โหลด index ครั้งเดียว (cache ข้ามการ rerun)"""
    changed = rag.update_database()
    rag.build_vectorstore(force=changed)


@st.cache_resource(show_spinner="🔌 กำลังเชื่อมต่อโมเดล ...")
def get_llm(model_name: str):
    """สร้าง LLM client ตามรุ่นที่เลือก — cache แยกต่อ model"""
    rag.LLM_MODEL = model_name          # build_llm() อ่าน global ตัวนี้
    return rag.build_llm()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_models() -> list[str]:
    """ดึงรายชื่อ model ที่ endpoint เปิดให้ใช้ (/v1/models) — กรอง embedding ออก"""
    try:
        req = urllib.request.Request(
            f"{rag.LLM_BASE_URL}/models",
            headers={"Authorization": f"Bearer {rag.LLM_API_KEY}"},
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
        ids = sorted(
            m["id"] for m in data.get("data", [])
            if m.get("id") and "embed" not in m["id"].lower()
        )
        if ids:
            return ids
    except Exception:
        pass
    return [rag.LLM_MODEL]


def get_groups() -> list[str]:
    """รายชื่อกลุ่มเอกสารทั้งหมดที่โหลดอยู่ (จาก chunks)"""
    return sorted({c.get("group", "") for c in rag._chunks if c.get("group")})


def get_years() -> list[int]:
    """ปีเอกสารทั้งหมดที่มีในดัชนี (>0 เท่านั้น; year=0 = ไม่ระบุปี ไม่นับ)"""
    return sorted({int(c.get("year", 0) or 0) for c in rag._chunks if c.get("year")})


# route_groups / recent_turns / rewrite_followup / build_user_prompt / GROUP_DESC
# ย้ายไป service.py ทั้งหมด (แชร์กับ FastAPI) — app.py เรียกผ่าน service.answer_stream()


# ข้อความ progress ต่อ "stage" ที่ service.answer_stream ส่งมา (เฉพาะฝั่ง Streamlit)
# ป้ายบอกว่ากำลังทำอะไร — คำถามหนึ่งใช้เวลา 15-60 วินาที ถ้าไม่บอกอะไรเลยผู้ใช้จะคิดว่าค้าง
# stage ที่ไม่มีในนี้จะแสดงข้อความดิบ (เช่น "เจอ มาตรา ๙, มาตรา ๙/๑" ที่ retrieve ส่งมา)
_STAGE_ICON = {
    "เข้าใจคำถามต่อเนื่อง": "💬",
    "เลือกกลุ่มเอกสาร": "🧭",
    "แตกคำถามเป็นหลายมุมค้นหา": "🧩",
    "เขียนคำตอบ": "✍️",
}


def stage_label(stage: str, t0: float) -> str:
    """ป้ายสถานะ + เวลาที่ผ่านไป — ตัวเลขที่เดินอยู่คือสัญญาณว่ายังทำงาน ไม่ได้ค้าง"""
    icon = next((v for k, v in _STAGE_ICON.items() if stage.startswith(k)), "🔎")
    return f"{icon} {stage}  ·  {time.time() - t0:.0f} วินาที"


def answer_question(llm, question, placeholder, status, stream,
                    auto_group, all_groups, manual_groups, year_filter=None,
                    history=None):
    """ตัวบริโภคฝั่ง Streamlit — วนรับเหตุการณ์จาก service.answer_stream() แล้วเขียนลง UI
    (logic ทั้งหมดอยู่ service.py; ตรงนี้แค่ 'แสดงผล') คืน tuple เดิมให้ caller ไม่ต้องแก้"""
    answer, chunks, reasoning, groups_used, elapsed, search_q = "", [], "", [], 0.0, ""
    t0 = time.time()
    for ev in service.answer_stream(
        llm, question, auto_group=auto_group, all_groups=all_groups,
        manual_groups=manual_groups, year_filter=year_filter,
        history=history, stream=stream,
    ):
        if "stage" in ev:
            status.update(label=stage_label(ev["stage"], t0))
        elif "token" in ev:
            answer += ev["token"]
            placeholder.markdown(answer + " ▌")
        elif "meta" in ev:
            groups_used = ev["meta"]["groups"]
            status.update(label=stage_label(
                f"อ่านตัวบท {ev['meta']['n_sources']} ก้อน จาก {', '.join(groups_used)}", t0))
        elif "final" in ev:
            f = ev["final"]
            answer, chunks, reasoning = f["answer"], f["chunks"], f["reasoning"]
            groups_used, elapsed, search_q = f["groups_used"], f["elapsed"], f["search_q"]

    placeholder.markdown(answer or "_(ไม่ได้รับคำตอบจากโมเดล)_")
    return answer, chunks, reasoning, groups_used, elapsed, search_q


# ── rendering helpers ─────────────────────────────────────────────────────────
def render_sources(chunks: list[dict]):
    """แถวอ้างอิง: ไฟล์ | มาตรา | หน้า — จัดคอลัมน์ด้วย mono เพราะเป็นข้อมูลเทคนิคจริง
    (เลขมาตรา/หน้าคือหลักฐานของโปรดักต์ ให้มันเป็นตัวเด่นแทนการตกแต่ง)"""
    if not chunks:
        return
    n_old = sum(1 for c in chunks if not c.get("in_force"))
    head = f"📄 แหล่งอ้างอิง ({len(chunks)} ชิ้น"
    head += f" · {n_old} ชิ้นเป็นตัวบทย้อนหลัง)" if n_old else ")"
    with st.expander(head):
        for c in chunks:
            # ชื่อเอกสารอ่านง่ายกว่าชื่อไฟล์ + ป้ายบอกว่าตัวบทนี้ยังใช้บังคับอยู่ไหม
            doc = html.escape(str(c.get("doc_label") or c.get("source", "?")))
            art = html.escape(str(c.get("article") or "—"))
            pg = c.get("page_start") or 0
            badge = "✅" if c.get("in_force") else "⚠️"
            st.markdown(
                f'<div class="src-row" title="{html.escape(str(c.get("source","")))}">'
                f'<span class="g">{badge} {doc}</span>'
                f'<span class="a">{art}</span>'
                f'<span class="p">{"p." + str(pg) if pg else ""}</span></div>',
                unsafe_allow_html=True,
            )
            snippet = c["text"][:600].strip()
            st.caption(snippet + (" ..." if len(c["text"]) > 600 else ""))


def log_feedback(msg: dict, rating: int):
    """บันทึกความเห็นลง feedback.jsonl (1 บรรทัด/ครั้ง)"""
    rec = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "rating": "up" if rating == 1 else "down",
        "model": msg.get("model"),
        "groups": msg.get("groups"),
        "question": msg.get("question"),
        "answer": msg.get("content"),
        "sources": [c.get("id") for c in msg.get("chunks", [])],
    }
    try:
        with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        st.warning(f"บันทึก feedback ไม่สำเร็จ: {e}")
        return False


def feedback_and_copy(idx: int, msg: dict):
    """แถว 👍/👎 + ปุ่มคัดลอกคำตอบ ใต้ข้อความ assistant แต่ละอัน"""
    fcol, ccol, _ = st.columns([2, 2, 6])
    with fcol:
        val = st.feedback("thumbs", key=f"fb_{idx}")
        if val is not None and st.session_state.get(f"fb_logged_{idx}") != val:
            if log_feedback(msg, val):
                st.session_state[f"fb_logged_{idx}"] = val
                st.toast("บันทึกความเห็นแล้ว ขอบคุณครับ 🙏")
    with ccol:
        with st.popover("📋 คัดลอก"):
            st.code(msg["content"], language="markdown", wrap_lines=True)


def conversation_to_md(messages: list[dict]) -> str:
    lines = [f"# บทสนทนา Thai Law RAG — {datetime.now():%Y-%m-%d %H:%M}", ""]
    for m in messages:
        if m["role"] == "user":
            lines.append(f"## 🙋 {m['content']}\n")
        else:
            meta = " · ".join(x for x in [m.get("model"), ", ".join(m.get("groups", []))] if x)
            lines.append(f"**🏎️ ตอบ** ({meta}):\n\n{m['content']}\n")
            if m.get("chunks"):
                srcs = ", ".join(sorted({c.get("source", "?") for c in m["chunks"]}))
                lines.append(f"> 📄 แหล่งอ้างอิง: {srcs}\n")
    return "\n".join(lines)


# ── โหลด index ก่อน (ต้องมีก่อนสร้าง sidebar เพราะใช้รายชื่อกลุ่ม) ────────────────
# ครั้งแรกใช้เวลา (อ่าน PDF + โหลดเวกเตอร์) → โชว์หน้า loading ที่บอกบริบท
# ไม่ใช่ spinner โดดๆ  รอบถัดไป cache ทำงาน = ผ่านฉับเดียว
_boot = st.empty()
try:
    with _boot.container():
        st.markdown(
            '<div class="welcome"><h2>⚖️ กฎหมายไทย RAG</h2>'
            '<p>กำลังเตรียมดัชนีเอกสาร — ครั้งแรกใช้เวลาสักครู่ '
            '(อ่าน PDF แล้วโหลดเวกเตอร์เข้าหน่วยความจำ) ครั้งต่อไปจะเปิดได้ทันที</p></div>',
            unsafe_allow_html=True,
        )
        with st.spinner("กำลังเตรียมดัชนี…"):
            init_index()
    _boot.empty()
    all_groups = get_groups()
except Exception as e:
    _boot.empty()
    st.error(f"โหลดดัชนีไม่สำเร็จ: {e}")
    st.stop()

# ── เตรียม store ประวัติแชท (หลายแชท) ก่อนสร้าง sidebar ──
if "conversations" not in st.session_state:
    _store = load_store()
    st.session_state.conversations = _store["conversations"]
    st.session_state.current = _store["current"]
    if not st.session_state.conversations:
        start_new_chat()
bind_current()   # ทุก run: ให้ st.session_state.messages ชี้แชทปัจจุบันเสมอ


# ── sidebar = ประวัติแชท (แบบ Claude) ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">⚖️ กฎหมายไทย RAG</div>', unsafe_allow_html=True)
    # แชทใหม่ = ปุ่มธรรมดา (ไม่ใช่ primary) — primary สงวนไว้บอก "แชทที่เปิดอยู่"
    if st.button("＋  แชทใหม่", use_container_width=True, key="newchat"):
        start_new_chat()
        st.rerun()

    st.caption("ประวัติแชท")
    for c in st.session_state.conversations:
        cid = c["id"]
        sel = (cid == st.session_state.current)
        lc, dc = st.columns([0.82, 0.18])
        # แชทที่เปิดอยู่บอกด้วย primary (พื้น + เส้นแดงซ้าย + ตัวหนา) ไม่ใช่สีอย่างเดียว
        if lc.button("💬 " + (c["title"] or "แชทใหม่"),
                     key=f"sel_{cid}", use_container_width=True,
                     type="primary" if sel else "secondary"):
            st.session_state.current = cid
            bind_current()
            st.rerun()
        if dc.button("🗑", key=f"del_{cid}", use_container_width=True, help="ลบแชทนี้"):
            delete_chat(cid)
            st.rerun()

    st.divider()
    with st.expander("⚙️ ตั้งค่า / ขอบเขตค้นหา"):
        st.caption("🌙 ธีมสว่าง/มืด: เมนู ☰ มุมขวาบน → Settings → Appearance "
                   "(เลือก System ให้ตามเครื่องได้)")
        _models = fetch_models()
        _didx = _models.index(rag.LLM_MODEL) if rag.LLM_MODEL in _models else 0
        selected_model = st.selectbox("🤖 LLM model", _models, index=_didx)
        if st.button("🔄 รีเฟรชรายชื่อโมเดล", use_container_width=True):
            fetch_models.clear()
            st.rerun()
        st.text_input("Embedding model", value=rag.EMBED_MODEL, disabled=True)

        st.markdown("**📁 ขอบเขตค้นหา**")
        auto_group = st.toggle(
            "🤖 เลือกกลุ่มเอกสารอัตโนมัติ", value=True,
            help="ระบบเดาเองว่าคำถามอยู่กลุ่มไหน แล้วค้นเฉพาะกลุ่มนั้น (แม่นกว่า ค้นไม่ปนกลุ่ม)",
        )
        if auto_group:
            manual_groups = None
            st.caption("กลุ่มที่มี: " + ", ".join(all_groups))
        else:
            manual_groups = st.multiselect(
                "ค้นเฉพาะกลุ่ม", all_groups, default=all_groups,
                help="เลือกกลุ่มที่ต้องการค้น (ไม่เลือก = ค้นทุกกลุ่ม)",
            )

        all_years = get_years()
        if all_years:
            year_filter = st.multiselect(
                "📅 ปีเอกสาร", all_years, default=[],
                help="เลือกปีที่ต้องการค้น (ไม่เลือก = ทุกปี) — ใช้ตอนมีข้อมูลหลายปี",
            ) or None
        else:
            year_filter = None

        stream_on = st.toggle("Streaming (โชว์คำตอบทีละคำ)", value=True)
        rerank_on = st.toggle(
            "Rerank (cross-encoder)", value=True,
            help="cross-encoder ดัน chunk ที่ตรงจริงขึ้นบน — แม่นขึ้นชัด "
                 "(query แรกช้าเพราะโหลดโมเดล bge-reranker ครั้งเดียว)",
        )
        rag.RERANK_ENABLED = rerank_on   # retrieve() อ่าน global ตอนเรียก

        if st.session_state.messages:
            st.download_button(
                "⬇️ บันทึกแชทนี้ (.md)", conversation_to_md(st.session_state.messages),
                file_name=f"chat_{datetime.now():%Y%m%d_%H%M}.md",
                mime="text/markdown", use_container_width=True,
            )
        u = rag._token_usage
        if u["calls"]:
            st.caption(
                f"Token สะสม — prompt {u['prompt']:,} · completion {u['completion']:,} "
                f"· รวม {u['total']:,} ({u['calls']} calls)"
            )


# ── main ──────────────────────────────────────────────────────────────────────
prompt = None
if st.session_state.messages:
    # โหมดสนทนา: ให้บทสนทนาเป็นตัวเด่น (ไม่มี header ใหญ่ — brand อยู่ที่ sidebar แล้ว)
    with st.popover("💡 คำถามตัวอย่าง", use_container_width=False):
        for cat, exs in EXAMPLE_GROUPS.items():
            st.caption(cat)
            for i, ex in enumerate(exs):
                if st.button(ex, use_container_width=True, key=f"cex_{cat}_{i}"):
                    prompt = ex
    for idx, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"], avatar="🏎️" if msg["role"] == "assistant" else None):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # meta = บรรทัดเดียว เงียบๆ (แทน pill เกลื่อน) — คงอีโมจิบอกชนิดไว้
                bits = []
                if msg.get("model"):
                    bits.append(f'🤖 {msg["model"]}')
                bits += [f"🔎 {g}" for g in msg.get("groups", [])]
                bits += [f"📅 {y}" for y in msg.get("years", [])]
                if msg.get("search_q"):      # คำถามต่อเนื่องที่ถูกตีความใหม่ก่อนค้น
                    bits.append(f'↻ ค้นด้วย “{html.escape(str(msg["search_q"]))}”')
                if bits:
                    st.markdown(f'<div class="meta">{" · ".join(bits)}</div>',
                                unsafe_allow_html=True)
                if msg.get("reasoning") and msg["reasoning"] != msg["content"]:
                    with st.expander("💭 การคิด (reasoning)"):
                        st.markdown(msg["reasoning"])
                render_sources(msg.get("chunks", []))
                feedback_and_copy(idx, msg)
else:
    # หน้า welcome — ชิดซ้ายตามคอลัมน์อ่าน, ไม่มี badge โฆษณาฟีเจอร์
    # (ให้ "คำถามตัวอย่าง" เป็นตัวเด่นแทน เพราะมันคือ next action จริง)
    st.markdown(
        '<div class="welcome"><h2>⚖️ ถามกฎหมายไทยได้เลย</h2>'
        '<p>ค้นจากตัวบทกฎหมายจริง ตอบพร้อมอ้างอิงมาตราและเลขหน้า '
        '— จากราชกิจจานุเบกษา (พ.ร.บ. / กฎกระทรวง / ประกาศ)</p></div>',
        unsafe_allow_html=True,
    )
    for cat, exs in EXAMPLE_GROUPS.items():
        st.markdown(f'<div class="ex-cat">{cat}</div>', unsafe_allow_html=True)
        cols = st.columns(len(exs))
        for i, ex in enumerate(exs):
            if cols[i].button(ex, use_container_width=True, key=f"ex_{cat}_{i}"):
                prompt = ex

# สร้าง LLM client ตาม model ที่เลือกใน sidebar (sidebar รันก่อน → selected_model พร้อมแล้ว)
try:
    llm = get_llm(selected_model)
except Exception as e:
    st.error(
        f"เชื่อมต่อโมเดลไม่สำเร็จ: {e}\n\n"
        "ตรวจว่าได้ตั้ง env var **LLM_API_KEY** แล้ว "
        '(`setx LLM_API_KEY \"<token>\"` แล้วเปิด terminal ใหม่)'
    )
    st.stop()

typed = st.chat_input("พิมพ์คำถาม ...")
if typed:
    prompt = typed

# ── ตอบคำถาม ──────────────────────────────────────────────────────────────────
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar="🏎️"):
        placeholder = st.empty()
        with st.status("🚦 เริ่มประมวลผล...", expanded=False) as status:
            try:
                # ประวัติ = ทุกข้อความก่อนหน้า (คำถามปัจจุบันถูก append ไปแล้วด้านบน)
                _history = st.session_state.messages[:-1]
                answer, chunks, reasoning, groups_used, elapsed, search_q = answer_question(
                    llm, prompt, placeholder, status, stream_on,
                    auto_group, all_groups, manual_groups, year_filter, _history,
                )
                status.update(
                    label=f"✓ เสร็จใน {elapsed:.1f}s · [{', '.join(groups_used)}]",
                    state="complete", expanded=False,
                )
            except Exception as e:
                answer, chunks, reasoning, groups_used = f"เกิดข้อผิดพลาด: {e}", [], "", []
                search_q = prompt
                placeholder.error(answer)
                status.update(label="เกิดข้อผิดพลาด", state="error")

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "chunks": chunks,
        "reasoning": reasoning, "model": selected_model,
        "groups": groups_used, "years": year_filter or [], "question": prompt,
        # เก็บเฉพาะตอนที่คำถามถูกตีความใหม่ → โชว์ให้ผู้ใช้เห็นว่าค้นด้วยอะไรจริง
        "search_q": search_q if search_q != prompt else "",
    })
    save_store()   # เซฟทั้ง store (กันหายตอน refresh)
    st.rerun()
