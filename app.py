"""المساعد القانوني الذكي — قانون التأمين الاجتماعي المصري (قانون رقم 79 لسنة 1975).

التشغيل:  streamlit run app.py

- دخول بحساب (تسجيل / تسجيل دخول) — كل مستخدم له سجل محادثات محفوظ.
- قاعدة المعرفة جاهزة في مجلد chroma_db (لا يوجد رفع ملفات).
- إجابة واحدة فقط لكل سؤال + شارات المراجع (أرقام المواد).
"""

import hashlib
import json
import os
import re
import sys
import time
import uuid
from datetime import datetime

# استيراد rag_core من نفس المجلد مهما كان مكان فتح الترمينال
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

from rag_core import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_LLM_MODEL,
    ask_question,
    build_chain,
    load_or_build_vectorstore,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_FILE = os.path.join(BASE_DIR, "users.json")
HISTORY_DIR = os.path.join(BASE_DIR, "history")
PDF_PATH = os.path.join(BASE_DIR, "data", "law-79-1975.pdf")
FIXED_TOP_K = 5  # عدد المقاطع المسترجعة ثابت داخلياً

# ============================================================
# ============================================================
GROQ_API_KEY = ""

SUGGESTED_QUESTIONS = [
    "ما هي الفئات التي يسري عليها قانون التأمين الاجتماعي؟",
    "ما هي شروط استحقاق المعاش؟",
    "ما هي شروط استحقاق الأرملة للمعاش؟",
    "ما هو الحد الأدنى للمعاش؟",
    "ما هي مدة الاشتراك المطلوبة؟",
]


# ================= الحسابات =================
def _load_users() -> dict:
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def _hash_pw(username: str, password: str) -> str:
    salt = f"low_rag::{username}::"
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def verify_user(username: str, password: str) -> bool:
    users = _load_users()
    return users.get(username, "") == _hash_pw(username, password)


def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip()
    if len(username) < 3:
        return False, "اسم المستخدم لازم 3 حروف على الأقل."
    if len(password) < 4:
        return False, "كلمة السر لازم 4 حروف على الأقل."
    users = _load_users()
    if username in users:
        return False, "اسم المستخدم ده متسجل قبل كده — سجل الدخول."
    users[username] = _hash_pw(username, password)
    _save_users(users)
    return True, "تم إنشاء الحساب ✅ — سجل الدخول دلوقتي."


# ================= سجل المحادثات (لكل مستخدم) =================
def _history_path(username: str) -> str:
    os.makedirs(HISTORY_DIR, exist_ok=True)
    safe = re.sub(r"[^\w\-@.]+", "_", username)
    return os.path.join(HISTORY_DIR, f"{safe}.json")


def load_convos(username: str) -> list:
    try:
        with open(_history_path(username), "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_convos(username: str, convos: list) -> None:
    try:
        with open(_history_path(username), "w", encoding="utf-8") as f:
            json.dump(convos, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def new_convo(title: str = "محادثة جديدة") -> dict:
    return {
        "id": uuid.uuid4().hex[:8],
        "title": title,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [],
    }


def get_current() -> dict | None:
    convos = st.session_state.get("convos", [])
    cid = st.session_state.get("current_id")
    for c in convos:
        if c["id"] == cid:
            return c
    return None


def persist() -> None:
    if st.session_state.get("user"):
        save_convos(st.session_state["user"], st.session_state.get("convos", []))


# ================= أدوات =================
def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, None)
        if value:
            return str(value).strip().strip("'\"")
    except Exception:
        pass
    return (os.environ.get(name, default) or "").strip().strip("'\"")


def clean_key(raw: str) -> str:
    if not raw:
        return ""
    return str(raw).strip().replace(" ", "").replace("\n", "").strip("'\"")


def now_time() -> str:
    return datetime.now().strftime("%I:%M %p")


def extract_articles(answer: str) -> list[str]:
    """استخراج أرقام المواد المذكورة في الإجابة لعرضها كشارات مراجع."""
    nums = re.findall(r"المادة\s*\(?\s*(\d+)\s*\)?", answer)
    seen, out = set(), []
    for n in nums:
        if n not in seen:
            seen.add(n)
            out.append(f"المادة {n}")
    return out[:8]


# ================= إعداد الصفحة والتنسيق =================
st.set_page_config(
    page_title="المساعد القانوني الذكي | قانون التأمين الاجتماعي",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&display=swap');
    html, body, [class*="st-"] { font-family: 'Cairo', sans-serif; }
    [data-testid="stMain"] .block-container {
        direction: rtl; text-align: right;
        max-width: 1400px; padding-top: 1rem;
    }
    [data-testid="stHorizontalBlock"] { direction: rtl; }
    header[data-testid="stHeader"] { background: rgba(0,0,0,0); }

    /* الشريط العلوي */
    .topbar {
        background: linear-gradient(90deg, #0a3527, #14543f);
        color: #fff; border-radius: 14px;
        padding: 0.7rem 1.2rem; margin-bottom: 1rem;
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 0.5rem;
    }
    .topbar .title { font-weight: 800; font-size: 1.25rem; }
    .topbar .law { font-size: 0.9rem; opacity: 0.9; }
    .status-pill {
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.5);
        border-radius: 20px; padding: 0.15rem 0.9rem;
        font-size: 0.85rem; font-weight: 700;
    }
    .dot { color: #4ade80; font-weight: 900; }

    /* اللوحة الجانبية */
    .side-panel {
        background: #fbf8ef;
        border: 1px solid #ece3cb;
        border-radius: 14px; padding: 1rem;
    }
    .side-head { text-align: center; margin-bottom: 0.6rem; }
    .side-head .t1 { font-weight: 800; font-size: 1.3rem; color: #0c3f31; }
    .side-head .t2 { color: #6b7280; font-size: 0.95rem; }
    .doc-card {
        background: #eef6f1; border: 1px solid #d7e7dd;
        border-radius: 12px; padding: 0.8rem; margin: 0.6rem 0;
    }
    .doc-card .d1 { font-weight: 800; color: #0c3f31; }
    .doc-card .d2 { font-size: 0.85rem; color: #4b5563; }
    .side-foot {
        text-align: center; color: #0c3f31; font-size: 0.85rem;
        border-top: 1px solid #ece3cb; margin-top: 0.8rem; padding-top: 0.6rem;
    }

    /* فقاعة سؤال المستخدم */
    .q-row { display: flex; align-items: flex-start; gap: 0.5rem; margin: 0.8rem 0 0.2rem; }
    .q-bubble {
        background: #e9f5ec; border-radius: 12px;
        padding: 0.6rem 1rem; font-weight: 600; color: #14352a;
        display: inline-block;
    }
    .avatar {
        width: 38px; height: 38px; border-radius: 50%;
        background: #dcebe1; display: inline-flex;
        align-items: center; justify-content: center; font-size: 1.2rem; flex-shrink: 0;
    }
    .msg-time { font-size: 0.75rem; color: #9ca3af; margin: 0.1rem 0 0.4rem; }

    /* كارت الإجابة */
    .a-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(12,63,49,0.06);
        line-height: 2.1; margin-bottom: 0.4rem;
    }
    .a-head { display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.4rem; }
    .a-icon {
        width: 38px; height: 38px; border-radius: 50%;
        background: #0c3f31; color: #fff;
        display: inline-flex; align-items: center; justify-content: center;
        font-size: 1.1rem;
    }
    .a-title { font-weight: 800; color: #0c3f31; font-size: 1.05rem; }
    .ref-chip {
        display: inline-block; background: #e9f5ec; color: #0c3f31;
        border: 1px solid #cfe5d6; border-radius: 8px;
        padding: 0.05rem 0.7rem; font-size: 0.82rem; font-weight: 700;
        margin: 0.15rem 0.2rem;
    }
    .ref-src { font-size: 0.8rem; color: #6b7280; margin-top: 0.3rem; }

    /* شاشة الدخول */
    .login-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 16px; padding: 2rem;
        box-shadow: 0 4px 20px rgba(12,63,49,0.08);
        max-width: 480px; margin: 4rem auto;
    }
    .login-card h2 { color: #0c3f31; text-align: center; margin-bottom: 0.2rem; }
    .login-card p { color: #6b7280; text-align: center; margin-bottom: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ================= شاشة الدخول =================
if "user" not in st.session_state:
    st.session_state["user"] = None

if st.session_state["user"] is None:
    st.markdown(
        '<div class="login-card"><h2>⚖️ المساعد القانوني الذكي</h2>'
        "<p>قانون التأمين الاجتماعي المصري — رقم 79 لسنة 1975<br>سجل الدخول لحفظ أسئلتك وسجل محادثاتك</p></div>",
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        tab = st.radio("الوضع", ["تسجيل الدخول", "حساب جديد"], horizontal=True)
        username = st.text_input("اسم المستخدم")
        password = st.text_input("كلمة السر", type="password")
        if tab == "تسجيل الدخول":
            if st.button("دخول", use_container_width=True, type="primary"):
                if verify_user(username.strip(), password):
                    st.session_state["user"] = username.strip()
                    st.session_state["convos"] = load_convos(username.strip())
                    if not st.session_state["convos"]:
                        c = new_convo()
                        st.session_state["convos"] = [c]
                    st.session_state["current_id"] = st.session_state["convos"][0]["id"]
                    st.rerun()
                else:
                    st.error("❌ اسم المستخدم أو كلمة السر غير صحيحة.")
        else:
            if st.button("إنشاء الحساب", use_container_width=True, type="primary"):
                ok, msg = register_user(username, password)
                if ok:
                    st.success(msg)
                else:
                    st.error(f"❌ {msg}")
    st.stop()

username = st.session_state["user"]
if "convos" not in st.session_state:
    st.session_state["convos"] = load_convos(username) or [new_convo()]
if "current_id" not in st.session_state or get_current() is None:
    st.session_state["current_id"] = st.session_state["convos"][0]["id"]


# ================= تحميل قاعدة المعرفة =================
@st.cache_resource(show_spinner="📚 جاري تحميل قاعدة المعرفة...")
def get_retriever_cached():
    persist_dir = os.path.join(BASE_DIR, "chroma_db")
    if not (os.path.isdir(persist_dir) and os.listdir(persist_dir)):
        raise FileNotFoundError("مجلد chroma_db غير موجود أو فارغ.")
    vectorstore = load_or_build_vectorstore(
        splits=None,
        embedding_model=DEFAULT_EMBEDDING_MODEL,
        persist_dir=persist_dir,
    )
    return vectorstore.as_retriever(search_kwargs={"k": 8}), vectorstore._collection.count()


try:
    retriever, n_chunks = get_retriever_cached()
    kb_ok = True
except Exception as e:
    retriever, n_chunks, kb_ok = None, 0, False
    kb_error = str(e)


# ================= الشريط العلوي =================
status = (
    '<span class="status-pill"><span class="dot">●</span> المستند متصل</span>'
    if kb_ok
    else '<span class="status-pill">🔴 غير متصل</span>'
)
st.markdown(
    f'<div class="topbar">'
    f'<div><span class="title">🦅 المساعد القانوني الذكي</span>'
    f' &nbsp;|&nbsp; <span class="law">قانون رقم 79 لسنة 1975</span>'
    f' &nbsp;|&nbsp; <span class="law">قانون التأمين الاجتماعي المصري</span></div>'
    f"<div>{status}</div></div>",
    unsafe_allow_html=True,
)

if not kb_ok:
    st.error(f"❌ تعذر تحميل قاعدة المعرفة: {kb_error}")
    st.stop()

# شريط أدوات: المراجع / الإعدادات / الحساب
t1, t2, t3, _sp = st.columns([1, 1, 1.4, 4])
with t1:
    if st.button("📖 المراجع", use_container_width=True):
        st.session_state["show_refs"] = not st.session_state.get("show_refs", False)
with t2:
    if st.button("⚙️ الإعدادات", use_container_width=True):
        st.session_state["show_settings"] = not st.session_state.get("show_settings", False)
with t3:
    st.caption(f"👤 {username}")

if st.session_state.get("show_settings"):
    with st.expander("⚙️ الإعدادات", expanded=True):
        st.caption(f"النموذج: {DEFAULT_LLM_MODEL}")
        if st.button("🚪 تسجيل الخروج"):
            persist()
            for k in ("user", "convos", "current_id"):
                st.session_state.pop(k, None)
            st.rerun()


def get_api_key() -> str:
    """المفتاح من الكود أولاً (GROQ_API_KEY فوق)، ثم secrets، ثم متغير البيئة."""
    return clean_key(
        (GROQ_API_KEY or "") or get_secret("GROQ_API_KEY")
    )


api_key = get_api_key()

if st.session_state.get("show_refs"):
    with st.expander("📖 مراجع المحادثة الحالية", expanded=True):
        cur = get_current()
        all_refs: list[str] = []
        if cur:
            for m in cur["messages"]:
                if m["role"] == "assistant":
                    for a in m.get("articles", []):
                        if a not in all_refs:
                            all_refs.append(a)
        if all_refs:
            st.markdown(" ".join(f'<span class="ref-chip">📄 {a}</span>' for a in all_refs),
                        unsafe_allow_html=True)
            st.caption("مصدر الإجابة: قانون التأمين الاجتماعي رقم 79 لسنة 1975")
        else:
            st.caption("لا توجد مراجع بعد — اسأل سؤالاً أولاً.")


# ================= معالجة سؤال =================
def answer_question(question: str) -> None:
    if not api_key:
        st.warning("⚠️ مفتاح Groq API مش متظبط — حطه في أول ملف app.py (سطر GROQ_API_KEY).")
        return
    cur = get_current()
    if cur is None:
        return
    t_q = now_time()
    cur["messages"].append({"role": "user", "content": question, "time": t_q})
    if len([m for m in cur["messages"] if m["role"] == "user"]) == 1:
        cur["title"] = question[:45]
    with st.spinner("⏳ جاري البحث في القانون وصياغة الإجابة..."):
        try:
            t0 = time.time()
            chain = build_chain(api_key, llm_model=DEFAULT_LLM_MODEL)
            answer, _sources = ask_question(chain, retriever, question, top_k=FIXED_TOP_K)
            elapsed = time.time() - t0
        except Exception as e:
            cur["messages"].pop()  # إزالة السؤال الذي فشل
            msg = str(e)
            low = msg.lower()
            if "401" in msg or "unauthorized" in low or "invalid api key" in low:
                st.error("🔑 مفتاح Groq API غير صحيح أو منتهي — حدّثه في أول ملف app.py (سطر GROQ_API_KEY).")
            elif "model_not_found" in low or ("404" in msg and "model" in low):
                st.error("🔄 موديل Groq اتغير — أعد تشغيل البرنامج (Ctrl+C ثم streamlit run app.py).")
            elif "timeout" in low or "timed out" in low:
                st.error("⏳ انتهت المهلة — حاول مرة أخرى.")
            else:
                st.error(f"❌ حدث خطأ: {msg[:800]}")
            persist()
            return
    cur["messages"].append({
        "role": "assistant",
        "content": answer,
        "time": now_time(),
        "elapsed": round(elapsed, 1),
        "articles": extract_articles(answer),
        "rating": None,
    })
    persist()


pending = st.session_state.pop("pending_question", None)
if pending:
    answer_question(pending)
    st.rerun()


# ================= التخطيط: محادثة + لوحة جانبية =================
cols = st.columns([1.15, 3])
side, main = cols[0], cols[1]

# ---------- اللوحة الجانبية (يمين) ----------
with side:
    st.markdown(
        '<div class="side-panel">'
        '<div class="side-head"><div style="font-size:2rem">⚖️</div>'
        '<div class="t1">القانون المصري</div>'
        '<div class="t2">قانون التأمين الاجتماعي</div></div>',
        unsafe_allow_html=True,
    )
    if st.button("➕ محادثة جديدة", use_container_width=True, type="primary"):
        c = new_convo()
        st.session_state["convos"].insert(0, c)
        st.session_state["current_id"] = c["id"]
        persist()
        st.rerun()

    st.markdown("**💡 أسئلة مقترحة**")
    for q in SUGGESTED_QUESTIONS:
        if st.button(f"→ {q}", key=f"sug_{hash(q)}", use_container_width=True):
            st.session_state["pending_question"] = q
            st.rerun()

    st.markdown(
        '<div class="doc-card"><div class="d1">📄 قانون التأمين الاجتماعي</div>'
        '<div class="d2">قانون رقم 79 لسنة 1975<br>عدد المواد: 64 مادة</div></div>',
        unsafe_allow_html=True,
    )
    if os.path.exists(PDF_PATH):
        with open(PDF_PATH, "rb") as f:
            st.download_button(
                "🔗 عرض المستند",
                data=f.read(),
                file_name="law-79-1975.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    with st.expander("🕘 سجل المحادثات", expanded=True):
        convos = st.session_state.get("convos", [])
        if not convos:
            st.caption("لا توجد محادثات بعد.")
        for c in convos[:15]:
            label = ("✅ " if c["id"] == st.session_state.get("current_id") else "") + c["title"][:35]
            if st.button(label, key=f"hist_{c['id']}", use_container_width=True):
                st.session_state["current_id"] = c["id"]
                st.rerun()

    if st.button("🗑️ مسح المحادثة", use_container_width=True):
        convos = [c for c in st.session_state.get("convos", [])
                  if c["id"] != st.session_state.get("current_id")]
        if not convos:
            convos = [new_convo()]
        st.session_state["convos"] = convos
        st.session_state["current_id"] = convos[0]["id"]
        persist()
        st.rerun()

    st.markdown('<div class="side-foot">⚖️ نحو عدالة اجتماعية أفضل</div></div>',
                unsafe_allow_html=True)

# ---------- منطقة المحادثة ----------
with main:
    cur = get_current()
    if cur is None:
        cur = new_convo()
        st.session_state["convos"] = [cur]
        st.session_state["current_id"] = cur["id"]

    if not cur["messages"]:
        st.info("👋 أهلاً بيك! اسأل أي سؤال عن قانون التأمين الاجتماعي من الأسفل أو من الأسئلة المقترحة.")

    for idx, m in enumerate(cur["messages"]):
        if m["role"] == "user":
            st.markdown(
                f'<div class="q-row"><div class="avatar">👤</div>'
                f'<div class="q-bubble">{m["content"]}</div></div>'
                f'<div class="msg-time">{m.get("time", "")}</div>',
                unsafe_allow_html=True,
            )
        else:
            articles = m.get("articles", []) or extract_articles(m["content"])
            chips = " ".join(f'<span class="ref-chip">📄 {a}</span>' for a in articles)
            st.markdown(
                f'<div class="a-card">'
                f'<div class="a-head"><span class="a-icon">⚖️</span>'
                f'<span class="a-title">الإجابة</span></div>'
                f'<div>{m["content"]}</div>'
                + (f'<div style="margin-top:0.5rem"><b>📚 المراجع</b><br>{chips}'
                    f'<div class="ref-src">🔗 مصدر الإجابة: قانون التأمين الاجتماعي رقم 79 لسنة 1975</div></div>'
                    if chips else "")
                + "</div>",
                unsafe_allow_html=True,
            )
            f1, f2, f3, f4 = st.columns([1, 1, 1, 4])
            with f1:
                up = "✅ مفيد" if m.get("rating") == "up" else "👍 مفيد"
                if st.button(up, key=f"up_{cur['id']}_{idx}"):
                    m["rating"] = "up"
                    persist()
                    st.rerun()
            with f2:
                down = "✅ غير دقيق" if m.get("rating") == "down" else "👎 غير دقيق"
                if st.button(down, key=f"down_{cur['id']}_{idx}"):
                    m["rating"] = "down"
                    persist()
                    st.rerun()
            with f3:
                if st.button("📋 نسخ", key=f"copy_{cur['id']}_{idx}"):
                    st.session_state[f"show_copy_{cur['id']}_{idx}"] = True
            with f4:
                st.caption(f"{m.get('time', '')}" + (f" • ⏱️ {m.get('elapsed', '')} ث" if m.get("elapsed") else ""))
            if st.session_state.get(f"show_copy_{cur['id']}_{idx}"):
                st.code(m["content"], language="text")

    # شريط الإدخال
    with st.form("ask_form", clear_on_submit=True):
        ic1, ic2 = st.columns([6, 1])
        with ic1:
            q_text = st.text_input(
                "سؤال",
                placeholder="اكتب سؤالك هنا... (مثال: ما هي شروط استحقاق المعاش؟)",
                label_visibility="collapsed",
            )
        with ic2:
            sent = st.form_submit_button("➤", use_container_width=True, type="primary")
    if sent and q_text and q_text.strip():
        answer_question(q_text.strip())
        st.rerun()

    st.caption("المستند المقدم: قانون التأمين الاجتماعي رقم 79 لسنة 1975 &nbsp;|&nbsp; نظام RAG مدعوم بالذكاء الاصطناعي")
