# examen.py
# ------------------------------------------------------------------
# Mega Formation — Exams (100% Google Sheets / Single Spreadsheet)
# - Employee builds questions -> saved to sheet: Questions + Meta
# - Admin creates Candidate accounts (single-use) -> sheet: Candidates
# - Candidate login (phone+password) -> take exam -> results saved -> Results_MB / Results_BZ
# - Admin sees results + manage candidates
#
# FIXED:
# ✅ Read quota (429) protections:
#   - Cached reads (ttl=20s)
#   - Retry/backoff on 429
#   - NO ws.clear()
#   - Chunked writes + cache invalidation after write
#
# ADDED:
# ✅ WhatsApp buttons:
#   - Admin Panel: Send candidate login/password via WhatsApp
#   - Results: Send exam result via WhatsApp
# ------------------------------------------------------------------

import os, re, time, hashlib, uuid, random, urllib.parse
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

import streamlit as st
import pandas as pd
import gspread
from gspread.exceptions import APIError
from google.oauth2.service_account import Credentials

# ---------------- Page config ----------------
st.set_page_config(page_title="Mega Formation — Exams", layout="wide")

# ---------------- Constants ----------------
LEVELS   = ["A1", "A2", "B1", "B2"]
SECTIONS = ["Listening", "Reading", "Use of English", "Writing"]
BRANCHES = {"Menzel Bourguiba": "MB", "Bizerte": "BZ"}
DEFAULT_DUR = {"A1": 60, "A2": 60, "B1": 90, "B2": 90}

PASS_MARK = 60.0  # النجاح من 60/100

# Google Sheets tab names (same Spreadsheet)
SHEET_USERS      = "Users"
SHEET_CANDIDATES = "Candidates"
SHEET_QUESTIONS  = "Questions"
SHEET_META       = "Meta"
SHEET_RES_MB     = "Results_MB"
SHEET_RES_BZ     = "Results_BZ"

# Columns
Q_COLS = [
    "QID","Level","Section","Type","Question","Options","Answer",
    "SourceText","Mode","MaxSelect","MinWords","MaxWords","Keywords","UpdatedAt"
]
META_COLS  = ["Level","Key","Value"]
USERS_COLS = ["username", "pass_hash", "role", "updated_at"]
CAND_COLS  = ["phone","pass_hash","level","branch","created_at","last_login_at","used_at","is_used","created_by"]

RESULT_COLS = [
    "timestamp","name","phone","branch","level","exam_id","overall","pass",
    "Listening","Reading","Use_of_English","Writing"
]

# Default logins (كما تحبّ)
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASS = "megaadmin"
DEFAULT_EMP_USER   = "employee"
DEFAULT_EMP_PASS   = "mega123"

# Google API scopes
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

# ---------------- Utils ----------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def clean_phone(p: str) -> str:
    """Normalize UAE-style numbers to +9715xxxxxxxx when possible."""
    p = (p or "").strip()
    p = re.sub(r"[^\d+]", "", p)

    if p.startswith("05") and len(p) == 10:
        p = "+971" + p[1:]
    elif p.startswith("5") and len(p) == 9:
        p = "+971" + p
    elif p.startswith("971") and not p.startswith("+"):
        p = "+" + p
    return p

def make_password(length=8) -> str:
    import string
    chars = string.ascii_letters + string.digits
    return "".join(random.choice(chars) for _ in range(length))

def pipe_join(val) -> str:
    if val is None:
        return ""
    if isinstance(val, list):
        return "|".join([str(x).strip() for x in val if str(x).strip()])
    return str(val).strip()

def split_pipe(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def tokenise(text: str, mode: str):
    if not text:
        return []
    if mode == "word":
        tokens = re.findall(r"\w+[\w'-]*|[.,!?;:]", text)
        return [t for t in tokens if t.strip()]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# ---------------- WhatsApp helpers ----------------
def get_app_link() -> str:
    # حطّها في secrets: APP_LINK="https://...."
    try:
        v = str(st.secrets.get("APP_LINK", "")).strip()
        return v
    except Exception:
        return ""

def wa_digits(phone: str) -> str:
    """wa.me يحب أرقام برك (من غير +)"""
    p = clean_phone(phone)
    return re.sub(r"\D", "", str(p or ""))

def wa_link(phone: str, message: str) -> str:
    p = wa_digits(phone)
    if not p:
        return ""
    return f"https://wa.me/{p}?text={urllib.parse.quote(message)}"

def build_candidate_msg(phone: str, pwd: str, level: str, branch: str) -> str:
    link = get_app_link()
    msg = (
        "👋 مرحبا! هاذم بيانات الدخول لامتحان Mega Formation:\n\n"
        f"📞 Phone: {phone}\n"
        f"🔑 Password: {pwd}\n"
        f"🎯 Level: {level}\n"
        f"🏫 Branch: {branch}\n\n"
        "✅ الحساب Single-use (مرّة برك) — بالتوفيق 🤍"
    )
    if link:
        msg += f"\n\n🔗 Link: {link}"
    return msg

def build_result_msg(row: dict) -> str:
    name = (row.get("name") or "").strip() or "Candidate"
    level = row.get("level","")
    overall = row.get("overall","")
    passed = row.get("pass","")
    L = row.get("Listening","")
    R = row.get("Reading","")
    U = row.get("Use_of_English","")
    W = row.get("Writing","")
    return (
        "📌 نتيجة الامتحان — Mega Formation\n\n"
        f"👤 الاسم: {name}\n"
        f"🎯 Level: {level}\n"
        f"🏁 Result: {passed}\n"
        f"📊 Overall: {overall}/100\n\n"
        "تفاصيل:\n"
        f"🎧 Listening: {L}\n"
        f"📖 Reading: {R}\n"
        f"🧠 Use of English: {U}\n"
        f"✍️ Writing: {W}\n\n"
        "✅ شكرا، الإدارة باش تتواصل معاك."
    )

# ---------------- Google Sheets Client ----------------
def _fix_private_key(sa: dict) -> dict:
    pk = sa.get("private_key", "")
    if pk and "\\n" in pk:
        sa["private_key"] = pk.replace("\\n", "\n")
    return sa

@st.cache_resource
def gs_client() -> gspread.Client:
    sa = dict(st.secrets["gcp_service_account"])
    sa = _fix_private_key(sa)
    creds = Credentials.from_service_account_info(sa, scopes=SCOPES)
    return gspread.authorize(creds)

@st.cache_resource
def gs_open_spreadsheet():
    gc = gs_client()
    sid = st.secrets["SPREADSHEET_ID"]
    return gc.open_by_key(sid)

def _is_429(e: Exception) -> bool:
    try:
        if isinstance(e, APIError):
            resp = getattr(e, "response", None)
            return bool(resp and resp.status_code == 429)
        return ("429" in str(e)) or ("Quota exceeded" in str(e))
    except Exception:
        return ("429" in str(e)) or ("Quota exceeded" in str(e))

def _sleep_backoff(attempt: int):
    base = min(10, 0.8 * (2 ** attempt))
    time.sleep(base + random.random() * 0.6)

def with_retry(fn, *args, **kwargs):
    last = None
    for attempt in range(7):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            last = e
            if _is_429(e) and attempt < 6:
                _sleep_backoff(attempt)
                continue
            raise
    raise last

def ws_ensure(title: str, cols: List[str]):
    sh = gs_open_spreadsheet()
    try:
        ws = sh.worksheet(title)
    except Exception:
        ws = sh.add_worksheet(title=title, rows=1000, cols=max(10, len(cols)+2))
        with_retry(ws.update, "A1", [cols])
        return ws

    header = with_retry(ws.row_values, 1) or []
    if header[:len(cols)] != cols:
        with_retry(ws.update, "A1", [cols])
    return ws

def ws_update_matrix_in_chunks(ws, start_row: int, matrix: List[List[Any]], chunk_rows: int = 350):
    total = len(matrix)
    r0 = start_row
    i = 0
    while i < total:
        chunk = matrix[i:i+chunk_rows]
        a1_row = r0 + i
        with_retry(ws.update, f"A{a1_row}", chunk)
        i += chunk_rows

def _df_from_values(values: List[List[str]], cols: List[str]) -> pd.DataFrame:
    if not values:
        return pd.DataFrame(columns=cols)

    header = values[0] if values else []
    data = values[1:] if len(values) > 1 else []
    df = pd.DataFrame(data, columns=header).fillna("")
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    return df[cols].fillna("")

# ---------------- Cached reads (FIX quota) ----------------
@st.cache_data(ttl=20, show_spinner=False)
def ws_read_df_cached(sheet_title: str, cols: tuple) -> pd.DataFrame:
    ws = ws_ensure(sheet_title, list(cols))
    values = with_retry(ws.get_all_values)
    return _df_from_values(values, list(cols))

def ws_read_df(sheet_title: str, cols: List[str]) -> pd.DataFrame:
    return ws_read_df_cached(sheet_title, tuple(cols))

def _invalidate_cache():
    try:
        ws_read_df_cached.clear()
    except Exception:
        pass

def ws_write_df(sheet_title: str, df: pd.DataFrame, cols: List[str]):
    ws = ws_ensure(sheet_title, cols)
    df = df.copy().fillna("")
    df = df.reindex(columns=cols, fill_value="")
    matrix = [cols] + df.astype(str).values.tolist()

    ws_update_matrix_in_chunks(ws, 1, matrix, chunk_rows=350)

    try:
        with_retry(ws.resize, rows=max(2, len(matrix)+5), cols=max(10, len(cols)+2))
    except Exception:
        pass

    _invalidate_cache()

def ws_append_row(sheet_title: str, row: Dict[str, Any], cols: List[str]):
    ws = ws_ensure(sheet_title, cols)
    vals = [str(row.get(c, "")) for c in cols]
    with_retry(ws.append_row, vals, value_input_option="USER_ENTERED")
    _invalidate_cache()

# ---------------- Bootstrap sheets ----------------
@st.cache_data(ttl=60, show_spinner=False)
def bootstrap_and_load_all():
    ws_ensure(SHEET_USERS, USERS_COLS)
    ws_ensure(SHEET_CANDIDATES, CAND_COLS)
    ws_ensure(SHEET_QUESTIONS, Q_COLS)
    ws_ensure(SHEET_META, META_COLS)
    ws_ensure(SHEET_RES_MB, RESULT_COLS)
    ws_ensure(SHEET_RES_BZ, RESULT_COLS)

    dfU = ws_read_df(SHEET_USERS, USERS_COLS)
    if dfU.empty:
        seed = pd.DataFrame([
            {"username": DEFAULT_ADMIN_USER, "pass_hash": DEFAULT_ADMIN_PASS, "role": "admin", "updated_at": now_iso()},
            {"username": DEFAULT_EMP_USER,   "pass_hash": DEFAULT_EMP_PASS,   "role": "employee", "updated_at": now_iso()},
        ])
        ws_write_df(SHEET_USERS, seed, USERS_COLS)
        dfU = ws_read_df(SHEET_USERS, USERS_COLS)

    dfC = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
    dfQ = ws_read_df(SHEET_QUESTIONS, Q_COLS)
    dfM = ws_read_df(SHEET_META, META_COLS)
    dfR_MB = ws_read_df(SHEET_RES_MB, RESULT_COLS)
    dfR_BZ = ws_read_df(SHEET_RES_BZ, RESULT_COLS)

    return {
        "users": dfU,
        "candidates": dfC,
        "questions": dfQ,
        "meta": dfM,
        "res_mb": dfR_MB,
        "res_bz": dfR_BZ,
    }

# ---------------- Meta helpers ----------------
def meta_get(dfM: pd.DataFrame, level: str, key: str, default=""):
    sub = dfM[(dfM["Level"].astype(str).str.strip() == level) & (dfM["Key"].astype(str).str.strip() == key)]
    if sub.empty:
        return default
    v = str(sub.iloc[-1]["Value"]).strip()
    return v if v else default

def meta_set(dfM: pd.DataFrame, level: str, key: str, value: str) -> pd.DataFrame:
    level = level.strip()
    key = key.strip()
    dfM2 = dfM.copy()
    dfM2 = dfM2[~((dfM2["Level"].astype(str).str.strip() == level) & (dfM2["Key"].astype(str).str.strip() == key))]
    dfM2 = pd.concat([dfM2, pd.DataFrame([{"Level": level, "Key": key, "Value": str(value)}])], ignore_index=True)
    ws_write_df(SHEET_META, dfM2, META_COLS)
    return ws_read_df(SHEET_META, META_COLS)

# ---------------- Exam loading from Sheets ----------------
def load_exam_from_sheets(level: str) -> Optional[Dict[str, Any]]:
    dfQ = ws_read_df(SHEET_QUESTIONS, Q_COLS)
    dfM = ws_read_df(SHEET_META, META_COLS)

    sub = dfQ[dfQ["Level"].astype(str).str.strip() == level].copy()
    if sub.empty:
        return None

    title = meta_get(dfM, level, "title", f"Mega Formation English Exam — {level}")
    dur   = meta_get(dfM, level, "duration_min", str(DEFAULT_DUR.get(level, 60)))
    try:
        dur = int(float(dur))
    except Exception:
        dur = DEFAULT_DUR.get(level, 60)

    listening_audio = meta_get(dfM, level, "listening_audio", "")
    listening_trans = meta_get(dfM, level, "listening_transcript", "")
    reading_passage = meta_get(dfM, level, "reading_passage", "")

    exam = {
        "meta": {"title": title, "level": level, "duration_min": dur, "exam_id": f"GS_{level}_{datetime.now().strftime('%Y%m%d')}"},
        "listening": {"audio_path": listening_audio, "transcript": listening_trans, "tasks": []},
        "reading": {"passage": reading_passage, "tasks": []},
        "use": {"tasks": []},
        "writing": {"prompt": "", "min_words": 120, "max_words": 150, "keywords": []}
    }

    for _, r in sub.iterrows():
        sec = str(r["Section"]).strip()
        ttype = str(r["Type"]).strip()
        qid = str(r["QID"]).strip()

        if sec == "Writing" and ttype == "writing":
            exam["writing"]["qid"] = qid
            exam["writing"]["prompt"] = str(r["Question"]).strip()
            try:
                exam["writing"]["min_words"] = int(float(r["MinWords"])) if str(r["MinWords"]).strip() else 120
                exam["writing"]["max_words"] = int(float(r["MaxWords"])) if str(r["MaxWords"]).strip() else 150
            except Exception:
                pass
            exam["writing"]["keywords"] = split_pipe(str(r["Keywords"]))
            continue

        task = {"qid": qid, "type": ttype, "q": str(r["Question"]).strip()}

        if ttype in ("radio","checkbox"):
            task["options"] = split_pipe(str(r["Options"]))
            if ttype == "radio":
                task["answer"] = str(r["Answer"]).strip()
            else:
                task["answer"] = split_pipe(str(r["Answer"]))
        elif ttype == "tfn":
            task["options"] = ["T","F","NG"]
            task["answer"] = str(r["Answer"]).strip() or "T"
        elif ttype == "text":
            task["options"] = []
            task["answer"] = split_pipe(str(r["Answer"]))
        elif ttype == "highlight":
            src = str(r["SourceText"])
            mode = str(r["Mode"] or "word").strip() or "word"
            try:
                mx = int(float(r["MaxSelect"])) if str(r["MaxSelect"]).strip() else 3
            except Exception:
                mx = 3
            task["options"] = {"text": src, "mode": mode, "max_select": mx}
            task["answer"] = split_pipe(str(r["Answer"]))
        else:
            continue

        if sec == "Listening":
            exam["listening"]["tasks"].append(task)
        elif sec == "Reading":
            exam["reading"]["tasks"].append(task)
        elif sec == "Use of English":
            exam["use"]["tasks"].append(task)

    return exam

# ---------------- Exam saving to Sheets ----------------
def row_from_task(level: str, section: str, task: Dict[str, Any]) -> Dict[str, Any]:
    qid = task.get("qid") or str(uuid.uuid4())
    task["qid"] = qid

    ttype = task.get("type","")
    q = task.get("q","")

    options = task.get("options","")
    answer  = task.get("answer","")

    source_text, mode, max_sel = "", "", ""
    if ttype == "highlight" and isinstance(options, dict):
        source_text = options.get("text","")
        mode = options.get("mode","word")
        max_sel = options.get("max_select",3)
        options = ""  # stored separately

    return {
        "QID": qid,
        "Level": level,
        "Section": section,
        "Type": ttype,
        "Question": q,
        "Options": pipe_join(options),
        "Answer": pipe_join(answer),
        "SourceText": source_text,
        "Mode": mode,
        "MaxSelect": max_sel,
        "MinWords": "",
        "MaxWords": "",
        "Keywords": "",
        "UpdatedAt": now_iso(),
    }

def row_from_writing(level: str, writing: Dict[str, Any]) -> Dict[str, Any]:
    qid = writing.get("qid") or str(uuid.uuid4())
    writing["qid"] = qid
    return {
        "QID": qid,
        "Level": level,
        "Section": "Writing",
        "Type": "writing",
        "Question": writing.get("prompt",""),
        "Options": "",
        "Answer": "",
        "SourceText": "",
        "Mode": "",
        "MaxSelect": "",
        "MinWords": int(writing.get("min_words",120)),
        "MaxWords": int(writing.get("max_words",150)),
        "Keywords": pipe_join(writing.get("keywords",[])),
        "UpdatedAt": now_iso(),
    }

def save_exam_to_sheets(level: str, exam: Dict[str, Any]):
    rows = []
    for sec_key, sec_name in [("listening","Listening"), ("reading","Reading"), ("use","Use of English")]:
        for t in exam.get(sec_key, {}).get("tasks", []):
            rows.append(row_from_task(level, sec_name, t))
    rows.append(row_from_writing(level, exam.get("writing", {})))

    dfQ = ws_read_df(SHEET_QUESTIONS, Q_COLS)
    dfQ = dfQ[dfQ["Level"].astype(str).str.strip() != level].copy()
    dfQ2 = pd.concat([dfQ, pd.DataFrame(rows, columns=Q_COLS)], ignore_index=True).fillna("")
    ws_write_df(SHEET_QUESTIONS, dfQ2, Q_COLS)

    dfM = ws_read_df(SHEET_META, META_COLS)
    dfM = meta_set(dfM, level, "title", exam["meta"].get("title", f"Mega Formation English Exam — {level}"))
    dfM = meta_set(dfM, level, "duration_min", str(exam["meta"].get("duration_min", DEFAULT_DUR.get(level,60))))
    dfM = meta_set(dfM, level, "listening_audio", exam.get("listening", {}).get("audio_path",""))
    dfM = meta_set(dfM, level, "listening_transcript", exam.get("listening", {}).get("transcript",""))
    dfM = meta_set(dfM, level, "reading_passage", exam.get("reading", {}).get("passage",""))

# ---------------- Users login ----------------
def _looks_like_sha256(x: str) -> bool:
    x = (x or "").strip().lower()
    return bool(re.fullmatch(r"[0-9a-f]{64}", x))

def verify_user(username: str, password: str) -> Optional[str]:
    df = ws_read_df(SHEET_USERS, USERS_COLS)

    u = (username or "").strip()
    pw = (password or "").strip()
    pw_hash = sha256(pw)

    hit = df[df["username"].astype(str).str.strip() == u].copy()
    if hit.empty:
        return None

    stored = str(hit.iloc[0].get("pass_hash", "")).strip()

    ok = False
    if _looks_like_sha256(stored):
        ok = (stored == pw_hash)
    else:
        ok = (stored == pw)

    if not ok:
        return None

    return str(hit.iloc[0].get("role", "")).strip() or None

# ---------------- Candidates ----------------
def admin_create_candidate(phone: str, level: str, branch: str, created_by: str, pass_plain: Optional[str]=None) -> str:
    phone = clean_phone(phone)
    pwd = pass_plain or make_password(8)

    df = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
    df = df[df["phone"].astype(str).str.strip() != phone].copy()

    row = {
        "phone": phone,
        "pass_hash": sha256(pwd),
        "level": level,
        "branch": branch,
        "created_at": now_iso(),
        "last_login_at": "",
        "used_at": "",
        "is_used": "0",
        "created_by": created_by
    }
    df2 = pd.concat([df, pd.DataFrame([row])], ignore_index=True).fillna("")
    ws_write_df(SHEET_CANDIDATES, df2, CAND_COLS)
    return pwd

def verify_candidate_login(phone: str, password: str):
    phone = clean_phone(phone)
    df = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
    ph = sha256(password or "")

    hit = df[(df["phone"].astype(str).str.strip() == phone) & (df["pass_hash"].astype(str).str.strip() == ph)]
    if hit.empty:
        return None, "Login غالط."
    row = hit.iloc[-1].to_dict()
    if str(row.get("is_used","0")) == "1":
        return None, "هذا الحساب مستعمل قبل (Single-use)."

    idx = hit.index[-1]
    df.loc[idx, "last_login_at"] = now_iso()
    ws_write_df(SHEET_CANDIDATES, df, CAND_COLS)

    payload = {"phone": phone, "level": str(row.get("level","B1")), "branch": str(row.get("branch","MB"))}
    return payload, None

def mark_candidate_used(phone: str):
    phone = clean_phone(phone)
    df = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
    mask = (df["phone"].astype(str).str.strip() == phone)
    if mask.any():
        df.loc[mask, "is_used"] = "1"
        df.loc[mask, "used_at"] = now_iso()
        ws_write_df(SHEET_CANDIDATES, df, CAND_COLS)

# ---------------- Scoring ----------------
def score_item_pct(item, user_val):
    itype = item.get("type")
    correct = item.get("answer")
    if itype in ("radio","tfn"):
        return 100.0 if (user_val is not None and user_val == correct) else 0.0
    if itype in ("checkbox","highlight"):
        corr = set(correct or [])
        sel  = set(user_val or [])
        if not corr:
            return (100.0 if not sel else 0.0)
        tp = len(corr & sel)
        fp = len(sel - corr)
        raw = (tp - 0.5*fp) / max(1, len(corr))
        return max(0.0, min(1.0, raw))*100.0
    if itype == "text":
        kws = [k.strip().lower() for k in (correct or []) if k.strip()]
        txt = (user_val or "").strip().lower()
        if not kws:
            return 100.0 if txt else 0.0
        hits = sum(1 for k in kws if k in txt)
        return (hits/len(kws))*100.0
    return 0.0

def score_section_percent(tasks, user_map):
    q_pcts = []
    for i, t in enumerate(tasks or []):
        q_pcts.append(score_item_pct(t, user_map.get(i)))
    return round(sum(q_pcts)/len(q_pcts), 1) if q_pcts else 0.0

def score_writing_pct(text, min_w, max_w, keywords):
    wc = len((text or "").split())
    base = 40 if (min_w and max_w and min_w <= wc <= max_w) else (20 if wc>0 else 0)
    hits = sum(1 for k in (keywords or []) if k.lower() in (text or "").lower())
    kw_score = min(60, hits*12)
    return float(min(100, base + kw_score)), wc, hits

# ---------------- Results ----------------
def save_result_row(branch_code: str, row: Dict[str, Any]):
    target_sheet = SHEET_RES_MB if branch_code == "MB" else SHEET_RES_BZ
    ws_append_row(target_sheet, row, RESULT_COLS)

# ---------------- Session state ----------------
def init_state():
    st.session_state.setdefault("role", "candidate")
    st.session_state.setdefault("user", "")
    st.session_state.setdefault("candidate_ok", False)
    st.session_state.setdefault("candidate_payload", None)
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("exam", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("last_candidate", None)  # ✅ last created candidate for WhatsApp
init_state()

# ---------------- Safe bootstrap (shows errors nicely) ----------------
try:
    data = bootstrap_and_load_all()
except Exception as e:
    st.error("❌ Google Sheets: فشل الاتصال.")
    st.code(str(e))
    st.info(
        "✅ Check-list:\n\n"
        "1) SPREADSHEET_ID صحيح\n"
        "2) Share للSheet مع service account email (Editor)\n"
        "3) secrets: private_key فيها \\n (والكود يصلّحهم)\n"
        "4) Sheets API + Drive API enabled\n\n"
        "إذا المشكلة Quota 429: استنى شوية أو قلّل reruns."
    )
    st.stop()
c1, c2, c3= st.columns(3)

with c1:
    st.image("mega_logo.png", width=130)

with c2:
    st.markdown(
        "<h2 style='text-align:left;margin-top:20px'>Mega Formation</h2>",
        unsafe_allow_html=True
    )

with c3:
    st.image("logo_mega.png", width=140)
# ---------------- Header ----------------
st.markdown("<h1 style='text-align:center;margin-bottom:0'>ACBPT — English Exams</h1>", unsafe_allow_html=True)
# ---------------- Sidebar: Login ----------------
with st.sidebar:
    st.header("Login")
    tab_emp, tab_admin, tab_cand = st.tabs(["👩‍💼 Employee", "🛡️ Admin", "🎓 Candidate"])

    with tab_emp:
        eu = st.text_input("Username", key="emp_u")
        ep = st.text_input("Password", type="password", key="emp_p")
        if st.button("Login Employee", key="emp_login"):
            role = verify_user(eu, ep)
            if role == "employee":
                st.session_state.role = "employee"
                st.session_state.user = eu
                st.success("Employee logged in ✅")
            else:
                st.error("Login failed.")

    with tab_admin:
        au = st.text_input("Username ", key="adm_u")
        ap = st.text_input("Password ", type="password", key="adm_p")
        if st.button("Login Admin", key="adm_login"):
            role = verify_user(au, ap)
            if role == "admin":
                st.session_state.role = "admin"
                st.session_state.user = au
                st.success("Admin logged in ✅")
            else:
                st.error("Login failed.")

    with tab_cand:
        phone = st.text_input("Phone", key="cand_phone")
        pwd   = st.text_input("Password", type="password", key="cand_pwd")
        if st.button("Login Candidate", key="cand_login"):
            payload, err = verify_candidate_login(phone, pwd)
            if err:
                st.error(err)
            else:
                st.session_state.candidate_ok = True
                st.session_state.candidate_payload = payload
                st.success("OK ✅ You can start the exam now.")

    if st.session_state.role in ("admin","employee"):
        if st.button("Logout", key="logout_any"):
            st.session_state.role = "candidate"
            st.session_state.user = ""
            st.session_state.candidate_ok = False
            st.session_state.candidate_payload = None
            st.session_state.candidate_started = False
            st.session_state.deadline = None
            st.session_state.exam = None
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.last_candidate = None
            st.success("Logged out.")

# ---------------- Employee Panel ----------------
def load_exam_for_edit(level: str) -> Dict[str, Any]:
    exam = load_exam_from_sheets(level)
    if not exam:
        exam = {
            "meta": {"title": f"Mega Formation English Exam — {level}", "level": level, "duration_min": DEFAULT_DUR.get(level,60), "exam_id": f"GS_{level}"},
            "listening": {"audio_path": "", "transcript": "", "tasks": []},
            "reading": {"passage": "", "tasks": []},
            "use": {"tasks": []},
            "writing": {"prompt": "", "min_words": 120, "max_words": 150, "keywords": []}
        }
    return exam

def render_task_editor(level: str, section_key: str, tasks: List[Dict[str, Any]], idx=None):
    TYPES = ["radio","checkbox","text","tfn","highlight"]
    MODES = ["word","sentence"]

    with st.container(border=True):
        if idx is None:
            st.subheader(f"{section_key} — Add task")
            itype = st.selectbox("Type", TYPES, key=f"{section_key}_new_type")
            q     = st.text_area("Question / Prompt", key=f"{section_key}_new_q")

            options, correct = [], None
            source_text, mode, max_sel = "", "word", 3

            if itype in ("radio","checkbox"):
                opts_raw = st.text_area("Options (one per line)", key=f"{section_key}_new_opts")
                options  = [o.strip() for o in opts_raw.splitlines() if o.strip()]
                if itype == "radio":
                    correct = st.selectbox("Correct option", options, index=0 if options else None, key=f"{section_key}_new_corr_radio")
                else:
                    correct = st.multiselect("Correct options", options, default=[], key=f"{section_key}_new_corr_ck")
            elif itype == "tfn":
                options = ["T","F","NG"]
                correct = st.selectbox("Correct", options, index=0, key=f"{section_key}_new_corr_tfn")
            elif itype == "text":
                kw_raw = st.text_input("Keywords (comma-separated)", key=f"{section_key}_new_corr_txt", placeholder="since, was, paid")
                correct = [k.strip() for k in kw_raw.split(",") if k.strip()]
            elif itype == "highlight":
                source_text = st.text_area("Source text", key=f"{section_key}_new_h_text")
                mode = st.radio("Selection unit", MODES, horizontal=True, key=f"{section_key}_new_h_mode")
                max_sel = st.number_input("Max selections", value=3, min_value=1, step=1, key=f"{section_key}_new_h_max")
                tokens = tokenise(source_text, mode)
                st.caption(f"Preview tokens = {len(tokens)}")
                correct = st.multiselect("Correct selections (exact match)", tokens, default=[], key=f"{section_key}_new_h_corr")
                options = {"text": source_text, "mode": mode, "max_select": int(max_sel)}

            if st.button("➕ Add task", key=f"{section_key}_add_btn"):
                tasks.append({
                    "qid": str(uuid.uuid4()),
                    "type": itype,
                    "q": q.strip(),
                    "options": options,
                    "answer": correct
                })
                st.success("Task added ✅")

        else:
            data = tasks[idx]
            st.subheader(f"{section_key} — Edit task #{idx+1}")

            itype = st.selectbox("Type", ["radio","checkbox","text","tfn","highlight"],
                                 index=["radio","checkbox","text","tfn","highlight"].index(data.get("type","radio")),
                                 key=f"{section_key}_edit_type_{idx}")
            q = st.text_area("Question / Prompt", value=data.get("q",""), key=f"{section_key}_edit_q_{idx}")

            options = data.get("options", [])
            correct = data.get("answer", [])

            if itype in ("radio","checkbox"):
                opts_raw = st.text_area("Options (one per line)",
                                        value="\n".join(options if isinstance(options,list) else []),
                                        key=f"{section_key}_edit_opts_{idx}")
                options  = [o.strip() for o in opts_raw.splitlines() if o.strip()]
                if itype == "radio":
                    ix = options.index(correct) if (correct in options) else (0 if options else 0)
                    correct = st.selectbox("Correct option", options, index=ix, key=f"{section_key}_edit_corr_radio_{idx}")
                else:
                    corr_default = [o for o in (correct or []) if o in options] if isinstance(correct,list) else []
                    correct = st.multiselect("Correct options", options, default=corr_default, key=f"{section_key}_edit_corr_ck_{idx}")

            elif itype == "tfn":
                options = ["T","F","NG"]
                ix = options.index(correct) if correct in options else 0
                correct = st.selectbox("Correct", options, index=ix, key=f"{section_key}_edit_corr_tfn_{idx}")

            elif itype == "text":
                kw_txt = ", ".join(correct) if isinstance(correct, list) else ""
                kw_raw = st.text_input("Keywords (comma-separated)", value=kw_txt, key=f"{section_key}_edit_corr_txt_{idx}")
                options = []
                correct = [k.strip() for k in kw_raw.split(",") if k.strip()]

            elif itype == "highlight":
                opts = options if isinstance(options, dict) else {}
                src_text = st.text_area("Source text", value=opts.get("text",""), key=f"{section_key}_edit_h_text_{idx}")
                mode = st.radio("Selection unit", ["word","sentence"],
                                index=(0 if (opts.get("mode","word")=="word") else 1),
                                horizontal=True, key=f"{section_key}_edit_h_mode_{idx}")
                max_sel = st.number_input("Max selections", value=int(opts.get("max_select",3)),
                                          min_value=1, step=1, key=f"{section_key}_edit_h_max_{idx}")
                tokens = tokenise(src_text, mode)
                st.caption(f"Preview tokens = {len(tokens)}")
                corr_default = [c for c in (correct or []) if c in tokens] if isinstance(correct, list) else []
                correct = st.multiselect("Correct selections", tokens, default=corr_default, key=f"{section_key}_edit_h_corr_{idx}")
                options = {"text": src_text, "mode": mode, "max_select": int(max_sel)}

            c1, c2 = st.columns(2)
            with c1:
                if st.button("💾 Save task", key=f"{section_key}_save_{idx}"):
                    tasks[idx] = {"qid": data.get("qid") or str(uuid.uuid4()),
                                  "type": itype, "q": q.strip(), "options": options, "answer": correct}
                    st.success("Saved ✅")
            with c2:
                if st.button("🗑️ Delete task", key=f"{section_key}_del_{idx}"):
                    tasks.pop(idx)
                    st.warning("Deleted ⚠️")

def employee_panel():
    st.subheader("👩‍💼 Employee Panel — Builder")

    level = st.selectbox("Level to edit", LEVELS, key="emp_edit_level")
    exam = load_exam_for_edit(level)

    st.markdown("#### Exam meta")
    cA,cB = st.columns([2,1])
    with cA:
        exam["meta"]["title"] = st.text_input("Title", value=exam["meta"].get("title", f"Mega Formation English Exam — {level}"), key="meta_title")
    with cB:
        exam["meta"]["duration_min"] = st.number_input("Duration (min)", min_value=10, step=5,
                                                       value=int(exam["meta"].get("duration_min", DEFAULT_DUR.get(level,60))),
                                                       key="meta_dur")

    st.markdown("#### Listening meta")
    exam["listening"]["transcript"] = st.text_area("Listening transcript (optional)",
                                                   value=exam["listening"].get("transcript",""),
                                                   key="listen_trans")
    exam["listening"]["audio_path"] = st.text_input("Listening audio filename (optional)", value=exam["listening"].get("audio_path",""))

    st.markdown("#### Reading meta")
    exam["reading"]["passage"] = st.text_area("Reading passage",
                                              value=exam["reading"].get("passage",""),
                                              key="reading_passage")

    st.markdown("---")
    st.markdown("### Listening Tasks")
    tasksL = exam["listening"]["tasks"]
    for i, t in enumerate(tasksL):
        with st.expander(f"Task {i+1} — {t.get('type','')} — {t.get('q','')[:60]}"):
            render_task_editor(level, "Listening", tasksL, idx=i)
    render_task_editor(level, "Listening", tasksL, idx=None)

    st.markdown("### Reading Tasks")
    tasksR = exam["reading"]["tasks"]
    for i, t in enumerate(tasksR):
        with st.expander(f"Task {i+1} — {t.get('type','')} — {t.get('q','')[:60]}"):
            render_task_editor(level, "Reading", tasksR, idx=i)
    render_task_editor(level, "Reading", tasksR, idx=None)

    st.markdown("### Use of English Tasks")
    tasksU = exam["use"]["tasks"]
    for i, t in enumerate(tasksU):
        with st.expander(f"Task {i+1} — {t.get('type','')} — {t.get('q','')[:60]}"):
            render_task_editor(level, "Use of English", tasksU, idx=i)
    render_task_editor(level, "Use of English", tasksU, idx=None)

    st.markdown("### Writing")
    W = exam["writing"]
    W["prompt"] = st.text_area("Writing prompt", value=W.get("prompt",""), key="w_prompt")
    c1,c2 = st.columns(2)
    W["min_words"] = c1.number_input("Min words", value=int(W.get("min_words",120)), min_value=0, step=5, key="w_min")
    W["max_words"] = c2.number_input("Max words", value=int(W.get("max_words",150)), min_value=0, step=5, key="w_max")
    kraw = st.text_input("Keywords (comma-separated)", value=", ".join(W.get("keywords",[])), key="w_kw")
    W["keywords"] = [k.strip() for k in kraw.split(",") if k.strip()]

    st.markdown("---")
    if st.button("💾 Save THIS LEVEL to Google Sheets", type="primary", key="save_level_gs"):
        save_exam_to_sheets(level, exam)
        st.success(f"Saved ✅ → Spreadsheet (Level {level})")

# ---------------- Admin Panel ----------------
def admin_panel():
    st.subheader("🛡️ Admin Panel")

    tab_cands, tab_results = st.tabs(["👥 Candidates", "📊 Results"])

    with tab_cands:
        st.markdown("### Create candidate login (Phone + Password)")
        c1,c2,c3 = st.columns(3)
        with c1:
            phone = st.text_input("Candidate phone", key="adm_cand_phone")
        with c2:
            lvl = st.selectbox("Level", LEVELS, key="adm_cand_level")
        with c3:
            br = st.selectbox("Branch", list(BRANCHES.keys()), key="adm_cand_branch")

        colA, colB = st.columns([1,1])
        with colA:
            auto_pw = st.checkbox("Auto-generate password", value=True, key="adm_auto_pw")
        with colB:
            manual_pw = st.text_input("Manual password (if not auto)", type="password", key="adm_manual_pw")

        if st.button("Create / Reset Candidate", type="primary", key="adm_create_cand"):
            p = clean_phone(phone)
            if not p:
                st.error("اكتب رقم هاتف صحيح.")
            else:
                pwd = admin_create_candidate(
                    phone=p,
                    level=lvl,
                    branch=BRANCHES[br],
                    created_by=st.session_state.user,
                    pass_plain=(None if auto_pw else manual_pw)
                )
                st.success("Candidate created ✅ (give phone+password to the candidate)")
                st.code(f"Phone: {p}\nPassword: {pwd}\nLevel: {lvl}\nBranch: {BRANCHES[br]}\nSingle-use: YES (locks after submit)")

                # ✅ خزّن آخر Candidate للزر متاع WhatsApp
                st.session_state.last_candidate = {
                    "phone": p,
                    "pwd": pwd,
                    "level": lvl,
                    "branch": BRANCHES[br],
                }

        # ✅ زر في الوسط لإرسال لوغين الممتحن بالواتساب
        if st.session_state.get("last_candidate"):
            l, m, r = st.columns([1,2,1])
            with m:
                if st.button("📲 إرسال لوغين الممتحن على WhatsApp", use_container_width=True, key="wa_send_login"):
                    c = st.session_state.get("last_candidate", {})
                    msg = build_candidate_msg(c.get("phone",""), c.get("pwd",""), c.get("level",""), c.get("branch",""))
                    url = wa_link(c.get("phone",""), msg)
                    st.code(msg)
                    if url:
                        st.markdown(f"[📲 Open WhatsApp]({url})")
                    else:
                        st.error("رقم الهاتف غير صحيح.")

        st.markdown("---")
        dfc = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
        if dfc.empty:
            st.info("No candidates yet.")
        else:
            view = dfc.copy()
            view["status"] = view["is_used"].apply(lambda x: "USED" if str(x)=="1" else "ACTIVE")
            st.dataframe(view.sort_values("created_at", ascending=False), use_container_width=True, height=280)

            st.markdown("### Unlock candidate (ACTIVE) or delete")
            c1,c2,c3 = st.columns([2,1,1])
            with c1:
                target = st.text_input("Phone to manage", key="adm_manage_phone")
            with c2:
                if st.button("Unlock", key="adm_unlock"):
                    p = clean_phone(target)
                    df = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
                    mask = (df["phone"].astype(str).str.strip() == p)
                    if mask.any():
                        df.loc[mask, "is_used"] = "0"
                        df.loc[mask, "used_at"] = ""
                        ws_write_df(SHEET_CANDIDATES, df, CAND_COLS)
                        st.success("Unlocked ✅")
                    else:
                        st.error("Phone not found.")
            with c3:
                if st.button("Delete", key="adm_delete"):
                    p = clean_phone(target)
                    df = ws_read_df(SHEET_CANDIDATES, CAND_COLS)
                    df = df[df["phone"].astype(str).str.strip() != p].copy()
                    ws_write_df(SHEET_CANDIDATES, df, CAND_COLS)
                    st.warning("Deleted.")

    with tab_results:
        st.markdown("### Results dashboard")
        sel_branch = st.selectbox("Branch", list(BRANCHES.keys()), key="adm_branch_sel")
        bcode = BRANCHES[sel_branch]
        target_sheet = SHEET_RES_MB if bcode == "MB" else SHEET_RES_BZ

        df = ws_read_df(target_sheet, RESULT_COLS)
        if df.empty:
            st.warning("No results yet.")
        else:
            df_sorted = df.sort_values("timestamp", ascending=False).copy()
            st.dataframe(df_sorted, use_container_width=True)
            st.download_button("⬇️ Download CSV", df_sorted.to_csv(index=False).encode("utf-8"), f"results_{bcode}.csv", "text/csv")

            st.markdown("### 📲 إرسال نتيجة ممتحن على WhatsApp")

            # options for selectbox
            options = []
            df_show = df_sorted.head(200)  # limit UI
            for idx, rr in df_show.iterrows():
                label = f"{rr.get('timestamp','')} | {rr.get('name','')} | {rr.get('phone','')} | {rr.get('level','')} | {rr.get('overall','')} | {rr.get('pass','')}"
                options.append((label, idx))

            if options:
                pick_label = st.selectbox("اختر ممتحن", [x[0] for x in options], key="pick_res_wa")
                pick_idx = dict(options)[pick_label]
                row = df_show.loc[pick_idx].to_dict()

                l, m, r = st.columns([1,2,1])
                with m:
                    if st.button("📩 إرسال النتيجة على WhatsApp", use_container_width=True, key="wa_send_result"):
                        msg = build_result_msg(row)
                        url = wa_link(row.get("phone",""), msg)
                        st.code(msg)
                        if url:
                            st.markdown(f"[📲 Open WhatsApp]({url})")
                        else:
                            st.error("رقم الهاتف غير موجود/غلط.")

# ---------------- Candidate Exam ----------------
def render_candidate():
    st.subheader("🎓 Candidate Exam")

    if not st.session_state.candidate_ok:
        st.info("Please select “Candidate” from the left panel and enter your login and password")
        return

    payload = st.session_state.candidate_payload or {}
    phone = payload.get("phone","")
    level = payload.get("level","B1")
    bcode = payload.get("branch","MB")

    cA, cB, cC = st.columns([1,1,1])
    with cA: st.markdown(f"**Phone**: `{phone}`")
    with cB: st.markdown(f"**Level**: **{level}**")
    with cC: st.markdown(f"**Branch**: **{bcode}**")

    name = st.text_input("Your name", key="cand_name_real")

    if not st.session_state.candidate_started:
        if st.button("▶️ Start Exam", type="primary", key="start_exam"):
            exam = load_exam_from_sheets(level)
            if not exam:
                st.error("هذا الLevel مازال موش محضّر في Google Sheets. اطلب من الموظف يحضّرو.")
                return
            st.session_state.exam = exam
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = True
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=int(exam["meta"]["duration_min"]))
            st.success("Exam started ✅")
        return

    exam = st.session_state.exam

    if st.session_state.deadline:
        left = st.session_state.deadline - datetime.utcnow()
        left_sec = max(0, int(left.total_seconds()))
        st.markdown(f"**Time left**: {left_sec//60:02d}:{left_sec%60:02d}")
        if left_sec == 0:
            st.warning("Time is up! Submit now.")

    tabs = st.tabs(SECTIONS)

    # Listening
    with tabs[0]:
        L = exam["listening"]
        if L.get("transcript"):
            st.info(L["transcript"])
        ap = L.get("audio_path","")
        if ap:
            st.caption(f"Audio filename (optional): {ap}")

        for i, t in enumerate(L.get("tasks", [])):
            key = f"L_{i}"
            ttype = t.get("type")
            if ttype == "radio":
                st.session_state.answers["Listening"][i] = st.radio(t["q"], t.get("options",[]), index=None, key=key)
            elif ttype == "checkbox":
                st.session_state.answers["Listening"][i] = st.multiselect(t["q"], t.get("options",[]), key=key)
            elif ttype == "tfn":
                st.session_state.answers["Listening"][i] = st.radio(t["q"], ["T","F","NG"], index=None, key=key)
            elif ttype == "text":
                st.session_state.answers["Listening"][i] = st.text_input(t["q"], key=key)
            elif ttype == "highlight":
                opts = t.get("options",{})
                tokens = tokenise(opts.get("text",""), opts.get("mode","word"))
                max_sel = int(opts.get("max_select",3))
                st.write(t["q"])
                sel = st.multiselect(f"Select up to {max_sel} {opts.get('mode','word')}(s):", tokens, key=key, max_selections=max_sel)
                st.session_state.answers["Listening"][i] = sel

    # Reading
    with tabs[1]:
        R = exam["reading"]
        if R.get("passage"):
            st.info(R["passage"])
        for i, t in enumerate(R.get("tasks", [])):
            key = f"R_{i}"
            ttype = t.get("type")
            if ttype == "radio":
                st.session_state.answers["Reading"][i] = st.radio(t["q"], t.get("options",[]), index=None, key=key)
            elif ttype == "checkbox":
                st.session_state.answers["Reading"][i] = st.multiselect(t["q"], t.get("options",[]), key=key)
            elif ttype == "tfn":
                st.session_state.answers["Reading"][i] = st.radio(t["q"], ["T","F","NG"], index=None, key=key)
            elif ttype == "text":
                st.session_state.answers["Reading"][i] = st.text_input(t["q"], key=key)
            elif ttype == "highlight":
                opts = t.get("options",{})
                tokens = tokenise(opts.get("text",""), opts.get("mode","word"))
                max_sel = int(opts.get("max_select",3))
                st.write(t["q"])
                sel = st.multiselect(f"Select up to {max_sel} {opts.get('mode','word')}(s):", tokens, key=key, max_selections=max_sel)
                st.session_state.answers["Reading"][i] = sel

    # Use of English
    with tabs[2]:
        U = exam["use"]
        for i, t in enumerate(U.get("tasks", [])):
            key = f"U_{i}"
            ttype = t.get("type")
            if ttype == "radio":
                st.session_state.answers["Use of English"][i] = st.radio(t["q"], t.get("options",[]), index=None, key=key)
            elif ttype == "checkbox":
                st.session_state.answers["Use of English"][i] = st.multiselect(t["q"], t.get("options",[]), key=key)
            elif ttype == "tfn":
                st.session_state.answers["Use of English"][i] = st.radio(t["q"], ["T","F","NG"], index=None, key=key)
            elif ttype == "text":
                st.session_state.answers["Use of English"][i] = st.text_input(t["q"], key=key)
            elif ttype == "highlight":
                opts = t.get("options",{})
                tokens = tokenise(opts.get("text",""), opts.get("mode","word"))
                max_sel = int(opts.get("max_select",3))
                st.write(t["q"])
                sel = st.multiselect(f"Select up to {max_sel} {opts.get('mode','word')}(s):", tokens, key=key, max_selections=max_sel)
                st.session_state.answers["Use of English"][i] = sel

    # Writing
    with tabs[3]:
        W = exam["writing"]
        if W.get("prompt"):
            st.write(W["prompt"])
        st.caption(f"Target: {W.get('min_words',0)}–{W.get('max_words',0)} words")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=220, key="W_0")

    st.markdown("---")
    if st.button("✅ Submit Exam", type="primary", key="submit_exam"):
        L_pct = score_section_percent(exam["listening"]["tasks"], st.session_state.answers["Listening"])
        R_pct = score_section_percent(exam["reading"]["tasks"], st.session_state.answers["Reading"])
        U_pct = score_section_percent(exam["use"]["tasks"], st.session_state.answers["Use of English"])

        W = exam["writing"]
        W_text = st.session_state.answers["Writing"].get(0,"")
        W_pct, wc, hits = score_writing_pct(W_text, W.get("min_words",0), W.get("max_words",0), W.get("keywords",[]))

        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)
        passed = "PASS" if overall >= PASS_MARK else "FAIL"

        row = {
            "timestamp": now_iso(),
            "name": (name or "").strip(),
            "phone": phone,
            "branch": bcode,
            "level": level,
            "exam_id": exam["meta"].get("exam_id",""),
            "overall": overall,
            "pass": passed,
            "Listening": L_pct,
            "Reading": R_pct,
            "Use_of_English": U_pct,
            "Writing": W_pct,
        }
        save_result_row(bcode, row)
        mark_candidate_used(phone)

        st.session_state.candidate_started = False
        st.session_state.deadline = None
        st.session_state.exam = None
        st.session_state.answers = {s:{} for s in SECTIONS}
        st.session_state.candidate_ok = False
        st.session_state.candidate_payload = None

        st.success("✅ تم الإجتياز بنجاح، سيتم التواصل معك من قبل إدارة الهيكل.")
        st.stop()

# ---------------- Router ----------------
if st.session_state.role == "employee":
    employee_panel()
elif st.session_state.role == "admin":
    admin_panel()
else:
    render_candidate()









