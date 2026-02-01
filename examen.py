# examen.py
# ------------------------------------------------------------------
# Mega Formation — Exams (Google Sheets Bank + Employee Builder + Admin Candidate Accounts + Admin Results)
# - Employee builds questions from UI -> saved to Google Sheets (persistent)
# - Admin creates Candidate accounts: phone + password + level + branch (single-use)
# - Candidate login: phone + password
# - Candidate never sees score; only final message
# - Admin sees results + can manage candidates
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import re, time, hashlib, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import gspread
from google.oauth2.service_account import Credentials
import gspread.exceptions as gse

# ---------------- Page config ----------------
st.set_page_config(page_title="Mega Formation — Exams", layout="wide")

# ---------------- Constants ----------------
LEVELS   = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
DEFAULT_DUR = {"A1":60, "A2":60, "B1":90, "B2":90}

PASS_MARK = 60.0  # النجاح من 60/100 (تنجم تبدّلها)

# Google Sheets: worksheets
WS_QUESTIONS  = "Questions"
WS_META       = "Meta"
WS_USERS      = "Users"
WS_CANDIDATES = "Candidates"
WS_RES_MB     = "Results_MB"
WS_RES_BZ     = "Results_BZ"

# Columns
Q_COLS = [
    "QID","Level","Section","Type","Question","Options","Answer",
    "SourceText","Mode","MaxSelect","MinWords","MaxWords","Keywords","UpdatedAt"
]
META_COLS = ["Level","Key","Value"]
USER_COLS = ["username","pass_hash","role"]
CAND_COLS = ["phone","pass_hash","level","branch","created_at","last_login_at","used_at","is_used","created_by"]
RES_COLS  = ["timestamp","name","phone","branch","level","exam_id","overall","pass",
             "Listening","Reading","Use_of_English","Writing"]

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
    import random, string
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

# ---------------- Google Sheets Layer ----------------
GSCOPE = ["https://www.googleapis.com/auth/spreadsheets"]

def gs_client():
    sa = st.secrets["gcp_service_account"]
    creds = Credentials.from_service_account_info(dict(sa), scopes=GSCOPE)
    return gspread.authorize(creds)

def gs_open():
    gc = gs_client()
    sh = gc.open_by_key(st.secrets["SPREADSHEET_ID"])
    return sh

def backoff_retry(fn, tries=5):
    last = None
    for i in range(tries):
        try:
            return fn()
        except gse.APIError as e:
            last = e
            time.sleep(0.5 * (2**i))
    raise last

def ensure_ws(sh, title: str, headers: List[str], rows=2000, cols=30):
    def _do():
        try:
            ws = sh.worksheet(title)
        except gspread.WorksheetNotFound:
            ws = sh.add_worksheet(title=title, rows=str(rows), cols=str(max(cols, len(headers)+2)))
            ws.update("1:1", [headers])
            return ws

        head = ws.row_values(1)
        if head != headers:
            ws.update("1:1", [headers])
        return ws

    return backoff_retry(_do)

def ws_to_df(ws, headers: List[str]) -> pd.DataFrame:
    def _do():
        vals = ws.get_all_values()
        if not vals or len(vals) <= 1:
            return pd.DataFrame(columns=headers)
        df = pd.DataFrame(vals[1:], columns=vals[0]).fillna("")
        for h in headers:
            if h not in df.columns:
                df[h] = ""
        return df[headers].fillna("")
    return backoff_retry(_do)

def df_overwrite_ws(ws, headers: List[str], df: pd.DataFrame):
    df = df.copy()
    for h in headers:
        if h not in df.columns:
            df[h] = ""
    df = df[headers].fillna("")

    def _do():
        ws.clear()
        ws.update("1:1", [headers])
        rows = df.values.tolist()
        if rows:
            ws.append_rows(rows, value_input_option="RAW")
    return backoff_retry(_do)

def append_row(ws, row: List[Any]):
    def _do():
        ws.append_row([str(x) if x is not None else "" for x in row], value_input_option="RAW")
    return backoff_retry(_do)

def append_rows(ws, rows: List[List[Any]]):
    def _do():
        ws.append_rows([[str(x) if x is not None else "" for x in r] for r in rows], value_input_option="RAW")
    return backoff_retry(_do)

# ---------------- Bootstrapping (worksheets + default users) ----------------
@st.cache_data(ttl=300)
def bootstrap_and_load_all() -> Dict[str, pd.DataFrame]:
    sh = gs_open()

    wsq  = ensure_ws(sh, WS_QUESTIONS,  Q_COLS)
    wsm  = ensure_ws(sh, WS_META,       META_COLS)
    wsu  = ensure_ws(sh, WS_USERS,      USER_COLS)
    wsc  = ensure_ws(sh, WS_CANDIDATES, CAND_COLS)
    wsmb = ensure_ws(sh, WS_RES_MB,     RES_COLS)
    wsbz = ensure_ws(sh, WS_RES_BZ,     RES_COLS)

    df_users = ws_to_df(wsu, USER_COLS)

    # Create default users if empty
    if df_users.empty:
        admin_u = str(st.secrets.get("DEFAULT_ADMIN_USER", "admin"))
        admin_p = str(st.secrets.get("DEFAULT_ADMIN_PASS", "megaadmin"))
        emp_u   = str(st.secrets.get("DEFAULT_EMP_USER", "employee"))
        emp_p   = str(st.secrets.get("DEFAULT_EMP_PASS", "mega123"))

        rows = [
            [admin_u, sha256(admin_p), "admin"],
            [emp_u,   sha256(emp_p),   "employee"],
        ]
        append_rows(wsu, rows)
        df_users = ws_to_df(wsu, USER_COLS)

    data = {
        "Questions":  ws_to_df(wsq,  Q_COLS),
        "Meta":       ws_to_df(wsm,  META_COLS),
        "Users":      df_users,
        "Candidates": ws_to_df(wsc,  CAND_COLS),
        "Results_MB": ws_to_df(wsmb, RES_COLS),
        "Results_BZ": ws_to_df(wsbz, RES_COLS),
    }
    return data

def reload_all():
    st.cache_data.clear()

# ---------------- Meta helpers ----------------
def meta_get(df_meta: pd.DataFrame, level: str, key: str, default="") -> str:
    sub = df_meta[
        (df_meta["Level"].astype(str).str.strip() == str(level).strip()) &
        (df_meta["Key"].astype(str).str.strip() == str(key).strip())
    ]
    if sub.empty:
        return default
    v = str(sub.iloc[-1]["Value"]).strip()
    return v if v else default

def meta_set(level: str, key: str, value: str):
    sh = gs_open()
    wsm = ensure_ws(sh, WS_META, META_COLS)
    dfm = ws_to_df(wsm, META_COLS)

    level = str(level).strip()
    key   = str(key).strip()
    value = str(value)

    dfm = dfm[~(
        (dfm["Level"].astype(str).str.strip() == level) &
        (dfm["Key"].astype(str).str.strip() == key)
    )].copy()

    dfm = pd.concat([dfm, pd.DataFrame([{"Level": level, "Key": key, "Value": value}])], ignore_index=True)
    df_overwrite_ws(wsm, META_COLS, dfm)
    reload_all()

# ---------------- Questions (Bank) ----------------
def load_exam_from_gs(level: str, df_questions: pd.DataFrame, df_meta: pd.DataFrame) -> Optional[Dict[str, Any]]:
    level = str(level).strip()
    sub = df_questions[df_questions["Level"].astype(str).str.strip() == level].copy()
    if sub.empty:
        return None

    title = meta_get(df_meta, level, "title", f"Mega Formation English Exam — {level}")
    dur   = meta_get(df_meta, level, "duration_min", str(DEFAULT_DUR.get(level, 60)))
    try:
        dur = int(float(dur))
    except Exception:
        dur = DEFAULT_DUR.get(level, 60)

    listening_audio = meta_get(df_meta, level, "listening_audio", "")
    listening_trans = meta_get(df_meta, level, "listening_transcript", "")
    reading_passage = meta_get(df_meta, level, "reading_passage", "")

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
        qid = str(r["QID"]).strip() or str(uuid.uuid4())

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
        options = ""

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
        "MaxSelect": str(max_sel) if max_sel != "" else "",
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
        "MinWords": str(int(writing.get("min_words",120))),
        "MaxWords": str(int(writing.get("max_words",150))),
        "Keywords": pipe_join(writing.get("keywords",[])),
        "UpdatedAt": now_iso(),
    }

def upsert_questions_rows(rows: List[Dict[str, Any]]):
    sh = gs_open()
    wsq = ensure_ws(sh, WS_QUESTIONS, Q_COLS)
    dfq = ws_to_df(wsq, Q_COLS)

    # map qid -> index
    idx_map = {str(r["QID"]).strip(): i for i, r in dfq.iterrows() if str(r.get("QID","")).strip()}
    for r in rows:
        qid = str(r.get("QID","")).strip()
        if not qid:
            continue
        if qid in idx_map:
            i = idx_map[qid]
            for c in Q_COLS:
                dfq.at[i, c] = r.get(c, "")
        else:
            dfq = pd.concat([dfq, pd.DataFrame([r], columns=Q_COLS)], ignore_index=True)

    df_overwrite_ws(wsq, Q_COLS, dfq.fillna(""))
    reload_all()

def delete_question_qid(qid: str):
    qid = str(qid).strip()
    if not qid:
        return
    sh = gs_open()
    wsq = ensure_ws(sh, WS_QUESTIONS, Q_COLS)
    dfq = ws_to_df(wsq, Q_COLS)
    dfq = dfq[dfq["QID"].astype(str).str.strip() != qid].copy()
    df_overwrite_ws(wsq, Q_COLS, dfq)
    reload_all()

# ---------------- Users ----------------
def verify_user(df_users: pd.DataFrame, username: str, password: str):
    u = (username or "").strip()
    ph = sha256(password or "")
    hit = df_users[(df_users["username"] == u) & (df_users["pass_hash"] == ph)]
    if hit.empty:
        return None
    return str(hit.iloc[0]["role"])

# ---------------- Candidates ----------------
def admin_create_candidate(phone: str, level: str, branch: str, created_by: str, pass_plain: Optional[str]=None) -> str:
    phone = clean_phone(phone)
    pwd = pass_plain or make_password(8)

    sh = gs_open()
    wsc = ensure_ws(sh, WS_CANDIDATES, CAND_COLS)
    df = ws_to_df(wsc, CAND_COLS)

    # overwrite existing for same phone (reset)
    df = df[df["phone"].astype(str).str.strip() != phone].copy()

    row = {
        "phone": phone,
        "pass_hash": sha256(pwd),
        "level": str(level),
        "branch": str(branch),
        "created_at": now_iso(),
        "last_login_at": "",
        "used_at": "",
        "is_used": "0",
        "created_by": str(created_by),
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)

    df_overwrite_ws(wsc, CAND_COLS, df)
    reload_all()
    return pwd

def verify_candidate_login(df_candidates: pd.DataFrame, phone: str, password: str):
    phone = clean_phone(phone)
    ph = sha256(password or "")

    df = df_candidates.copy()
    hit = df[(df["phone"].astype(str).str.strip() == phone) & (df["pass_hash"].astype(str).str.strip() == ph)]
    if hit.empty:
        return None, "Login غالط."
    row = hit.iloc[-1].to_dict()

    if str(row.get("is_used","0")) == "1":
        return None, "هذا الحساب مستعمل قبل (Single-use)."

    # update last_login_at in sheet
    sh = gs_open()
    wsc = ensure_ws(sh, WS_CANDIDATES, CAND_COLS)
    df2 = ws_to_df(wsc, CAND_COLS)
    mask = (df2["phone"].astype(str).str.strip() == phone) & (df2["pass_hash"].astype(str).str.strip() == ph)
    if mask.any():
        df2.loc[mask, "last_login_at"] = now_iso()
        df_overwrite_ws(wsc, CAND_COLS, df2)
        reload_all()

    payload = {"phone": phone, "level": str(row.get("level","B1")), "branch": str(row.get("branch","MB"))}
    return payload, None

def mark_candidate_used(phone: str):
    phone = clean_phone(phone)
    sh = gs_open()
    wsc = ensure_ws(sh, WS_CANDIDATES, CAND_COLS)
    df = ws_to_df(wsc, CAND_COLS)
    mask = (df["phone"].astype(str).str.strip() == phone)
    if mask.any():
        df.loc[mask, "is_used"] = "1"
        df.loc[mask, "used_at"] = now_iso()
        df_overwrite_ws(wsc, CAND_COLS, df)
        reload_all()

def unlock_candidate(phone: str):
    phone = clean_phone(phone)
    sh = gs_open()
    wsc = ensure_ws(sh, WS_CANDIDATES, CAND_COLS)
    df = ws_to_df(wsc, CAND_COLS)
    mask = (df["phone"].astype(str).str.strip() == phone)
    if mask.any():
        df.loc[mask, "is_used"] = "0"
        df.loc[mask, "used_at"] = ""
        df_overwrite_ws(wsc, CAND_COLS, df)
        reload_all()
        return True
    return False

def delete_candidate(phone: str):
    phone = clean_phone(phone)
    sh = gs_open()
    wsc = ensure_ws(sh, WS_CANDIDATES, CAND_COLS)
    df = ws_to_df(wsc, CAND_COLS)
    df = df[df["phone"].astype(str).str.strip() != phone].copy()
    df_overwrite_ws(wsc, CAND_COLS, df)
    reload_all()

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

# ---------------- Results (Google Sheets) ----------------
def save_result_row(branch_code: str, row: Dict[str, Any]):
    sh = gs_open()
    wsname = WS_RES_MB if branch_code == "MB" else WS_RES_BZ
    wsr = ensure_ws(sh, wsname, RES_COLS)

    out = [row.get(c, "") for c in RES_COLS]
    append_row(wsr, out)
    reload_all()

# ---------------- Session state ----------------
def init_state():
    st.session_state.setdefault("role", "candidate")  # candidate/admin/employee
    st.session_state.setdefault("user", "")
    st.session_state.setdefault("candidate_ok", False)
    st.session_state.setdefault("candidate_payload", None)
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("exam", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
init_state()

# ---------------- Load all from GS ----------------
data = bootstrap_and_load_all()
DFQ = data["Questions"]
DFM = data["Meta"]
DFU = data["Users"]
DFC = data["Candidates"]
DFR_MB = data["Results_MB"]
DFR_BZ = data["Results_BZ"]

# ---------------- Header ----------------
c1,c2 = st.columns([1,4])
with c1:
    st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — English Exams</h2>", unsafe_allow_html=True)
    st.caption("Employee builds questions → Google Sheets | Admin creates candidate passwords | Admin results")

# ---------------- Sidebar: Login ----------------
with st.sidebar:
    st.header("Login")
    tab_emp, tab_admin, tab_cand = st.tabs(["👩‍💼 Employee", "🛡️ Admin", "🎓 Candidate"])

    with tab_emp:
        eu = st.text_input("Username", key="emp_u")
        ep = st.text_input("Password", type="password", key="emp_p")
        if st.button("Login Employee", key="emp_login"):
            role = verify_user(DFU, eu, ep)
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
            role = verify_user(DFU, au, ap)
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
            payload, err = verify_candidate_login(DFC, phone, pwd)
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
            st.success("Logged out.")

# ---------------- Employee Builder ----------------
def load_exam_for_edit(level: str) -> Dict[str, Any]:
    exam = load_exam_from_gs(level, DFQ, DFM)
    if not exam:
        exam = {
            "meta": {"title": f"Mega Formation English Exam — {level}", "level": level, "duration_min": DEFAULT_DUR.get(level,60), "exam_id": f"GS_{level}"},
            "listening": {"audio_path": "", "transcript": "", "tasks": []},
            "reading": {"passage": "", "tasks": []},
            "use": {"tasks": []},
            "writing": {"prompt": "", "min_words": 120, "max_words": 150, "keywords": []}
        }
    return exam

def save_exam_to_gs(level: str, exam: Dict[str, Any]):
    rows = []
    for sec_key, sec_name in [("listening","Listening"), ("reading","Reading"), ("use","Use of English")]:
        for t in exam.get(sec_key, {}).get("tasks", []):
            rows.append(row_from_task(level, sec_name, t))
    rows.append(row_from_writing(level, exam.get("writing", {})))

    # Upsert rows into Questions
    upsert_questions_rows(rows)

    # Meta update
    meta_set(level, "title", exam["meta"].get("title", f"Mega Formation English Exam — {level}"))
    meta_set(level, "duration_min", str(exam["meta"].get("duration_min", DEFAULT_DUR.get(level,60))))
    meta_set(level, "listening_audio", exam.get("listening", {}).get("audio_path",""))
    meta_set(level, "listening_transcript", exam.get("listening", {}).get("transcript",""))
    meta_set(level, "reading_passage", exam.get("reading", {}).get("passage",""))

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
                    qid = data.get("qid")
                    if qid:
                        delete_question_qid(qid)
                    tasks.pop(idx)
                    st.warning("Deleted ⚠️")

def employee_panel():
    st.subheader("👩‍💼 Employee Panel — Builder (Google Sheets)")

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
    exam["listening"]["audio_path"] = st.text_input("Listening audio path/link (optional)",
                                                    value=exam["listening"].get("audio_path",""),
                                                    key="listen_audio_path")
    st.caption("ملاحظة: إذا تحط رابط (Google Drive direct / أي URL) ينجم يتستعمل كـ reference، أما st.audio يلزم file مباشر يخدم.")

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
        save_exam_to_gs(level, exam)
        st.success(f"Saved ✅ → Google Sheets (Level {level})")

# ---------------- Admin Panel ----------------
def admin_panel():
    st.subheader("🛡️ Admin Panel (Google Sheets)")

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
                st.code(
                    f"Phone: {p}\nPassword: {pwd}\nLevel: {lvl}\nBranch: {BRANCHES[br]}\nSingle-use: YES (locks after submit)"
                )

        st.markdown("---")
        dfc = DFC.copy()
        if dfc.empty:
            st.info("No candidates yet.")
        else:
            view = dfc.copy()
            view["status"] = view["is_used"].apply(lambda x: "USED" if str(x)=="1" else "ACTIVE")
            st.dataframe(view.sort_values("created_at", ascending=False), use_container_width=True, height=280)

            st.markdown("### Manage candidate: Unlock or Delete")
            c1,c2,c3 = st.columns([2,1,1])
            with c1:
                target = st.text_input("Phone to manage", key="adm_manage_phone")
            with c2:
                if st.button("Unlock (set ACTIVE)", key="adm_unlock"):
                    ok = unlock_candidate(target)
                    if ok:
                        st.success("Unlocked ✅")
                    else:
                        st.error("Phone not found.")
            with c3:
                if st.button("Delete", key="adm_delete"):
                    delete_candidate(target)
                    st.warning("Deleted.")

    with tab_results:
        st.markdown("### Results dashboard")
        sel_branch = st.selectbox("Branch", list(BRANCHES.keys()), key="adm_branch_sel")
        bcode = BRANCHES[sel_branch]

        df = DFR_MB.copy() if bcode == "MB" else DFR_BZ.copy()
        if df.empty:
            st.warning("No results yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
            st.download_button(
                "⬇️ Download CSV",
                df.to_csv(index=False).encode("utf-8"),
                f"results_{bcode}.csv",
                "text/csv"
            )

# ---------------- Candidate Exam ----------------
def render_candidate():
    st.subheader("🎓 Candidate Exam")

    if not st.session_state.candidate_ok:
        st.info("سجّل الدخول بالـ Phone + Password من الشريط الجانبي (يوفرهم الأدمين).")
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
            exam = load_exam_from_gs(level, DFQ, DFM)
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

    # timer
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

        ap = (L.get("audio_path","") or "").strip()
        if ap.startswith("http"):
            st.caption("Audio link saved in Google Sheets (URL).")
            st.markdown(ap)
        elif ap:
            st.caption("Audio path saved (text). If you want audio playback, use a direct URL.")
            st.markdown(ap)

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
                sel = st.multiselect(
                    f"Select up to {max_sel} {opts.get('mode','word')}(s):",
                    tokens, key=key, max_selections=max_sel
                )
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
                sel = st.multiselect(
                    f"Select up to {max_sel} {opts.get('mode','word')}(s):",
                    tokens, key=key, max_selections=max_sel
                )
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
                sel = st.multiselect(
                    f"Select up to {max_sel} {opts.get('mode','word')}(s):",
                    tokens, key=key, max_selections=max_sel
                )
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
        # compute score (hidden from candidate)
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

        # lock candidate account (single-use)
        mark_candidate_used(phone)

        # reset local session
        st.session_state.candidate_started = False
        st.session_state.deadline = None
        st.session_state.exam = None
        st.session_state.answers = {s:{} for s in SECTIONS}
        st.session_state.candidate_ok = False
        st.session_state.candidate_payload = None

        # candidate sees only this
        st.success("✅ تم الإجتياز بنجاح، سيتم التواصل معك من قبل إدارة الهيكل.")
        st.stop()

# ---------------- Router ----------------
if st.session_state.role == "employee":
    employee_panel()
elif st.session_state.role == "admin":
    admin_panel()
else:
    render_candidate()
