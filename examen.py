# examen.py
# ------------------------------------------------------------------
# Mega Formation — Exams (Google Sheets 100% / Spreadsheet واحد)
# - Employee builds questions from UI -> saved to Google Sheets (Questions+Meta)
# - Admin creates Candidate accounts (single-use) -> Candidates sheet
# - Candidate login -> single-use lock after submit
# - Candidate never sees score; only final message
# - Admin sees results per branch -> Results_MB / Results_BZ sheets
#
# FIXED:
# - Robust Service Account loading from Streamlit secrets
# - Fix private_key newline issues (\n)
# - Add Drive scope + Sheets scope
# - Retry/backoff on API errors
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import re, time, hashlib, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

import gspread
import gspread.exceptions as gse
from google.oauth2.service_account import Credentials

# ---------------- Page config ----------------
st.set_page_config(page_title="Mega Formation — Exams", layout="wide")

# ---------------- Constants ----------------
LEVELS   = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
DEFAULT_DUR = {"A1":60, "A2":60, "B1":90, "B2":90}

PASS_MARK = 60.0  # النجاح من 60/100

# Google Sheets tabs
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
USERS_COLS = ["username","pass_hash","role"]
CAND_COLS = ["phone","pass_hash","level","branch","created_at","last_login_at","used_at","is_used","created_by"]
RESULT_COLS = ["timestamp","name","phone","branch","level","exam_id","overall","pass",
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
        p = "+971" + p[1:]          # 05xxxxxxxx -> +9715xxxxxxxx
    elif p.startswith("5") and len(p) == 9:
        p = "+971" + p              # 5xxxxxxxx -> +9715xxxxxxxx
    elif p.startswith("971") and not p.startswith("+"):
        p = "+" + p                 # 9715xxxxxxx -> +9715xxxxxxx

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

# ---------------- Google Sheets Auth (FIXED) ----------------
SCOPE = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive",
]

def _load_sa_info_from_secrets() -> dict:
    sa = st.secrets.get("gcp_service_account")
    if not sa:
        raise RuntimeError("Missing secrets: gcp_service_account")

    sa_info = dict(sa)

    # Fix newline issues in private_key
    pk = sa_info.get("private_key", "")
    if isinstance(pk, str):
        sa_info["private_key"] = pk.replace("\\n", "\n")

    # Some deployments omit token_uri; ensure it exists
    if not sa_info.get("token_uri"):
        sa_info["token_uri"] = "https://oauth2.googleapis.com/token"

    return sa_info

@st.cache_resource
def gs_client():
    sa_info = _load_sa_info_from_secrets()
    creds = Credentials.from_service_account_info(sa_info, scopes=SCOPE)
    return gspread.authorize(creds)

def _retry(call, tries=6, base_sleep=0.6):
    last = None
    for i in range(tries):
        try:
            return call()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2**i))
    raise last

def gs_open():
    gc = gs_client()
    sid = st.secrets.get("SPREADSHEET_ID", "").strip()
    if not sid:
        raise RuntimeError("Missing secrets: SPREADSHEET_ID")
    # Open with retry/backoff
    return _retry(lambda: gc.open_by_key(sid))

def ws_get_or_create(sh, title: str, headers: List[str], rows=2000, cols=30):
    def _open():
        return sh.worksheet(title)

    try:
        ws = _retry(_open)
    except gspread.WorksheetNotFound:
        ws = _retry(lambda: sh.add_worksheet(title=title, rows=str(rows), cols=str(cols)))
        _retry(lambda: ws.update("1:1", [headers]))
        return ws

    # Ensure header
    current = ws.row_values(1)
    if current != headers:
        _retry(lambda: ws.update("1:1", [headers]))
    return ws

def ws_to_df(ws, headers: List[str]) -> pd.DataFrame:
    vals = _retry(lambda: ws.get_all_values())
    if not vals or len(vals) <= 1:
        return pd.DataFrame(columns=headers)

    file_header = vals[0]
    data = vals[1:]

    # Map columns by header name (robust)
    hmap = {str(h).strip(): i for i, h in enumerate(file_header)}
    fixed = []
    for r in data:
        r = list(r or [])
        row = []
        for col in headers:
            ix = hmap.get(col)
            row.append(r[ix] if (ix is not None and ix < len(r)) else "")
        fixed.append(row)

    df = pd.DataFrame(fixed, columns=headers).fillna("")
    return df

def df_to_ws(ws, df: pd.DataFrame, headers: List[str]):
    df = df.copy()
    for c in headers:
        if c not in df.columns:
            df[c] = ""
    df = df[headers].fillna("")
    values = [headers] + df.astype(str).values.tolist()

    # Clear then update
    _retry(lambda: ws.clear())
    _retry(lambda: ws.update(values))

# ---------------- Bootstrap: create all sheets ----------------
@st.cache_data(ttl=300)
def bootstrap_and_load_all() -> Dict[str, Any]:
    sh = gs_open()

    ws_q  = ws_get_or_create(sh, WS_QUESTIONS,  Q_COLS)
    ws_m  = ws_get_or_create(sh, WS_META,       META_COLS)
    ws_u  = ws_get_or_create(sh, WS_USERS,      USERS_COLS)
    ws_c  = ws_get_or_create(sh, WS_CANDIDATES, CAND_COLS)
    ws_r1 = ws_get_or_create(sh, WS_RES_MB,     RESULT_COLS)
    ws_r2 = ws_get_or_create(sh, WS_RES_BZ,     RESULT_COLS)

    dfq = ws_to_df(ws_q, Q_COLS)
    dfm = ws_to_df(ws_m, META_COLS)
    dfu = ws_to_df(ws_u, USERS_COLS)
    dfc = ws_to_df(ws_c, CAND_COLS)

    # Seed Users if empty
    if dfu.empty:
        admin_user = str(st.secrets.get("ADMIN_USER","admin"))
        admin_pass = str(st.secrets.get("ADMIN_PASS","megaadmin"))
        emp_user   = str(st.secrets.get("EMP_USER","employee"))
        emp_pass   = str(st.secrets.get("EMP_PASS","mega123"))

        seed = pd.DataFrame([
            {"username":admin_user, "pass_hash":sha256(admin_pass), "role":"admin"},
            {"username":emp_user,   "pass_hash":sha256(emp_pass),   "role":"employee"},
        ], columns=USERS_COLS)
        df_to_ws(ws_u, seed, USERS_COLS)
        dfu = seed

    return {
        "sh": sh,
        "ws_q": ws_q, "ws_m": ws_m, "ws_u": ws_u, "ws_c": ws_c, "ws_r_mb": ws_r1, "ws_r_bz": ws_r2,
        "dfq": dfq, "dfm": dfm, "dfu": dfu, "dfc": dfc,
    }

def reload_data():
    st.cache_data.clear()

# ---------------- Bank (Questions + Meta) ----------------
def load_bank_questions(data) -> pd.DataFrame:
    return data["dfq"].copy().fillna("")

def load_bank_meta(data) -> pd.DataFrame:
    return data["dfm"].copy().fillna("")

def save_bank(data, dfq: pd.DataFrame, dfm: pd.DataFrame):
    df_to_ws(data["ws_q"], dfq, Q_COLS)
    df_to_ws(data["ws_m"], dfm, META_COLS)

def meta_get(data, level: str, key: str, default=""):
    dfm = load_bank_meta(data)
    sub = dfm[(dfm["Level"].astype(str).str.strip() == level) & (dfm["Key"].astype(str).str.strip() == key)]
    if sub.empty:
        return default
    v = str(sub.iloc[-1]["Value"]).strip()
    return v if v else default

def meta_set(data, level: str, key: str, value: str):
    dfm = load_bank_meta(data)
    level = level.strip()
    key = key.strip()
    dfm = dfm[~((dfm["Level"].astype(str).str.strip() == level) & (dfm["Key"].astype(str).str.strip() == key))]
    dfm = pd.concat([dfm, pd.DataFrame([{"Level":level, "Key":key, "Value":str(value)}])], ignore_index=True)
    dfq = load_bank_questions(data)
    save_bank(data, dfq, dfm)

def upsert_questions(data, rows: List[Dict[str, Any]]):
    dfq = load_bank_questions(data)
    if dfq.empty:
        dfq2 = pd.DataFrame(rows, columns=Q_COLS).fillna("")
        dfm = load_bank_meta(data)
        save_bank(data, dfq2, dfm)
        return

    idx_map = {str(r["QID"]): i for i, r in dfq.iterrows() if str(r.get("QID","")).strip()}
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

    dfm = load_bank_meta(data)
    save_bank(data, dfq.fillna(""), dfm.fillna(""))

def delete_question_qid(data, qid: str):
    dfq = load_bank_questions(data)
    dfq = dfq[dfq["QID"].astype(str).str.strip() != str(qid).strip()].copy()
    dfm = load_bank_meta(data)
    save_bank(data, dfq, dfm)

def load_exam_from_gs(data, level: str) -> Optional[Dict[str, Any]]:
    dfq = load_bank_questions(data)
    sub = dfq[dfq["Level"].astype(str).str.strip() == level].copy()
    if sub.empty:
        return None

    title = meta_get(data, level, "title", f"Mega Formation English Exam — {level}")
    dur   = meta_get(data, level, "duration_min", str(DEFAULT_DUR.get(level, 60)))
    try:
        dur = int(float(dur))
    except Exception:
        dur = DEFAULT_DUR.get(level, 60)

    listening_audio = meta_get(data, level, "listening_audio", "")
    listening_trans = meta_get(data, level, "listening_transcript", "")
    reading_passage = meta_get(data, level, "reading_passage", "")

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

# ---------------- Users (Admin/Employee) ----------------
def load_users(data) -> pd.DataFrame:
    return data["dfu"].copy().fillna("")

def verify_user(data, username: str, password: str):
    df = load_users(data)
    u = (username or "").strip()
    ph = sha256(password or "")
    hit = df[(df["username"] == u) & (df["pass_hash"] == ph)]
    if hit.empty:
        return None
    return str(hit.iloc[0]["role"])

# ---------------- Candidates ----------------
def load_candidates(data) -> pd.DataFrame:
    return data["dfc"].copy().fillna("")

def save_candidates(data, df: pd.DataFrame):
    df_to_ws(data["ws_c"], df, CAND_COLS)

def admin_create_candidate(data, phone: str, level: str, branch: str, created_by: str, pass_plain: Optional[str]=None) -> str:
    phone = clean_phone(phone)
    pwd = pass_plain or make_password(8)
    df = load_candidates(data)

    # overwrite existing for same phone (reset)
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
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_candidates(data, df)
    return pwd

def verify_candidate_login(data, phone: str, password: str):
    phone = clean_phone(phone)
    df = load_candidates(data)
    ph = sha256(password or "")
    hit = df[(df["phone"].astype(str).str.strip() == phone) & (df["pass_hash"].astype(str).str.strip() == ph)]
    if hit.empty:
        return None, "Login غالط."
    row = hit.iloc[-1].to_dict()

    if str(row.get("is_used","0")) == "1":
        return None, "هذا الحساب مستعمل قبل (Single-use)."

    # update last_login_at
    idx = hit.index[-1]
    df.loc[idx, "last_login_at"] = now_iso()
    save_candidates(data, df)

    payload = {"phone": phone, "level": str(row.get("level","B1")), "branch": str(row.get("branch","MB"))}
    return payload, None

def mark_candidate_used(data, phone: str):
    phone = clean_phone(phone)
    df = load_candidates(data)
    mask = (df["phone"].astype(str).str.strip() == phone)
    if mask.any():
        df.loc[mask, "is_used"] = "1"
        df.loc[mask, "used_at"] = now_iso()
        save_candidates(data, df)

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
def results_ws(data, branch_code: str):
    return data["ws_r_mb"] if branch_code == "MB" else data["ws_r_bz"]

def results_df_from_ws(data, branch_code: str) -> pd.DataFrame:
    ws = results_ws(data, branch_code)
    return ws_to_df(ws, RESULT_COLS)

def save_result_row(data, branch_code: str, row: Dict[str, Any]):
    ws = results_ws(data, branch_code)
    df = ws_to_df(ws, RESULT_COLS)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df_to_ws(ws, df, RESULT_COLS)

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

# ---------------- Load / Bootstrap ----------------
try:
    data = bootstrap_and_load_all()
except Exception as e:
    st.error("❌ فشل الاتصال بـ Google Sheets.")
    st.code(str(e))
    st.info("✅ Check-list سريع:\n"
            "1) SPREADSHEET_ID صحيح\n"
            "2) Share للـ Sheet مع client_email (Editor)\n"
            "3) private_key في secrets فيه \\n\n"
            "4) فعّل Sheets API + Drive API\n")
    st.stop()

# ---------------- Header ----------------
c1,c2 = st.columns([1,4])
with c1:
    st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — English Exams (Google Sheets)</h2>", unsafe_allow_html=True)
    st.caption("Employee builds questions → Google Sheets | Admin creates candidate passwords | Admin results")

# ---------------- Sidebar: Login ----------------
with st.sidebar:
    st.header("Login")

    tab_emp, tab_admin, tab_cand = st.tabs(["👩‍💼 Employee", "🛡️ Admin", "🎓 Candidate"])

    with tab_emp:
        eu = st.text_input("Username", key="emp_u")
        ep = st.text_input("Password", type="password", key="emp_p")
        if st.button("Login Employee", key="emp_login"):
            role = verify_user(data, eu, ep)
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
            role = verify_user(data, au, ap)
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
            payload, err = verify_candidate_login(data, phone, pwd)
            if err:
                st.error(err)
            else:
                st.session_state.candidate_ok = True
                st.session_state.candidate_payload = payload
                st.success("OK ✅ You can start the exam now.")

    if st.session_state.role in ("admin","employee") or st.session_state.candidate_ok:
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

    st.markdown("---")
    if st.button("🔄 Refresh from Google Sheets"):
        reload_data()
        st.rerun()

# ---------------- Builder helpers ----------------
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

def load_exam_for_edit(level: str) -> Dict[str, Any]:
    exam = load_exam_from_gs(data, level)
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

    upsert_questions(data, rows)

    meta_set(data, level, "title", exam["meta"].get("title", f"Mega Formation English Exam — {level}"))
    meta_set(data, level, "duration_min", str(exam["meta"].get("duration_min", DEFAULT_DUR.get(level,60))))
    meta_set(data, level, "listening_audio", exam.get("listening", {}).get("audio_path",""))
    meta_set(data, level, "listening_transcript", exam.get("listening", {}).get("transcript",""))
    meta_set(data, level, "reading_passage", exam.get("reading", {}).get("passage",""))

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
            data_t = tasks[idx]
            st.subheader(f"{section_key} — Edit task #{idx+1}")

            itype = st.selectbox("Type", ["radio","checkbox","text","tfn","highlight"],
                                 index=["radio","checkbox","text","tfn","highlight"].index(data_t.get("type","radio")),
                                 key=f"{section_key}_edit_type_{idx}")
            q = st.text_area("Question / Prompt", value=data_t.get("q",""), key=f"{section_key}_edit_q_{idx}")

            options = data_t.get("options", [])
            correct = data_t.get("answer", [])

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
                    tasks[idx] = {"qid": data_t.get("qid") or str(uuid.uuid4()),
                                  "type": itype, "q": q.strip(), "options": options, "answer": correct}
                    st.success("Saved ✅")
            with c2:
                if st.button("🗑️ Delete task", key=f"{section_key}_del_{idx}"):
                    qid = data_t.get("qid")
                    if qid:
                        delete_question_qid(data, qid)
                    tasks.pop(idx)
                    st.warning("Deleted ⚠️")

# ---------------- Panels ----------------
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
    # ملاحظة: audio_path هنا نخليه string فقط (أنت تنجم تربطو Drive/URL لاحقاً)
    exam["listening"]["audio_path"] = st.text_input("Listening audio path/URL (optional)",
                                                    value=exam["listening"].get("audio_path",""),
                                                    key="listen_audio_path")

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
        reload_data()
        st.success(f"Saved ✅ → Google Sheets ({WS_QUESTIONS} + {WS_META})")

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
                    data=data,
                    phone=p,
                    level=lvl,
                    branch=BRANCHES[br],
                    created_by=st.session_state.user,
                    pass_plain=(None if auto_pw else manual_pw)
                )
                reload_data()
                st.success("Candidate created ✅ (give phone+password to the candidate)")
                st.code(f"Phone: {p}\nPassword: {pwd}\nLevel: {lvl}\nBranch: {BRANCHES[br]}\nSingle-use: YES (locks after submit)")

        st.markdown("---")
        dfc = load_candidates(data)
        if dfc.empty:
            st.info("No candidates yet.")
        else:
            view = dfc.copy()
            view["status"] = view["is_used"].apply(lambda x: "USED" if str(x)=="1" else "ACTIVE")
            st.dataframe(view.sort_values("created_at", ascending=False), use_container_width=True, height=280)

            st.markdown("### Manage candidate (unlock or delete)")
            c1,c2,c3 = st.columns([2,1,1])
            with c1:
                target = st.text_input("Phone to manage", key="adm_manage_phone")
            with c2:
                if st.button("Unlock (set ACTIVE)", key="adm_unlock"):
                    p = clean_phone(target)
                    df = load_candidates(data)
                    mask = (df["phone"].astype(str).str.strip() == p)
                    if mask.any():
                        df.loc[mask, "is_used"] = "0"
                        df.loc[mask, "used_at"] = ""
                        save_candidates(data, df)
                        reload_data()
                        st.success("Unlocked ✅")
                    else:
                        st.error("Phone not found.")
            with c3:
                if st.button("Delete", key="adm_delete"):
                    p = clean_phone(target)
                    df = load_candidates(data)
                    df = df[df["phone"].astype(str).str.strip() != p].copy()
                    save_candidates(data, df)
                    reload_data()
                    st.warning("Deleted.")

    with tab_results:
        st.markdown("### Results dashboard")
        sel_branch = st.selectbox("Branch", list(BRANCHES.keys()), key="adm_branch_sel")
        bcode = BRANCHES[sel_branch]

        df = results_df_from_ws(data, bcode)
        if df.empty:
            st.warning("No results yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
            st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), f"results_{bcode}.csv", "text/csv")

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
            exam = load_exam_from_gs(data, level)
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
        if L.get("audio_path"):
            st.caption(f"Audio path/URL: {L['audio_path']}")

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
        save_result_row(data, bcode, row)

        # lock candidate
        mark_candidate_used(data, phone)

        # reset
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
