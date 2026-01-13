# examen.py
# ------------------------------------------------------------------
# Mega Formation — Exam System (Excel Question Bank + Employee Builder + Candidate OTP + Admin Results)
# - Employee builds questions from UI -> saved to data/exam_bank.xlsx (no loss next time)
# - Candidate login: phone + one-time OTP (issued by employee)
# - Candidate never sees score; only final message
# - Admin sees results dashboard only
# ------------------------------------------------------------------

import streamlit as st
import pandas as pd
import os, re, time, hashlib, uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# ---------------- Page config ----------------
st.set_page_config(page_title="Mega Formation — Exams", layout="wide")

# ---------------- Paths ----------------
BASE_DIR    = Path(__file__).resolve().parent
DATA_DIR    = BASE_DIR / "data"
MEDIA_DIR   = BASE_DIR / "media"
RESULTS_DIR = BASE_DIR / "results"

DATA_DIR.mkdir(exist_ok=True)
MEDIA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

BANK_XLSX  = DATA_DIR / "exam_bank.xlsx"
USERS_XLSX = DATA_DIR / "users.xlsx"
OTPS_CSV   = DATA_DIR / "otps.csv"

# ---------------- Constants ----------------
LEVELS   = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
DEFAULT_DUR = {"A1":60, "A2":60, "B1":90, "B2":90}

PASS_MARK = 60.0  # النجاح من 60/100 (تبدّلها كي تحب)

RESULT_PATHS = {
    "MB": RESULTS_DIR / "results_MB.csv",
    "BZ": RESULTS_DIR / "results_BZ.csv"
}

OTP_MINUTES_VALID = 30

Q_COLS = [
    "QID","Level","Section","Type","Question","Options","Answer",
    "SourceText","Mode","MaxSelect","MinWords","MaxWords","Keywords","UpdatedAt"
]
META_COLS = ["Level","Key","Value"]

# ---------------- Utils ----------------
def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def clean_phone(p: str) -> str:
    p = (p or "").strip()
    p = re.sub(r"[^\d+]", "", p)
    return p

def make_otp(length=6) -> str:
    import random
    return "".join(str(random.randint(0, 9)) for _ in range(length))

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

def clean_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in ("-","_",".") else "_" for c in (name or ""))
    return (safe[:80] or f"file_{int(time.time())}")

def tokenise(text: str, mode: str):
    if not text:
        return []
    if mode == "word":
        tokens = re.findall(r"\w+[\w'-]*|[.,!?;:]", text)
        return [t for t in tokens if t.strip()]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# ---------------- Excel Bank ----------------
def ensure_bank_file():
    if BANK_XLSX.exists():
        return
    dfq = pd.DataFrame(columns=Q_COLS)
    dfm = pd.DataFrame(columns=META_COLS)
    with pd.ExcelWriter(BANK_XLSX, engine="openpyxl") as w:
        dfq.to_excel(w, sheet_name="Questions", index=False)
        dfm.to_excel(w, sheet_name="Meta", index=False)

def load_bank_questions() -> pd.DataFrame:
    ensure_bank_file()
    try:
        df = pd.read_excel(BANK_XLSX, sheet_name="Questions").fillna("")
    except Exception:
        df = pd.DataFrame(columns=Q_COLS)
    for c in Q_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[Q_COLS].fillna("")

def load_bank_meta() -> pd.DataFrame:
    ensure_bank_file()
    try:
        df = pd.read_excel(BANK_XLSX, sheet_name="Meta").fillna("")
    except Exception:
        df = pd.DataFrame(columns=META_COLS)
    for c in META_COLS:
        if c not in df.columns:
            df[c] = ""
    return df[META_COLS].fillna("")

def save_bank(dfq: pd.DataFrame, dfm: pd.DataFrame):
    dfq = dfq.fillna("")
    dfm = dfm.fillna("")
    with pd.ExcelWriter(BANK_XLSX, engine="openpyxl") as w:
        dfq.to_excel(w, sheet_name="Questions", index=False)
        dfm.to_excel(w, sheet_name="Meta", index=False)

def meta_get(level: str, key: str, default=""):
    dfm = load_bank_meta()
    sub = dfm[(dfm["Level"].astype(str).str.strip() == level) & (dfm["Key"].astype(str).str.strip() == key)]
    if sub.empty:
        return default
    return str(sub.iloc[-1]["Value"]).strip() if str(sub.iloc[-1]["Value"]).strip() else default

def meta_set(level: str, key: str, value: str):
    dfm = load_bank_meta()
    level = level.strip()
    key = key.strip()
    # remove existing same key+level, append latest
    dfm = dfm[~((dfm["Level"].astype(str).str.strip() == level) & (dfm["Key"].astype(str).str.strip() == key))]
    dfm = pd.concat([dfm, pd.DataFrame([{"Level":level, "Key":key, "Value":str(value)}])], ignore_index=True)
    dfq = load_bank_questions()
    save_bank(dfq, dfm)

def upsert_questions(rows: List[Dict[str, Any]]):
    dfq = load_bank_questions()
    if dfq.empty:
        dfq2 = pd.DataFrame(rows, columns=Q_COLS).fillna("")
        dfm = load_bank_meta()
        save_bank(dfq2, dfm)
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

    dfm = load_bank_meta()
    save_bank(dfq.fillna(""), dfm.fillna(""))

def delete_question_qid(qid: str):
    dfq = load_bank_questions()
    dfq = dfq[dfq["QID"].astype(str).str.strip() != str(qid).strip()].copy()
    dfm = load_bank_meta()
    save_bank(dfq, dfm)

def load_exam_from_excel(level: str) -> Dict[str, Any] | None:
    dfq = load_bank_questions()
    sub = dfq[dfq["Level"].astype(str).str.strip() == level].copy()
    if sub.empty:
        return None

    title = meta_get(level, "title", f"Mega Formation English Exam — {level}")
    dur   = meta_get(level, "duration_min", str(DEFAULT_DUR.get(level, 60)))
    try:
        dur = int(float(dur))
    except Exception:
        dur = DEFAULT_DUR.get(level, 60)

    listening_audio = meta_get(level, "listening_audio", "")
    listening_trans = meta_get(level, "listening_transcript", "")
    reading_passage = meta_get(level, "reading_passage", "")

    exam = {
        "meta": {"title": title, "level": level, "duration_min": dur, "exam_id": f"EXCEL_{level}_{datetime.now().strftime('%Y%m%d')}"},
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
def ensure_users_file():
    if USERS_XLSX.exists():
        return
    # default users
    df = pd.DataFrame([
        {"username":"admin",    "pass_hash":sha256("megaadmin"), "role":"admin"},
        {"username":"employee", "pass_hash":sha256("mega123"),   "role":"employee"},
    ])
    with pd.ExcelWriter(USERS_XLSX, engine="openpyxl") as w:
        df.to_excel(w, sheet_name="Users", index=False)

def load_users() -> pd.DataFrame:
    ensure_users_file()
    try:
        df = pd.read_excel(USERS_XLSX, sheet_name="Users").fillna("")
        for c in ["username","pass_hash","role"]:
            if c not in df.columns:
                df[c] = ""
        return df[["username","pass_hash","role"]]
    except Exception:
        return pd.DataFrame(columns=["username","pass_hash","role"])

def verify_user(username: str, password: str):
    df = load_users()
    u = (username or "").strip()
    ph = sha256(password or "")
    hit = df[(df["username"] == u) & (df["pass_hash"] == ph)]
    if hit.empty:
        return None
    return str(hit.iloc[0]["role"])

# ---------------- OTP storage ----------------
def load_otps() -> pd.DataFrame:
    if OTPS_CSV.exists():
        try:
            return pd.read_csv(OTPS_CSV).fillna("")
        except Exception:
            pass
    return pd.DataFrame(columns=["phone","otp_hash","level","branch","issued_by","created_at","expires_at","used_at","is_used"])

def save_otps(df: pd.DataFrame):
    df.to_csv(OTPS_CSV, index=False)

def issue_otp(phone: str, level: str, branch: str, issued_by: str):
    phone = clean_phone(phone)
    otp = make_otp(6)
    created = datetime.now()
    expires = created + timedelta(minutes=OTP_MINUTES_VALID)

    df = load_otps()
    row = {
        "phone": phone,
        "otp_hash": sha256(otp),
        "level": level,
        "branch": branch,
        "issued_by": issued_by,
        "created_at": created.isoformat(timespec="seconds"),
        "expires_at": expires.isoformat(timespec="seconds"),
        "used_at": "",
        "is_used": "0"
    }
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    save_otps(df)
    return otp

def validate_and_consume_otp(phone: str, otp: str):
    phone = clean_phone(phone)
    df = load_otps()
    if df.empty:
        return None, "OTP غير موجود."

    otp_h = sha256(otp)
    cand = df[(df["phone"] == phone) & (df["otp_hash"] == otp_h) & (df["is_used"].astype(str) == "0")]
    if cand.empty:
        return None, "OTP غالط أو مستعمل."

    idx = cand.index[-1]
    exp = df.loc[idx, "expires_at"]
    try:
        exp_dt = datetime.fromisoformat(str(exp))
    except Exception:
        return None, "OTP فيه مشكلة في الصلاحية."

    if datetime.now() > exp_dt:
        return None, "OTP منتهي الصلوحية."

    df.loc[idx, "is_used"] = "1"
    df.loc[idx, "used_at"] = now_iso()
    save_otps(df)

    payload = {"level": str(df.loc[idx, "level"]), "branch": str(df.loc[idx, "branch"])}
    return payload, None

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

# ---------------- Results storage ----------------
def results_df(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path).fillna("")
        except Exception:
            return pd.DataFrame()
    cols = ["timestamp","name","phone","branch","level","exam_id","overall","pass",
            "Listening","Reading","Use_of_English","Writing"]
    return pd.DataFrame(columns=cols)

def save_result_row(branch_code, row):
    path = RESULT_PATHS[branch_code]
    df = results_df(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

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

# ---------------- Header ----------------
c1,c2 = st.columns([1,4])
with c1:
    logo_path = MEDIA_DIR / "mega_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    else:
        st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — English Exams</h2>", unsafe_allow_html=True)
    st.caption("Employee builds questions → saved to Excel | Candidate OTP | Admin results")

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
        otp   = st.text_input("One-time password (OTP)", type="password", key="cand_otp")
        if st.button("Login Candidate", key="cand_login"):
            payload, err = validate_and_consume_otp(phone, otp)
            if err:
                st.error(err)
            else:
                st.session_state.candidate_ok = True
                st.session_state.candidate_payload = {"phone": clean_phone(phone), **payload}
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
    exam = load_exam_from_excel(level)
    if not exam:
        # create fresh skeleton
        exam = {
            "meta": {"title": f"Mega Formation English Exam — {level}", "level": level, "duration_min": DEFAULT_DUR.get(level,60), "exam_id": f"EXCEL_{level}"},
            "listening": {"audio_path": "", "transcript": "", "tasks": []},
            "reading": {"passage": "", "tasks": []},
            "use": {"tasks": []},
            "writing": {"prompt": "", "min_words": 120, "max_words": 150, "keywords": []}
        }
    return exam

def save_exam_to_excel(level: str, exam: Dict[str, Any]):
    rows = []
    for sec_key, sec_name in [("listening","Listening"), ("reading","Reading"), ("use","Use of English")]:
        for t in exam.get(sec_key, {}).get("tasks", []):
            rows.append(row_from_task(level, sec_name, t))
    rows.append(row_from_writing(level, exam.get("writing", {})))

    upsert_questions(rows)

    # save meta too
    meta_set(level, "title", exam["meta"].get("title", f"Mega Formation English Exam — {level}"))
    meta_set(level, "duration_min", str(exam["meta"].get("duration_min", DEFAULT_DUR.get(level,60))))

    # listening + reading meta
    meta_set(level, "listening_audio", exam.get("listening", {}).get("audio_path",""))
    meta_set(level, "listening_transcript", exam.get("listening", {}).get("transcript",""))
    meta_set(level, "reading_passage", exam.get("reading", {}).get("passage",""))

def render_task_editor(level: str, section_key: str, tasks: List[Dict[str, Any]], idx=None):
    TYPES = ["radio","checkbox","text","tfn","highlight"]
    MODES = ["word","sentence"]

    box = st.container(border=True)

    with box:
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
                opts_raw = st.text_area("Options (one per line)", value="\n".join(options if isinstance(options,list) else []), key=f"{section_key}_edit_opts_{idx}")
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
                if isinstance(correct, list):
                    kw_txt = ", ".join(correct)
                else:
                    kw_txt = ""
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
                    tasks[idx] = {"qid": data.get("qid") or str(uuid.uuid4()), "type": itype, "q": q.strip(), "options": options, "answer": correct}
                    st.success("Saved ✅")
            with c2:
                if st.button("🗑️ Delete task", key=f"{section_key}_del_{idx}"):
                    # delete from excel immediately if has QID
                    qid = data.get("qid")
                    if qid:
                        delete_question_qid(qid)
                    tasks.pop(idx)
                    st.warning("Deleted ⚠️")

def employee_panel():
    st.subheader("👩‍💼 Employee Panel")

    # OTP issuing
    st.markdown("### 1) Generate OTP for Candidate (one-time)")
    c1,c2,c3 = st.columns(3)
    with c1:
        phone = st.text_input("Candidate phone", key="otp_phone")
    with c2:
        lvl = st.selectbox("Level", LEVELS, key="otp_lvl")
    with c3:
        br = st.selectbox("Branch", list(BRANCHES.keys()), key="otp_br")

    if st.button("Generate OTP", type="primary", key="gen_otp_btn"):
        if not clean_phone(phone):
            st.error("اكتب رقم الهاتف صحيح.")
        else:
            otp = issue_otp(phone, lvl, BRANCHES[br], st.session_state.user)
            st.success("OTP generated ✅ (give it to candidate)")
            st.code(f"Phone: {clean_phone(phone)}\nOTP: {otp}\nValid: {OTP_MINUTES_VALID} minutes\nLevel: {lvl}\nBranch: {BRANCHES[br]}")

    st.markdown("---")
    st.markdown("### 2) Build / Edit Exam Questions")

    level = st.selectbox("Level to edit", LEVELS, key="emp_edit_level")
    exam = load_exam_for_edit(level)

    # Meta controls
    st.markdown("#### Exam meta")
    cA,cB = st.columns([2,1])
    with cA:
        exam["meta"]["title"] = st.text_input("Title", value=exam["meta"].get("title", f"Mega Formation English Exam — {level}"), key="meta_title")
    with cB:
        exam["meta"]["duration_min"] = st.number_input("Duration (min)", min_value=10, step=5,
                                                       value=int(exam["meta"].get("duration_min", DEFAULT_DUR.get(level,60))),
                                                       key="meta_dur")

    # Listening meta + audio upload
    st.markdown("#### Listening meta")
    exam["listening"]["transcript"] = st.text_area("Listening transcript (optional)",
                                                   value=exam["listening"].get("transcript",""),
                                                   key="listen_trans")
    up = st.file_uploader("Upload audio (MP3/WAV) — saved to media/", type=["mp3","wav"], key="listen_audio_up")
    if up:
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(up.name)}"
        fpath = MEDIA_DIR / fname
        with open(fpath, "wb") as f:
            f.write(up.read())
        exam["listening"]["audio_path"] = fname
        st.success(f"Saved audio: media/{fname}")
    if exam["listening"].get("audio_path") and (MEDIA_DIR / exam["listening"]["audio_path"]).exists():
        st.audio(str(MEDIA_DIR / exam["listening"]["audio_path"]))

    # Reading meta
    st.markdown("#### Reading meta")
    exam["reading"]["passage"] = st.text_area("Reading passage",
                                              value=exam["reading"].get("passage",""),
                                              key="reading_passage")

    st.markdown("---")
    # Tasks editors
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
    if st.button("💾 Save THIS LEVEL to Excel", type="primary", key="save_level_excel"):
        save_exam_to_excel(level, exam)
        st.success(f"Saved ✅ → data/exam_bank.xlsx (Level {level})")

    st.caption("ملاحظة: Delete task يمسحها من Excel مباشرة (بالـ QID).")

# ---------------- Admin Dashboard (results only) ----------------
def admin_dashboard():
    st.subheader("🛡️ Admin Dashboard — Results Only")

    sel_branch = st.selectbox("Branch", list(BRANCHES.keys()), key="adm_branch_sel")
    bcode = BRANCHES[sel_branch]
    path = RESULT_PATHS[bcode]

    df = results_df(path)
    if df.empty:
        st.warning("No results yet.")
        return

    st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
    st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8"), f"results_{bcode}.csv", "text/csv")

# ---------------- Candidate Exam ----------------
def render_candidate():
    st.subheader("🎓 Candidate Exam")

    if not st.session_state.candidate_ok:
        st.info("سجّل الدخول بالـ Phone + OTP من الشريط الجانبي.")
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
            exam = load_exam_from_excel(level)
            if not exam:
                st.error("هذا الLevel مازال موش محضّر في Excel. اطلب من الموظف يحضّرو.")
                return
            st.session_state.exam = exam
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = True
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=int(exam["meta"]["duration_min"]))
            st.success("Exam started ✅")
        return

    exam = st.session_state.exam
    meta = exam["meta"]

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
        ap = L.get("audio_path","")
        if ap and (MEDIA_DIR / ap).exists():
            st.audio(str(MEDIA_DIR / ap))
        elif ap:
            st.caption(f"Audio not found: media/{ap}")

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
            "exam_id": meta.get("exam_id",""),
            "overall": overall,
            "pass": passed,
            "Listening": L_pct,
            "Reading": R_pct,
            "Use_of_English": U_pct,
            "Writing": W_pct,
        }
        save_result_row(bcode, row)

        # lock candidate view
        st.session_state.candidate_started = False
        st.session_state.deadline = None
        st.session_state.exam = None
        st.session_state.answers = {s:{} for s in SECTIONS}

        # candidate sees only this
        st.success("✅ تم الإجتياز بنجاح، سيتم التواصل معك من قبل إدارة الهيكل.")
        st.stop()

# ---------------- Router ----------------
if st.session_state.role == "employee":
    employee_panel()
elif st.session_state.role == "admin":
    admin_dashboard()
else:
    render_candidate()
