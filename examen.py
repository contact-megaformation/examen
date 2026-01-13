# Mega_Level_Excel_OTP.py
# ------------------------------------------------------------------
# Mega Formation — Excel Question Bank + Staff/Admin + Candidate OTP
# - Questions stored in data/exam_bank.xlsx (by Level + Section)
# - Employee login: manage questions + issue one-time OTP for candidate
# - Candidate login: phone + OTP (valid once + expires)
# - Candidate never sees results; end message only
# - Admin sees Results dashboard
# ------------------------------------------------------------------

import streamlit as st
import os, re, json, time, hashlib
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

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

BANK_XLSX   = DATA_DIR / "exam_bank.xlsx"
USERS_XLSX  = DATA_DIR / "users.xlsx"
OTPS_CSV    = DATA_DIR / "otps.csv"

LEVELS   = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}

RESULT_PATHS = {
    "MB": RESULTS_DIR / "results_MB.csv",
    "BZ": RESULTS_DIR / "results_BZ.csv"
}

OTP_MINUTES_VALID = 30  # صلاحية OTP

# ---------------- Security helpers ----------------
def sha256(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()

def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")

def clean_phone(p: str) -> str:
    p = (p or "").strip()
    p = re.sub(r"[^\d+]", "", p)
    return p

def make_otp(length=6) -> str:
    import random
    return "".join(str(random.randint(0,9)) for _ in range(length))

# ---------------- Excel / storage ----------------
def ensure_bank_file():
    """Create empty exam_bank.xlsx with the required sheets/headers if missing."""
    if BANK_XLSX.exists():
        return
    q_cols = ["Level","Section","Type","Question","Options","Answer",
              "SourceText","Mode","MaxSelect","MinWords","MaxWords","Keywords"]
    m_cols = ["Level","Key","Value"]
    with pd.ExcelWriter(BANK_XLSX, engine="openpyxl") as w:
        pd.DataFrame(columns=q_cols).to_excel(w, sheet_name="Questions", index=False)
        pd.DataFrame(columns=m_cols).to_excel(w, sheet_name="Meta", index=False)

def load_bank_questions() -> pd.DataFrame:
    ensure_bank_file()
    try:
        df = pd.read_excel(BANK_XLSX, sheet_name="Questions")
        df = df.fillna("")
        return df
    except Exception:
        return pd.DataFrame(columns=["Level","Section","Type","Question","Options","Answer",
                                     "SourceText","Mode","MaxSelect","MinWords","MaxWords","Keywords"])

def load_bank_meta() -> pd.DataFrame:
    ensure_bank_file()
    try:
        df = pd.read_excel(BANK_XLSX, sheet_name="Meta")
        df = df.fillna("")
        return df
    except Exception:
        return pd.DataFrame(columns=["Level","Key","Value"])

def save_bank(df_questions: pd.DataFrame, df_meta: pd.DataFrame):
    with pd.ExcelWriter(BANK_XLSX, engine="openpyxl") as w:
        df_questions.to_excel(w, sheet_name="Questions", index=False)
        df_meta.to_excel(w, sheet_name="Meta", index=False)

def ensure_users_file():
    """users.xlsx columns: username, pass_hash, role (admin/employee)"""
    if USERS_XLSX.exists():
        return
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
        return df
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
            df = pd.read_csv(OTPS_CSV).fillna("")
            return df
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
    # find latest matching not used
    cand = df[(df["phone"] == phone) & (df["otp_hash"] == otp_h) & (df["is_used"].astype(str) == "0")]
    if cand.empty:
        return None, "OTP غالط أو مستعمل."

    # take the last one
    idx = cand.index[-1]
    exp = df.loc[idx, "expires_at"]
    try:
        exp_dt = datetime.fromisoformat(str(exp))
    except Exception:
        return None, "OTP فيه مشكلة في الصلاحية."

    if datetime.now() > exp_dt:
        return None, "OTP منتهي الصلوحية."

    # consume
    df.loc[idx, "is_used"] = "1"
    df.loc[idx, "used_at"] = now_iso()
    save_otps(df)

    payload = {
        "level": str(df.loc[idx, "level"]),
        "branch": str(df.loc[idx, "branch"])
    }
    return payload, None

# ---------------- Exam build from Excel ----------------
def split_pipe(s: str):
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split("|") if x.strip()]

def meta_for_level(level: str) -> dict:
    dfm = load_bank_meta()
    m = {}
    lvl = (level or "").strip()
    sub = dfm[dfm["Level"].astype(str).str.strip() == lvl]
    for _, r in sub.iterrows():
        k = str(r.get("Key","")).strip()
        v = str(r.get("Value","")).strip()
        if k:
            m[k] = v
    # defaults
    title = m.get("title") or f"Mega Formation English Exam — {level}"
    dur   = m.get("duration_min") or ("60" if level in ("A1","A2") else "90")
    return {
        "title": title,
        "duration_min": int(float(dur)) if str(dur).replace(".","",1).isdigit() else 60,
        "listening_audio": m.get("listening_audio",""),
        "listening_transcript": m.get("listening_transcript",""),
        "reading_passage": m.get("reading_passage",""),
    }

def exam_from_excel(level: str) -> dict:
    df = load_bank_questions()
    lvl = (level or "").strip()
    df = df[df["Level"].astype(str).str.strip() == lvl].copy()
    if df.empty:
        return None

    meta = meta_for_level(lvl)

    exam = {
        "meta": {
            "title": meta["title"],
            "level": lvl,
            "duration_min": meta["duration_min"],
            "exam_id": f"EXCEL_{lvl}_{datetime.now().strftime('%Y%m%d')}"
        },
        "listening": {"audio_path": meta["listening_audio"], "transcript": meta["listening_transcript"], "tasks":[]},
        "reading": {"passage": meta["reading_passage"], "tasks":[]},
        "use": {"tasks":[]},
        "writing": {"prompt":"", "min_words":120, "max_words":150, "keywords":[]}
    }

    # tasks
    for _, r in df.iterrows():
        section = str(r.get("Section","")).strip()
        ttype   = str(r.get("Type","")).strip()

        if section not in SECTIONS:
            continue

        if section == "Writing" and ttype == "writing":
            exam["writing"]["prompt"] = str(r.get("Question","")).strip()
            mn = r.get("MinWords","")
            mx = r.get("MaxWords","")
            kws= r.get("Keywords","")
            try:
                exam["writing"]["min_words"] = int(float(mn)) if str(mn).strip() else 120
                exam["writing"]["max_words"] = int(float(mx)) if str(mx).strip() else 150
            except Exception:
                pass
            exam["writing"]["keywords"] = split_pipe(str(kws))
            continue

        task = {"type": ttype, "q": str(r.get("Question","")).strip()}

        if ttype in ("radio","checkbox"):
            task["options"] = split_pipe(str(r.get("Options","")))
            if ttype == "radio":
                task["answer"] = str(r.get("Answer","")).strip()
            else:
                task["answer"] = split_pipe(str(r.get("Answer","")))
        elif ttype == "tfn":
            task["options"] = ["T","F","NG"]
            task["answer"]  = str(r.get("Answer","")).strip() or "T"
        elif ttype == "text":
            task["options"] = []
            task["answer"]  = split_pipe(str(r.get("Answer","")))  # keywords
        elif ttype == "highlight":
            src = str(r.get("SourceText",""))
            mode= str(r.get("Mode","word")).strip() or "word"
            mxs = r.get("MaxSelect", 3)
            try:
                mxs = int(float(mxs))
            except Exception:
                mxs = 3
            task["options"] = {"text": src, "mode": mode, "max_select": mxs}
            task["answer"]  = split_pipe(str(r.get("Answer","")))
        else:
            # unknown type -> skip
            continue

        if section == "Listening":
            exam["listening"]["tasks"].append(task)
        elif section == "Reading":
            exam["reading"]["tasks"].append(task)
        elif section == "Use of English":
            exam["use"]["tasks"].append(task)

    return exam

# ---------------- Tokenise for highlight ----------------
def tokenise(text: str, mode: str):
    if not text:
        return []
    if mode == "word":
        tokens = re.findall(r"\w+[\w'-]*|[.,!?;:]", text)
        return [t for t in tokens if t.strip()]
    # sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

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
        u = user_map.get(i)
        q_pcts.append(score_item_pct(t, u))
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
    cols = ["timestamp","name","phone","branch","level","exam_id","overall","Listening","Reading","Use_of_English","Writing"]
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

# ---------------- Header (logo) ----------------
c1,c2 = st.columns([1,4])
with c1:
    logo_path = MEDIA_DIR / "mega_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), width=90)
    else:
        st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — English Exams</h2>", unsafe_allow_html=True)
    st.caption("Excel question bank + Employee/Admin + Candidate OTP")

# ---------------- Sidebar: Login areas ----------------
with st.sidebar:
    st.header("Login")

    tab_emp, tab_admin, tab_cand = st.tabs(["👩‍💼 Employee", "🛡️ Admin", "🎓 Candidate"])

    # Employee login
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

    # Admin login
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

    # Candidate OTP login
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

# ---------------- Employee panel ----------------
def employee_panel():
    st.subheader("👩‍💼 Employee Panel")

    ensure_bank_file()
    dfq = load_bank_questions()
    dfm = load_bank_meta()

    st.markdown("### 1) Issue OTP for candidate")
    c1,c2,c3 = st.columns(3)
    with c1:
        phone = st.text_input("Candidate phone", key="otp_phone")
    with c2:
        lvl = st.selectbox("Level", LEVELS, key="otp_lvl")
    with c3:
        br = st.selectbox("Branch", list(BRANCHES.keys()), key="otp_br")

    if st.button("Generate OTP (one-time)", type="primary", key="gen_otp_btn"):
        if not clean_phone(phone):
            st.error("اكتب رقم الهاتف صحيح.")
        else:
            otp = issue_otp(phone, lvl, BRANCHES[br], st.session_state.user)
            st.success("OTP generated ✅")
            st.code(f"OTP: {otp}\nValid: {OTP_MINUTES_VALID} minutes\nPhone: {clean_phone(phone)}\nLevel: {lvl}\nBranch: {BRANCHES[br]}")

    st.markdown("---")
    st.markdown("### 2) Manage Questions (Excel-backed)")

    sel_level = st.selectbox("Filter Level", ["All"] + LEVELS, key="emp_filter_lvl")
    sel_sec   = st.selectbox("Filter Section", ["All"] + SECTIONS, key="emp_filter_sec")

    view = dfq.copy()
    if sel_level != "All":
        view = view[view["Level"].astype(str).str.strip() == sel_level]
    if sel_sec != "All":
        view = view[view["Section"].astype(str).str.strip() == sel_sec]

    st.dataframe(view, use_container_width=True, height=280)

    st.caption("تقدر تعدّل مباشرة من هنا ثم Save -> يتكتب في Excel.")
    edited = st.data_editor(view, num_rows="dynamic", use_container_width=True, key="editor_questions")

    if st.button("💾 Save Questions to Excel", key="save_q_excel"):
        # merge back to full dfq:
        # easiest: overwrite whole Questions with edited if filter=All; otherwise replace matching rows by index
        if sel_level == "All" and sel_sec == "All":
            dfq2 = edited.copy()
        else:
            dfq2 = dfq.copy()
            # align on original index by keeping an internal column if needed; here we fallback to replace by row signature
            # We'll do: drop all filtered rows, then append edited rows.
            dfq2 = dfq2[~(
                ((sel_level=="All") | (dfq2["Level"].astype(str).str.strip()==sel_level)) &
                ((sel_sec=="All")   | (dfq2["Section"].astype(str).str.strip()==sel_sec))
            )]
            dfq2 = pd.concat([dfq2, edited], ignore_index=True)

        save_bank(dfq2.fillna(""), dfm.fillna(""))
        st.success("Saved ✅ (data/exam_bank.xlsx)")

    st.markdown("---")
    st.markdown("### 3) Manage Meta (duration/audio/transcript/passage)")
    st.dataframe(dfm, use_container_width=True, height=220)
    edited_meta = st.data_editor(dfm, num_rows="dynamic", use_container_width=True, key="editor_meta")
    if st.button("💾 Save Meta to Excel", key="save_m_excel"):
        save_bank(load_bank_questions().fillna(""), edited_meta.fillna(""))
        st.success("Meta saved ✅")

# ---------------- Admin results dashboard ----------------
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

# ---------------- Candidate exam ----------------
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
            exam = exam_from_excel(level)
            if not exam:
                st.error("هذا الLevel مازال موش محضّر في Excel.")
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
        # score (candidate won't see it)
        L_pct = score_section_percent(exam["listening"]["tasks"], st.session_state.answers["Listening"])
        R_pct = score_section_percent(exam["reading"]["tasks"], st.session_state.answers["Reading"])
        U_pct = score_section_percent(exam["use"]["tasks"], st.session_state.answers["Use of English"])

        W = exam["writing"]
        W_text = st.session_state.answers["Writing"].get(0,"")
        W_pct, wc, hits = score_writing_pct(W_text, W.get("min_words",0), W.get("max_words",0), W.get("keywords",[]))

        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)

        row = {
            "timestamp": now_iso(),
            "name": (name or "").strip(),
            "phone": phone,
            "branch": bcode,
            "level": level,
            "exam_id": meta.get("exam_id",""),
            "overall": overall,
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

        st.success("✅ تم الإجتياز بنجاح، سيتم التواصل معك من قبل إدارة الهيكل.")
        st.stop()

# ---------------- Main router ----------------
if st.session_state.role == "employee":
    employee_panel()
elif st.session_state.role == "admin":
    admin_dashboard()
else:
    render_candidate()
