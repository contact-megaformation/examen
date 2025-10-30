# Mega_Level_Builder_Simple_PREVIEW.py
# ------------------------------------------------------------------
# Mega Formation — Level-based Simple Builder (Admin hidden)
# - مسارات ثابتة (exams / results / media) بجنب السكريبت
# - لوغو أوتوماتيك من media/mega_logo.png
# - النتائج مخفية عن المترشّح، تظهر فقط للأدمين (Dashboard)
# - متصفّح exams + Export/Import JSON + Scan & Recover
# - زر Preview Exam (قراءة فقط)
# - إصلاح قيمة duration_min الافتراضية
# ------------------------------------------------------------------

import streamlit as st
import os, json, time, re, shutil
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# ---------------- Config & Constants ----------------
st.set_page_config(page_title="Mega Formation — Level Exams", layout="wide")

BASE_DIR   = Path(__file__).resolve().parent
EXAMS_DIR  = BASE_DIR / "exams"
RESULTS_DIR= BASE_DIR / "results"
MEDIA_DIR  = BASE_DIR / "media"
EXAMS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
MEDIA_DIR.mkdir(exist_ok=True)

LEVELS   = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
DEFAULT_DUR = {"A1":60, "A2":60, "B1":90, "B2":90}
RESULT_PATHS = {
    "MB": RESULTS_DIR / "results_MB.csv",
    "BZ": RESULTS_DIR / "results_BZ.csv"
}

ADMIN_PASS = "megaadmin"  # بدّلها كيما تحب

# ---------------- Helpers ----------------
def exam_path_for(level:str)->str:
    return str(EXAMS_DIR / f"EXAM_{level}.json")

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def results_df(path: Path):
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    cols = ["timestamp","name","branch","level","exam_id","overall","Listening","Reading","Use_of_English","Writing"]
    return pd.DataFrame(columns=cols)

def save_result_row(branch_code, row):
    path = RESULT_PATHS[branch_code]
    df = results_df(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

def clean_filename(name:str)->str:
    safe = "".join(c if c.isalnum() or c in ("-","_",".") else "_" for c in (name or ""))
    return (safe[:80] or f"file_{int(time.time())}")

def tokenise(text:str, mode:str):
    if not text: return []
    if mode == "word":
        tokens = re.findall(r"\w+[\w'-]*|[.,!?;:]", text)
        return [t for t in tokens if t.strip()]
    # sentence
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

# ---------------- Scoring (percent-based) ----------------
def score_item_pct(item, user_val):
    """
    - radio/tfn: 100% لو مطابقة وإلا 0.
    - checkbox/highlight: (tp - 0.5*fp)/len(correct) مع حد أدنى 0.
    - text: keywords hit / total keywords * 100. بدون Keywords: 100 إن الجواب مش فارغ.
    """
    itype = item.get("type")
    correct = item.get("answer")
    if itype in ("radio","tfn"):
        return 100.0 if (user_val is not None and user_val == correct) else 0.0
    if itype in ("checkbox","highlight"):
        corr = set(correct or [])
        sel  = set(user_val or [])
        if not corr: return (100.0 if not sel else 0.0)
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
        pct = score_item_pct(t, u)
        q_pcts.append(pct)
    section_pct = round(sum(q_pcts)/len(q_pcts), 1) if q_pcts else 0.0
    return section_pct

def score_writing_pct(text, min_w, max_w, keywords):
    wc = len((text or "").split())
    base = 40 if (min_w and max_w and min_w <= wc <= max_w) else (20 if wc>0 else 0)
    hits = sum(1 for k in (keywords or []) if k.lower() in (text or "").lower())
    kw_score = min(60, hits*12)
    return float(min(100, base + kw_score)), wc, hits

# ---------------- Default exam skeleton ----------------
def empty_exam(level:str):
    return {
        "meta":{
            "title": f"Mega Formation English Exam — {level}",
            "level": level,
            "duration_min": DEFAULT_DUR.get(level,60),
            "exam_id": f"EXAM_{level}"
        },
        "listening":{"audio_path":"","transcript":"","tasks":[]},
        "reading":{"passage":"","tasks":[]},
        "use":{"tasks":[]},
        "writing":{"prompt":"","min_words":120,"max_words":150,"keywords":[]}
    }

def sample_exam(level="B1"):
    return {
        "meta": {"title": f"Sample Exam {level}", "level": level, "duration_min": DEFAULT_DUR.get(level,90), "exam_id": f"EXAM_{level}"},
        "listening": {
            "audio_path": "",
            "transcript": "Short announcement about a meeting at 4 p.m. on Friday.",
            "tasks": [
                {"type":"radio","q":"The meeting is on ___","options":["Monday","Wednesday","Friday"],"answer":"Friday"}
            ]
        },
        "reading": {
            "passage": "Email from a manager about moving the deadline to next week.",
            "tasks": [
                {"type":"tfn","q":"The deadline was extended.","options":["T","F","NG"],"answer":"T"}
            ]
        },
        "use": {
            "tasks": [
                {"type":"text","q":"I have lived in Tunis ___ 2019.","answer":["since"]},
                {"type":"text","q":"We couldn’t go out because it ___ raining.","answer":["was"]}
            ]
        },
        "writing": {
            "prompt": "Write an email to a colleague asking to finalize the monthly report by Thursday 4 p.m.",
            "min_words": 120, "max_words": 150, "keywords": ["deadline","report","help"]
        }
    }

# ---------------- State ----------------
def init_state():
    st.session_state.setdefault("is_admin", False)         # مخفي افتراضيًا
    st.session_state.setdefault("admin_level", "B1")       # مستوى الأدمين يحرّر فيه
    st.session_state.setdefault("candidate_level", "B1")   # مستوى المترشح
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("exam", empty_exam("B1"))  # نموذج محلي للتحرير الحالي
    st.session_state.setdefault("show_preview", False)     # وضع المعاينة للأدمين
init_state()

# ---------------- Header (Auto Logo) ----------------
c1,c2 = st.columns([1,4])
with c1:
    logo_path = MEDIA_DIR / "mega_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=False)
    else:
        st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — Level Exams</h2>", unsafe_allow_html=True)
    st.caption("Candidate mode by default — Admin behind password")

# ---------------- Sidebar: Candidate controls ----------------
with st.sidebar:
    st.header("Candidate")
    st.session_state.candidate_level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.candidate_level), key="cand_level_sel")
    cand_name = st.text_input("Your name", key="cand_name")
    cand_branch = st.selectbox("Branch", list(BRANCHES.keys()), key="cand_branch_sel")

    if not st.session_state.candidate_started:
        if st.button("▶️ Start Exam", key="start_exam_btn"):
            path = exam_path_for(st.session_state.candidate_level)
            st.caption(f"Loading from: {path}")  # للتأكد
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.session_state.answers = {s:{} for s in SECTIONS}
                st.session_state.candidate_started = True
                # ----- إصلاح القيمة الافتراضية للمدة -----
                dur_default = data.get("meta", {}).get("duration_min")
                if not isinstance(dur_default, (int, float)):
                    lvl = data.get("meta", {}).get("level", st.session_state.candidate_level)
                    dur_default = DEFAULT_DUR.get(lvl, 60)
                st.session_state.deadline = datetime.utcnow() + timedelta(minutes=int(dur_default))
                st.success(f"Loaded {os.path.basename(path)} and started.")
            else:
                st.error("هذا المستوى غير مُحضّر بعد من الأدمين (لا يوجد ملف امتحان).")
    else:
        if st.button("🔁 Restart", key="restart_exam_btn"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = False
            st.session_state.deadline = None

# ---------------- Sidebar: Admin login + Exams browser ----------------
with st.sidebar:
    with st.expander("🔐 Admin login", expanded=False):
        pw = st.text_input("Password", type="password", key="admin_pw_input")
        if st.button("Login as admin", key="admin_login_btn"):
            if pw == ADMIN_PASS:
                st.session_state.is_admin = True
                st.success("Admin mode enabled.")
            else:
                st.error("Wrong password.")
        if st.session_state.is_admin and st.button("Logout", key="admin_logout_btn"):
            st.session_state.is_admin = False

    with st.expander("📁 Exams folder (admin)", expanded=False):
        try:
            files = sorted(os.listdir(EXAMS_DIR))
            st.write(files if files else "No exam files yet in 'exams/'.")
        except Exception as e:
            st.error(f"Cannot list exams: {e}")

# ---------------- Admin tools ----------------
def render_task_editor(section_key, idx=None):
    st.markdown("<div style='background:#fff;padding:16px;border-radius:12px;margin:8px 0;box-shadow:0 6px 24px rgba(0,0,0,.06)'>", unsafe_allow_html=True)
    st.subheader(f"{section_key} — Task editor")

    TYPES = ["radio","checkbox","text","tfn","highlight"]
    MODES = ["word","sentence"]

    if idx is None:
        itype = st.selectbox("Type", TYPES, key=f"{section_key}_new_type")
        q     = st.text_area("Question / Prompt", key=f"{section_key}_new_q")
        options, correct = [], None
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
            options = []
            kw_raw = st.text_input("Keywords (comma-separated)", key=f"{section_key}_new_corr_txt", placeholder="payment, extension, invoice")
            correct = [k.strip() for k in kw_raw.split(",") if k.strip()]
        elif itype == "highlight":
            src_text = st.text_area("Source text (select from this text)", key=f"{section_key}_new_h_text")
            unit     = st.radio("Selection unit", MODES, horizontal=True, key=f"{section_key}_new_h_mode")
            max_sel  = st.number_input("Max selections", value=3, min_value=1, step=1, key=f"{section_key}_new_h_max")
            tokens   = tokenise(src_text, unit)
            st.caption(f"Preview ({unit}) count = {len(tokens)}")
            corr_sel = st.multiselect("Correct selections (exact match)", tokens, default=[], key=f"{section_key}_new_h_corr")
            options  = {"text": src_text, "mode": unit, "max_select": int(max_sel)}
            correct  = corr_sel

        if st.button("➕ Add task", key=f"{section_key}_add_btn"):
            st.session_state.exam[section_key]["tasks"].append({
                "type": itype, "q": q.strip(), "options": options, "answer": correct
            })
            st.success("Task added.")

    else:
        data   = st.session_state.exam[section_key]["tasks"][idx]
        itype  = st.selectbox("Type", TYPES, index=TYPES.index(data.get("type","radio")), key=f"{section_key}_edit_type_{idx}")
        q      = st.text_area("Question / Prompt", value=data.get("q",""), key=f"{section_key}_edit_q_{idx}")
        options= data.get("options", [])
        correct= data.get("answer", [])
        if itype in ("radio","checkbox"):
            opts_raw = st.text_area("Options (one per line)", value="\n".join(options), key=f"{section_key}_edit_opts_{idx}")
            options  = [o.strip() for o in opts_raw.splitlines() if o.strip()]
            if itype == "radio":
                ix = options.index(correct) if (correct in options) else (0 if options else 0)
                correct = st.selectbox("Correct option", options, index=ix, key=f"{section_key}_edit_corr_radio_{idx}")
            else:
                correct = st.multiselect("Correct options", options, default=[o for o in (correct or []) if o in options], key=f"{section_key}_edit_corr_ck_{idx}")
        elif itype == "tfn":
            options = ["T","F","NG"]
            ix = options.index(correct) if correct in options else 0
            correct = st.selectbox("Correct", options, index=ix, key=f"{section_key}_edit_corr_tfn_{idx}")
        elif itype == "text":
            kw_raw = st.text_input("Keywords (comma-separated)", value=", ".join(correct if isinstance(correct,list) else []), key=f"{section_key}_edit_corr_txt_{idx}")
            options = []
            correct = [k.strip() for k in kw_raw.split(",") if k.strip()]
        elif itype == "highlight":
            src_text = (options or {}).get("text","")
            unit     = (options or {}).get("mode","word")
            max_sel  = int((options or {}).get("max_select",3))
            src_text = st.text_area("Source text (select from this text)", value=src_text, key=f"{section_key}_edit_h_text_{idx}")
            unit     = st.radio("Selection unit", ["word","sentence"], index=(0 if unit=="word" else 1), horizontal=True, key=f"{section_key}_edit_h_mode_{idx}")
            max_sel  = st.number_input("Max selections", value=max_sel, min_value=1, step=1, key=f"{section_key}_edit_h_max_{idx}")
            tokens   = tokenise(src_text, unit)
            st.caption(f"Preview ({unit}) count = {len(tokens)}")
            correct  = st.multiselect("Correct selections (exact match)", tokens, default=[c for c in (correct or []) if c in tokens], key=f"{section_key}_edit_h_corr_{idx}")
            options  = {"text": src_text, "mode": unit, "max_select": int(max_sel)}

        c1,c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", key=f"{section_key}_save_{idx}"):
                st.session_state.exam[section_key]["tasks"][idx] = {"type": itype, "q": q.strip(), "options": options, "answer": correct}
                st.success("Saved.")
        with c2:
            if st.button("🗑️ Delete", key=f"{section_key}_del_{idx}"):
                st.session_state.exam[section_key]["tasks"].pop(idx)
                st.warning("Deleted.")
    st.markdown("</div>", unsafe_allow_html=True)

def render_tasks_admin(section_key, title=None):
    st.markdown(f"<h4>{title or section_key} — Tasks</h4>", unsafe_allow_html=True)
    tasks = st.session_state.exam[section_key]["tasks"]
    if tasks:
        for i, t in enumerate(tasks):
            with st.expander(f"Task {i+1} — {t.get('type','radio')}: {t.get('q','')[:60]}", expanded=False):
                render_task_editor(section_key, idx=i)
    else:
        st.info("No tasks yet.")
    render_task_editor(section_key, idx=None)

# ---------------- Preview (read-only) ----------------
def render_preview_read_only(exam: dict):
    st.markdown("---")
    st.subheader("👀 Preview (read-only)")
    meta = exam.get("meta", {})
    st.caption(f"Title: {meta.get('title','')} | Level: {meta.get('level','')} | Duration: {meta.get('duration_min','—')} min")
    tabs = st.tabs(SECTIONS)

    # Listening
    with tabs[0]:
        L = exam.get("listening", {})
        if L.get("transcript"):
            st.info(L["transcript"])
        apath = L.get("audio_path","")
        if apath and (MEDIA_DIR / apath).exists():
            st.audio(str(MEDIA_DIR / apath))
        for i, t in enumerate(L.get("tasks", [])):
            ttype = t.get("type")
            if ttype == "radio":
                st.radio(t.get("q",""), t.get("options",[]), index=None, key=f"pv_L_r_{i}", disabled=True)
            elif ttype == "checkbox":
                st.multiselect(t.get("q",""), t.get("options",[]), key=f"pv_L_c_{i}", disabled=True)
            elif ttype == "tfn":
                st.radio(t.get("q",""), ["T","F","NG"], index=None, key=f"pv_L_tfn_{i}", disabled=True)
            elif ttype == "text":
                st.text_input(t.get("q",""), key=f"pv_L_txt_{i}", disabled=True)
            elif ttype == "highlight":
                opts = t.get("options",{})
                tokens = tokenise(opts.get("text",""), opts.get("mode","word"))
                st.multiselect(t.get("q",""), tokens, key=f"pv_L_h_{i}", disabled=True)

    # Reading
    with tabs[1]:
        R = exam.get("reading", {})
        if R.get("passage"): st.info(R["passage"])
        for i, t in enumerate(R.get("tasks", [])):
            ttype = t.get("type")
            if ttype == "radio":
                st.radio(t.get("q",""), t.get("options",[]), index=None, key=f"pv_R_r_{i}", disabled=True)
            elif ttype == "checkbox":
                st.multiselect(t.get("q",""), t.get("options",[]), key=f"pv_R_c_{i}", disabled=True)
            elif ttype == "tfn":
                st.radio(t.get("q",""), ["T","F","NG"], index=None, key=f"pv_R_tfn_{i}", disabled=True)
            elif ttype == "text":
                st.text_input(t.get("q",""), key=f"pv_R_txt_{i}", disabled=True)
            elif ttype == "highlight":
                opts = t.get("options",{})
                tokens = tokenise(opts.get("text",""), opts.get("mode","word"))
                st.multiselect(t.get("q",""), tokens, key=f"pv_R_h_{i}", disabled=True)

    # Use of English
    with tabs[2]:
        U = exam.get("use", {})
        for i, t in enumerate(U.get("tasks", [])):
            ttype = t.get("type")
            if ttype == "radio":
                st.radio(t.get("q",""), t.get("options",[]), index=None, key=f"pv_U_r_{i}", disabled=True)
            elif ttype == "checkbox":
                st.multiselect(t.get("q",""), t.get("options",[]), key=f"pv_U_c_{i}", disabled=True)
            elif ttype == "tfn":
                st.radio(t.get("q",""), ["T","F","NG"], index=None, key=f"pv_U_tfn_{i}", disabled=True)
            elif ttype == "text":
                st.text_input(t.get("q",""), key=f"pv_U_txt_{i}", disabled=True)
            elif ttype == "highlight":
                opts = t.get("options",{})
                tokens = tokenise(opts.get("text",""), opts.get("mode","word"))
                st.multiselect(t.get("q",""), tokens, key=f"pv_U_h_{i}", disabled=True)

    # Writing
    with tabs[3]:
        W = exam.get("writing", {})
        if W.get("prompt"): st.write(W["prompt"])
        st.text_area("Your essay (disabled)", height=200, key="pv_W", disabled=True)

# ---------------- Admin Panel ----------------
def admin_panel():
    st.markdown("---")
    st.subheader("🛡️ Admin Mode (Level-focused)")
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        st.session_state.admin_level = st.selectbox(
            "Level to edit", LEVELS, index=LEVELS.index(st.session_state.admin_level), key="admin_level_sel"
        )
    with col2:
        # ----- إصلاح القيمة الافتراضية للمدة -----
        dur_default = st.session_state.exam.get("meta", {}).get("duration_min")
        if not isinstance(dur_default, (int, float)):
            dur_default = DEFAULT_DUR.get(st.session_state.admin_level, 60)
        dur = st.number_input(
            "Duration (minutes)",
            value=int(dur_default),
            min_value=10,
            step=5,
            key="admin_duration"
        )
        st.session_state.exam.setdefault("meta", {})
        st.session_state.exam["meta"]["duration_min"] = int(dur)
        st.session_state.exam["meta"]["level"] = st.session_state.admin_level
        st.session_state.exam["meta"]["title"] = f"Mega Formation English Exam — {st.session_state.admin_level}"
        st.session_state.exam["meta"]["exam_id"] = f"EXAM_{st.session_state.admin_level}"
    with col3:
        path = exam_path_for(st.session_state.admin_level)
        cA, cB = st.columns(2)
        with cA:
            if st.button("📂 Load this level", key="admin_load_level"):
                data = load_json(path)
                if data:
                    st.session_state.exam = data
                    st.success(f"Loaded {os.path.basename(path)}")
                else:
                    st.session_state.exam = empty_exam(st.session_state.admin_level)
                    st.warning("No saved exam for this level. Started a fresh template.")
        with cB:
            if st.button("💾 Save this level", key="admin_save_level"):
                save_json(path, st.session_state.exam)
                st.success(f"Saved {os.path.basename(path)}")

        # Export / Import
        st.markdown("**Import / Export**")
        colx, coly = st.columns(2)
        with colx:
            if st.button("📤 Export JSON (current)"):
                data = st.session_state.exam
                st.download_button("Download file",
                    data=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
                    file_name=f"EXAM_{st.session_state.admin_level}.json",
                    mime="application/json", key="dl_exam_json")
        with coly:
            up_json = st.file_uploader("📥 Import JSON (replace current)", type=["json"], key="import_exam_json")
            if up_json:
                try:
                    data = json.loads(up_json.read().decode("utf-8"))
                    st.session_state.exam = data
                    st.success("Imported JSON into current editor (remember to Save this level).")
                except Exception as e:
                    st.error(f"Invalid JSON: {e}")

        # Scan & Recover
        st.markdown("**🧭 Scan & Recover old EXAM_*.json**")
        if st.button("🔎 Scan project and copy to exams/"):
            found = 0
            for p in BASE_DIR.rglob("EXAM_*.json"):
                if p.parent == EXAMS_DIR:  # already there
                    continue
                target = EXAMS_DIR / p.name
                try:
                    shutil.copy2(p, target)
                    found += 1
                except Exception:
                    pass
            if found:
                st.success(f"Recovered {found} file(s) into exams/. Refresh sidebar to see them.")
            else:
                st.info("No external EXAM_*.json files found outside exams/.")

    # Listening
    st.markdown("### Listening")
    L = st.session_state.exam["listening"]
    L["transcript"] = st.text_area("Transcript", value=L.get("transcript",""), key="adm_listen_transcript_simple")
    up = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3","wav"], key="adm_listen_upload_simple")
    if up:
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(up.name)}"
        fpath = MEDIA_DIR / fname
        with open(fpath, "wb") as f: f.write(up.read())
        L["audio_path"] = fname
        st.success(f"Saved audio: media/{fname}")
    if L.get("audio_path") and (MEDIA_DIR / L["audio_path"]).exists():
        st.audio(str(MEDIA_DIR / L["audio_path"]))
    st.session_state.exam["listening"] = L
    render_tasks_admin("listening", "Listening")

    # Reading
    st.markdown("### Reading")
    R = st.session_state.exam["reading"]
    R["passage"] = st.text_area("Passage", value=R.get("passage",""), key="adm_read_passage_simple")
    st.session_state.exam["reading"] = R
    render_tasks_admin("reading", "Reading")

    # Use of English
    st.markdown("### Use of English")
    render_tasks_admin("use", "Use of English")

    # Writing
    st.markdown("### Writing")
    W = st.session_state.exam["writing"]
    W["prompt"] = st.text_area("Prompt", value=W.get("prompt",""), key="adm_writing_prompt_simple")
    c1,c2 = st.columns(2)
    W["min_words"] = c1.number_input("Min words", value=int(W.get("min_words",120)), min_value=0, step=5, key="adm_writing_min_simple")
    W["max_words"] = c2.number_input("Max words", value=int(W.get("max_words",150)), min_value=0, step=5, key="adm_writing_max_simple")
    kraw = st.text_input("Keywords (comma-separated)", value=", ".join(W.get("keywords",[])), key="adm_writing_kws_simple")
    W["keywords"] = [k.strip() for k in kraw.split(",") if k.strip()]
    st.session_state.exam["writing"] = W

    # Quick Seeder
    st.markdown("### 🧪 Quick Seeder")
    if st.button("Create sample exam for selected level"):
        lvl = st.session_state.admin_level
        save_json(exam_path_for(lvl), sample_exam(lvl))
        st.success(f"Sample exam created → exams/EXAM_{lvl}.json")

    # ------- زر Preview Exam -------
    st.markdown("### 👀 Preview")
    if st.button("🔎 Preview Exam (read-only)"):
        st.session_state.show_preview = True

    if st.session_state.show_preview:
        render_preview_read_only(st.session_state.exam)

# ---------------- Admin Results Viewer ----------------
def admin_results_viewer():
    st.markdown("---")
    st.subheader("📊 Results Dashboard")
    sel_branch = st.selectbox("Select branch", list(BRANCHES.keys()), key="admin_results_branch")
    bcode = BRANCHES[sel_branch]
    path = RESULT_PATHS[bcode]
    df = results_df(path)
    if df.empty:
        st.warning("No results yet for this branch.")
    else:
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode(), f"{bcode}_results.csv", "text/csv", key="dl_results")

# ---------------- Candidate View ----------------
def render_candidate():
    meta = st.session_state.exam["meta"]
    st.markdown("---")
    st.subheader("🎓 Candidate View")
    c1,c2 = st.columns([1,1])
    with c1:
        st.markdown(f"**Level**: **{meta.get('level','')}**")
    with c2:
        if st.session_state.candidate_started and st.session_state.deadline:
            left = st.session_state.deadline - datetime.utcnow()
            left_sec = max(0, int(left.total_seconds()))
            st.markdown(f"**Time Left**: {left_sec//60:02d}:{left_sec%60:02d}")

    tabs = st.tabs(SECTIONS)

    # Listening
    with tabs[0]:
        L = st.session_state.exam["listening"]
        st.write("**Transcript (optional)**")
        if L.get("transcript"): st.info(L["transcript"])
        apath = L.get("audio_path","")
        if apath and (MEDIA_DIR / apath).exists():
            st.audio(str(MEDIA_DIR / apath))
        else:
            st.caption("No audio uploaded for this level.")
        st.write("**Listening Tasks**")
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
                selected = st.multiselect(f"Select up to {max_sel} {opts.get('mode','word')}(s):", tokens, key=key, max_selections=max_sel)
                if selected:
                    df_sel = pd.DataFrame({"#": range(1,len(selected)+1), opts.get("mode","word"): selected})
                    st.dataframe(df_sel, use_container_width=True)
                st.session_state.answers["Listening"][i] = selected

    # Reading
    with tabs[1]:
        R = st.session_state.exam["reading"]
        st.write("**Reading Passage**")
        if R.get("passage"): st.info(R["passage"])
        st.write("**Reading Tasks**")
        for i, t in enumerate(R.get("tasks", [])):
            key=f"R_{i}"; ttype=t.get("type")
            if ttype=="radio":
                st.session_state.answers["Reading"][i]=st.radio(t["q"], t.get("options",[]), index=None, key=key)
            elif ttype=="checkbox":
                st.session_state.answers["Reading"][i]=st.multiselect(t["q"], t.get("options",[]), key=key)
            elif ttype=="tfn":
                st.session_state.answers["Reading"][i]=st.radio(t["q"], ["T","F","NG"], index=None, key=key)
            elif ttype=="text":
                st.session_state.answers["Reading"][i]=st.text_input(t["q"], key=key)
            elif ttype=="highlight":
                opts=t.get("options",{}); tokens=tokenise(opts.get("text",""), opts.get("mode","word"))
                max_sel=int(opts.get("max_select",3))
                st.write(t["q"])
                selected=st.multiselect(f"Select up to {max_sel} {opts.get('mode','word')}(s):", tokens, key=key, max_selections=max_sel)
                if selected:
                    df_sel=pd.DataFrame({"#":range(1,len(selected)+1), opts.get("mode","word"):selected})
                    st.dataframe(df_sel, use_container_width=True)
                st.session_state.answers["Reading"][i]=selected

    # Use of English
    with tabs[2]:
        U = st.session_state.exam["use"]
        st.write("**Use of English Tasks**")
        for i, t in enumerate(U.get("tasks", [])):
            key=f"U_{i}"; ttype=t.get("type")
            if ttype=="radio":
                st.session_state.answers["Use of English"][i]=st.radio(t["q"], t.get("options",[]), index=None, key=key)
            elif ttype=="checkbox":
                st.session_state.answers["Use of English"][i]=st.multiselect(t["q"], t.get("options",[]), key=key)
            elif ttype=="tfn":
                st.session_state.answers["Use of English"][i]=st.radio(t["q"], ["T","F","NG"], index=None, key=key)
            elif ttype=="text":
                st.session_state.answers["Use of English"][i]=st.text_input(t["q"], key=key)
            elif ttype=="highlight":
                opts=t.get("options",{}); tokens=tokenise(opts.get("text",""), t.get("mode","word"))
                max_sel=int(opts.get("max_select",3))
                st.write(t["q"])
                selected=st.multiselect(f"Select up to {max_sel} {opts.get('mode','word')}(s):", tokens, key=key, max_selections=max_sel)
                if selected:
                    df_sel=pd.DataFrame({"#":range(1,len(selected)+1), opts.get("mode","word"):selected})
                    st.dataframe(df_sel, use_container_width=True)
                st.session_state.answers["Use of English"][i]=selected

    # Writing
    with tabs[3]:
        W = st.session_state.exam["writing"]
        st.write("**Writing Prompt**")
        if W.get("prompt"): st.write(W["prompt"])
        st.caption(f"Target words: {W.get('min_words',0)}–{W.get('max_words',0)} | Keywords: {', '.join(W.get('keywords',[])) or '—'}")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=220, key="W_0")

    # Submit (hidden scoring for candidate)
    cL, cR = st.columns([2,1])
    with cL:
        if st.button("✅ Submit All", type="primary", key="cand_submit_all"):
            # compute
            L_pct = score_section_percent(st.session_state.exam["listening"]["tasks"], st.session_state.answers["Listening"])
            R_pct = score_section_percent(st.session_state.exam["reading"]["tasks"], st.session_state.answers["Reading"])
            U_pct = score_section_percent(st.session_state.exam["use"]["tasks"], st.session_state.answers["Use of English"])
            W = st.session_state.exam["writing"]
            W_text = st.session_state.answers["Writing"].get(0,"")
            W_pct, wc, hits = score_writing_pct(W_text, W.get("min_words",0), W.get("max_words",0), W.get("keywords",[]))
            overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)

            # save only
            bcode = BRANCHES[st.session_state.get("cand_branch_sel","Menzel Bourguiba")]
            row = {
                "timestamp": now_iso(),
                "name": st.session_state.get("cand_name",""),
                "branch": bcode,
                "level": st.session_state.exam["meta"]["level"],
                "exam_id": st.session_state.exam["meta"].get("exam_id",""),
                "overall": overall,
                "Listening": L_pct,
                "Reading": R_pct,
                "Use_of_English": U_pct,
                "Writing": W_pct,
            }
            save_result_row(bcode, row)

            st.info("✅ Your answers have been submitted successfully. Thank you!")
            st.caption("Your exam has been recorded. You can leave the page now.")

    with cR:
        if st.session_state.candidate_started and st.session_state.deadline:
            left = st.session_state.deadline - datetime.utcnow()
            if left.total_seconds() <= 0:
                st.warning("Time is up! Consider submitting.")

# ---------------- Show Admin Panel if logged in ----------------
if st.session_state.is_admin:
    admin_panel()
    admin_results_viewer()

# ---------------- Always show Candidate View ----------------
render_candidate()
