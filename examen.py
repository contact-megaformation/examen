# Mega_Admin_Builder_Exam_v2.py
# ------------------------------------------------------------------
# Mega Formation — Admin Builder (Hidden Admin) + Candidate Clean UI
# - Admin (behind password): يبني الامتحان ويضيف الأسئلة (radio / checkbox / text / tfn / highlight)
# - Highlight type: المترشّح يختار كلمات/جمل من نص، والاختيارات تُعرض في جدول + تتصحّح
# - Listening: رفع Audio + Transcript
# - Exams saved/loaded as JSON in exams/
# - Results per branch (MB/BZ) saved in results/
# - Candidate view نظيفة (لا يرى أدوات الأدمين)
# ------------------------------------------------------------------

import streamlit as st
import os, json, time
import pandas as pd
from datetime import datetime, timedelta

# ==================== ثوابت ====================
LEVELS = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
DEFAULT_DUR = {"A1":60, "A2":60, "B1":90, "B2":90}
RESULT_PATHS = {"MB":"results/results_MB.csv", "BZ":"results/results_BZ.csv"}

EXAMS_DIR = "exams"
RESULTS_DIR = "results"
MEDIA_DIR = "media"

os.makedirs(EXAMS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

# كلمة سر وضع الأدمين (بدّلها هنا)
ADMIN_PASS = "megaadmin"

# ==================== إعداد الصفحة ====================
st.set_page_config(page_title="Mega Formation — Exam Builder (Admin hidden)", layout="wide")
st.markdown("""
<style>
.title {text-align:center; font-size:32px; font-weight:800; margin:0}
.subtitle {text-align:center; color:#666; margin:4px 0 16px}
.card {background:#fff; padding:18px 20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,.06); margin:12px 0}
.badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px}
.kpi {font-size:24px; font-weight:700}
.small {font-size:12px; color:#666}
</style>
""", unsafe_allow_html=True)

# ==================== Helpers ====================
def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def now_iso():
    return datetime.now().isoformat(timespec="seconds")

def results_df(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp","name","branch","level","exam_id","overall","Listening","Reading","Use_of_English","Writing"])

def save_result_row(branch_code, row):
    path = RESULT_PATHS[branch_code]
    df = results_df(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)

def clean_filename(name:str)->str:
    safe = "".join(c if c.isalnum() or c in ("-","_",".") else "_" for c in (name or ""))
    return safe[:80] or f"exam_{int(time.time())}.json"

def init_state():
    st.session_state.setdefault("is_admin", False)     # مخفي افتراضيًا
    st.session_state.setdefault("logo_bytes", None)
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("exam_loaded_path","")
    st.session_state.setdefault("candidate_name","")
    st.session_state.setdefault("candidate_branch","Menzel Bourguiba")
    st.session_state.setdefault("exam", {
        "meta":{
            "title":"Mega Formation English Exam",
            "level":"B1",
            "duration_min": DEFAULT_DUR["B1"],
            "branch":"Menzel Bourguiba",
            "exam_id":""
        },
        "listening":{
            "audio_path":"",  # relative path in media/
            "transcript":"",
            "tasks":[]
        },
        "reading":{
            "passage":"",
            "tasks":[]
        },
        "use":{
            "tasks":[]
        },
        "writing":{
            "prompt":"",
            "min_words":120,
            "max_words":150,
            "keywords":[]
        }
    })

init_state()

# ==================== Admin login (مخفي) ====================
with st.sidebar:
    with st.expander("🔐 Admin login", expanded=False):
        pw = st.text_input("Password", type="password")
        if st.button("Login as admin"):
            if pw == ADMIN_PASS:
                st.session_state.is_admin = True
                st.success("Admin mode enabled.")
            else:
                st.error("Wrong password.")
        if st.session_state.is_admin and st.button("Logout"):
            st.session_state.is_admin = False

# ==================== ترويسة ====================
c1,c2 = st.columns([1,4])
with c1:
    lg = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if lg: st.session_state.logo_bytes = lg.read()
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, use_container_width=False)
with c2:
    st.markdown("<div class='title'>Mega Formation — English Exam</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Candidate mode by default — Admin is hidden</div>", unsafe_allow_html=True)

# ==================== Sidebar (Candidate controls only) ====================
with st.sidebar:
    st.header("Candidate")
    st.session_state.candidate_name = st.text_input("Your name", value=st.session_state.candidate_name)
    st.session_state.candidate_branch = st.selectbox("Branch", list(BRANCHES.keys()),
                                                     index=list(BRANCHES.keys()).index(st.session_state.candidate_branch))

    meta = st.session_state.exam["meta"]
    level = meta.get("level", "B1")
    default_min = DEFAULT_DUR.get(level, 60)

    if not st.session_state.candidate_started:
        if st.button("▶️ Start Exam"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = True
            dur = int(meta.get("duration_min", default_min))
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=dur)
    else:
        if st.button("🔁 Restart"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = False
            st.session_state.deadline = None

# ==================== أدوات الأدمين (مخفية إلا بعد لوجين) ====================
def tokenise(text:str, mode:str):
    if not text: return []
    if mode == "word":
        # تقسيم بسيط على الفراغات وعلامات ترقيم خفيفة
        import re
        tokens = re.findall(r"\w+[\w'-]*|[.,!?;:]", text)
        return [t for t in tokens if t.strip()]
    # sentence
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def points_of(item): return int(item.get("points",1))

def score_item(item, user_val):
    itype = item.get("type")
    correct = item.get("answer")
    pts = points_of(item)
    if itype == "radio":
        return pts if user_val == correct else 0
    if itype == "checkbox":
        return pts if set(user_val or []) == set(correct or []) else 0
    if itype == "tfn":
        return pts if (user_val in ["T","F","NG"] and user_val == correct) else 0
    if itype == "text":
        kws = [k.strip().lower() for k in (correct or []) if k.strip()]
        text = (user_val or "").lower()
        hit = sum(1 for k in kws if k in text)
        return min(pts, hit) if kws else (pts if text.strip() else 0)
    if itype == "highlight":
        # user_val = list of selected tokens; correct = list of target tokens (exact match)
        return pts if set(user_val or []) == set(correct or []) else 0
    return 0

def render_task_editor(section_key, idx=None):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"{section_key} — Add/Edit Task")

    TYPES = ["radio","checkbox","text","tfn","highlight"]
    MODES = ["word","sentence"]  # for highlight

    if idx is None:
        itype = st.selectbox("Question type", TYPES, key=f"{section_key}_new_type")
        q = st.text_area("Question / Prompt", key=f"{section_key}_new_q")
        pts = st.number_input("Points", value=1, step=1, min_value=1, key=f"{section_key}_new_points")

        options, correct = [], None
        h_text, h_mode, h_max, h_correct = "", "word", 3, []

        if itype in ("radio","checkbox"):
            opts_raw = st.text_area("Options (one per line)", key=f"{section_key}_new_opts")
            options = [o.strip() for o in opts_raw.splitlines() if o.strip()]
            if itype == "radio":
                correct = st.selectbox("Correct option", options, index=0 if options else None, key=f"{section_key}_new_corr_radio")
            else:
                correct = st.multiselect("Correct options", options, default=[], key=f"{section_key}_new_corr_ck")

        elif itype == "tfn":
            options = ["T","F","NG"]
            correct = st.selectbox("Correct", options, index=0, key=f"{section_key}_new_corr_tfn")

        elif itype == "text":
            options = []
            correct_raw = st.text_input("Keywords (comma-separated)", key=f"{section_key}_new_corr_txt", placeholder="payment, extension, invoice")
            correct = [k.strip() for k in correct_raw.split(",") if k.strip()]

        elif itype == "highlight":
            st.caption("Highlight settings")
            h_text = st.text_area("Source text (to select from)", key=f"{section_key}_new_h_text")
            h_mode = st.radio("Selection unit", MODES, horizontal=True, key=f"{section_key}_new_h_mode")
            h_max = st.number_input("Max selections allowed", value=3, min_value=1, step=1, key=f"{section_key}_new_h_max")
            tokens = tokenise(h_text, h_mode)
            st.write(f"Preview {h_mode}s ({len(tokens)}):")
            st.write(tokens[:15] if len(tokens)>15 else tokens)
            h_correct = st.multiselect("Correct selections (exact match)", tokens, default=[], key=f"{section_key}_new_h_corr")
            options = {"text": h_text, "mode": h_mode, "max_select": int(h_max)}
            correct = h_correct

        if st.button("➕ Add task", key=f"{section_key}_add_btn"):
            task = {"type": itype, "q": q.strip(), "options": options, "answer": correct, "points": int(pts)}
            st.session_state.exam[section_key]["tasks"].append(task)
            st.success("Task added.")
    else:
        data = st.session_state.exam[section_key]["tasks"][idx]
        itype = st.selectbox("Question type", TYPES, index=TYPES.index(data.get("type","radio")), key=f"{section_key}_edit_type_{idx}")
        q = st.text_area("Question / Prompt", value=data.get("q",""), key=f"{section_key}_edit_q_{idx}")
        pts = st.number_input("Points", value=int(data.get("points",1)), step=1, min_value=1, key=f"{section_key}_edit_points_{idx}")

        options = data.get("options", [])
        correct = data.get("answer", [])

        if itype in ("radio","checkbox"):
            opts_raw = st.text_area("Options (one per line)", value="\n".join(options), key=f"{section_key}_edit_opts_{idx}")
            options = [o.strip() for o in opts_raw.splitlines() if o.strip()]
            if itype == "radio":
                correct = st.selectbox("Correct option", options, index=options.index(correct) if (correct in options) else 0 if options else 0, key=f"{section_key}_edit_corr_radio_{idx}")
            else:
                correct = st.multiselect("Correct options", options, default=[o for o in correct if o in options], key=f"{section_key}_edit_corr_ck_{idx}")

        elif itype == "tfn":
            options = ["T","F","NG"]
            correct = st.selectbox("Correct", options, index=options.index(correct) if correct in options else 0, key=f"{section_key}_edit_corr_tfn_{idx}")

        elif itype == "text":
            correct_raw = st.text_input("Keywords (comma-separated)", value=", ".join(correct if isinstance(correct,list) else []), key=f"{section_key}_edit_corr_txt_{idx}")
            options = []
            correct = [k.strip() for k in correct_raw.split(",") if k.strip()]

        elif itype == "highlight":
            h_text = (options or {}).get("text","")
            h_mode = (options or {}).get("mode","word")
            h_max  = int((options or {}).get("max_select",3))
            h_text = st.text_area("Source text (to select from)", value=h_text, key=f"{section_key}_edit_h_text_{idx}")
            h_mode = st.radio("Selection unit", MODES, index=MODES.index(h_mode) if h_mode in MODES else 0, horizontal=True, key=f"{section_key}_edit_h_mode_{idx}")
            h_max  = st.number_input("Max selections allowed", value=h_max, min_value=1, step=1, key=f"{section_key}_edit_h_max_{idx}")
            tokens = tokenise(h_text, h_mode)
            st.write(f"Preview {h_mode}s ({len(tokens)}):")
            st.write(tokens[:15] if len(tokens)>15 else tokens)
            correct = st.multiselect("Correct selections (exact match)", tokens, default=[c for c in (correct or []) if c in tokens], key=f"{section_key}_edit_h_corr_{idx}")
            options = {"text": h_text, "mode": h_mode, "max_select": int(h_max)}

        c1, c2 = st.columns(2)
        with c1:
            if st.button("💾 Save", key=f"{section_key}_save_{idx}"):
                st.session_state.exam[section_key]["tasks"][idx] = {"type": itype, "q": q.strip(), "options": options, "answer": correct, "points": int(pts)}
                st.success("Saved.")
        with c2:
            if st.button("🗑️ Delete", key=f"{section_key}_del_{idx}"):
                st.session_state.exam[section_key]["tasks"].pop(idx)
                st.warning("Deleted.")

    st.markdown("</div>", unsafe_allow_html=True)

def render_section_tasks_admin(section_key, title=None):
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(title or f"{section_key} — Tasks")
    tasks = st.session_state.exam[section_key]["tasks"]
    if tasks:
        for i, t in enumerate(tasks):
            with st.expander(f"Task {i+1} — {t.get('type','radio')}: {t.get('q','')[:60]}"):
                render_task_editor(section_key, idx=i)
    else:
        st.info("No tasks yet.")
    st.markdown("</div>", unsafe_allow_html=True)
    render_task_editor(section_key, idx=None)

def render_listening_admin():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Listening — Audio & Transcript")
    L = st.session_state.exam["listening"]
    L["transcript"] = st.text_area("Transcript", value=L.get("transcript",""))
    up = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3","wav"], key="listen_upload")
    if up:
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(up.name)}"
        fpath = os.path.join(MEDIA_DIR, fname)
        with open(fpath, "wb") as f: f.write(up.read())
        L["audio_path"] = fname
        st.success(f"Saved audio: media/{fname}")
    apath = L.get("audio_path","")
    if apath and os.path.exists(os.path.join(MEDIA_DIR, apath)):
        with open(os.path.join(MEDIA_DIR, apath), "rb") as f:
            st.audio(f.read())
    else:
        st.caption("No audio uploaded yet.")
    st.session_state.exam["listening"] = L
    st.markdown("</div>", unsafe_allow_html=True)

    render_section_tasks_admin("listening", "Listening — Tasks")

def render_reading_admin():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Reading — Passage")
    R = st.session_state.exam["reading"]
    R["passage"] = st.text_area("Passage", value=R.get("passage",""))
    st.session_state.exam["reading"] = R
    st.markdown("</div>", unsafe_allow_html=True)
    render_section_tasks_admin("reading", "Reading — Tasks")

def render_use_admin():
    render_section_tasks_admin("use", "Use of English — Tasks")

def render_writing_admin():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Writing — Prompt & Auto-criteria")
    W = st.session_state.exam["writing"]
    W["prompt"] = st.text_area("Prompt", value=W.get("prompt",""))
    c1,c2 = st.columns(2)
    W["min_words"] = c1.number_input("Min words", value=int(W.get("min_words",100)), min_value=0, step=5)
    W["max_words"] = c2.number_input("Max words", value=int(W.get("max_words",150)), min_value=0, step=5)
    kw_raw = st.text_input("Keywords (comma-separated)", value=", ".join(W.get("keywords",[])))
    W["keywords"] = [k.strip() for k in kw_raw.split(",") if k.strip()]
    st.session_state.exam["writing"] = W
    st.markdown("</div>", unsafe_allow_html=True)

# ===== أدوات إدارة الامتحانات (حفظ/تحميل) =====
def admin_meta_bar():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Exam meta & Save/Load")
    meta = st.session_state.exam["meta"]
    meta["title"] = st.text_input("Exam Title", value=meta.get("title","Mega Formation English Exam"))
    meta["level"] = st.selectbox("Level", LEVELS, index=LEVELS.index(meta.get("level","B1")))
    meta["branch"] = st.selectbox("Branch", list(BRANCHES.keys()), index=list(BRANCHES.keys()).index(meta.get("branch","Menzel Bourguiba")))
    default_min = DEFAULT_DUR.get(meta["level"], 60)
    meta["duration_min"] = st.number_input("Duration (minutes)", value=int(meta.get("duration_min", default_min)), min_value=10, step=5)
    meta["exam_id"] = st.text_input("Exam ID (for saving)", value=meta.get("exam_id",""))
    st.session_state.exam["meta"] = meta

    st.markdown("---")
    fname_default = clean_filename(meta["exam_id"] or f"{meta['level']}_{BRANCHES[meta['branch']]}_{datetime.now().strftime('%Y%m%d')}.json")
    c1,c2 = st.columns(2)
    with c1:
        if st.button("💾 Save to exams/"):
            path = os.path.join(EXAMS_DIR, fname_default)
            save_json(path, st.session_state.exam)
            st.session_state.exam_loaded_path = path
            st.success(f"Saved: {path}")
    with c2:
        existing = [f for f in os.listdir(EXAMS_DIR) if f.lower().endswith(".json")]
        sel = st.selectbox("Load exam", ["-- select --"]+existing, key="load_sel")
        if sel != "-- select --" and st.button("📂 Load selected"):
            path = os.path.join(EXAMS_DIR, sel)
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.session_state.exam_loaded_path = path
                st.success(f"Loaded: {path}")
    st.markdown("</div>", unsafe_allow_html=True)

# ==================== Candidate View ====================
def render_candidate():
    meta = st.session_state.exam["meta"]
    st.markdown("---")
    st.subheader("🎓 Candidate View")
    colA, colB = st.columns([1,1])
    with colA:
        st.markdown(f"**Level**: <span class='badge'>{meta.get('level','')}</span>", unsafe_allow_html=True)
    with colB:
        if st.session_state.candidate_started and st.session_state.deadline:
            left = st.session_state.deadline - datetime.utcnow()
            left_sec = max(0, int(left.total_seconds()))
            mm, ss = left_sec//60, left_sec%60
            st.markdown(f"**Time Left**: <span class='kpi'>{mm:02d}:{ss:02d}</span>", unsafe_allow_html=True)

    tabs = st.tabs(SECTIONS)

    # Listening
    with tabs[0]:
        L = st.session_state.exam["listening"]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Transcript (optional)**")
        if L.get("transcript"): st.info(L["transcript"])
        apath = L.get("audio_path","")
        apath_abs = os.path.join(MEDIA_DIR, apath) if apath else ""
        if apath and os.path.exists(apath_abs):
            with open(apath_abs, "rb") as f: st.audio(f.read())
        else:
            st.caption("No audio uploaded.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
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
                src_text = opts.get("text","")
                mode = opts.get("mode","word")
                max_sel = int(opts.get("max_select",3))
                tokens = tokenise(src_text, mode)
                # multiselect للاختيار
                st.write(t["q"])
                selected = st.multiselect(f"Select up to {max_sel} {mode}(s):", tokens, key=key, max_selections=max_sel)
                # جدول فوري
                if selected:
                    df_sel = pd.DataFrame({"#": range(1, len(selected)+1), mode: selected})
                    st.dataframe(df_sel, use_container_width=True)
                st.session_state.answers["Listening"][i] = selected
        st.markdown("</div>", unsafe_allow_html=True)

    # Reading
    with tabs[1]:
        R = st.session_state.exam["reading"]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Reading Passage**")
        if R.get("passage"): st.info(R["passage"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Reading Tasks**")
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
                src_text = opts.get("text","")
                mode = opts.get("mode","word")
                max_sel = int(opts.get("max_select",3))
                tokens = tokenise(src_text, mode)
                st.write(t["q"])
                selected = st.multiselect(f"Select up to {max_sel} {mode}(s):", tokens, key=key, max_selections=max_sel)
                if selected:
                    df_sel = pd.DataFrame({"#": range(1, len(selected)+1), mode: selected})
                    st.dataframe(df_sel, use_container_width=True)
                st.session_state.answers["Reading"][i] = selected
        st.markdown("</div>", unsafe_allow_html=True)

    # Use of English
    with tabs[2]:
        U = st.session_state.exam["use"]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Use of English Tasks**")
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
                src_text = opts.get("text","")
                mode = opts.get("mode","word")
                max_sel = int(opts.get("max_select",3))
                tokens = tokenise(src_text, mode)
                st.write(t["q"])
                selected = st.multiselect(f"Select up to {max_sel} {mode}(s):", tokens, key=key, max_selections=max_sel)
                if selected:
                    df_sel = pd.DataFrame({"#": range(1, len(selected)+1), mode: selected})
                    st.dataframe(df_sel, use_container_width=True)
                st.session_state.answers["Use of English"][i] = selected
        st.markdown("</div>", unsafe_allow_html=True)

    # Writing
    with tabs[3]:
        W = st.session_state.exam["writing"]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Writing Prompt**")
        if W.get("prompt"): st.write(W["prompt"])
        st.caption(f"Target words: {W.get('min_words',0)}–{W.get('max_words',0)} | Keywords: {', '.join(W.get('keywords',[])) or '—'}")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=220, key="W_0")
        st.markdown("</div>", unsafe_allow_html=True)

    # Submit
    def score_section(tasks, user_map):
        total_pts = sum(points_of(t) for t in tasks)
        got_pts = 0
        rows = []
        for i, t in enumerate(tasks):
            u = user_map.get(i)
            s = score_item(t, u)
            got_pts += s
            # تنظيف عرض الصحيح
            corr = t.get("answer")
            if isinstance(corr, list): corr_disp = ", ".join(corr)
            else: corr_disp = corr
            if isinstance(u, list): u_disp = ", ".join(u)
            else: u_disp = u
            rows.append({"Q#":i+1,"type": t.get("type"), "q": t.get("q","")[:80],
                         "user": u_disp, "correct": corr_disp, "points": points_of(t), "score": s})
        pct = round(100.0 * got_pts / total_pts, 1) if total_pts>0 else 0.0
        return pct, pd.DataFrame(rows)

    def score_writing(text, min_w, max_w, keywords):
        wc = len((text or "").split())
        base = 40 if (min_w and max_w and min_w <= wc <= max_w) else (20 if wc>0 else 0)
        hits = sum(1 for k in (keywords or []) if k.lower() in (text or "").lower())
        kw_score = min(60, hits * 12)
        return min(100, base + kw_score), wc, hits

    colL, colR = st.columns([2,1])
    with colL:
        can_submit = st.session_state.candidate_started
        if st.button("✅ Submit All", type="primary", disabled=not can_submit):
            st.session_state.candidate_started = False
            # Listening
            L_tasks = st.session_state.exam["listening"]["tasks"]
            L_pct, L_df = score_section(L_tasks, st.session_state.answers["Listening"])
            # Reading
            R_tasks = st.session_state.exam["reading"]["tasks"]
            R_pct, R_df = score_section(R_tasks, st.session_state.answers["Reading"])
            # Use
            U_tasks = st.session_state.exam["use"]["tasks"]
            U_pct, U_df = score_section(U_tasks, st.session_state.answers["Use of English"])
            # Writing
            W = st.session_state.exam["writing"]
            W_text = st.session_state.answers["Writing"].get(0,"")
            W_pct, wc, hits = score_writing(W_text, W.get("min_words",0), W.get("max_words",0), W.get("keywords",[]))

            overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)
            st.success(f"**Overall: {overall}%**")
            st.write({"Listening":L_pct, "Reading":R_pct, "Use of English":U_pct, "Writing":W_pct})
            st.caption(f"Writing: words={wc}, keyword hits={hits}/{len(W.get('keywords',[]))}")

            # حفظ حسب الفرع
            bcode = BRANCHES[st.session_state.candidate_branch]
            exam_id = st.session_state.exam["meta"].get("exam_id") or os.path.basename(st.session_state.exam_loaded_path or "")
            row = {
                "timestamp": now_iso(),
                "name": st.session_state.candidate_name,
                "branch": bcode,
                "level": st.session_state.exam["meta"]["level"],
                "exam_id": exam_id,
                "overall": overall,
                "Listening": L_pct,
                "Reading": R_pct,
                "Use_of_English": U_pct,
                "Writing": W_pct,
            }
            save_result_row(bcode, row)

            # تنزيل تقارير (اختياري)
            st.download_button("⬇️ Listening report", L_df.to_csv(index=False).encode(), "listening_report.csv", "text/csv")
            st.download_button("⬇️ Reading report",   R_df.to_csv(index=False).encode(), "reading_report.csv",   "text/csv")
            st.download_button("⬇️ Use report",       U_df.to_csv(index=False).encode(), "use_report.csv",       "text/csv")

    with colR:
        if st.session_state.candidate_started and st.session_state.deadline:
            left = st.session_state.deadline - datetime.utcnow()
            if left.total_seconds() <= 0:
                st.warning("Time is up! Consider submitting.")

# ==================== Admin UI (hidden unless logged in) ====================
if st.session_state.is_admin:
    st.markdown("---")
    st.subheader("🛡️ Admin Mode")
    admin_meta_bar()
    tabs_admin = st.tabs(["Listening","Reading","Use of English","Writing","Results"])
    with tabs_admin[0]:
        render_listening_admin()
    with tabs_admin[1]:
        render_reading_admin()
    with tabs_admin[2]:
        render_use_admin()
    with tabs_admin[3]:
        render_writing_admin()
    with tabs_admin[4]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Results by Branch")
        sel_branch = st.selectbox("Branch", list(BRANCHES.keys()))
        code = BRANCHES[sel_branch]
        df = results_df(RESULT_PATHS[code])
        if df.empty:
            st.info("No results yet.")
        else:
            lvl_filter = st.multiselect("Filter levels", LEVELS, default=LEVELS)
            view = df[df["level"].isin(lvl_filter)].copy()
            st.dataframe(view, use_container_width=True)
            st.download_button("⬇️ Download CSV", data=view.to_csv(index=False).encode(),
                               file_name=f"results_{code}.csv", mime="text/csv")
        st.markdown("</div>", unsafe_allow_html=True)

# ====== نهاية الكود ======
