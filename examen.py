# Mega_Level_Builder_Simple.py
# ------------------------------------------------------------------
# Mega Formation — Level-based Simple Builder (Admin hidden)
# Admin: يختار المستوى A1/A2/B1/B2، يبني/يعدّل أسئلة المستوى ويحفظها في exams/EXAM_<LEVEL>.json
# Candidate: يختار المستوى → يحمّل الامتحان لذلك المستوى إن كان متوفرًا ويعدّي
# أقسام: Listening / Reading / Use of English / Writing
# أنواع أسئلة: radio / checkbox / text / tfn / highlight(word|sentence)
# تصحيح بالـ% لكل سؤال ولكل قسم؛ النتائج تحفظ حسب الفرع (MB/BZ) في results/
# ------------------------------------------------------------------

import streamlit as st
import os, json, time, re
import pandas as pd
from datetime import datetime, timedelta

# ---------------- Config & Constants ----------------
st.set_page_config(page_title="Mega Formation — Level Exams", layout="wide")

LEVELS   = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
DEFAULT_DUR = {"A1":60, "A2":60, "B1":90, "B2":90}
RESULT_PATHS = {"MB":"results/results_MB.csv", "BZ":"results/results_BZ.csv"}

EXAMS_DIR  = "exams"
RESULTS_DIR= "results"
MEDIA_DIR  = "media"
os.makedirs(EXAMS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

ADMIN_PASS = "megaadmin"  # بدّلها كما تحب

# ---------------- Helpers ----------------
def exam_path_for(level:str)->str:
    return os.path.join(EXAMS_DIR, f"EXAM_{level}.json")

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def now_iso(): return datetime.now().isoformat(timespec="seconds")

def results_df(path):
    if os.path.exists(path): return pd.read_csv(path)
    cols=["timestamp","name","branch","level","exam_id","overall","Listening","Reading","Use_of_English","Writing"]
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
    يرجّع % (0..100) حسب نوع السؤال.
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
    rows, q_pcts = [], []
    for i, t in enumerate(tasks or []):
        u = user_map.get(i)
        pct = score_item_pct(t, u)
        q_pcts.append(pct)
        corr = t.get("answer")
        corr_disp = ", ".join(corr) if isinstance(corr, list) else corr
        u_disp    = ", ".join(u) if isinstance(u, list) else u
        text_src  = t.get("q","")
        if t.get("type")=="highlight":
            text_src = t.get("options",{}).get("text","")
        rows.append({"Q#":i+1,"type":t.get("type"),"question":(text_src or "")[:100],
                     "user":u_disp,"correct":corr_disp,"Q%":round(pct,1)})
    section_pct = round(sum(q_pcts)/len(q_pcts), 1) if q_pcts else 0.0
    return section_pct, pd.DataFrame(rows)

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

# ---------------- State ----------------
def init_state():
    st.session_state.setdefault("is_admin", False)         # مخفي افتراضيًا
    st.session_state.setdefault("logo_bytes", None)
    st.session_state.setdefault("admin_level", "B1")       # مستوى الأدمين يحرّر فيه
    st.session_state.setdefault("candidate_level", "B1")   # مستوى المترشح
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("exam", empty_exam("B1"))  # نموذج محلي للتحرير الحالي
init_state()

# ---------------- Header ----------------
c1,c2 = st.columns([1,4])
with c1:
    lg = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if lg: st.session_state.logo_bytes = lg.read()
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, use_container_width=False)
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — Level Exams</h2>", unsafe_allow_html=True)
    st.caption("Candidate mode by default — Admin behind password")

# ---------------- Sidebar: Candidate controls ----------------
with st.sidebar:
    st.header("Candidate")
    st.session_state.candidate_level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.candidate_level), key="cand_level_sel")
    cand_name = st.text_input("Your name", key="cand_name")
    cand_branch = st.selectbox("Branch", list(BRANCHES.keys()), key="cand_branch_sel")

    # حمل إمتحان المستوى للمترشح وقت يبدأ
    if not st.session_state.candidate_started:
        if st.button("▶️ Start Exam", key="start_exam_btn"):
            path = exam_path_for(st.session_state.candidate_level)
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.session_state.answers = {s:{} for s in SECTIONS}
                st.session_state.candidate_started = True
                dur = int(data["meta"].get("duration_min", DEFAULT_DUR.get(data["meta"].get("level","B1"),60)))
                st.session_state.deadline = datetime.utcnow() + timedelta(minutes=dur)
                st.success(f"Loaded {os.path.basename(path)} and started.")
            else:
                st.error("هذا المستوى غير مُحضّر بعد من الأدمين (لا يوجد ملف امتحان).")
    else:
        if st.button("🔁 Restart", key="restart_exam_btn"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = False
            st.session_state.deadline = None

# ---------------- Sidebar: Admin login ----------------
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

# ---------------- Admin tools (hidden unless logged in) ----------------
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
        data = st.session_state.exam[section_key]["tasks"][idx]
        itype = st.selectbox("Type", TYPES, index=TYPES.index(data.get("type","radio")), key=f"{section_key}_edit_type_{idx}")
        q     = st.text_area("Question / Prompt", value=data.get("q",""), key=f"{section_key}_edit_q_{idx}")
        options = data.get("options", [])
        correct = data.get("answer", [])
        if itype in ("radio","checkbox"):
            opts_raw = st.text_area("Options (one per line)", value="\n".join(options), key=f"{section_key}_edit_opts_{idx}")
            options  = [o.strip() for o in opts_raw.splitlines() if o.strip()]
            if itype == "radio":
                ix = options.index(correct) if (correct in options) else 0 if options else 0
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

def admin_panel():
    st.markdown("---")
    st.subheader("🛡️ Admin Mode (Level-focused)")
    # اختيار المستوى الذي تريد تحريره
    col1,col2,col3 = st.columns([1,1,1])
    with col1:
        st.session_state.admin_level = st.selectbox("Level to edit", LEVELS, index=LEVELS.index(st.session_state.admin_level), key="admin_level_sel")
    with col2:
        dur = st.number_input("Duration (minutes)", value=int(st.session_state.exam["meta"].get("duration_min", DEFAULT_DUR.get(st.session_state.admin_level,60))), min_value=10, step=5, key="admin_duration")
        st.session_state.exam["meta"]["duration_min"] = int(dur)
        st.session_state.exam["meta"]["level"] = st.session_state.admin_level
        st.session_state.exam["meta"]["title"] = f"Mega Formation English Exam — {st.session_state.admin_level}"
        st.session_state.exam["meta"]["exam_id"] = f"EXAM_{st.session_state.admin_level}"
    with col3:
        path = exam_path_for(st.session_state.admin_level)
        if st.button("📂 Load this level", key="admin_load_level"):
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.success(f"Loaded {os.path.basename(path)}")
            else:
                st.session_state.exam = empty_exam(st.session_state.admin_level)
                st.warning("No saved exam for this level. Started a fresh template.")
        if st.button("💾 Save this level", key="admin_save_level"):
            save_json(path, st.session_state.exam)
            st.success(f"Saved {os.path.basename(path)}")

    # Listening
    st.markdown("### Listening")
    L = st.session_state.exam["listening"]
    L["transcript"] = st.text_area("Transcript", value=L.get("transcript",""), key="adm_listen_transcript_simple")
    up = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3","wav"], key="adm_listen_upload_simple")
    if up:
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(up.name)}"
        fpath = os.path.join(MEDIA_DIR, fname)
        with open(fpath, "wb") as f: f.write(up.read())
        L["audio_path"] = fname
        st.success(f"Saved audio: media/{fname}")
    if L.get("audio_path") and os.path.exists(os.path.join(MEDIA_DIR, L["audio_path"])):
        with open(os.path.join(MEDIA_DIR, L["audio_path"]), "rb") as f:
            st.audio(f.read())
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
        if apath and os.path.exists(os.path.join(MEDIA_DIR, apath)):
            with open(os.path.join(MEDIA_DIR, apath), "rb") as f: st.audio(f.read())
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
                opts=t.get("options",{}); tokens=tokenise(opts.get("text",""), opts.get("mode","word"))
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

    # Submit
    cL, cR = st.columns([2,1])
    with cL:
        if st.button("✅ Submit All", type="primary", key="cand_submit_all"):
            # Listening
            L_tasks = st.session_state.exam["listening"]["tasks"]
            L_pct, L_df = score_section_percent(L_tasks, st.session_state.answers["Listening"])
            # Reading
            R_tasks = st.session_state.exam["reading"]["tasks"]
            R_pct, R_df = score_section_percent(R_tasks, st.session_state.answers["Reading"])
            # Use
            U_tasks = st.session_state.exam["use"]["tasks"]
            U_pct, U_df = score_section_percent(U_tasks, st.session_state.answers["Use of English"])
            # Writing
            W = st.session_state.exam["writing"]
            W_text = st.session_state.answers["Writing"].get(0,"")
            W_pct, wc, hits = score_writing_pct(W_text, W.get("min_words",0), W.get("max_words",0), W.get("keywords",[]))
            overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)

            st.success(f"**Overall: {overall}%**")
            st.write({"Listening":L_pct, "Reading":R_pct, "Use of English":U_pct, "Writing":W_pct})
            st.caption(f"Writing: words={wc}, keyword hits={hits}/{len(W.get('keywords',[]))}")

            # Save per-branch
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

            # Downloads
            st.download_button("⬇️ Listening report", L_df.to_csv(index=False).encode(), "listening_report.csv", "text/csv", key="dl_listen")
            st.download_button("⬇️ Reading report",   R_df.to_csv(index=False).encode(), "reading_report.csv",   "text/csv", key="dl_read")
            st.download_button("⬇️ Use report",       U_df.to_csv(index=False).encode(), "use_report.csv",       "text/csv", key="dl_use")

    with cR:
        if st.session_state.candidate_started and st.session_state.deadline:
            left = st.session_state.deadline - datetime.utcnow()
            if left.total_seconds() <= 0:
                st.warning("Time is up! Consider submitting.")

# ---------------- Show Admin Panel if logged in ----------------
if st.session_state.is_admin:
    admin_panel()

# ---------------- Always show Candidate View ----------------
render_candidate()
