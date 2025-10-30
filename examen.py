# Mega_Level_Builder_Simple_Final.py
# ------------------------------------------------------------------
# Mega Formation — Level-based Exam Builder (Admin hidden)
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

ADMIN_PASS = "megaadmin"

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
    st.session_state.setdefault("is_admin", False)
    st.session_state.setdefault("candidate_level", "B1")
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("exam", empty_exam("B1"))
init_state()

# ---------------- Header ----------------
c1, c2 = st.columns([1, 4])
with c1:
    default_logo_path = os.path.join(MEDIA_DIR, "mega_logo.png")
    if os.path.exists(default_logo_path):
        st.image(default_logo_path, use_container_width=False)
    else:
        st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — Level Exams</h2>", unsafe_allow_html=True)
    st.caption("Candidate mode by default — Admin behind password")

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Candidate")
    st.session_state.candidate_level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.candidate_level))
    cand_name = st.text_input("Your name")
    cand_branch = st.selectbox("Branch", list(BRANCHES.keys()))
    if not st.session_state.candidate_started:
        if st.button("▶️ Start Exam"):
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
                st.error("Exam not prepared yet by admin.")
    else:
        if st.button("🔁 Restart"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = False
            st.session_state.deadline = None

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

# ---------------- Admin Panel ----------------
def render_task_editor(section_key):
    st.markdown(f"#### {section_key} — Add new task")
    TYPES = ["radio","checkbox","text","tfn","highlight"]
    itype = st.selectbox("Type", TYPES, key=f"{section_key}_new_type")
    q = st.text_area("Question / Prompt", key=f"{section_key}_new_q")
    options, correct = [], None
    if itype in ("radio","checkbox"):
        opts_raw = st.text_area("Options (one per line)", key=f"{section_key}_new_opts")
        options = [o.strip() for o in opts_raw.splitlines() if o.strip()]
        if itype == "radio":
            correct = st.selectbox("Correct option", options, index=0 if options else None)
        else:
            correct = st.multiselect("Correct options", options, default=[])
    elif itype == "tfn":
        options = ["T","F","NG"]
        correct = st.selectbox("Correct", options, index=0)
    elif itype == "text":
        kw_raw = st.text_input("Keywords (comma-separated)", key=f"{section_key}_new_corr_txt")
        correct = [k.strip() for k in kw_raw.split(",") if k.strip()]
    elif itype == "highlight":
        src_text = st.text_area("Source text (select from this text)", key=f"{section_key}_new_h_text")
        unit = st.radio("Selection unit", ["word","sentence"], horizontal=True)
        max_sel = st.number_input("Max selections", value=3, min_value=1, step=1)
        tokens = tokenise(src_text, unit)
        corr_sel = st.multiselect("Correct selections", tokens, default=[])
        options = {"text": src_text, "mode": unit, "max_select": int(max_sel)}
        correct = corr_sel
    if st.button("➕ Add task", key=f"{section_key}_add_btn"):
        st.session_state.exam[section_key]["tasks"].append({"type": itype, "q": q.strip(), "options": options, "answer": correct})
        st.success("Task added.")

def admin_panel():
    st.markdown("---")
    st.subheader("🛡️ Admin Mode")
    col1,col2 = st.columns([1,1])
    with col1:
        st.session_state.admin_level = st.selectbox("Level to edit", LEVELS, key="admin_level_sel")
    with col2:
        dur = st.number_input("Duration (minutes)", value=int(st.session_state.exam["meta"].get("duration_min", DEFAULT_DUR.get(st.session_state.admin_level,60))), min_value=10, step=5)
        st.session_state.exam["meta"]["duration_min"] = int(dur)
        st.session_state.exam["meta"]["level"] = st.session_state.admin_level
        st.session_state.exam["meta"]["title"] = f"Mega Formation English Exam — {st.session_state.admin_level}"
        st.session_state.exam["meta"]["exam_id"] = f"EXAM_{st.session_state.admin_level}"
    path = exam_path_for(st.session_state.admin_level)
    c1,c2 = st.columns(2)
    with c1:
        if st.button("📂 Load level"):
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.success(f"Loaded {os.path.basename(path)}")
            else:
                st.session_state.exam = empty_exam(st.session_state.admin_level)
                st.warning("No saved exam for this level.")
    with c2:
        if st.button("💾 Save level"):
            save_json(path, st.session_state.exam)
            st.success(f"Saved {os.path.basename(path)}")

    # Sections
    for section in ["listening","reading","use","writing"]:
        st.markdown(f"### {section.capitalize()}")
        if section == "listening":
            L = st.session_state.exam["listening"]
            L["transcript"] = st.text_area("Transcript", value=L.get("transcript",""))
            up = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3","wav"], key=f"{section}_upload")
            if up:
                fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(up.name)}"
                fpath = os.path.join(MEDIA_DIR, fname)
                with open(fpath, "wb") as f: f.write(up.read())
                L["audio_path"] = fname
                st.success(f"Saved audio: media/{fname}")
            st.session_state.exam["listening"] = L
        elif section == "reading":
            R = st.session_state.exam["reading"]
            R["passage"] = st.text_area("Passage", value=R.get("passage",""))
            st.session_state.exam["reading"] = R
        elif section == "writing":
            W = st.session_state.exam["writing"]
            W["prompt"] = st.text_area("Prompt", value=W.get("prompt",""))
            c1,c2 = st.columns(2)
            W["min_words"] = c1.number_input("Min words", value=int(W.get("min_words",120)), min_value=0, step=5)
            W["max_words"] = c2.number_input("Max words", value=int(W.get("max_words",150)), min_value=0, step=5)
            kraw = st.text_input("Keywords (comma-separated)", value=", ".join(W.get("keywords",[])))
            W["keywords"] = [k.strip() for k in kraw.split(",") if k.strip()]
            st.session_state.exam["writing"] = W
        render_task_editor(section)

# ---------------- Admin Results Viewer ----------------
def admin_results_viewer():
    st.markdown("---")
    st.subheader("📊 Results Dashboard")
    sel_branch = st.selectbox("Select branch", list(BRANCHES.keys()))
    bcode = BRANCHES[sel_branch]
    path = RESULT_PATHS[bcode]
    df = results_df(path)
    if df.empty:
        st.warning("No results yet for this branch.")
    else:
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Download results CSV", df.to_csv(index=False).encode(), f"{bcode}_results.csv", "text/csv")

# ---------------- Candidate View ----------------
def render_candidate():
    if not st.session_state.candidate_started:
        return
    meta = st.session_state.exam["meta"]
    st.markdown("---")
    st.subheader(f"🎓 {meta.get('title','')}")
    tabs = st.tabs(SECTIONS)

    # Listening
    with tabs[0]:
        L = st.session_state.exam["listening"]
        apath = L.get("audio_path","")
        if apath and os.path.exists(os.path.join(MEDIA_DIR, apath)):
            with open(os.path.join(MEDIA_DIR, apath), "rb") as f: st.audio(f.read())
        for i, t in enumerate(L.get("tasks", [])):
            st.session_state.answers["Listening"][i] = st.text_input(t["q"], key=f"L_{i}")

    # Reading
    with tabs[1]:
        R = st.session_state.exam["reading"]
        st.info(R.get("passage",""))
        for i, t in enumerate(R.get("tasks", [])):
            st.session_state.answers["Reading"][i] = st.text_input(t["q"], key=f"R_{i}")

    # Use of English
    with tabs[2]:
        U = st.session_state.exam["use"]
        for i, t in enumerate(U.get("tasks", [])):
            st.session_state.answers["Use of English"][i] = st.text_input(t["q"], key=f"U_{i}")

    # Writing
    with tabs[3]:
        W = st.session_state.exam["writing"]
        st.write(W.get("prompt",""))
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=200)

    # Submit
    if st.button("✅ Submit All", type="primary"):
        L_pct = score_section_percent(st.session_state.exam["listening"]["tasks"], st.session_state.answers["Listening"])
        R_pct = score_section_percent(st.session_state.exam["reading"]["tasks"], st.session_state.answers["Reading"])
        U_pct = score_section_percent(st.session_state.exam["use"]["tasks"], st.session_state.answers["Use of English"])
        W_pct, wc, hits = score_writing_pct(
            st.session_state.answers["Writing"].get(0,""),
            st.session_state.exam["writing"].get("min_words",0),
            st.session_state.exam["writing"].get("max_words",0),
            st.session_state.exam["writing"].get("keywords",[])
        )
        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)
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
        st.info("✅ Your answers have been submitted successfully.")
        st.caption("Your exam has been recorded. You can leave the page now.")

# ---------------- Display ----------------
if st.session_state.is_admin:
    admin_panel()
    admin_results_viewer()

render_candidate()
