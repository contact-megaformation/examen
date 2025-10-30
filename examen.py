# Mega_Level_Builder_Simple_v2.py
# ------------------------------------------------------------------
# Mega Formation — Level-based Simple Builder (Admin hidden)
# تحديث: لوجو تلقائي + نتائج فقط للأدمين + فحص مسارات ثابتة + نموذج امتحان جاهز
# ------------------------------------------------------------------

import streamlit as st
import os, json, time, re
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
RESULT_PATHS = {"MB": RESULTS_DIR/"results_MB.csv", "BZ": RESULTS_DIR/"results_BZ.csv"}

ADMIN_PASS = "megaadmin"

# ---------------- Helpers ----------------
def exam_path_for(level:str)->str:
    return str(EXAMS_DIR / f"EXAM_{level}.json")

def load_json(path):
    if not os.path.exists(path): return None
    with open(path, "r", encoding="utf-8") as f: return json.load(f)

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f: json.dump(data, f, ensure_ascii=False, indent=2)

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
    safe = "".join(c if c.isalnum() or c in ("-","_", ".") else "_" for c in (name or ""))
    return (safe[:80] or f"file_{int(time.time())}")

def tokenise(text:str, mode:str):
    if not text: return []
    if mode == "word":
        tokens = re.findall(r"\w+[\w'-]*|[.,!?;:]", text)
        return [t for t in tokens if t.strip()]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def score_item_pct(item, user_val):
    itype = item.get("type"); correct = item.get("answer")
    if itype in ("radio","tfn"): return 100.0 if user_val == correct else 0.0
    if itype in ("checkbox","highlight"):
        corr, sel = set(correct or []), set(user_val or [])
        if not corr: return (100.0 if not sel else 0.0)
        tp, fp = len(corr & sel), len(sel - corr)
        raw = (tp - 0.5*fp) / max(1, len(corr))
        return max(0.0, min(1.0, raw))*100.0
    if itype == "text":
        kws = [k.strip().lower() for k in (correct or []) if k.strip()]
        txt = (user_val or "").strip().lower()
        if not kws: return 100.0 if txt else 0.0
        hits = sum(1 for k in kws if k in txt)
        return (hits/len(kws))*100.0
    return 0.0

def score_section_percent(tasks, user_map):
    q_pcts = []
    for i, t in enumerate(tasks or []):
        pct = score_item_pct(t, user_map.get(i))
        q_pcts.append(pct)
    return round(sum(q_pcts)/len(q_pcts),1) if q_pcts else 0.0

def score_writing_pct(text, min_w, max_w, keywords):
    wc = len((text or "").split())
    base = 40 if (min_w <= wc <= max_w) else (20 if wc>0 else 0)
    hits = sum(1 for k in (keywords or []) if k.lower() in (text or "").lower())
    kw_score = min(60, hits*12)
    return float(min(100, base + kw_score))

def empty_exam(level:str):
    return {
        "meta":{"title": f"Mega Formation English Exam — {level}","level": level,"duration_min": DEFAULT_DUR.get(level,60),"exam_id": f"EXAM_{level}"},
        "listening":{"audio_path":"","transcript":"","tasks":[]},
        "reading":{"passage":"","tasks":[]},
        "use":{"tasks":[]},
        "writing":{"prompt":"","min_words":120,"max_words":150,"keywords":[]}
    }

def sample_exam(level="B1"):
    return {
        "meta": {"title": f"Sample Exam {level}", "level": level, "duration_min": 90, "exam_id": f"EXAM_{level}"},
        "listening": {"audio_path": "", "transcript": "Short audio about a meeting.", "tasks": [
            {"type": "radio", "q": "The meeting is on ___", "options": ["Monday", "Tuesday", "Friday"], "answer": "Friday"}
        ]},
        "reading": {"passage": "An email from the manager about project updates.", "tasks": [
            {"type": "tfn", "q": "The deadline was extended.", "options": ["T", "F", "NG"], "answer": "T"}
        ]},
        "use": {"tasks": [
            {"type": "text", "q": "I have lived in Tunis ___ 2019.", "answer": ["since"]}
        ]},
        "writing": {"prompt": "Write an email to your colleague about finishing a report.", "min_words": 120, "max_words": 150, "keywords": ["report", "deadline", "help"]}
    }

# ---------------- State ----------------
st.session_state.setdefault("is_admin", False)
st.session_state.setdefault("candidate_level", "B1")
st.session_state.setdefault("candidate_started", False)
st.session_state.setdefault("deadline", None)
st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
st.session_state.setdefault("exam", empty_exam("B1"))

# ---------------- Header ----------------
c1, c2 = st.columns([1,4])
with c1:
    logo_path = MEDIA_DIR / "mega_logo.png"
    if logo_path.exists():
        st.image(str(logo_path), use_container_width=False)
    else:
        st.markdown("🧭 **Mega Formation**")
with c2:
    st.markdown("<h2 style='margin:0'>Mega Formation — Level Exams</h2>", unsafe_allow_html=True)
    st.caption("Candidate mode by default — Admin behind password")

# ---------------- Sidebar: Candidate ----------------
with st.sidebar:
    st.header("Candidate")
    st.session_state.candidate_level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.candidate_level))
    cand_name = st.text_input("Your name")
    cand_branch = st.selectbox("Branch", list(BRANCHES.keys()))

    if not st.session_state.candidate_started:
        if st.button("▶️ Start Exam"):
            path = exam_path_for(st.session_state.candidate_level)
            st.caption(f"Loading from: {path}")
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.session_state.answers = {s:{} for s in SECTIONS}
                st.session_state.candidate_started = True
                dur = int(data["meta"].get("duration_min", DEFAULT_DUR.get(data["meta"].get("level","B1"),60)))
                st.session_state.deadline = datetime.utcnow() + timedelta(minutes=dur)
                st.success(f"Exam {st.session_state.candidate_level} loaded.")
            else:
                st.error("⚠️ هذا المستوى غير مُحضّر بعد من الأدمين.")
    else:
        if st.button("🔁 Restart"):
            st.session_state.candidate_started = False
            st.session_state.deadline = None

# ---------------- Admin Login ----------------
with st.sidebar:
    with st.expander("🔐 Admin login", expanded=False):
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if pw == ADMIN_PASS:
                st.session_state.is_admin = True
                st.success("Admin mode enabled.")
            else:
                st.error("Wrong password.")
        if st.session_state.is_admin and st.button("Logout"):
            st.session_state.is_admin = False

    if st.session_state.is_admin:
        with st.expander("📁 Exams folder", expanded=False):
            files = os.listdir(EXAMS_DIR)
            st.write(files or "(empty)")

# ---------------- Admin Panel ----------------
def admin_panel():
    st.markdown("---")
    st.subheader("🛡️ Admin Mode")

    col1, col2, col3 = st.columns(3)
    with col1:
        lvl = st.selectbox("Level to edit", LEVELS)
    with col2:
        dur = st.number_input("Duration (min)", value=DEFAULT_DUR.get(lvl,60))
    with col3:
        if st.button("📂 Load"):
            data = load_json(exam_path_for(lvl)) or empty_exam(lvl)
            st.session_state.exam = data
            st.success(f"Loaded {lvl}")
        if st.button("💾 Save"):
            st.session_state.exam["meta"]["level"] = lvl
            st.session_state.exam["meta"]["duration_min"] = int(dur)
            save_json(exam_path_for(lvl), st.session_state.exam)
            st.success(f"Saved exam {lvl}")

    if st.button("🧪 Create sample exam"):
        save_json(exam_path_for(lvl), sample_exam(lvl))
        st.success(f"Sample exam created for {lvl}")

    # Results
    st.markdown("### 📊 Results Dashboard")
    sel_branch = st.selectbox("Branch", list(BRANCHES.keys()))
    df = results_df(RESULT_PATHS[BRANCHES[sel_branch]])
    if df.empty:
        st.warning("No results yet.")
    else:
        st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("⬇️ Download CSV", df.to_csv(index=False).encode(), f"results_{sel_branch}.csv", "text/csv")

# ---------------- Candidate View ----------------
def render_candidate():
    if not st.session_state.candidate_started: return

    meta = st.session_state.exam["meta"]
    st.markdown("---")
    st.subheader(f"🎓 Exam: {meta.get('level')}")

    tabs = st.tabs(SECTIONS)

    # Listening
    with tabs[0]:
        L = st.session_state.exam["listening"]
        if L.get("audio_path"):
            path = MEDIA_DIR / L["audio_path"]
            if path.exists():
                st.audio(str(path))
        for i, t in enumerate(L.get("tasks", [])):
            st.session_state.answers["Listening"][i] = st.radio(t["q"], t.get("options", []))

    # Reading
    with tabs[1]:
        R = st.session_state.exam["reading"]
        st.info(R.get("passage", ""))
        for i, t in enumerate(R.get("tasks", [])):
            st.session_state.answers["Reading"][i] = st.radio(t["q"], t.get("options", []))

    # Use of English
    with tabs[2]:
        U = st.session_state.exam["use"]
        for i, t in enumerate(U.get("tasks", [])):
            st.session_state.answers["Use of English"][i] = st.text_input(t["q"])

    # Writing
    with tabs[3]:
        W = st.session_state.exam["writing"]
        st.write(W.get("prompt", ""))
        st.session_state.answers["Writing"][0] = st.text_area("Your essay", height=200)

    if st.button("✅ Submit All"):
        L_pct = score_section_percent(st.session_state.exam["listening"]["tasks"], st.session_state.answers["Listening"])
        R_pct = score_section_percent(st.session_state.exam["reading"]["tasks"], st.session_state.answers["Reading"])
        U_pct = score_section_percent(st.session_state.exam["use"]["tasks"], st.session_state.answers["Use of English"])
        W_pct = score_writing_pct(st.session_state.answers["Writing"].get(0, ""), *[W.get(x,0) for x in ("min_words","max_words","keywords")])
        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)

        st.info("✅ Your answers have been submitted successfully. Thank you!")

        bcode = BRANCHES[cand_branch]
        row = {"timestamp": now_iso(), "name": cand_name, "branch": bcode, "level": meta.get("level"), "exam_id": meta.get("exam_id"),
               "overall": overall, "Listening": L_pct, "Reading": R_pct, "Use_of_English": U_pct, "Writing": W_pct}
        save_result_row(bcode, row)

# ---------------- Show ----------------
if st.session_state.is_admin:
    admin_panel()
render_candidate()
