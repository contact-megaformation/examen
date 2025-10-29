# Mega_Exam_Admin_Builder.py
# ---------------------------------------------------------
# Mega Formation — Exam Builder (Admin) + Candidate UI
# - Admin: حضّر Listening/Reading/Use/Writing لكل مستوى + ارفع Audio
# - Exam: تنولد الواجهة تلقائيًا من data/exams.json
# - النتائج حسب الفرع (MB/BZ) في results/*.csv
# ---------------------------------------------------------

import streamlit as st
import os, json, io, wave, struct, math, pandas as pd
from datetime import datetime, timedelta

# ------------ Config ------------
LEVELS = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
TOTAL_TIME_MIN = {"A1":60,"A2":60,"B1":90,"B2":90}
PASS_MARK = 60
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}

DATA_DIR   = "data"
AUDIO_DIR  = "audio"
RESULT_DIR = "results"
EXAMS_JSON = os.path.join(DATA_DIR, "exams.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# ------------ Helpers ------------
def load_exams():
    if os.path.exists(EXAMS_JSON):
        with open(EXAMS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    # skeleton
    return {lvl: {
        "listening": {
            "prep_pairs": [],                # [ [term, def], ... ]
            "transcript": "",
            "audio_file": "",                # relative path under audio/
            "task1_truefalse": [],           # [ [statement, "True"/"False"], ... ]
            "task2_speaker": []              # [ [line, "The customer"/"The supplier"], ... ]
        },
        "reading": {
            "passage": "",
            "mcq": []                        # [ [q,[o1,o2,o3,o4],correct], ... ]
        },
        "use": {
            "cloze": []                      # [ [stem,[o1,o2,o3,o4],correct], ... ]
        },
        "writing": {
            "prompt": "",
            "min_words": 50,
            "max_words": 70,
            "keywords": []                   # [ "kw1","kw2",... ]
        }
    } for lvl in LEVELS}

def save_exams(db):
    with open(EXAMS_JSON, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)

def safe_audio_save(uploaded_file, level):
    if not uploaded_file: return ""
    ext = os.path.splitext(uploaded_file.name)[1].lower()
    if ext not in [".mp3",".wav"]: ext = ".wav"
    fname = f"{level}_listening{ext}"
    path = os.path.join(AUDIO_DIR, fname)
    with open(path, "wb") as f: f.write(uploaded_file.read())
    return fname  # relative for storage

def parse_pairs(text):
    # "term | definition" per line
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or "|" not in line: continue
        term,defn = [x.strip() for x in line.split("|",1)]
        out.append([term,defn])
    return out

def parse_truefalse(text):
    # "statement | True/False" per line
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or "|" not in line: continue
        stmt,lab = [x.strip() for x in line.split("|",1)]
        lab = "True" if lab.lower().startswith("t") else "False"
        out.append([stmt, lab])
    return out

def parse_speaker(text):
    # "line | The customer/The supplier" per line
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or "|" not in line: continue
        ln, who = [x.strip() for x in line.split("|",1)]
        who = "The supplier" if "supplier" in who.lower() else "The customer"
        out.append([ln, who])
    return out

def parse_mcq(text):
    # "question | opt1 ; opt2 ; opt3 ; opt4 | correct" per line
    out=[]
    for line in text.splitlines():
        line=line.strip()
        if not line or "|" not in line: continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 3: continue
        q, opts_raw, corr = parts[0], parts[1], parts[2]
        opts = [o.strip() for o in opts_raw.split(";")]
        out.append([q, opts, corr])
    return out

def parse_keywords(text):
    return [k.strip() for k in text.split(",") if k.strip()]

def beep_wav():
    # fallback beep (single valid WAV) in case no audio file
    segs=[(0.35,660),(0.10,None),(0.35,880)]
    frames=[]
    rate=22050; vol=0.30
    for dur,freq in segs:
        n=int(dur*rate)
        for i in range(n):
            sample=0.0 if freq is None else vol*math.sin(2*math.pi*freq*(i/rate))
            frames.append(struct.pack("<h", int(sample*32767)))
    with io.BytesIO() as buf:
        with wave.open(buf,"wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

def score_mcq(rows, answers_map):
    ok=0; recs=[]
    for i,(q,opts,corr) in enumerate(rows):
        usr=answers_map.get(i)
        good = (usr==corr); ok+=int(good)
        recs.append({"#":i+1,"Q":q,"Correct":corr,"User":usr or "","IsCorrect":good})
    return round(100*ok/max(1,len(rows)),1), pd.DataFrame(recs)

def score_tf(rows, answers_map):
    ok=0; recs=[]
    for i,(stmt,corr) in enumerate(rows):
        usr=answers_map.get(i)
        good=(usr==corr); ok+=int(good)
        recs.append({"#":i+1,"Statement":stmt,"Correct":corr,"User":usr or "","IsCorrect":good})
    return round(100*ok/max(1,len(rows)),1), pd.DataFrame(recs)

def score_speaker(rows, answers_map):
    ok=0; recs=[]
    for i,(line,corr) in enumerate(rows):
        usr=answers_map.get(i)
        good=(usr==corr); ok+=int(good)
        recs.append({"#":i+1,"Line":line,"Correct":corr,"User":usr or "","IsCorrect":good})
    return round(100*ok/max(1,len(rows)),1), pd.DataFrame(recs)

# ------------ State ------------
def init_state():
    st.session_state.setdefault("mode","Exam")  # "Admin" / "Exam"
    st.session_state.setdefault("branch","Menzel Bourguiba")
    st.session_state.setdefault("level","B1")
    st.session_state.setdefault("name","")
    st.session_state.setdefault("seed", 12345)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
init_state()

# ------------ Header ------------
st.set_page_config(page_title="Mega Formation — Exam Builder", layout="wide")
c1,c2,c3 = st.columns([1,4,2])
with c1:
    st.toggle("Admin mode", key="is_admin")
    st.session_state.mode = "Admin" if st.session_state.is_admin else "Exam"
with c2:
    st.markdown("## Mega Formation — Exam Builder / Runner")
with c3:
    st.markdown("")

db = load_exams()

# =========================================================
# ADMIN MODE
# =========================================================
if st.session_state.mode == "Admin":
    st.info("🛠️ وضع الأدمين: حضّر/عدّل بنك الأسئلة واحفظ. الأوديو يتخزّن في مجلد audio/.")

    tabL, tabR, tabU, tabW = st.tabs(["Listening","Reading","Use of English","Writing"])
    level = st.selectbox("Level to edit", LEVELS, index=LEVELS.index(st.session_state.level))
    cur = db[level]

    # -------- Listening --------
    with tabL:
        st.subheader("Listening — إعداد سريع بالنص")
        lcol1,lcol2 = st.columns(2)
        with l:

