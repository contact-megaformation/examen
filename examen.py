# Mega_BC_Style_Exam_Branches.py
# ---------------------------------------------------------
# Mega Formation — BC-like Exam (A1–B2) with Branch results (MB/BZ)
# Listening (TTS audio) • Reading • Use of English • Writing
# Admin dashboard (no email): results saved to /mnt/data/results_{MB|BZ}.csv
# Total time: A1/A2=60 min, B1/B2=90 min
# ---------------------------------------------------------

import streamlit as st
import random, io, os, wave, struct, math, tempfile
import pandas as pd
from datetime import datetime, timedelta

# ------------- Page & Style -------------
st.set_page_config(page_title="Mega Formation — BC-style Exam (Branches)", layout="wide")
st.markdown("""
<style>
.title {text-align:center; font-size:32px; font-weight:800; margin:0}
.subtitle {text-align:center; color:#666; margin:2px 0 14px}
.card {background:#fff; padding:18px 20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,.06); margin:12px 0}
.kpi {font-size:24px; font-weight:700}
.badge {display:inline-block; padding:3px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px}
.small {font-size:12px; color:#666}
</style>
""", unsafe_allow_html=True)

LEVELS = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
TOTAL_TIME_MIN = {"A1":60,"A2":60,"B1":90,"B2":90}
PASS_MARK = 60
BRANCHES = {"Menzel Bourguiba":"MB", "Bizerte":"BZ"}
RESULT_PATHS = {"MB":"/mnt/data/results_MB.csv", "BZ":"/mnt/data/results_BZ.csv"}

# ------------- Audio (TTS + solid fallback) -------------
def make_tones_wav(segments, rate=22050, volume=0.30):
    frames=[]
    for dur,freq in segments:
        n=int(dur*rate)
        for i in range(n):
            sample = 0.0 if freq is None else volume*math.sin(2*math.pi*freq*(i/rate))
            frames.append(struct.pack("<h", int(sample*32767)))
    with io.BytesIO() as buf:
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(22050)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

def tts_or_beep_bytes(text, rate_wpm=165):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate_wpm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp=tf.name
        engine.save_to_file(text or "Listening item.", tmp)
        engine.runAndWait()
        with open(tmp,"rb") as f: data=f.read()
        try: os.remove(tmp)
        except: pass
        if len(data) < 2000:  # invalid/empty
            raise RuntimeError("Invalid TTS WAV")
        return data
    except Exception:
        # single valid WAV (not 0:00)
        return make_tones_wav([(0.35,660),(0.10,None),(0.35,880)])

# ------------- Banks (BC-like; B1 aligned with your screenshots) -------------
LISTENING = {
    # A1/A2 مختصرة – تنجم تكبّر لاحقًا
    "A1": {
        "prep_pairs": [("lunch","midday meal"), ("bus","public transport"), ("park","green area in a town")],
        "transcript": "Hello, I'm Anna. Today is Monday. I take the eight o'clock bus to work. At lunch, I meet Tom in the park.",
        "task1_truefalse": [
            ("Anna takes the bus at 8:00.", "True"),
            ("She meets John in the park.", "False"),
            ("Today is Sunday.", "False"),
            ("She meets Tom at lunch.", "True"),
            ("She goes by car.", "False"),
            ("The meeting is in the park.", "True"),
        ],
        "task2_speaker": [
            ("Please be on time.", "The supplier"),
            ("I meet Tom at lunch.", "The customer"),
            ("I can help you.", "The supplier"),
            ("I need to catch the bus.", "The customer"),
            ("Let me see what I can do.", "The supplier"),
            ("Thanks for your help.", "The customer"),
            ("I'm happy to help you.", "The supplier"),
            ("I appreciate your help.", "The customer"),
        ]
    },
    "A2": {
        "prep_pairs": [("guided tour","visit with a guide"), ("ticket","paid pass"), ("arrive","to reach a place")],
        "transcript": "Good afternoon. The city museum opens at ten. Guided tours start at eleven and last one hour. Tickets are cheaper online.",
        "task1_truefalse": [
            ("The museum opens at 10.", "True"),
            ("Tours start at 10.", "False"),
            ("Tours last one hour.", "True"),
            ("Tickets are cheaper online.", "True"),
            ("The museum is closed on Monday.", "True"),
            ("Arrive ten minutes late.", "False"),
        ],
        "task2_speaker": [
            ("Please arrive ten minutes early.", "The supplier"),
            ("Could you book online?", "The customer"),
            ("Tours start at eleven.", "The supplier"),
            ("I will come on Sunday.", "The customer"),
            ("Tickets are cheaper online.", "The supplier"),
            ("I need two tickets, please.", "The customer"),
            ("Let me check availability.", "The supplier"),
            ("Thanks for the information.", "The customer"),
        ]
    },
    # --------- B1 copy of BC-style phone call structure ---------
    "B1": {
        "prep_pairs": [
            ("extension","more time allowed"),
            ("invoice","document showing what to pay"),
            ("cash flow","money in and out timing")
        ],
        "transcript": (
            "Customer: Hello Junko, it's Andrea. I'm calling about our last order. We have a small cash flow problem this month "
            "and I wanted to ask if we could have an extension on the payment terms. "
            "Supplier: I see. Let me check what we can do. Delivery should have arrived already. "
            "Customer: Yes, delivery is fine. We just need a little more time. You'd really be helping us. "
            "Supplier: I'm happy to help you. I think we can make an exception this time. I can extend the last invoice to sixty days. "
            "Customer: Thank you. I appreciate your help. I'll look out for your email confirmation."
        ),
        # Task 1: True/False – مطابق تقريبًا للي في السكرينات
        "task1_truefalse": [
            ("The delivery hasn’t arrived yet.", "False"),
            ("Andrea is having cash flow issues and needs a payment extension.", "True"),
            ("Andrea usually asks for an extension of the payment terms.", "False"),
            ("Andrea has a new order, even bigger than the last one.", "False"),
            ("Junko can extend the payment terms on the last order to 60 days.", "True"),
            ("Junko will send Andrea an email confirmation.", "True"),
        ],
        # Task 2: Grouping "Who says this line?"
        "task2_speaker": [
            ("You’d really be helping us.", "The customer"),
            ("I need a favour.", "The customer"),
            ("I appreciate your help.", "The customer"),
            ("Let me see what I can do.", "The supplier"),
            ("I’m happy to help you.", "The supplier"),
            ("I’m not sure if I can do that.", "The supplier"),
            ("I think we can make an exception this time.", "The supplier"),
            ("I promise this won’t become the norm.", "The customer"),
        ]
    },
    "B2": {
        "prep_pairs": [("protocols","agreed rules"), ("cohesion","team unity"), ("check-ins","regular brief updates")],
        "transcript": "The panel concluded remote work increased access to global talent but raised communication challenges. They recommended clearer protocols and regular check-ins.",
        "task1_truefalse": [
            ("Remote work increased access to global talent.", "True"),
            ("No communication challenges were mentioned.", "False"),
            ("They recommended clearer protocols.", "True"),
            ("They suggested fewer check-ins.", "False"),
            ("The context involves teams across time zones.", "True"),
            ("They cancelled remote work.", "False"),
        ],
        "task2_speaker": [
            ("We need clearer protocols.", "The supplier"),
            ("Remote work helps us hire globally.", "The customer"),
            ("Let's schedule regular check-ins.", "The supplier"),
            ("We struggle with communication sometimes.", "The customer"),
            ("Thanks, that should help cohesion.", "The customer"),
            ("I'll prepare a proposal.", "The supplier"),
            ("Could we try weekly updates?", "The customer"),
            ("Yes, let's do that.", "The supplier"),
        ]
    },
}

READING = {
    "A1":{"passage":"Maria lives near the sea and works in a café. After work she goes to the beach with friends.",
          "mcq":[("Where does Maria live?",["near the sea","in the desert","in a factory","in a hospital"],"near the sea"),
                 ("Where does she work?",["in a café","at home","in a bank","in a school"],"in a café"),
                 ("What does she do after work?",["goes to the beach","works again","drives to Paris","goes to sleep"],"goes to the beach")]},
    "A2":{"passage":"The city library moved to a larger building and offers weekend workshops and free Wi-Fi.",
          "mcq":[("Why did the library move?",["to a larger building","it closed","to a smaller place","for a party"],"to a larger building"),
                 ("Which service is new?",["weekend workshops","gym","cinema","paid Wi-Fi"],"weekend workshops"),
                 ("What do many students do there?",["study","cook","sleep","dance"],"study")]},
    "B1":{"passage":"Volunteering connects people with local needs and strengthens communities when volunteers receive training.",
          "mcq":[("What connects people with local needs?",["Volunteering","Prices","Holidays","Traffic"],"Volunteering"),
                 ("What helps volunteers?",["training","prizes","cars","uniforms"],"training"),
                 ("What can volunteering strengthen?",["communities","storms","costs","exams"],"communities")]},
    "B2":{"passage":"Integrating intermittent renewables into aging grids requires coordination and investment; storage helps balance fluctuations.",
          "mcq":[("What requires coordination?",["integration of renewables","closing schools","tourism","tax cuts"],"integration of renewables"),
                 ("What helps balance fluctuations?",["storage","advertising","meetings","tax cuts"],"storage"),
                 ("What kind of grids are mentioned?",["aging","brand-new","perfect","private"],"aging")]},
}

USE = {
    "A1":[("He __ a student.",["is","are","am","be"],"is"),
          ("We __ in Tunis.",["live","lives","living","to live"],"live"),
          ("There __ two apples.",["are","is","be","been"],"are"),
          ("I __ coffee every day.",["drink","drinks","drank","drinking"],"drink")],
    "A2":[("I have lived here __ 2019.",["since","for","during","from"],"since"),
          ("If it rains, we __ at home.",["will stay","stay","stayed","stays"],"will stay"),
          ("He can't __ the meeting.",["attend","to attend","attends","attending"],"attend"),
          ("We didn't go out __ the rain.",["because of","because","so","although"],"because of")],
    "B1":[("I wish I __ more time.",["had","have","would have","am having"],"had"),
          ("Hardly __ the meeting begun when the alarm rang.",["had","has","did","was"],"had"),
          ("He denied __ the window.",["breaking","to break","break","to have broke"],"breaking"),
          ("We need someone __ can code.",["who","which","whom","what"],"who")],
    "B2":[("No sooner __ the announcement made than shares fell.",["had","was","has","having"],"had"),
          ("The project, __ objectives were unclear, was delayed.",["whose","who's","which","that"],"whose"),
          ("Had I known, I __ earlier.",["would have left","left","would leave","had left"],"would have left"),
          ("He insisted that she __ present.",["be","was","is","would be"],"be")],
}

WRITING = {
    "A1":("Write about your daily routine (50–70 words).",50,70,["morning","work","eat","go","home"]),
    "A2":("Describe your last holiday (80–100 words).",80,100,["where","when","with","activities","feelings"]),
    "B1":("Do you prefer studying alone or in groups? Explain (120–150 words).",120,150,["prefer","because","example","time","learn"]),
    "B2":("Some companies allow remote work. Discuss advantages and disadvantages (180–220 words).",180,220,["productivity","balance","communication","costs","team"]),
}

# ------------- State -------------
def init_state():
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("branch", "Menzel Bourguiba")
    st.session_state.setdefault("level", "B1")
    st.session_state.setdefault("seed", random.randint(1,10_000_000))
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("logo_bytes", None)
    st.session_state.setdefault("is_admin", False)

init_state()

# ------------- Header -------------
c1,c2,c3 = st.columns([1,4,1])
with c1:
    lg = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
    if lg: st.session_state.logo_bytes = lg.read()
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, use_container_width=False)
with c2:
    st.markdown("<div class='title'>Mega Formation — BC-style Exam</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Listening • Reading • Use of English • Writing</div>", unsafe_allow_html=True)
with c3:
    st.toggle("Admin dashboard", key="is_admin")

# ------------- Sidebar (Setup) -------------
with st.sidebar:
    st.header("Setup")
    st.session_state.name   = st.text_input("Candidate name", value=st.session_state.name)
    st.session_state.branch = st.selectbox("Branch", list(BRANCHES.keys()), index=list(BRANCHES.keys()).index(st.session_state.branch))
    st.session_state.level  = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.level))
    st.session_state.seed   = st.number_input("Random seed", value=st.session_state.seed, step=1, format="%d")

    if not st.session_state.started:
        if st.button("▶️ Start Exam"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.started = True
            mins = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=mins)
    else:
        if st.button("🔁 Restart"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.seed = random.randint(1,10_000_000)
            mins = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=mins)

    # Quick audio test
    if st.button("🔊 Audio self-test"):
        st.audio(make_tones_wav([(0.35,660),(0.1,None),(0.35,880)]))

# ------------- Admin Dashboard -------------
def load_results_csv(code):
    path = RESULT_PATHS[code]
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame(columns=["timestamp","name","branch","level","overall",
                                 "Listening","Reading","Use of English","Writing"])

if st.session_state.is_admin:
    st.subheader("🛡️ Admin — Results by Branch")
    bcode = BRANCHES[st.session_state.branch]
    df = load_results_csv(bcode)
    if df.empty:
        st.info("لا توجد نتائج بعد لهذا الفرع.")
    else:
        col1,col2 = st.columns(2)
        with col1:
            lvl_filter = st.multiselect("Filter levels", LEVELS, default=LEVELS)
        with col2:
            date_from = st.date_input("From", value=None)
        view = df[df["level"].isin(lvl_filter)].copy()
        if date_from:
            view = view[pd.to_datetime(view["timestamp"]).dt.date >= date_from]
        st.dataframe(view, use_container_width=True)
        st.download_button("⬇️ Download CSV", data=view.to_csv(index=False).encode(),
                           file_name=f"results_{bcode}.csv", mime="text/csv")
    st.divider()

# ------------- Helpers -------------
def time_left_str():
    if not st.session_state.deadline: return ""
    left = st.session_state.deadline - datetime.utcnow()
    if left.total_seconds() <= 0: return "00:00"
    mm, ss = divmod(int(left.total_seconds()), 60)
    return f"{mm:02d}:{ss:02d}"

def score_tf(rows, answers_map):
    recs=[]; ok=0
    for i,(stmt,corr) in enumerate(rows):
        usr = answers_map.get(i)
        good = (usr == corr); ok += int(good)
        recs.append({"#":i+1,"Statement":stmt,"Correct":corr,"User":usr or "","IsCorrect":good})
    pct = round(100*ok/max(1,len(rows)),1)
    return pct, pd.DataFrame(recs)

def score_mcq(rows, answers_map):
    recs=[]; ok=0
    for i,(q,opts,corr) in enumerate(rows):
        usr = answers_map.get(i)
        good = (usr == corr); ok += int(good)
        recs.append({"#":i+1,"Question":q,"Correct":corr,"User":usr or "","IsCorrect":good})
    pct = round(100*ok/max(1,len(rows)),1)
    return pct, pd.DataFrame(recs)

def score_speaker(rows, answers_map):
    # rows: [(line, "The customer"/"The supplier"), ...]
    recs=[]; ok=0
    for i,(line,corr) in enumerate(rows):
        usr = answers_map.get(i)
        good = (usr == corr); ok += int(good)
        recs.append({"#":i+1,"Line":line,"Correct":corr,"User":usr or "","IsCorrect":good})
    pct = round(100*ok/max(1,len(rows)),1)
    return pct, pd.DataFrame(recs)

# ------------- Exam Body -------------
if st.session_state.started:
    k1,k2,k3 = st.columns([1,1,2])
    with k1:
        st.markdown("**Level**"); st.markdown(f"<span class='badge'>{st.session_state.level}</span>", unsafe_allow_html=True)
    with k2:
        st.markdown("**Time Left**"); st.markdown(f"<div class='kpi'>{time_left_str()}</div>", unsafe_allow_html=True)
    with k3:
        st.info("Complete the 4 sections then Submit. Listening = Preparation + Transcript + T/F + Speaker grouping.")

    if time_left_str()=="00:00":
        st.warning("Time is up! Auto-submitting your exam.")

    lvl = st.session_state.level
    rnd = random.Random(st.session_state.seed)
    tabs = st.tabs(SECTIONS)

    # ---------- Listening ----------
    with tabs[0]:
        bank = LISTENING[lvl]
        # Preparation
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Preparation — Vocabulary")
        L_prep_ans = {}
        for i,(term,defn) in enumerate(bank["prep_pairs"]):
            opts = [defn,"a different meaning","an unrelated meaning","another possible meaning"]
            rnd.shuffle(opts)
            L_prep_ans[i] = st.radio(f"{term}", opts, index=None, key=f"Lprep_{i}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Transcript + Audio
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Transcript & Audio")
        st.caption("استمع ثم أجب عن الأسئلة.")
        st.audio(tts_or_beep_bytes(bank["transcript"]))
        with st.expander("Show transcript"):
            st.write(bank["transcript"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Task 1: True/False
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Task 1 — True / False")
        L_t1_ans = {}
        for i,(stmt, corr) in enumerate(bank["task1_truefalse"]):
            L_t1_ans[i] = st.radio(stmt, ["True","False"], index=None, key=f"LT1_{i}")
        st.markdown("</div>", unsafe_allow_html=True)

        # Task 2: Speaker grouping
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Task 2 — Who says this line?")
        L_t2_ans = {}
        for i,(line, corr) in enumerate(bank["task2_speaker"]):
            L_t2_ans[i] = st.radio(f"“{line}”", ["The customer","The supplier"], index=None, key=f"LT2_{i}")
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.answers["Listening"] = {
            "prep":L_prep_ans,
            "t1":L_t1_ans,
            "t2":L_t2_ans
        }

    # ---------- Reading ----------
    with tabs[1]:
        data = READING[lvl]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Reading Passage")
        st.info(data["passage"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Task — Multiple choice")
        R_ans = {}
        for i,(q,opts,corr) in enumerate(data["mcq"]):
            R_ans[i] = st.radio(q, opts, index=None, key=f"R_{i}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.answers["Reading"] = {"mcq":R_ans}

    # ---------- Use of English ----------
    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Use of English — Cloze")
        U_ans = {}
        for i,(stem,opts,corr) in enumerate(USE[lvl]):
            U_ans[i] = st.selectbox(stem, opts, index=None, key=f"U_{i}")
        st.markdown("</div>", unsafe_allow_html=True)
        st.session_state.answers["Use of English"] = {"cloze":U_ans}

    # ---------- Writing ----------
    with tabs[3]:
        prompt, lo, hi, kws = WRITING[lvl]
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Writing")
        st.write(f"**Prompt:** {prompt}")
        st.caption(f"Target words: {lo}–{hi}. Include: {', '.join(kws)}")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=220, key="W_0")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- Submit ----------
    if st.button("✅ Submit All", type="primary") or time_left_str()=="00:00":
        # Listening scores
        L = LISTENING[lvl]
        L_prep_pct, _ = score_mcq([("",["", "", "", ""],d) for _,d in L["prep_pairs"]], st.session_state.answers["Listening"]["prep"])
        L_t1_pct, _   = score_tf(L["task1_truefalse"], st.session_state.answers["Listening"]["t1"])
        L_t2_pct, _   = score_speaker(L["task2_speaker"], st.session_state.answers["Listening"]["t2"])
        L_pct = round((L_prep_pct + L_t1_pct + L_t2_pct)/3, 1)

        # Reading
        R = READING[lvl]
        R_pct, _ = score_mcq(R["mcq"], st.session_state.answers["Reading"]["mcq"])

        # Use of English
        U = USE[lvl]
        U_pct, _ = score_mcq([(s,opts,c) for (s,opts,c) in U], st.session_state.answers["Use of English"]["cloze"])

        # Writing (rule-of-thumb)
        text = st.session_state.answers["Writing"].get(0,"")
        lo,hi = WRITING[lvl][1], WRITING[lvl][2]
        kws = WRITING[lvl][3]
        wc = len(text.split())
        base = 40 if (lo <= wc <= hi) else (20 if wc>0 else 0)
        hits = sum(1 for k in kws if k.lower() in text.lower())
        W_pct = min(100, base + min(60, hits*12))

        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)

        st.success(f"**Overall: {overall}%** — {'✅ PASS' if overall>=PASS_MARK else '❌ FAIL'}")
        st.write({"Listening":L_pct, "Reading":R_pct, "Use of English":U_pct, "Writing":W_pct})
        st.caption(f"Writing: words={wc}, keywords matched={hits}/{len(kws)} (manual review recommended)")

        # Save to branch CSV (no email)
        code = BRANCHES[st.session_state.branch]
        path = RESULT_PATHS[code]
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "name": st.session_state.name or "",
            "branch": code,
            "level": lvl,
            "overall": overall,
            "Listening": L_pct,
            "Reading": R_pct,
            "Use of English": U_pct,
            "Writing": W_pct,
        }
        if os.path.exists(path):
            pd.concat([pd.read_csv(path), pd.DataFrame([row])]).to_csv(path, index=False)
        else:
            pd.DataFrame([row]).to_csv(path, index=False)
        st.info(f"💾 Saved to {path}")

        # reset
        st.session_state.started=False
        st.session_state.deadline=None
else:
    st.info("اختار الفرع والمستوى واضغط Start Exam. الصوت يتولّد TTS من النص؛ وإن ما توفرش، تسمع ding-ding (WAV صحيح). النتائج تتسجل محليًا وتظهر في لوحة الأدمين.")
