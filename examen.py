# EnglishExam_AutoAudio_Email.py
# ----------------------------------------
# 4 Épreuves (Listening with AI Audio / Reading / Use of English / Writing)
# - كلّ الأسئلة والأوديو Built-in (الأوديو يُولَّد بالـTTS أوتوماتيكياً)
# - الوقت الكلي: A1/A2 = 60 دقيقة، B1/B2 = 90 دقيقة
# - إرسال النتيجة أوتوماتيكياً على الإيميل مع CSV مرفق
# - Logo upload من الـsidebar

import streamlit as st
import random, json, math, struct, io, wave, smtplib, ssl, os
from datetime import datetime, timedelta
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import pandas as pd

# ================== PAGE & STYLES ==================
st.set_page_config(page_title="Mega Formation — English Exam (Auto Audio + Email)", layout="wide")
st.markdown("""
<style>
  .title {text-align:center; font-size: 34px; font-weight:800; margin-bottom:0}
  .subtitle {text-align:center; color:#555; margin-top:4px}
  .card {background:#fff; padding:18px 20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,0.06); margin:12px 0}
  .kpi {font-size:28px; font-weight:700}
  .badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px}
</style>
""", unsafe_allow_html=True)

# ================== CONFIG ==================
LEVELS = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]

TOTAL_TIME_MIN = {"A1":60, "A2":60, "B1":90, "B2":90}
PASS_MARK = 60
Q_PER = {"Listening":6, "Reading":6, "Use of English":8}

# ================== BUILT-IN BANKS ==================
# Listening: كل سؤال فيه tts_text (باش نولّدولو WAV وقت العرض) + سؤال MCQ
# — المستوى A1
LISTENING_BANK = {
    "A1": [
        {
            "tts_text":"Hello! My name is Sam. I am from London. I work in a café.",
            "q":"Where is Sam from?",
            "options":["London","Paris","Rome","Berlin"],
            "answer":"London",
            "transcript":"Hello! My name is Sam. I am from London. I work in a café."
        },
        {
            "tts_text":"The bus leaves at half past three. Please be on time.",
            "q":"What time does the bus leave?",
            "options":["3:30","3:15","3:45","2:30"],
            "answer":"3:30",
            "transcript":"The bus leaves at half past three."
        },
        {
            "tts_text":"Please open the window. It is very hot in here.",
            "q":"What does the speaker want?",
            "options":["Open the window","Close the door","Turn on the TV","Bring water"],
            "answer":"Open the window",
            "transcript":"Please open the window. It is very hot in here."
        },
        {
            "tts_text":"We have English on Monday and Wednesday.",
            "q":"When do they have English?",
            "options":["Monday and Wednesday","Tuesday and Thursday","Friday only","Saturday only"],
            "answer":"Monday and Wednesday",
            "transcript":"We have English on Monday and Wednesday."
        },
        {
            "tts_text":"I like tea in the morning and coffee in the evening.",
            "q":"What does the speaker like in the morning?",
            "options":["Tea","Coffee","Juice","Water"],
            "answer":"Tea",
            "transcript":"I like tea in the morning and coffee in the evening."
        },
        {
            "tts_text":"My phone number is zero nine eight seven.",
            "q":"What is the phone number?",
            "options":["0987","0978","9870","9087"],
            "answer":"0987",
            "transcript":"My phone number is zero nine eight seven."
        },
    ],
    # — المستوى A2
    "A2": [
        {
            "tts_text":"The museum opens at ten, but the guided tour starts at eleven.",
            "q":"When does the guided tour start?",
            "options":["11:00","10:00","09:30","12:00"],
            "answer":"11:00",
            "transcript":"The museum opens at 10 but the guided tour starts at 11."
        },
        {
            "tts_text":"Could you send me the report by Friday? Thank you.",
            "q":"What does the speaker want?",
            "options":["The report by Friday","A call on Friday","A meeting today","A delay"],
            "answer":"The report by Friday",
            "transcript":"Could you send me the report by Friday?"
        },
        {
            "tts_text":"There is heavy traffic today, so I will be about fifteen minutes late.",
            "q":"Why will the speaker be late?",
            "options":["Heavy traffic","No petrol","Car broke down","Lost keys"],
            "answer":"Heavy traffic",
            "transcript":"There is heavy traffic, so I will be about 15 minutes late."
        },
        {
            "tts_text":"Your package has arrived; you can collect it this afternoon.",
            "q":"What arrived?",
            "options":["A package","A letter","A taxi","A person"],
            "answer":"A package",
            "transcript":"Your package has arrived; you can collect it this afternoon."
        },
        {
            "tts_text":"I prefer tea to coffee, especially in the evening.",
            "q":"What does the speaker prefer?",
            "options":["Tea","Coffee","Juice","Water"],
            "answer":"Tea",
            "transcript":"I prefer tea to coffee, especially in the evening."
        },
        {
            "tts_text":"I am looking for a cheaper hotel near the station.",
            "q":"What is the speaker looking for?",
            "options":["A cheaper hotel near the station","A restaurant","A taxi","An expensive hotel"],
            "answer":"A cheaper hotel near the station",
            "transcript":"I'm looking for a cheaper hotel near the station."
        },
    ],
    # — المستوى B1
    "B1": [
        {
            "tts_text":"Due to maintenance work, platform three is closed today.",
            "q":"What is closed?",
            "options":["Platform 3","The station","The ticket office","The train"],
            "answer":"Platform 3",
            "transcript":"Due to maintenance work, platform 3 is closed today."
        },
        {
            "tts_text":"I will forward you the agenda and minutes after the meeting.",
            "q":"What will the speaker send?",
            "options":["Agenda and minutes","Invoice","Invitation","Photos"],
            "answer":"Agenda and minutes",
            "transcript":"I'll forward you the agenda and minutes after the meeting."
        },
        {
            "tts_text":"The lecture has been postponed until next Thursday.",
            "q":"What happened to the lecture?",
            "options":["Postponed","Cancelled","Finished","Moved to today"],
            "answer":"Postponed",
            "transcript":"The lecture has been postponed until next Thursday."
        },
        {
            "tts_text":"We need to cut costs without compromising quality.",
            "q":"What do they need to do?",
            "options":["Cut costs","Stop production","Hire more staff","Increase costs"],
            "answer":"Cut costs",
            "transcript":"We need to cut costs without compromising quality."
        },
        {
            "tts_text":"There will be scattered showers in the north this evening.",
            "q":"What is the weather in the north?",
            "options":["Showers","Sunny","Windy","Snow"],
            "answer":"Showers",
            "transcript":"There will be scattered showers in the north."
        },
        {
            "tts_text":"Passengers must keep their luggage with them at all times.",
            "q":"What must passengers do?",
            "options":["Keep luggage with them","Leave luggage","Check luggage","Pay for luggage"],
            "answer":"Keep luggage with them",
            "transcript":"Passengers must keep their luggage with them at all times."
        },
    ],
    # — المستوى B2
    "B2": [
        {
            "tts_text":"Preliminary results indicate a significant rise in consumer confidence this quarter.",
            "q":"What do the results indicate?",
            "options":["A rise","No change","A fall","Unclear"],
            "answer":"A rise",
            "transcript":"Preliminary results indicate a significant rise in consumer confidence."
        },
        {
            "tts_text":"The panel reached a consensus after extensive deliberation.",
            "q":"What did the panel reach?",
            "options":["A consensus","A delay","A conflict","A vote"],
            "answer":"A consensus",
            "transcript":"The panel reached a consensus after extensive deliberation."
        },
        {
            "tts_text":"Remote work has broadened access to global talent pools.",
            "q":"What has remote work done?",
            "options":["Broadened access","Reduced access","Eliminated access","Complicated access"],
            "answer":"Broadened access",
            "transcript":"Remote work has broadened access to global talent pools."
        },
        {
            "tts_text":"The committee urged immediate implementation of the safety protocol.",
            "q":"What did the committee urge?",
            "options":["Immediate implementation","Delay","Cancellation","Review"],
            "answer":"Immediate implementation",
            "transcript":"The committee urged immediate implementation of the safety protocol."
        },
        {
            "tts_text":"New findings challenge the prevailing hypothesis.",
            "q":"What do the findings do?",
            "options":["Challenge it","Support it","Ignore it","Prove it"],
            "answer":"Challenge it",
            "transcript":"New findings challenge the prevailing hypothesis."
        },
        {
            "tts_text":"Although promising, the pilot study had a limited sample size.",
            "q":"What was limited?",
            "options":["Sample size","Budget","Time","Staff"],
            "answer":"Sample size",
            "transcript":"Although promising, the pilot study had a limited sample size."
        },
    ],
}

# Reading / Use / Writing (نفسهم ثابتين وبسيطين)
R_PASSAGES = {
    "A1":{"text":"Maria lives in a small town near the sea. She works in a café and goes to the beach after work.",
          "qs":[
              ("Where does Maria live?",["In a big city","In a small town near the sea","In the mountains","In the desert"],"In a small town near the sea"),
              ("Where does Maria work?",["In a shop","In a café","In a bank","At school"],"In a café"),
              ("What does she do after work?",["Goes home","Goes to the gym","Goes to the beach","Studies"],"Goes to the beach"),
              ("Maria lives __ the sea.",["at","near","on","under"],"near"),
              ("The text says Maria works __.",["in a café","in an office","from home","at night only"],"in a café"),
              ("The opposite of 'small' is __.",["little","tiny","big","short"],"big"),
          ]},
    "A2":{"text":"The city library moved to a larger building. Now it offers weekend workshops, free Wi-Fi, and study rooms.",
          "qs":[
              ("Why did the library move?",["It was closed","To a smaller place","To a larger building","For repairs"],"To a larger building"),
              ("Which service is mentioned?",["Paid internet","Free Wi-Fi","Gym","Cinema"],"Free Wi-Fi"),
              ("When are workshops offered?",["Weekdays","Weekends","Every night","Holidays only"],"Weekends"),
              ("Study rooms are available __.",["for staff only","for students only","for users","for teachers"],"for users"),
              ("The library now has more __.",["space","noise","rules","fees"],"space"),
              ("'Offers' is closest to __.",["gives","buys","sells","hides"],"gives"),
          ]},
    "B1":{"text":"Volunteering can strengthen communities by connecting people with local needs. However, volunteers require training to be effective.",
          "qs":[
              ("What strengthens communities?",["Traffic","Volunteering","Taxes","Tourism"],"Volunteering"),
              ("What do volunteers require?",["Money","Uniforms","Training","Cars"],"Training"),
              ("Volunteering connects people with __.",["local needs","sports","politics","fashion"],"local needs"),
              ("To be effective, volunteers need __.",["experience only","training","nothing","luck"],"training"),
              ("The tone of the passage is __.",["critical","informative","funny","angry"],"informative"),
              ("'However' shows __.",["addition","contrast","time","cause"],"contrast"),
          ]},
    "B2":{"text":"While renewable energy adoption is accelerating, integrating intermittent sources into aging grids demands investment and regulatory coordination.",
          "qs":[
              ("What is accelerating?",["Fossil fuel use","Renewable energy adoption","Electricity prices","Grid failures"],"Renewable energy adoption"),
              ("What makes integration challenging?",["Cheap technology","Intermittent sources","Abundant storage","Public support"],"Intermittent sources"),
              ("Grids described are __.",["new","aging","perfect","private"],"aging"),
              ("What does integration demand?",["No changes","Investment and coordination","Less regulation","Fewer workers"],"Investment and coordination"),
              ("'Intermittent' most nearly means __.",["constant","irregular","fast","expensive"],"irregular"),
              ("The passage focuses on __.",["transport","energy policy","education","health"],"energy policy"),
          ]},
}

U_BANK = {
    "A1":[
        ("He __ a student.",["am","is","are","be"],"is"),
        ("We __ in Tunis.",["live","lives","living","to live"],"live"),
        ("There __ two apples.",["is","are","be","been"],"are"),
        ("I __ coffee every day.",["drink","drinks","drank","drinking"],"drink"),
        ("Choose the plural: one man → two __.",["mans","men","manses","menses"],"men"),
        ("She __ from Spain.",["are","am","is","be"],"is"),
        ("I go __ school by bus.",["to","in","on","at"],"to"),
        ("Opposite of 'hot' is __.",["warm","cold","heat","cool"],"cold"),
    ],
    "A2":[
        ("I have lived here __ 2019.",["for","since","during","from"],"since"),
        ("If it rains, we __ at home.",["stay","will stay","stayed","stays"],"will stay"),
        ("He can't __ the meeting.",["to attend","attends","attend","attending"],"attend"),
        ("We didn't go out __ the rain.",["because","because of","so","although"],"because of"),
        ("Choose the past: buy → __.",["buyed","bought","buys","buy"],"bought"),
        ("You're coming, __?",["isn't you","aren't you","don't you","won't you"],"aren't you"),
        ("I'm interested __ history.",["in","on","at","about"],"in"),
        ("This test is __ than the last.",["easyer","easier","more easy","most easy"],"easier"),
    ],
    "B1":[
        ("I wish I __ more time.",["have","had","would have","am having"],"had"),
        ("Hardly __ the meeting begun when the alarm rang.",["had","has","did","was"],"had"),
        ("He denied __ the window.",["to break","break","breaking","to have broke"],"breaking"),
        ("We need someone __ can code.",["who","which","whom","what"],"who"),
        ("Despite __ late, she finished.",["to arrive","arrive","arriving","arrived"],"arriving"),
        ("The manager suggested that he __ earlier.",["comes","come","came","would come"],"come"),
        ("It's high time you __.",["come","came","would come","had come"],"came"),
        ("Make __ decision.",["do a","make a","take a","create a"],"make a"),
    ],
    "B2":[
        ("No sooner __ the announcement made than shares fell.",["was","had","has","having"],"had"),
        ("The project, __ objectives were unclear, was delayed.",["whose","who's","which","that"],"whose"),
        ("Had I known, I __ earlier.",["left","would have left","would leave","had left"],"would have left"),
        ("He insisted that she __ present.",["be","was","is","would be"],"be"),
        ("The proposal was rejected on the __ that ...",["grounds","reasons","basis","causes"],"grounds"),
        ("Seldom __ such a case.",["I hear","do I hear","I have heard","did I heard"],"do I hear"),
        ("By the time it finishes, we __ ten modules.",["will have completed","completed","have completed","had completed"],"will have completed"),
        ("We should consider __ a pilot program.",["to launch","launch","launching","to be launching"],"launching"),
    ],
}

W_PROMPTS = {
    "A1":("Write about your daily routine (50–70 words).",["morning","work","eat","go","home"]),
    "A2":("Describe your last holiday (80–100 words).",["where","when","with","activities","feelings"]),
    "B1":("Do you prefer studying alone or in groups? Explain (120–150 words).",["prefer","because","example","time","learn"]),
    "B2":("Some companies allow remote work. Discuss advantages and disadvantages (180–220 words).",["productivity","balance","communication","costs","team"]),
}

# ================== STATE ==================
def init_state():
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("level", "A1")
    st.session_state.setdefault("seed", random.randint(1, 10_000_000))
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("logo_bytes", None)

    # Email settings (Defaults to send to contact.megaformation@gmail.com)
    st.session_state.setdefault("email_enabled", True)
    st.session_state.setdefault("smtp_user", "")
    st.session_state.setdefault("smtp_app_password", "")
    st.session_state.setdefault("email_to", "contact.megaformation@gmail.com")

init_state()

# ================== AUDIO (TTS or Beep) ==================
def gen_beep_wav_bytes(duration=0.35, freq=660, rate=22050, volume=0.3):
    import math
    n = int(duration * rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(rate)
        for i in range(n):
            t = i / rate
            sample = volume * math.sin(2 * math.pi * freq * t)
            wf.writeframes(struct.pack('<h', int(sample * 32767)))
    return buf.getvalue()

def tts_or_beep_bytes(text, rate_wpm=160):
    # Offline TTS via pyttsx3 if available; otherwise short beeps
    try:
        import pyttsx3, tempfile
        engine = pyttsx3.init()
        engine.setProperty('rate', rate_wpm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp = tf.name
        engine.save_to_file(text, tmp)
        engine.runAndWait()
        with open(tmp, "rb") as f:
            data = f.read()
        try: os.remove(tmp)
        except: pass
        return data
    except Exception:
        return gen_beep_wav_bytes() + gen_beep_wav_bytes(freq=880)

# ================== HEADER ==================
c1, c2 = st.columns([1,4])
with c1:
    lg = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if lg: st.session_state.logo_bytes = lg.read()
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, caption="", use_container_width=False)
with c2:
    st.markdown("<div class='title'>Mega Formation — English Exam</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Listening (AI Audio) • Reading • Use of English • Writing</div>", unsafe_allow_html=True)

# ================== SIDEBAR ==================
with st.sidebar:
    st.header("Setup")
    st.session_state.name = st.text_input("Candidate name", value=st.session_state.name)
    st.session_state.level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.level))
    st.session_state.seed  = st.number_input("Random seed", value=st.session_state.seed, step=1, format="%d")

    st.subheader("Email (auto-send on submit)")
    st.caption("Gmail SMTP (App Password مطلوب)")
    st.session_state.email_enabled      = st.checkbox("Send results by email", value=st.session_state.email_enabled)
    st.session_state.smtp_user          = st.text_input("Sender Gmail", value=st.session_state.smtp_user, placeholder="you@gmail.com")
    st.session_state.smtp_app_password  = st.text_input("Gmail App Password", value=st.session_state.smtp_app_password, type="password", placeholder="xxxx xxxx xxxx xxxx")
    st.session_state.email_to           = st.text_input("Recipient", value=st.session_state.email_to)

    if not st.session_state.started:
        if st.button("▶️ Start Exam"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.started = True
            total_min = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=total_min)
    else:
        if st.button("🔁 Restart"):
            st.session_state.seed = random.randint(1, 10_000_000)
            st.session_state.answers = {s:{} for s in SECTIONS}
            total_min = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=total_min)

# ================== HELPERS ==================
def time_left_str():
    if not st.session_state.deadline: return ""
    left = st.session_state.deadline - datetime.utcnow()
    if left.total_seconds() <= 0: return "00:00"
    mm, ss = divmod(int(left.total_seconds()), 60)
    return f"{mm:02d}:{ss:02d}"

def reading_items(level, n):
    data = R_PASSAGES[level]
    qs = data["qs"][:]
    rnd = random.Random(st.session_state.seed); rnd.shuffle(qs)
    return data["text"], qs[:n]

def score_mcq(items, user_map):
    correct = 0; rows = []
    for i, it in enumerate(items):
        q = it["q"]; opts = it["options"]; ans = it["answer"]
        user = user_map.get(i)
        ok = (user == ans); correct += int(ok)
        rows.append({"Q#": i+1, "Question": q, "User": user or "", "Correct": ans, "IsCorrect": ok})
    pct = round(100*correct/max(1, len(items)), 1)
    return pct, pd.DataFrame(rows)

def score_writing(text, level):
    min_w = {"A1":50,"A2":80,"B1":120,"B2":180}[level]
    max_w = {"A1":70,"A2":100,"B1":150,"B2":220}[level]
    kws = W_PROMPTS[level][1]
    wc = len(text.strip().split()) if text.strip() else 0
    hits = sum(1 for k in kws if k.lower() in text.lower())
    base = 40 if min_w <= wc <= max_w else 20 if wc>0 else 0
    kw_score = min(60, hits*12)
    pct = min(100, base + kw_score)
    return pct, wc, hits, kws

def df_to_csv_bytes(df): return df.to_csv(index=False).encode()

# ================== EXAM UI ==================
if st.session_state.started:
    k1,k2,k3 = st.columns([1,1,2])
    with k1:
        st.markdown("**Level**")
        st.markdown(f"<span class='badge'>{st.session_state.level}</span>", unsafe_allow_html=True)
    with k2:
        st.markdown("**Time Left**")
        st.markdown(f"<div class='kpi'>{time_left_str()}</div>", unsafe_allow_html=True)
    with k3:
        st.info("Answer all sections then click Submit All at the bottom.")

    if time_left_str()=="00:00":
        st.warning("Time is up! Auto-submitting your exam.")

    lvl = st.session_state.level
    rnd = random.Random(st.session_state.seed)

    # Listening
    L_pool = LISTENING_BANK[lvl][:]
    rnd.shuffle(L_pool)
    L_items = L_pool[:Q_PER["Listening"]]

    # Reading
    R_text, R_items = reading_items(lvl, Q_PER["Reading"])

    # Use of English
    U_all = U_BANK[lvl][:]
    rnd.shuffle(U_all)
    U_items = [{"q":q, "options":[a,b,c,d], "answer":ans} for (q,[a,b,c,d],ans) in U_all[:Q_PER["Use of English"]]]

    tabs = st.tabs(SECTIONS)

    # Listening tab
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        for i, it in enumerate(L_items):
            st.markdown(f"**L{i+1}.** {it['q']}")
            audio_bytes = tts_or_beep_bytes(it["tts_text"])
            st.audio(audio_bytes)
            st.session_state.answers["Listening"][i] = st.radio("Select one:", it["options"], index=None, key=f"L_{i}")
            with st.expander("Transcript"):
                st.caption(it["transcript"])
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    # Reading tab
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Read the passage and answer the questions.**")
        st.info(R_text)
        for i, (q, opts, ans) in enumerate(R_items):
            st.markdown(f"**R{i+1}.** {q}")
            st.session_state.answers["Reading"][i] = st.radio("Select one:", opts, index=None, key=f"R_{i}")
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    # Use of English tab
    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Grammar & Vocabulary.** Choose the best answer.")
        for i, it in enumerate(U_items):
            st.markdown(f"**U{i+1}.** {it['q']}")
            st.session_state.answers["Use of English"][i] = st.radio("Select one:", it["options"], index=None, key=f"U_{i}")
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    # Writing tab
    with tabs[3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        prompt, kws = W_PROMPTS[lvl]
        st.write(f"**Prompt:** {prompt}")
        st.caption(f"Try to include: {', '.join(kws)}")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=230, key="W_0")
        st.markdown("</div>", unsafe_allow_html=True)

    # Submit button
    if st.button("✅ Submit All", type="primary") or time_left_str()=="00:00":
        # Score
        L_pct, L_df = score_mcq(
            [{"q":it["q"],"options":it["options"],"answer":it["answer"]} for it in L_items],
            st.session_state.answers["Listening"]
        )
        R_df_items = [{"q": q, "options": opts, "answer": ans} for (q,opts,ans) in R_items]
        R_pct, R_df = score_mcq(R_df_items, st.session_state.answers["Reading"])
        U_pct, U_df = score_mcq(U_items, st.session_state.answers["Use of English"])
        W_text = st.session_state.answers["Writing"].get(0,"")
        W_pct, wc, hits, kws = score_writing(W_text, lvl)

        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)

        st.success(f"**Overall Score: {overall}%** — {'✅ PASS' if overall>=PASS_MARK else '❌ FAIL'}")
        st.write({"Listening": L_pct, "Reading": R_pct, "Use of English": U_pct, "Writing": W_pct})
        st.caption(f"Writing → words={wc}, keywords matched={hits}/{len(kws)} (manual review recommended)")

        # Prepare CSV attachments
        def to_csv(df): return df.to_csv(index=False).encode()
        files = [
            ("listening_report.csv", to_csv(L_df)),
            ("reading_report.csv", to_csv(R_df)),
            ("use_of_english_report.csv", to_csv(U_df)),
        ]

        # Auto email
        if st.session_state.email_enabled and st.session_state.smtp_user and st.session_state.smtp_app_password and st.session_state.email_to:
            try:
                subject = f"Mega Formation — Exam Result ({st.session_state.name or 'Candidate'}) [{lvl}]"
                body = (
                    f"Candidate: {st.session_state.name or 'N/A'}\n"
                    f"Level: {lvl}\n"
                    f"Overall: {overall}% ({'PASS' if overall>=PASS_MARK else 'FAIL'})\n\n"
                    f"Section scores:\n"
                    f"- Listening: {L_pct}\n- Reading: {R_pct}\n- Use of English: {U_pct}\n- Writing: {W_pct}\n\n"
                    f"Writing: words={wc}, keywords matched={hits}/{len(kws)}\n"
                    f"Sent automatically from Mega Formation Exam app."
                )

                msg = MIMEMultipart()
                msg["From"] = st.session_state.smtp_user
                msg["To"] = st.session_state.email_to
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))
                for fname, data in files:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(data)
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
                    msg.attach(part)

                context = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                    server.login(st.session_state.smtp_user, st.session_state.smtp_app_password)
                    server.sendmail(st.session_state.smtp_user, [st.session_state.email_to], msg.as_string())
                st.success(f"✅ Results emailed to {st.session_state.email_to}")
            except Exception as e:
                st.error(f"Email failed: {e}")
        else:
            st.info("Email not sent — عبيّ 'Sender Gmail' و'App Password' وتأكد من تفعيل الخيار.")

        # Reset
        st.session_state.started = False
        st.session_state.deadline = None
else:
    st.info("اختار المستوى واضغط Start Exam. الصوت في Listening يتولّد آلياً من نص السؤال (TTS).")

