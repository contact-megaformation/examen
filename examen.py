# Mega_English_BC_Style_Exam.py
# ---------------------------------------------------------
# Mega Formation — British-Council-like Exam (A1–B2)
# 4 Épreuves: Listening (TTS audio) • Reading • Use of English • Writing
# Total time: A1/A2=60 min, B1/B2=90 min
# Auto-email results + CSV attachments to contact.megaformation@gmail.com
# ---------------------------------------------------------

import streamlit as st
import random, io, os, ssl, smtplib, wave, struct, math, tempfile
import pandas as pd
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text    import MIMEText
from email.mime.base    import MIMEBase
from email              import encoders

# -------------------- PAGE & STYLE --------------------
st.set_page_config(page_title="Mega Formation — English Exam (BC-Style)", layout="wide")
st.markdown("""
<style>
.title {text-align:center; font-size:32px; font-weight:800; margin:0}
.subtitle {text-align:center; color:#666; margin:2px 0 14px}
.section {background:#fff; padding:18px 20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,.06); margin:12px 0}
.kpi {font-size:24px; font-weight:700}
.badge {display:inline-block; padding:3px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px}
.small {font-size:12px; color:#666}
</style>
""", unsafe_allow_html=True)

LEVELS  = ["A1","A2","B1","B2"]
SECTIONS = ["Listening","Reading","Use of English","Writing"]
TOTAL_TIME_MIN = {"A1":60,"A2":60,"B1":90,"B2":90}
PASS_MARK = 60

# -------------------- AUDIO (TTS with robust fallback) --------------------
def make_tones_wav(segments, rate=22050, volume=0.30):
    """
    segments: list of (duration_sec, freq_hz). Use freq=None for silence.
    Returns a single valid WAV file (bytes).
    """
    frames = []
    for dur, freq in segments:
        n = int(dur * rate)
        for i in range(n):
            if freq is None:
                sample = 0.0
            else:
                sample = volume * math.sin(2 * math.pi * freq * (i / rate))
            frames.append(struct.pack('<h', int(sample * 32767)))
    with io.BytesIO() as buf:
        with wave.open(buf, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(rate)
            wf.writeframes(b"".join(frames))
        return buf.getvalue()

def tts_or_beep_bytes(text, rate_wpm=165):
    """
    Try offline TTS via pyttsx3; if unavailable/invalid, return a single valid WAV with 'ding-ding'.
    """
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty("rate", rate_wpm)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp = tf.name
        engine.save_to_file(text or "Listening item.", tmp)
        engine.runAndWait()
        with open(tmp, "rb") as f:
            data = f.read()
        try: os.remove(tmp)
        except: pass
        if len(data) < 2000:  # safety: treat tiny file as invalid
            raise RuntimeError("Empty/invalid TTS WAV")
        return data
    except Exception:
        return make_tones_wav([(0.35,660),(0.10,None),(0.35,880)])

# -------------------- BANKS (original content, BC-like) --------------------
# Listening per level: transcript + Preparation (vocab match) + Task1 (MCQ) + Task2 (Note completion)
LISTENING = {
    "A1": {
        "transcript": (
            "Hello, this is Anna. Today is Monday. I take the eight o'clock bus to work. "
            "At lunch, I meet my friend Tom in the park. After work, I go to the supermarket and buy milk and bread."
        ),
        "prep_pairs": [
            ("lunch","the meal in the middle of the day"),
            ("supermarket","a large shop to buy food"),
            ("bus","public transport with many seats"),
        ],
        "task1_mcq": [
            ("What day is it?", ["Monday","Tuesday","Friday","Sunday"], "Monday"),
            ("When does Anna take the bus?", ["8:00","9:00","7:30","10:00"], "8:00"),
            ("Where does she meet Tom?", ["in the park","at home","at school","at the station"], "in the park"),
            ("What does she buy?", ["milk and bread","tea and coffee","eggs","apples"], "milk and bread"),
            ("Who does she meet?", ["Tom","Sam","Anna","John"], "Tom"),
            ("When is the bus?", ["in the morning","at night","in the afternoon","at midnight"], "in the morning"),
        ],
        "task2_dropdown": [
            ("She takes the __ to work.", ["train","bus","car","bike"], "bus"),
            ("She meets Tom at __.", ["the gym","the park","the café","home"], "the park"),
            ("She buys __.", ["milk and bread","fish","pasta","rice"], "milk and bread"),
            ("Today is __.", ["Monday","Wednesday","Saturday","Sunday"], "Monday"),
        ],
    },
    "A2": {
        "transcript": (
            "Good afternoon. The city museum opens at ten from Tuesday to Sunday. "
            "Guided tours start at eleven and last one hour. Tickets are cheaper online. "
            "Please arrive ten minutes early to join your group."
        ),
        "prep_pairs": [
            ("guided tour","a visit where a guide explains things"),
            ("ticket","a paper or digital pass you pay for"),
            ("arrive","to get to a place"),
        ],
        "task1_mcq": [
            ("When does the museum open?", ["10:00","11:00","09:00","12:00"], "10:00"),
            ("Tours start at __.", ["11:00","10:00","12:00","14:00"], "11:00"),
            ("Tours last __.", ["1 hour","30 minutes","2 hours","all day"], "1 hour"),
            ("Tickets are cheaper __.", ["online","at the door","by phone","for students only"], "online"),
            ("Arrive __ early.", ["10 minutes","5 minutes","30 minutes","1 minute"], "10 minutes"),
            ("The museum is closed on __.", ["Monday","Tuesday","Sunday","Saturday"], "Monday"),
        ],
        "task2_dropdown": [
            ("Tours start at __.", ["10","11","12","13"], "11"),
            ("They last __ hour.", ["one","two","half","quarter"], "one"),
            ("Tickets are cheaper __.", ["online","in person","by cash only","for groups only"], "online"),
            ("Arrive __ early.", ["ten minutes","one hour","five minutes","not"], "ten minutes"),
        ],
    },
    "B1": {
        "transcript": (
            "Thanks for calling. The train to Lakeside departs at 08:45 from platform three. "
            "There is a short delay today due to maintenance, about ten minutes. "
            "Return tickets are valid for one month. Please keep your luggage with you."
        ),
        "prep_pairs": [
            ("depart","to leave, start a journey"),
            ("platform","place beside the rails to board a train"),
            ("valid","accepted or usable for a period"),
        ],
        "task1_mcq": [
            ("What time does the train depart?", ["08:45","08:30","09:00","08:15"], "08:45"),
            ("Which platform?", ["three","one","two","four"], "three"),
            ("How long is the delay?", ["about ten minutes","one hour","no delay","five minutes"], "about ten minutes"),
            ("Return tickets are valid for __.", ["one month","one week","two days","three months"], "one month"),
            ("Passengers must __.", ["keep luggage with them","leave luggage","check luggage","pay for luggage"], "keep luggage with them"),
            ("The delay is due to __.", ["maintenance","weather","strike","accident"], "maintenance"),
        ],
        "task2_dropdown": [
            ("The train departs at __.", ["08:30","08:45","09:15","08:00"], "08:45"),
            ("It leaves from platform __.", ["two","three","one","five"], "three"),
            ("Delay is about __ minutes.", ["ten","twenty","five","zero"], "ten"),
            ("Return tickets are valid for __.", ["one month","two days","one week","three months"], "one month"),
        ],
    },
    "B2": {
        "transcript": (
            "The panel concluded that remote work increased access to global talent while raising challenges in communication. "
            "They recommended clearer protocols and regular check-ins to maintain team cohesion across time zones."
        ),
        "prep_pairs": [
            ("protocols","agreed rules for how to do something"),
            ("cohesion","unity or sticking together"),
            ("check-ins","regular brief meetings to update status"),
        ],
        "task1_mcq": [
            ("Remote work increased access to __.", ["global talent","local offices","short commutes","vacation time"], "global talent"),
            ("A challenge mentioned is __.", ["communication","office rent","data storage","holidays"], "communication"),
            ("They recommended __ protocols.", ["clearer","fewer","secret","no"], "clearer"),
            ("Regular __ help cohesion.", ["check-ins","parties","overtime","bonuses"], "check-ins"),
            ("Cohesion means team __.", ["unity","conflict","size","pay"], "unity"),
            ("The context involves __ time zones.", ["across","without","few","none"], "across"),
        ],
        "task2_dropdown": [
            ("Remote work increases access to __ talent.", ["global","local","cheap","temporary"], "global"),
            ("Issues were raised in __.", ["communication","parking","heating","travel"], "communication"),
            ("Clearer __ were recommended.", ["protocols","budgets","holidays","laptops"], "protocols"),
            ("Regular __ support cohesion.", ["check-ins","vacations","audits","complaints"], "check-ins"),
        ],
    },
}

# Reading per level: passage + tasks (MCQ, TF/NG, Match headings)
READING = {
    "A1": {
        "passage": (
            "Maria lives near the sea and works in a café. After work, she often goes to the beach with her friends."
        ),
        "task1_mcq": [
            ("Where does Maria live?", ["near the sea","in the mountains","in the city centre","in the desert"], "near the sea"),
            ("Where does she work?", ["in a café","in an office","at home","in a shop"], "in a café"),
            ("What does she do after work?", ["goes to the beach","studies at school","cooks dinner at work","visits the museum"], "goes to the beach"),
        ],
        "task2_tfn": [
            ("Maria works in a café.", "T"),
            ("She hates the beach.", "F"),
            ("Maria drives a car to work.", "NG"),
        ],
        "task3_headings": {
            "P1: Maria's job and free time.": ["Free-time activities","Travel plans","Family details","Health issues"],
        },
    },
    "A2": {
        "passage": (
            "The city library moved to a larger building. It now offers weekend workshops and free Wi-Fi. Many students come to study."
        ),
        "task1_mcq": [
            ("Why did the library move?", ["to a larger building","to a smaller place","it closed","for repairs"], "to a larger building"),
            ("Which service is new?", ["weekend workshops","paid internet","gym","cinema"], "weekend workshops"),
            ("What do many students do there?", ["study","sleep","play games","cook"], "study"),
        ],
        "task2_tfn": [
            ("The library has free Wi-Fi.", "T"),
            ("Workshops are every night.", "F"),
            ("There is a new café inside the library.", "NG"),
        ],
        "task3_headings": {
            "P1: New services at the library.": ["Services and visitors","City history","Sports news","Cooking tips"],
        },
    },
    "B1": {
        "passage": (
            "Volunteering connects people with local needs and can strengthen communities. Volunteers, however, benefit from proper training."
        ),
        "task1_mcq": [
            ("What connects people with local needs?", ["Volunteering","Tourism","Taxes","Sports"], "Volunteering"),
            ("What do volunteers benefit from?", ["training","uniforms","cars","money"], "training"),
            ("What can volunteering strengthen?", ["communities","traffic","prices","weather"], "communities"),
        ],
        "task2_tfn": [
            ("Volunteering has no effect on communities.", "F"),
            ("Training helps volunteers be effective.", "T"),
            ("Volunteering always requires money.", "NG"),
        ],
        "task3_headings": {
            "P1: Why training matters.": ["Benefits of training","Festival plan","Travel guide","Job advert"],
        },
    },
    "B2": {
        "passage": (
            "Integrating intermittent renewables into aging grids requires coordination and investment. Storage helps balance fluctuations."
        ),
        "task1_mcq": [
            ("What requires coordination?", ["integration of renewables","closing factories","opening schools","building roads"], "integration of renewables"),
            ("What helps balance fluctuations?", ["storage","advertising","meetings","tax cuts"], "storage"),
            ("What kind of grids are mentioned?", ["aging","brand-new","private only","perfect"], "aging"),
        ],
        "task2_tfn": [
            ("Renewables can be intermittent.", "T"),
            ("No investment is needed.", "F"),
            ("Solar is never intermittent.", "NG"),
        ],
        "task3_headings": {
            "P1: Challenges of integration.": ["Energy system challenges","Healthy lifestyles","Tourist attractions","Cooking at home"],
        },
    },
}

# Use of English per level: Cloze (dropdown)
USE = {
    "A1":[
        ("He __ a student.", ["is","are","am","be"], "is"),
        ("We __ in Tunis.", ["live","lives","living","to live"], "live"),
        ("There __ two apples.", ["are","is","be","been"], "are"),
        ("I __ coffee every day.", ["drink","drinks","drank","drinking"], "drink"),
        ("I go __ school by bus.", ["to","in","on","at"], "to"),
        ("Opposite of 'hot' is __.", ["cold","heat","cool","warm"], "cold"),
        ("She __ from Spain.", ["is","are","am","be"], "is"),
        ("Choose the plural: one man → two __.", ["men","mans","manses","menses"], "men"),
    ],
    "A2":[
        ("I have lived here __ 2019.", ["since","for","during","from"], "since"),
        ("If it rains, we __ at home.", ["will stay","stay","stayed","stays"], "will stay"),
        ("He can't __ the meeting.", ["attend","to attend","attends","attending"], "attend"),
        ("We didn't go out __ the rain.", ["because of","because","so","although"], "because of"),
        ("Choose the past: buy → __.", ["bought","buyed","buys","buy"], "bought"),
        ("You're coming, __?", ["aren't you","isn't you","don't you","won't you"], "aren't you"),
        ("I'm interested __ history.", ["in","on","at","about"], "in"),
        ("This test is __ than the last.", ["easier","easyer","more easy","most easy"], "easier"),
    ],
    "B1":[
        ("I wish I __ more time.", ["had","have","would have","am having"], "had"),
        ("Hardly __ the meeting begun when the alarm rang.", ["had","has","did","was"], "had"),
        ("He denied __ the window.", ["breaking","to break","break","to have broke"], "breaking"),
        ("We need someone __ can code.", ["who","which","whom","what"], "who"),
        ("Despite __ late, she finished.", ["arriving","to arrive","arrive","arrived"], "arriving"),
        ("The manager suggested that he __ earlier.", ["come","comes","came","would come"], "come"),
        ("It's high time you __.", ["came","come","would come","had come"], "came"),
        ("Make __ decision.", ["a","do a","take a","create a"], "a"),
    ],
    "B2":[
        ("No sooner __ the announcement made than shares fell.", ["had","was","has","having"], "had"),
        ("The project, __ objectives were unclear, was delayed.", ["whose","who's","which","that"], "whose"),
        ("Had I known, I __ earlier.", ["would have left","left","would leave","had left"], "would have left"),
        ("He insisted that she __ present.", ["be","was","is","would be"], "be"),
        ("Seldom __ such a case.", ["do I hear","I hear","I have heard","did I heard"], "do I hear"),
        ("We should consider __ a pilot program.", ["launching","to launch","launch","to be launching"], "launching"),
        ("By the time it finishes, we __ ten modules.", ["will have completed","completed","have completed","had completed"], "will have completed"),
        ("The proposal was rejected on the __ that ...", ["grounds","reasons","basis","causes"], "grounds"),
    ],
}

# Writing: prompt + target range + keywords
WRITING = {
    "A1": ("Write about your daily routine (50–70 words).", 50, 70, ["morning","work","eat","go","home"]),
    "A2": ("Describe your last holiday (80–100 words).", 80, 100, ["where","when","with","activities","feelings"]),
    "B1": ("Do you prefer studying alone or in groups? Explain (120–150 words).", 120, 150, ["prefer","because","example","time","learn"]),
    "B2": ("Some companies allow remote work. Discuss advantages and disadvantages (180–220 words).", 180, 220, ["productivity","balance","communication","costs","team"]),
}

# -------------------- STATE --------------------
def init_state():
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("level", "A1")
    st.session_state.setdefault("seed", random.randint(1, 10_000_000))
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    # email defaults
    st.session_state.setdefault("email_to", "contact.megaformation@gmail.com")
    st.session_state.setdefault("smtp_user", "")
    st.session_state.setdefault("smtp_app_password", "")
    st.session_state.setdefault("email_enabled", True)
    # logo
    st.session_state.setdefault("logo_bytes", None)

init_state()

# -------------------- HEADER --------------------
c1,c2 = st.columns([1,4])
with c1:
    lg = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"])
    if lg: st.session_state.logo_bytes = lg.read()
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, use_container_width=False)
with c2:
    st.markdown("<div class='title'>Mega Formation — BC-Style Exam</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Listening • Reading • Use of English • Writing</div>", unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.header("Setup")
    st.session_state.name  = st.text_input("Candidate name", value=st.session_state.name)
    st.session_state.level = st.selectbox("Level", LEVELS, index=LEVELS.index(st.session_state.level))
    st.session_state.seed  = st.number_input("Random seed", value=st.session_state.seed, step=1, format="%d")

    st.subheader("Email result (auto on submit)")
    st.caption("Gmail SMTP — يلزم App Password")
    st.session_state.email_enabled     = st.checkbox("Send result by email", value=st.session_state.email_enabled)
    st.session_state.smtp_user         = st.text_input("Sender Gmail", value=st.session_state.smtp_user, placeholder="you@gmail.com")
    st.session_state.smtp_app_password = st.text_input("Gmail App Password", value=st.session_state.smtp_app_password, type="password", placeholder="xxxx xxxx xxxx xxxx")
    st.session_state.email_to          = st.text_input("Recipient", value=st.session_state.email_to)

    if not st.session_state.started:
        if st.button("▶️ Start Exam"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.started = True
            mins = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=mins)
    else:
        if st.button("🔁 Restart"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.seed = random.randint(1, 10_000_000)
            mins = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=mins)

# -------------------- HELPERS --------------------
def time_left_str():
    if not st.session_state.deadline: return ""
    left = st.session_state.deadline - datetime.utcnow()
    if left.total_seconds() <= 0: return "00:00"
    mm, ss = divmod(int(left.total_seconds()), 60)
    return f"{mm:02d}:{ss:02d}"

def shuffle_copy(lst, seed):
    r = random.Random(seed); L = lst[:]; r.shuffle(L); return L

def mcq_block(items, key_prefix, answers_store):
    rows=[]
    for i,(q,opts,ans) in enumerate(items):
        answers_store[i] = st.radio(q, opts, index=None, key=f"{key_prefix}_{i}")
        rows.append({"Q#":i+1,"Question":q,"Correct":ans})
    return rows

def tfn_block(items, key_prefix, answers_store):
    rows=[]
    for i,(s,label) in enumerate(items):
        answers_store[i] = st.radio(s, ["T","F","NG"], index=None, key=f"{key_prefix}_{i}")
        rows.append({"Q#":i+1,"Statement":s,"Correct":label})
    return rows

def match_headings_block(mapping, key_prefix, answers_store):
    rows=[]
    for i,(para,heads) in enumerate(mapping.items()):
        answers_store[i] = st.selectbox(f"{para}", heads, index=None, key=f"{key_prefix}_{i}")
        rows.append({"Q#":i+1,"Paragraph":para,"Correct":heads[0]})
    return rows

def dropdown_block(items, key_prefix, answers_store):
    rows=[]
    for i,(stem,opts,ans) in enumerate(items):
        answers_store[i] = st.selectbox(stem, opts, index=None, key=f"{key_prefix}_{i}")
        rows.append({"Q#":i+1,"Stem":stem,"Correct":ans})
    return rows

def score_mcq_report(rows, user_map):
    correct=0; table=[]
    for i,info in enumerate(rows):
        corr = info["Correct"]
        user = user_map.get(i)
        ok = (user==corr)
        correct += int(ok)
        rec = {k:info.get(k,"") for k in info}
        rec.update({"User":user or "", "IsCorrect":ok})
        table.append(rec)
    pct = round(100*correct/max(1,len(rows)),1) if rows else 0.0
    return pct, pd.DataFrame(table)

def score_writing(text, level):
    prompt, low, high, kws = WRITING[level]
    wc = len(text.split())
    base = 40 if (low<=wc<=high) else (20 if wc>0 else 0)
    kw_hits = sum(1 for k in kws if k.lower() in text.lower())
    kw_score = min(60, kw_hits*12)
    return min(100, base+kw_score), wc, kw_hits, kws

def to_csv_bytes(df): return df.to_csv(index=False).encode()

def send_email(smtp_user, app_pw, to_addr, subject, body, named_files):
    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"]   = to_addr
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    for fname, data in named_files:
        part = MIMEBase("application","octet-stream")
        part.set_payload(data); encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
        msg.attach(part)
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(smtp_user, app_pw)
        server.sendmail(smtp_user, [to_addr], msg.as_string())

# -------------------- EXAM BODY --------------------
if st.session_state.started:
    k1,k2,k3 = st.columns([1,1,2])
    with k1:
        st.markdown("**Level**"); st.markdown(f"<span class='badge'>{st.session_state.level}</span>", unsafe_allow_html=True)
    with k2:
        st.markdown("**Time Left**"); st.markdown(f"<div class='kpi'>{time_left_str()}</div>", unsafe_allow_html=True)
    with k3:
        st.info("Complete all sections then Submit at the bottom. BC-like layout: Preparation + Tasks.")

    if time_left_str()=="00:00":
        st.warning("Time is up! Auto-submitting your exam.")

    lvl = st.session_state.level
    rnd = random.Random(st.session_state.seed)
    tabs = st.tabs(SECTIONS)

    # ---------- LISTENING ----------
    with tabs[0]:
        lst = LISTENING[lvl]

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Preparation — Vocabulary match")
        L_prep_map = {}
        prep_rows = []
        for i,(term,defn) in enumerate(lst["prep_pairs"]):
            opts = [defn,"a different meaning","an unrelated meaning","another possible meaning"]
            rnd.shuffle(opts)
            L_prep_map[i] = st.radio(f"{term}", opts, index=None, key=f"L_prep_{i}")
            prep_rows.append({"Q#":i+1,"Term":term,"Correct":defn})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Audio")
        st.caption("استمع للنص ثم أجب عن الأسئلة.")
        st.audio(tts_or_beep_bytes(lst["transcript"]))
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Task 1 — Multiple choice")
        L_t1_map = {}
        L_t1_rows = mcq_block(lst["task1_mcq"], "L_t1", L_t1_map)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Task 2 — Note completion (Dropdown)")
        L_t2_map = {}
        L_t2_rows = dropdown_block(lst["task2_dropdown"], "L_t2", L_t2_map)
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.answers["Listening"] = {
            "prep":L_prep_map, "t1":L_t1_map, "t2":L_t2_map,
            "_keys":{"prep":prep_rows,"t1":L_t1_rows,"t2":L_t2_rows}
        }

    # ---------- READING ----------
    with tabs[1]:
        rd = READING[lvl]

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Reading Passage")
        st.info(rd["passage"])
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Task 1 — Multiple choice")
        R_t1_map = {}
        R_t1_rows = mcq_block(rd["task1_mcq"], "R_t1", R_t1_map)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Task 2 — True / False / Not Given")
        R_t2_map = {}
        R_t2_rows = tfn_block(rd["task2_tfn"], "R_t2", R_t2_map)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Task 3 — Match headings to paragraphs")
        R_t3_map = {}
        R_t3_rows = match_headings_block(rd["task3_headings"], "R_t3", R_t3_map)
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.answers["Reading"] = {
            "t1":R_t1_map, "t2":R_t2_map, "t3":R_t3_map,
            "_keys":{"t1":R_t1_rows,"t2":R_t2_rows,"t3":R_t3_rows}
        }

    # ---------- USE OF ENGLISH ----------
    with tabs[2]:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        st.subheader("Use of English — Cloze")
        U_map = {}
        U_rows = []
        for i,(stem,opts,ans) in enumerate(USE[lvl]):
            U_map[i] = st.selectbox(stem, opts, index=None, key=f"U_{i}")
            U_rows.append({"Q#":i+1,"Stem":stem,"Correct":ans})
        st.markdown("</div>", unsafe_allow_html=True)

        st.session_state.answers["Use of English"] = {"t1":U_map, "_keys":{"t1":U_rows}}

    # ---------- WRITING ----------
    with tabs[3]:
        st.markdown("<div class='section'>", unsafe_allow_html=True)
        prompt, low, high, kws = WRITING[lvl]
        st.subheader("Writing")
        st.write(f"**Prompt:** {prompt}")
        st.caption(f"Target words: {low}–{high}. Include: {', '.join(kws)}")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=230, key="W_0")
        st.markdown("</div>", unsafe_allow_html=True)

    # ---------- SUBMIT ----------
    if st.button("✅ Submit All", type="primary") or time_left_str()=="00:00":
        # Listening scoring
        L_ans = st.session_state.answers["Listening"]
        L_prep_pct, L_prep_df = score_mcq_report(L_ans["_keys"]["prep"], L_ans["prep"])
        L_t1_pct,   L_t1_df   = score_mcq_report(L_ans["_keys"]["t1"],   L_ans["t1"])
        L_t2_pct,   L_t2_df   = score_mcq_report(L_ans["_keys"]["t2"],   L_ans["t2"])
        L_pct = round((L_prep_pct + L_t1_pct + L_t2_pct)/3, 1)

        # Reading scoring
        R_ans = st.session_state.answers["Reading"]
        R_t1_pct, R_t1_df = score_mcq_report(R_ans["_keys"]["t1"], R_ans["t1"])
        R_t2_pct, R_t2_df = score_mcq_report(R_ans["_keys"]["t2"], R_ans["t2"])
        R_t3_pct, R_t3_df = score_mcq_report(R_ans["_keys"]["t3"], R_ans["t3"])
        R_pct = round((R_t1_pct + R_t2_pct + R_t3_pct)/3, 1)

        # Use of English scoring
        U_ans = st.session_state.answers["Use of English"]
        U_t1_pct, U_t1_df = score_mcq_report(U_ans["_keys"]["t1"], U_ans["t1"])
        U_pct = U_t1_pct

        # Writing scoring (rule-of-thumb)
        W_text = st.session_state.answers["Writing"].get(0,"")
        def score_writing(text, level):
            prompt, low, high, kws = WRITING[level]
            wc = len(text.split())
            base = 40 if (low<=wc<=high) else (20 if wc>0 else 0)
            kw_hits = sum(1 for k in kws if k.lower() in text.lower())
            kw_score = min(60, kw_hits*12)
            return min(100, base+kw_score), wc, kw_hits, kws
        W_pct, wc, hits, W_kws = score_writing(W_text, lvl)

        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)
        st.success(f"**Overall: {overall}%** — {'✅ PASS' if overall>=PASS_MARK else '❌ FAIL'}")
        st.write({"Listening":L_pct,"Reading":R_pct,"Use of English":U_pct,"Writing":W_pct})
        st.caption(f"Writing: words={wc}, keywords matched={hits}/{len(W_kws)} (manual review advised)")

        # CSVs
        files = [
            ("listening_prep.csv",  to_csv_bytes(L_prep_df)),
            ("listening_task1.csv", to_csv_bytes(L_t1_df)),
            ("listening_task2.csv", to_csv_bytes(L_t2_df)),
            ("reading_task1.csv",   to_csv_bytes(R_t1_df)),
            ("reading_task2.csv",   to_csv_bytes(R_t2_df)),
            ("reading_task3.csv",   to_csv_bytes(R_t3_df)),
            ("use_cloze.csv",       to_csv_bytes(U_t1_df)),
        ]

        # Email (auto)
        if st.session_state.email_enabled and st.session_state.smtp_user and st.session_state.smtp_app_password and st.session_state.email_to:
            try:
                subject = f"Mega Formation — Exam Result ({st.session_state.name or 'Candidate'}) [{lvl}]"
                body = (
                    f"Candidate: {st.session_state.name or 'N/A'}\n"
                    f"Level: {lvl}\n"
                    f"Overall: {overall}% ({'PASS' if overall>=PASS_MARK else 'FAIL'})\n\n"
                    f"Listening: {L_pct}  (Prep {L_prep_pct}, T1 {L_t1_pct}, T2 {L_t2_pct})\n"
                    f"Reading:   {R_pct}  (T1 {R_t1_pct}, T2 {R_t2_pct}, T3 {R_t3_pct})\n"
                    f"Use of English: {U_pct}\n"
                    f"Writing: {W_pct} | words={wc}, keywords matched={hits}/{len(W_kws)}\n"
                    f"Sent automatically from Mega Formation Exam app."
                )
                named_files = [(fname, data) for (fname, data) in files]
                # send
                msg = MIMEMultipart()
                msg["From"] = st.session_state.smtp_user
                msg["To"]   = st.session_state.email_to
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))
                for fname, data in named_files:
                    part = MIMEBase("application","octet-stream")
                    part.set_payload(data); encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f'attachment; filename="{fname}"')
                    msg.attach(part)
                ctx = ssl.create_default_context()
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
                    server.login(st.session_state.smtp_user, st.session_state.smtp_app_password)
                    server.sendmail(st.session_state.smtp_user, [st.session_state.email_to], msg.as_string())
                st.success(f"✅ Results emailed to {st.session_state.email_to}")
            except Exception as e:
                st.error(f"Email failed: {e}")
        else:
            st.info("Email not sent — عبيّ المرسل/كلمة سر التطبيق وتأكد تفعيل الإرسال.")

        # Reset after submit
        st.session_state.started=False
        st.session_state.deadline=None
else:
    st.info("اختار المستوى واضغط Start Exam. Listening فيه Audio يتولّد تلقائيًا (TTS) مع fallback WAV واحد صالح.")
