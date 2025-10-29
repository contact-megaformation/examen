# EnglishExam_WithAuthoring_Audio_Email.py
# ----------------------------------------
# Streamlit app: 4 Épreuves (Listening/Reading/Use of English/Writing)
# - Authoring for Listening (Upload/URL/TTS) + Import/Export JSON
# - Custom total time: A1/A2=60 min, B1/B2=90 min
# - Logo upload in header
# - Email results with CSV attachments via Gmail SMTP (App Password)

import streamlit as st
import random, json, math, struct, io, wave, smtplib, ssl
from datetime import datetime, timedelta
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email import encoders
import pandas as pd

# ============== PAGE CONFIG & STYLES ==============
st.set_page_config(page_title="English Exam — Authoring + Audio + Email", layout="wide")

st.markdown("""
<style>
  .title {text-align:center; font-size: 34px; font-weight:800; margin-bottom:0}
  .subtitle {text-align:center; color:#555; margin-top:4px}
  .card {background:#fff; padding:18px 20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,0.06); margin:12px 0}
  .kpi {font-size:28px; font-weight:700}
  .badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px}
  .logo {max-height:70px}
</style>
""", unsafe_allow_html=True)

# ============== CONFIG ==============
LEVEL_ORDER = ["A1", "A2", "B1", "B2"]
SECTION_ORDER = ["Listening", "Reading", "Use of English", "Writing"]

# Total exam minutes per level
TOTAL_TIME_MIN = {"A1": 60, "A2": 60, "B1": 90, "B2": 90}
PASS_MARK = 60
Q_PER = {"Listening": 6, "Reading": 6, "Use of English": 8}

# ============== SAMPLE BANKS (Reading / Use of English / Writing) ==============
R_PASSAGES = {
    "A1": {
        "text": "Maria lives in a small town near the sea. She works in a café and goes to the beach after work.",
        "qs": [
            ("Where does Maria live?", ["In a big city", "In a small town near the sea", "In the mountains", "In the desert"], "In a small town near the sea"),
            ("Where does Maria work?", ["In a shop", "In a café", "In a bank", "At school"], "In a café"),
            ("What does she do after work?", ["Goes home", "Goes to the gym", "Goes to the beach", "Studies"], "Goes to the beach"),
            ("Maria lives __ the sea.", ["at", "near", "on", "under"], "near"),
            ("The text says Maria works __.", ["in a café", "in an office", "from home", "at night only"], "in a café"),
            ("The opposite of 'small' is __.", ["little", "tiny", "big", "short"], "big"),
        ]
    },
    "A2": {
        "text": "The city library moved to a larger building. Now it offers weekend workshops, free Wi-Fi, and study rooms.",
        "qs": [
            ("Why did the library move?", ["It was closed", "To a smaller place", "To a larger building", "For repairs"], "To a larger building"),
            ("Which service is mentioned?", ["Paid internet", "Free Wi-Fi", "Gym", "Cinema"], "Free Wi-Fi"),
            ("When are workshops offered?", ["Weekdays", "Weekends", "Every night", "Holidays only"], "Weekends"),
            ("Study rooms are available __.", ["for staff only", "for students only", "for users", "for teachers"], "for users"),
            ("The library now has more __.", ["space", "noise", "rules", "fees"], "space"),
            ("'Offers' is closest to __.", ["gives", "buys", "sells", "hides"], "gives"),
        ]
    },
    "B1": {
        "text": "Volunteering can strengthen communities by connecting people with local needs. However, volunteers require training to be effective.",
        "qs": [
            ("What strengthens communities?", ["Traffic", "Volunteering", "Taxes", "Tourism"], "Volunteering"),
            ("What do volunteers require?", ["Money", "Uniforms", "Training", "Cars"], "Training"),
            ("Volunteering connects people with __.", ["local needs", "sports", "politics", "fashion"], "local needs"),
            ("To be effective, volunteers need __.", ["experience only", "training", "nothing", "luck"], "training"),
            ("The tone of the passage is __.", ["critical", "informative", "funny", "angry"], "informative"),
            ("'However' shows __.", ["addition", "contrast", "time", "cause"], "contrast"),
        ]
    },
    "B2": {
        "text": "While renewable energy adoption is accelerating, integrating intermittent sources into aging grids demands investment and regulatory coordination.",
        "qs": [
            ("What is accelerating?", ["Fossil fuel use", "Renewable energy adoption", "Electricity prices", "Grid failures"], "Renewable energy adoption"),
            ("What makes integration challenging?", ["Cheap technology", "Intermittent sources", "Abundant storage", "Public support"], "Intermittent sources"),
            ("Grids described are __.", ["new", "aging", "perfect", "private"], "aging"),
            ("What does integration demand?", ["No changes", "Investment and coordination", "Less regulation", "Fewer workers"], "Investment and coordination"),
            ("'Intermittent' most nearly means __.", ["constant", "irregular", "fast", "expensive"], "irregular"),
            ("The passage focuses on __.", ["transport", "energy policy", "education", "health"], "energy policy"),
        ]
    },
}

U_BANK = {
    "A1": [
        ("He __ a student.", ["am", "is", "are", "be"], "is"),
        ("We __ in Tunis.", ["live", "lives", "living", "to live"], "live"),
        ("There __ two apples.", ["is", "are", "be", "been"], "are"),
        ("I __ coffee every day.", ["drink", "drinks", "drank", "drinking"], "drink"),
        ("Choose the plural: one man → two __.", ["mans", "men", "manses", "menses"], "men"),
        ("She __ from Spain.", ["are", "am", "is", "be"], "is"),
        ("I go __ school by bus.", ["to", "in", "on", "at"], "to"),
        ("Opposite of 'hot' is __.", ["warm", "cold", "heat", "cool"], "cold"),
    ],
    "A2": [
        ("I have lived here __ 2019.", ["for", "since", "during", "from"], "since"),
        ("If it rains, we __ at home.", ["stay", "will stay", "stayed", "stays"], "will stay"),
        ("He can't __ the meeting.", ["to attend", "attends", "attend", "attending"], "attend"),
        ("We didn't go out __ the rain.", ["because", "because of", "so", "although"], "because of"),
        ("Choose the past: buy → __.", ["buyed", "bought", "buys", "buy"], "bought"),
        ("You're coming, __?", ["isn't you", "aren't you", "don't you", "won't you"], "aren't you"),
        ("I'm interested __ history.", ["in", "on", "at", "about"], "in"),
        ("This test is __ than the last.", ["easyer", "easier", "more easy", "most easy"], "easier"),
    ],
    "B1": [
        ("I wish I __ more time.", ["have", "had", "would have", "am having"], "had"),
        ("Hardly __ the meeting begun when the alarm rang.", ["had", "has", "did", "was"], "had"),
        ("He denied __ the window.", ["to break", "break", "breaking", "to have broke"], "breaking"),
        ("We need someone __ can code.", ["who", "which", "whom", "what"], "who"),
        ("Despite __ late, she finished.", ["to arrive", "arrive", "arriving", "arrived"], "arriving"),
        ("The manager suggested that he __ earlier.", ["comes", "come", "came", "would come"], "come"),
        ("It's high time you __.", ["come", "came", "would come", "had come"], "came"),
        ("Make __ decision.", ["do a", "make a", "take a", "create a"], "make a"),
    ],
    "B2": [
        ("No sooner __ the announcement made than shares fell.", ["was", "had", "has", "having"], "had"),
        ("The project, __ objectives were unclear, was delayed.", ["whose", "who's", "which", "that"], "whose"),
        ("Had I known, I __ earlier.", ["left", "would have left", "would leave", "had left"], "would have left"),
        ("He insisted that she __ present.", ["be", "was", "is", "would be"], "be"),
        ("The proposal was rejected on the __ that ...", ["grounds", "reasons", "basis", "causes"], "grounds"),
        ("Seldom __ such a case.", ["I hear", "do I hear", "I have heard", "did I heard"], "do I hear"),
        ("By the time it finishes, we __ ten modules.", ["will have completed", "completed", "have completed", "had completed"], "will have completed"),
        ("We should consider __ a pilot program.", ["to launch", "launch", "launching", "to be launching"], "launching"),
    ],
}

W_PROMPTS = {
    "A1": ("Write about your daily routine (50–70 words).", ["morning", "work", "eat", "go", "home"]),
    "A2": ("Describe your last holiday (80–100 words).", ["where", "when", "with", "activities", "feelings"]),
    "B1": ("Do you prefer studying alone or in groups? Explain (120–150 words).", ["prefer", "because", "example", "time", "learn"]),
    "B2": ("Some companies allow remote work. Discuss advantages and disadvantages (180–220 words).", ["productivity", "balance", "communication", "costs", "team"]),
}

# ============== STATE ==============
def init_state():
    st.session_state.setdefault("started", False)
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("level", "A1")
    st.session_state.setdefault("seed", random.randint(1, 10_000_000))
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s: {} for s in SECTION_ORDER})
    # Listening authoring bank
    st.session_state.setdefault("LISTENING_BANK_CUSTOM", {lv: [] for lv in LEVEL_ORDER})
    st.session_state.setdefault("listening_shuffle", {lv: False for lv in LEVEL_ORDER})
    # Logo bytes
    st.session_state.setdefault("logo_bytes", None)
    # Email settings
    st.session_state.setdefault("email_enabled", True)
    st.session_state.setdefault("smtp_user", "")
    st.session_state.setdefault("smtp_app_password", "")
    st.session_state.setdefault("email_to", "contact.megaformation@gmail.com")

init_state()

# ============== AUDIO HELPERS (TTS or Beep WAV) ==============
def gen_beep_wav_bytes(duration_sec=1.0, freq=440, rate=22050, volume=0.3):
    n = int(duration_sec * rate)
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(rate)
        for i in range(n):
            t = i / rate
            sample = volume * math.sin(2 * math.pi * freq * t)
            wf.writeframes(struct.pack('<h', int(sample * 32767)))
    return buf.getvalue()

def tts_or_beep_bytes(text, rate_wpm=150):
    # Try offline TTS via pyttsx3; fall back to beep
    try:
        import pyttsx3, tempfile, os
        engine = pyttsx3.init()
        engine.setProperty('rate', rate_wpm)
        # temp wav file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tf:
            tmp_path = tf.name
        engine.save_to_file(text, tmp_path)
        engine.runAndWait()
        with open(tmp_path, "rb") as f:
            data = f.read()
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return data
    except Exception:
        # fallback: short double-beep
        b1 = gen_beep_wav_bytes(0.35, 660)
        b2 = gen_beep_wav_bytes(0.35, 880)
        return b1 + b2

# ============== HEADER (Logo + Title) ==============
col_logo, col_title = st.columns([1,4])
with col_logo:
    if st.session_state.logo_bytes:
        st.image(st.session_state.logo_bytes, caption="", use_container_width=False)
with col_title:
    st.markdown("<div class='title'>English Placement / Exam</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>4 Épreuves • Listening (Audio) / Reading / Use of English / Writing</div>", unsafe_allow_html=True)

# ============== SIDEBAR CONFIG ==============
with st.sidebar:
    st.header("Setup")
    st.session_state.name = st.text_input("Candidate name", value=st.session_state.name)
    st.session_state.level = st.selectbox("Level", LEVEL_ORDER, index=LEVEL_ORDER.index(st.session_state.level))
    st.session_state.seed = st.number_input("Random seed", value=st.session_state.seed, step=1, format="%d")

    # Logo upload
    st.subheader("Logo")
    lg = st.file_uploader("Upload logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if lg: st.session_state.logo_bytes = lg.read()

    # Email section
    st.subheader("Email Results (Gmail SMTP)")
    st.caption("يَلزم App Password من Gmail (موش كلمة السرّ العادية).")
    st.session_state.email_enabled = st.checkbox("Send results by email", value=st.session_state.email_enabled)
    st.session_state.smtp_user = st.text_input("Your Gmail address (sender)", value=st.session_state.smtp_user, placeholder="you@gmail.com")
    st.session_state.smtp_app_password = st.text_input("Gmail App Password", value=st.session_state.smtp_app_password, type="password", placeholder="xxxx xxxx xxxx xxxx")
    st.session_state.email_to = st.text_input("Recipient", value=st.session_state.email_to)

    st.caption(f"📧 Default recipient: contact.megaformation@gmail.com")

    # Start/Restart
    if not st.session_state.started:
        if st.button("▶️ Start Exam"):
            st.session_state.answers = {s: {} for s in SECTION_ORDER}
            st.session_state.started = True
            total_min = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=total_min)
    else:
        if st.button("🔁 Restart"):
            st.session_state.seed = random.randint(1, 10_000_000)
            st.session_state.answers = {s: {} for s in SECTION_ORDER}
            total_min = TOTAL_TIME_MIN[st.session_state.level]
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=total_min)

# ============== ADMIN / AUTHORING LISTENING ==============
st.markdown("### Admin / Authoring — Listening (Audio + Questions)")
with st.expander("Open/Close Authoring", expanded=False):
    c1, c2 = st.columns([1,2])
    with c1:
        lvl = st.selectbox("Level", LEVEL_ORDER, index=LEVEL_ORDER.index(st.session_state.level), key="auth_level")
        st.session_state.listening_shuffle[lvl] = st.checkbox("Shuffle this level", value=st.session_state.listening_shuffle[lvl])
    with c2:
        st.caption("Add audio (Upload/URL/TTS), write question, 4 options, set answer, optional transcript. Export/Import below.")

    # Inputs
    q_text = st.text_input("Question", key="auth_q")
    o1 = st.text_input("Option 1", key="auth_o1")
    o2 = st.text_input("Option 2", key="auth_o2")
    o3 = st.text_input("Option 3", key="auth_o3")
    o4 = st.text_input("Option 4", key="auth_o4")
    correct_idx = st.selectbox("Correct answer", ["Option 1","Option 2","Option 3","Option 4"], index=0, key="auth_correct")
    transcript = st.text_area("Transcript (fallback if no audio)", key="auth_trans")
    audio_mode = st.radio("Audio source", ["upload","url","tts","none"], horizontal=True, key="auth_amode")

    audio_bytes, audio_url = None, None
    if audio_mode == "upload":
        f = st.file_uploader("Upload audio (mp3/wav/ogg/m4a)", type=["mp3","wav","ogg","m4a"], key="auth_file")
        if f: audio_bytes = f.read()
    elif audio_mode == "url":
        audio_url = st.text_input("Audio URL (direct link to mp3/ogg/wav)", key="auth_url")
    elif audio_mode == "tts":
        tts_preview = st.text_area("TTS text (what to speak). If empty, will use transcript.", key="auth_tts_text")
        if st.button("▶️ Generate TTS preview"):
            txt = tts_preview.strip() or transcript.strip() or q_text.strip() or "Listening item."
            st.audio(tts_or_beep_bytes(txt))

    cbtn = st.columns(3)
    with cbtn[0]:
        if st.button("➕ Add item"):
            opts = ( [o1,o2,o3,o4] + [""]*4 )[:4]
            ans = opts[["Option 1","Option 2","Option 3","Option 4"].index(correct_idx)]
            item = {
                "q": q_text.strip(),
                "options": opts,
                "answer": ans,
                "transcript": transcript.strip(),
                "audio_mode": audio_mode,
                "audio_bytes": audio_bytes,  # will be kept in RAM for this session
                "audio_url": audio_url,
                "tts_text": st.session_state.get("auth_tts_text","").strip()
            }
            st.session_state.LISTENING_BANK_CUSTOM[lvl].append(item)
            st.success("Item added.")
    with cbtn[1]:
        if st.button("🧹 Clear inputs"):
            for k in ["auth_q","auth_o1","auth_o2","auth_o3","auth_o4","auth_trans","auth_url","auth_file","auth_tts_text"]:
                if k in st.session_state:
                    st.session_state[k] = "" if k!="auth_file" else None

    st.markdown("---")
    st.subheader(f"📚 Items for {lvl}")
    bank = st.session_state.LISTENING_BANK_CUSTOM[lvl]
    if not bank:
        st.info("No items yet. Add some above.")
    else:
        for i, it in enumerate(bank):
            with st.container():
                st.markdown(f"**L{i+1}** — {it['q'] or '(no question)'}")
                cc = st.columns([1,1,2,2])
                with cc[0]:
                    if st.button("⬆️", key=f"up_{lvl}_{i}", disabled=(i==0)):
                        bank[i-1], bank[i] = bank[i], bank[i-1]
                        st.rerun()
                with cc[1]:
                    if st.button("⬇️", key=f"down_{lvl}_{i}", disabled=(i==len(bank)-1)):
                        bank[i+1], bank[i] = bank[i], bank[i+1]
                        st.rerun()
                with cc[2]:
                    st.write("Correct:", f"**{it['answer']}**")
                    st.caption(f"Options: {', '.join(it['options'])}")
                with cc[3]:
                    if it.get("audio_mode")=="upload" and it.get("audio_bytes"):
                        st.audio(it["audio_bytes"])
                    elif it.get("audio_mode")=="url" and it.get("audio_url"):
                        st.audio(it["audio_url"])
                    elif it.get("audio_mode")=="tts":
                        txt = (it.get("tts_text") or it.get("transcript") or it.get("q") or "Listening item.").strip()
                        st.audio(tts_or_beep_bytes(txt))
                    else:
                        st.caption("No audio. (Transcript available)")
                delcol = st.columns([1,6])[0]
                with delcol:
                    if st.button("🗑️ Delete", key=f"del_{lvl}_{i}"):
                        bank.pop(i)
                        st.rerun()
                st.divider()

    # Export / Import
    st.subheader("⬇️ Export / ⬆️ Import")
    e1, e2 = st.columns(2)
    with e1:
        if st.button("Export levels JSON"):
            export_levels = {lv: st.session_state.LISTENING_BANK_CUSTOM[lv] for lv in LEVEL_ORDER}
            data = json.dumps(export_levels, ensure_ascii=False, indent=2).encode()
            st.download_button("Download JSON", data=data, file_name="listening_bank_levels.json", mime="application/json", key="dl_json_levels")
    with e2:
        up = st.file_uploader("Import levels JSON", type=["json"], key="imp_json")
        if up:
            try:
                data = json.loads(up.read().decode())
                if all(k in data for k in LEVEL_ORDER):
                    new_bank = {lv: data.get(lv, []) for lv in LEVEL_ORDER}
                    # normalize
                    for lv in LEVEL_ORDER:
                        for it in new_bank[lv]:
                            it["options"] = (it.get("options", []) + [""]*4)[:4]
                            it.setdefault("audio_mode","none")
                            it.setdefault("audio_url", None)
                            it.setdefault("tts_text","")
                            if it["audio_mode"] != "upload":
                                it["audio_bytes"] = None
                    st.session_state.LISTENING_BANK_CUSTOM = new_bank
                    st.success("Imported successfully.")
                else:
                    st.error("Invalid JSON: expected top-level keys A1/A2/B1/B2.")
            except Exception as e:
                st.error(f"Failed to import: {e}")

st.markdown("---")

# ============== HELPERS (Exam) ==============
def time_left_str():
    if not st.session_state.deadline:
        return ""
    left = st.session_state.deadline - datetime.utcnow()
    if left.total_seconds() <= 0:
        return "00:00"
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

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode()

# ============== EXAM UI ==============
if st.session_state.started:
    k1,k2,k3 = st.columns([1,1,2])
    with k1:
        st.markdown("**Level**")
        st.markdown(f"<span class='badge'>{st.session_state.level}</span>", unsafe_allow_html=True)
    with k2:
        st.markdown("**Time Left**")
        st.markdown(f"<div class='kpi'>{time_left_str()}</div>", unsafe_allow_html=True)
    with k3:
        st.info("Complete the four sections, then click Submit All at the bottom.")

    if time_left_str() == "00:00":
        st.warning("Time is up! Auto-submitting your exam.")

    lvl_take = st.session_state.level

    # Listening items
    L_pool = list(st.session_state.LISTENING_BANK_CUSTOM[lvl_take])
    if st.session_state.listening_shuffle[lvl_take]:
        rnd = random.Random(st.session_state.seed); rnd.shuffle(L_pool)
    L_items = L_pool[:Q_PER["Listening"]]

    # Reading / Use of English
    R_text, R_items = reading_items(lvl_take, Q_PER["Reading"])
    U_all = U_BANK[lvl_take][:]
    rnd_u = random.Random(st.session_state.seed); rnd_u.shuffle(U_all)
    U_items = [{"q": q, "options": [a,b,c,d], "answer": ans} for (q,[a,b,c,d],ans) in U_all[:Q_PER["Use of English"]]]

    tabs = st.tabs(SECTION_ORDER)

    # Listening
    with tabs[0]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        if not L_items:
            st.error("No Listening items for this level. Add/import some in Admin / Authoring above.")
        for i, it in enumerate(L_items):
            st.markdown(f"**L{i+1}.** {it['q']}")
            # audio
            if it.get("audio_mode") == "upload" and it.get("audio_bytes"):
                st.audio(it["audio_bytes"])
            elif it.get("audio_mode") == "url" and it.get("audio_url"):
                st.audio(it["audio_url"])
            elif it.get("audio_mode") == "tts":
                txt = (it.get("tts_text") or it.get("transcript") or it.get("q") or "Listening item.").strip()
                st.audio(tts_or_beep_bytes(txt))
            else:
                st.caption("No audio provided.")
            st.session_state.answers["Listening"][i] = st.radio("Select one:", it["options"], index=None, key=f"L_{i}")
            if it.get("transcript"):
                with st.expander("Transcript"):
                    st.caption(it["transcript"])
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    # Reading
    with tabs[1]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Read the passage and answer the questions.**")
        st.info(R_text)
        for i, (q, opts, ans) in enumerate(R_items):
            st.markdown(f"**R{i+1}.** {q}")
            st.session_state.answers["Reading"][i] = st.radio("Select one:", opts, index=None, key=f"R_{i}")
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    # Use of English
    with tabs[2]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write("**Grammar & Vocabulary.** Choose the best answer.")
        for i, it in enumerate(U_items):
            st.markdown(f"**U{i+1}.** {it['q']}")
            st.session_state.answers["Use of English"][i] = st.radio("Select one:", it["options"], index=None, key=f"U_{i}")
            st.divider()
        st.markdown("</div>", unsafe_allow_html=True)

    # Writing
    with tabs[3]:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        prompt, kws = W_PROMPTS[lvl_take]
        st.write(f"**Prompt:** {prompt}")
        st.caption(f"Try to include: {', '.join(kws)}")
        st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=230, key="W_0")
        st.markdown("</div>", unsafe_allow_html=True)

    # Submit
    if st.button("✅ Submit All", type="primary") or time_left_str()=="00:00":
        # Score sections
        if L_items:
            L_pct, L_df = score_mcq(L_items, st.session_state.answers["Listening"])
        else:
            L_pct, L_df = None, pd.DataFrame([])
        R_df_items = [{"q": q, "options": opts, "answer": ans} for (q,opts,ans) in R_items]
        R_pct, R_df = score_mcq(R_df_items, st.session_state.answers["Reading"])
        U_pct, U_df = score_mcq(U_items, st.session_state.answers["Use of English"])
        W_text = st.session_state.answers["Writing"].get(0,"")
        W_pct, wc, hits, kws = score_writing(W_text, lvl_take)

        parts = [p for p in [L_pct, R_pct, U_pct, W_pct] if p is not None]
        overall = round(sum(parts)/len(parts), 1) if parts else 0.0

        st.success(f"**Overall Score: {overall}%** — {'✅ PASS' if overall >= PASS_MARK else '❌ FAIL'}")
        st.write({"Listening": L_pct, "Reading": R_pct, "Use of English": U_pct, "Writing": W_pct})
        st.caption(f"Writing → words={wc}, keywords matched={hits}/{len(kws)} (manual review recommended)")

        # Prepare CSV attachments
        files = []
        if L_pct is not None and not L_df.empty:
            files.append(("listening_report.csv", df_to_csv_bytes(L_df), "text/csv"))
        files.append(("reading_report.csv", df_to_csv_bytes(R_df), "text/csv"))
        files.append(("use_of_english_report.csv", df_to_csv_bytes(U_df), "text/csv"))

        # Email send
        if st.session_state.email_enabled and st.session_state.smtp_user and st.session_state.smtp_app_password and st.session_state.email_to:
            try:
                subject = f"Exam Result — {st.session_state.name or 'Candidate'} [{lvl_take}]"
                body = (
                    f"Candidate: {st.session_state.name or 'N/A'}\n"
                    f"Level: {lvl_take}\n"
                    f"Overall: {overall}% ({'PASS' if overall>=PASS_MARK else 'FAIL'})\n\n"
                    f"Section scores:\n"
                    f"- Listening: {L_pct if L_pct is not None else 'N/A'}\n"
                    f"- Reading: {R_pct}\n"
                    f"- Use of English: {U_pct}\n"
                    f"- Writing: {W_pct}\n\n"
                    f"Writing summary: words={wc}, keywords matched={hits}/{len(kws)}\n"
                    f"Sent from Mega Formation Exam app."
                )
                msg = MIMEMultipart()
                msg["From"] = st.session_state.smtp_user
                msg["To"] = st.session_state.email_to
                msg["Subject"] = subject
                msg.attach(MIMEText(body, "plain"))
                for fname, data, mime in files:
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
            st.info("Email not sent (check 'Send results by email', sender, app password, and recipient).")

        # Reset after submit
        st.session_state.started = False
        st.session_state.deadline = None

else:
    st.info("اختار المستوى واضغط Start Exam. في Listening تنجم تبني الأسئلة وتختار الصوت (Upload/URL/TTS) من تبويب Admin / Authoring.")

