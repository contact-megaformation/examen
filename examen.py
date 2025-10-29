import streamlit as st
import random
import pandas as pd
import json
from datetime import datetime, timedelta
from io import StringIO

# ==========================
# English Exam — Authoring + Audio Listening + Import/Export
# ==========================

st.set_page_config(page_title="English Exam — Authoring + Audio", layout="wide")

# ---------- Styles ----------
st.markdown(
    """
    <style>
      .title {text-align:center; font-size: 36px; font-weight:800; margin-bottom:0}
      .subtitle {text-align:center; color:#555; margin-top:4px}
      .card {background:#fff; padding:18px 20px; border-radius:16px; box-shadow:0 6px 24px rgba(0,0,0,0.06); margin-bottom:12px}
      .muted {color:#666}
      .kpi {font-size:28px; font-weight:700}
      .badge {display:inline-block; padding:4px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-weight:700; font-size:12px}
      .small {font-size:12px; color:#666}
      .danger {color:#b91c1c}
      .ok {color:#16a34a}
      .btnbar button {margin-right:6px}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='title'>English Placement / Exam</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Authoring Listening (Audio/URL) • Reading • Use of English • Writing</div>", unsafe_allow_html=True)

# ---------- Config ----------
LEVEL_ORDER = ["A1", "A2", "B1", "B2"]
SECTION_ORDER = ["Listening", "Reading", "Use of English", "Writing"]

LEVEL_TIME = {
    "A1": {"Listening": 8, "Reading": 8, "Use of English": 8, "Writing": 15},
    "A2": {"Listening": 10, "Reading": 10, "Use of English": 10, "Writing": 20},
    "B1": {"Listening": 12, "Reading": 12, "Use of English": 12, "Writing": 25},
    "B2": {"Listening": 15, "Reading": 15, "Use of English": 15, "Writing": 30},
}
PASS_MARK = 60
Q_PER = {"Listening": 6, "Reading": 6, "Use of English": 8}

# ---------- Default Reading / Use of English / Writing ----------
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

# ---------- State ----------
def init_state():
    if "started" not in st.session_state: st.session_state.started = False
    st.session_state.setdefault("name", "")
    st.session_state.setdefault("level", "A1")
    st.session_state.setdefault("seed", random.randint(1, 10_000_000))
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s: {} for s in SECTION_ORDER})
    # Authoring bank: per level list of items
    # item = {q, options(list of 4), answer, transcript(str), audio_mode("upload"/"url"/"none"), audio_bytes(optional), audio_url(optional)}
    st.session_state.setdefault("LISTENING_BANK_CUSTOM", {lvl: [] for lvl in LEVEL_ORDER})
    st.session_state.setdefault("listening_shuffle", {lvl: False for lvl in LEVEL_ORDER})

init_state()

# ---------- Helpers ----------
def set_deadline(level):
    minutes = sum(LEVEL_TIME[level].values())
    st.session_state.deadline = datetime.utcnow() + timedelta(minutes=minutes)

def time_left_str():
    if not st.session_state.deadline: return ""
    left = st.session_state.deadline - datetime.utcnow()
    if left.total_seconds() <= 0: return "00:00"
    mm, ss = divmod(int(left.total_seconds()), 60)
    return f"{mm:02d}:{ss:02d}"

def reading_items(level, n):
    data = R_PASSAGES[level]
    qs = data["qs"][:]
    rnd = random.Random(st.session_state.seed)
    rnd.shuffle(qs)
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

def ensure_four_options(opts):
    return (opts + [""]*4)[:4]

def safe_rerun():
    try:
        st.rerun()
    except Exception:
        try:
            st.experimental_rerun()
        except Exception:
            pass

# ================= Admin / Authoring =================
st.markdown("### Admin / Authoring — Listening (Audio + Questions)")
with st.expander("Open/Close Authoring", expanded=True):
    c1, c2 = st.columns([1,2])
    with c1:
        lvl = st.selectbox("Level", LEVEL_ORDER, index=LEVEL_ORDER.index(st.session_state.level), key="auth_level")
        st.session_state.listening_shuffle[lvl] = st.checkbox("Shuffle Listening for this level", value=st.session_state.listening_shuffle[lvl])
    with c2:
        st.caption("Build your Listening items: for each question, add audio (upload or URL), write question, 4 options, choose the correct answer, and (optional) transcript.")

    # Add / Edit item
    st.subheader("➕ Add / Edit Item")
    q_text = st.text_input("Question", key="auth_q")
    opt1 = st.text_input("Option 1", key="auth_o1")
    opt2 = st.text_input("Option 2", key="auth_o2")
    opt3 = st.text_input("Option 3", key="auth_o3")
    opt4 = st.text_input("Option 4", key="auth_o4")
    correct = st.selectbox("Correct answer", ["Option 1","Option 2","Option 3","Option 4"], index=0, key="auth_correct")
    transcript = st.text_area("Transcript (fallback if no audio)", key="auth_trans")

    audio_mode = st.radio("Audio source", ["upload","url","none"], horizontal=True, key="auth_amode")
    audio_bytes = None; audio_url = None
    if audio_mode == "upload":
        f = st.file_uploader("Upload audio (mp3/wav/ogg/m4a)", type=["mp3","wav","ogg","m4a"], key="auth_file")
        if f: audio_bytes = f.read()
    elif audio_mode == "url":
        audio_url = st.text_input("Audio URL (direct link to mp3/ogg/wav)", key="auth_url")

    add_cols = st.columns([1,1,1])
    with add_cols[0]:
        if st.button("Add item"):
            opts = ensure_four_options([opt1,opt2,opt3,opt4])
            ans_text = opts[["Option 1","Option 2","Option 3","Option 4"].index(correct)]
            item = {
                "q": q_text.strip(),
                "options": opts,
                "answer": ans_text,
                "transcript": transcript.strip(),
                "audio_mode": audio_mode,
                "audio_bytes": audio_bytes,
                "audio_url": audio_url,
            }
            st.session_state.LISTENING_BANK_CUSTOM[lvl].append(item)
            st.success("Item added.")
    with add_cols[1]:
        if st.button("Clear inputs"):
            for k in ["auth_q","auth_o1","auth_o2","auth_o3","auth_o4","auth_trans","auth_url","auth_file"]:
                if k in st.session_state: st.session_state[k] = "" if k!="auth_file" else None

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
                    if st.button("⬆️ Up", key=f"up_{lvl}_{i}", disabled=(i==0)):
                        bank[i-1], bank[i] = bank[i], bank[i-1]
                        safe_rerun()
                with cc[1]:
                    if st.button("⬇️ Down", key=f"down_{lvl}_{i}", disabled=(i==len(bank)-1)):
                        bank[i+1], bank[i] = bank[i], bank[i+1]
                        safe_rerun()
                with cc[2]:
                    st.write("Correct:", f"**{it['answer']}**")
                    st.caption(f"Options: {', '.join(it['options'])}")
                with cc[3]:
                    if it["audio_mode"]=="upload" and it.get("audio_bytes"):
                        st.audio(it["audio_bytes"])
                    elif it["audio_mode"]=="url" and it.get("audio_url"):
                        st.audio(it["audio_url"])
                    else:
                        st.caption("No audio. (Transcript available)")
                delcol = st.columns([1,6])[0]
                with delcol:
                    if st.button("🗑️ Delete", key=f"del_{lvl}_{i}"):
                        bank.pop(i)
                        safe_rerun()
                st.divider()

    # Export / Import JSON (robust)
    st.subheader("⬇️ Export / ⬆️ Import")
    ex1, ex2 = st.columns(2)
    with ex1:
        if st.button("Export Listening Bank (levels JSON)"):
            export_levels = {lv: st.session_state.LISTENING_BANK_CUSTOM[lv] for lv in LEVEL_ORDER}
            data = json.dumps(export_levels, ensure_ascii=False, indent=2).encode()
            st.download_button("Download JSON", data=data, file_name="listening_bank_levels.json", mime="application/json", key="dl_json_levels")
    with ex2:
        up = st.file_uploader("Import Listening Bank (JSON)", type=["json"], key="imp_json")
        if up:
            try:
                raw = up.read().decode()
                data = json.loads(raw)

                # Accept two formats:
                # 1) Pure levels dict: {"A1": [...], "A2": [...], "B1": [...], "B2": [...]}
                # 2) Wrapped: {"metadata": {...}, "LISTENING_BANK_CUSTOM": {"A1": [...], ...}}
                if all(k in data for k in LEVEL_ORDER):
                    new_bank = {lv: data.get(lv, []) for lv in LEVEL_ORDER}
                elif "LISTENING_BANK_CUSTOM" in data and isinstance(data["LISTENING_BANK_CUSTOM"], dict) and all(k in data["LISTENING_BANK_CUSTOM"] for k in LEVEL_ORDER):
                    new_bank = {lv: data["LISTENING_BANK_CUSTOM"].get(lv, []) for lv in LEVEL_ORDER}
                else:
                    st.error("Invalid JSON structure. Expected top-level A1/A2/B1/B2 or LISTENING_BANK_CUSTOM with levels.")
                    new_bank = None

                if new_bank is not None:
                    # quick sanity: ensure options length == 4
                    for lv in LEVEL_ORDER:
                        for it in new_bank[lv]:
                            it["options"] = ensure_four_options(it.get("options", []))
                            it.setdefault("audio_mode", "none")
                            it.setdefault("audio_url", None)
                            if it["audio_mode"] != "upload":
                                it["audio_bytes"] = None  # uploaded bytes cannot come from JSON
                    st.session_state.LISTENING_BANK_CUSTOM = new_bank
                    st.success("Imported successfully.")
            except Exception as e:
                st.error(f"Failed to import: {e}")

st.markdown("---")

# ================= Exam Tabs =================
tabs = st.tabs(["Take Exam", "Help / Notes"])

# ---------- Take Exam ----------
with tabs[0]:
    with st.sidebar:
        st.header("Exam Setup")
        st.session_state.name = st.text_input("Candidate name", value=st.session_state.name, key="cand_name")
        st.session_state.level = st.selectbox("Level", LEVEL_ORDER, index=LEVEL_ORDER.index(st.session_state.level), key="lvl_take")
        st.session_state.seed = st.number_input("Random seed", value=st.session_state.seed, step=1, format="%d")
        st.caption("⏱ Total time = sum of sections by level.")
        if not st.session_state.started:
            if st.button("▶️ Start Exam", key="start_exam"):
                st.session_state.answers = {s: {} for s in SECTION_ORDER}
                st.session_state.started = True
                minutes = sum(LEVEL_TIME[st.session_state.level].values())
                st.session_state.deadline = datetime.utcnow() + timedelta(minutes=minutes)
        else:
            if st.button("🔁 Restart", key="restart_exam"):
                st.session_state.seed = random.randint(1, 10_000_000)
                st.session_state.answers = {s: {} for s in SECTION_ORDER}
                minutes = sum(LEVEL_TIME[st.session_state.level].values())
                st.session_state.deadline = datetime.utcnow() + timedelta(minutes=minutes)

    if st.session_state.started:
        k1, k2, k3 = st.columns([1,1,2])
        with k1:
            st.markdown("**Level**")
            st.markdown(f"<span class='badge'>{st.session_state.level}</span>", unsafe_allow_html=True)
        with k2:
            st.markdown("**Time Left**")
            st.markdown(f"<div class='kpi'>{time_left_str()}</div>", unsafe_allow_html=True)
        with k3:
            st.info("Answer all sections then Submit All at the bottom.")

        if time_left_str() == "00:00":
            st.warning("Time is up! Auto-submitting your exam.")

        lvl_take = st.session_state.level

        # Listening from custom bank
        L_pool = list(st.session_state.LISTENING_BANK_CUSTOM[lvl_take])
        if st.session_state.listening_shuffle[lvl_take]:
            rnd = random.Random(st.session_state.seed); rnd.shuffle(L_pool)
        L_items = L_pool[:Q_PER["Listening"]]

        # Reading / Use of English
        R_text, R_items = reading_items(lvl_take, Q_PER["Reading"])
        U_all = U_BANK[lvl_take][:]
        rnd_u = random.Random(st.session_state.seed); rnd_u.shuffle(U_all)
        U_items = [{"q": q, "options": [o1,o2,o3,o4], "answer": ans} for (q,[o1,o2,o3,o4],ans) in U_all[:Q_PER["Use of English"]]]

        exam_tabs = st.tabs(SECTION_ORDER)

        # Listening
        with exam_tabs[0]:
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
                else:
                    st.caption("No audio provided.")
                # choices
                st.session_state.answers["Listening"][i] = st.radio("Select one:", it["options"], index=None, key=f"LL_{i}")
                # transcript fallback
                if it.get("transcript"):
                    with st.expander("Transcript"):
                        st.caption(it["transcript"])
                st.divider()
            st.markdown("</div>", unsafe_allow_html=True)

        # Reading
        with exam_tabs[1]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Read the passage and answer the questions.**")
            st.info(R_text)
            for i, (q, opts, ans) in enumerate(R_items):
                st.markdown(f"**R{i+1}.** {q}")
                st.session_state.answers["Reading"][i] = st.radio("Select one:", opts, index=None, key=f"RR_{i}")
                st.divider()
            st.markdown("</div>", unsafe_allow_html=True)

        # Use of English
        with exam_tabs[2]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.write("**Grammar & Vocabulary.** Choose the best answer.")
            for i, it in enumerate(U_items):
                st.markdown(f"**U{i+1}.** {it['q']}")
                st.session_state.answers["Use of English"][i] = st.radio("Select one:", it["options"], index=None, key=f"UU_{i}")
                st.divider()
            st.markdown("</div>", unsafe_allow_html=True)

        # Writing
        with exam_tabs[3]:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            prompt, kws = W_PROMPTS[lvl_take]
            st.write(f"**Prompt:** {prompt}")
            st.caption(f"Try to include: {', '.join(kws)}")
            st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=220, key="WW_0")
            st.markdown("</div>", unsafe_allow_html=True)

        # Submit
        if st.button("✅ Submit All", type="primary") or time_left_str()=="00:00":
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

            # Downloads
            def to_csv_bytes(df):
                buf = StringIO(); df.to_csv(buf, index=False); return buf.getvalue().encode()
            if L_pct is not None:
                st.download_button("⬇️ Listening report (CSV)", to_csv_bytes(L_df), file_name=f"{st.session_state.name or 'candidate'}_{lvl_take}_Listening.csv")
            st.download_button("⬇️ Reading report (CSV)", to_csv_bytes(R_df), file_name=f"{st.session_state.name or 'candidate'}_{lvl_take}_Reading.csv")
            st.download_button("⬇️ UseOfEnglish report (CSV)", to_csv_bytes(U_df), file_name=f"{st.session_state.name or 'candidate'}_{lvl_take}_UseOfEnglish.csv")

            st.session_state.started = False
            st.session_state.deadline = None

# ---------- Help / Notes ----------
with tabs[1]:
    st.markdown(
        """
        **كيفاش تربط أسئلة الـListening مع الصوت؟**
        - من **Admin / Authoring**:
          - اختار المستوى (A1/A2/B1/B2).
          - لكل سؤال: اختر مصدر الصوت (Upload/URL/None)، اكتب السؤال و4 اختيارات وحدّد الإجابة الصحيحة، وأضف Transcript اختياري.
          - تنجم ترتّب الأسئلة (Up/Down) وتمسح أي سؤال.
        - **Export** يعطيك JSON بسيط بالمستويات: {"A1": [], "A2": [], "B1": [...], "B2": []}
        - **Import** يقبل:
          1) JSON بسيط بالمستويات (المُفضّل لتطبيقك).
          2) أو JSON فيه "metadata" و "LISTENING_BANK_CUSTOM" — يتم استخراج المستويات تلقائيًا.

        **الامتحان:**
        - من "Take Exam": اختار المستوى، Start Exam. Listening يستعمل البنك اللي حضرتو بيدك. لو فعلت "Shuffle"، الأسئلة تتبدّل بصفة عشوائية.
        """
    )
