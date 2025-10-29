# Mega_Admin_Builder_Exam.py
# ------------------------------------------------------------------
# Mega Formation — Admin Builder for English Exam (A1–B2)
# - Admin: يبني الامتحان (أسئلة بأنواع Radio/Checkbox/Text/TFN)
# - Listening: رفع Audio + Transcript
# - حفظ/تحميل الامتحان بصيغة JSON من exams/
# - Candidate: يشوف امتحانك "جاهز" حسب اللي بنيتو، مع تصحيح آلي
# - نتائج حسب الفرع (MB/BZ) تحفظ في results/ وتظهر في لوحة أدمين
# ------------------------------------------------------------------

import streamlit as st
import os, json, time, io, base64, shutil
import pandas as pd
from datetime import datetime, timedelta

# -------------------- ثابتات --------------------
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

# -------------------- ستايل --------------------
st.set_page_config(page_title="Mega Formation — Admin Exam Builder", layout="wide")
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

# -------------------- Helpers --------------------
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

def ensure_exam_state():
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
    st.session_state.setdefault("logo_bytes", None)
    st.session_state.setdefault("is_admin", True)
    st.session_state.setdefault("candidate_started", False)
    st.session_state.setdefault("deadline", None)
    st.session_state.setdefault("answers", {s:{} for s in SECTIONS})
    st.session_state.setdefault("exam_loaded_path","")

def audio_player_for(path):
    # path must be relative under MEDIA_DIR
    if not path: return None
    fpath = path if os.path.isabs(path) else os.path.join(MEDIA_DIR, path)
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            return f.read()
    return None

def clean_filename(name):
    safe = "".join(c if c.isalnum() or c in ("-","_",".") else "_" for c in name)
    return safe[:80]

def points_of(item):
    return int(item.get("points", 1))

def score_item(item, user_val):
    itype = item.get("type")
    correct = item.get("answer")
    pts = points_of(item)

    if itype == "radio":
        return pts if user_val == correct else 0
    if itype == "checkbox":
        # compare sets
        return pts if set(user_val or []) == set(correct or []) else 0
    if itype == "tfn":
        return pts if (user_val in ["T","F","NG"] and user_val == correct) else 0
    if itype == "text":
        # keyword-based: answer contains list of keywords (lowercased)
        kws = [k.strip().lower() for k in (correct or []) if k.strip()]
        text = (user_val or "").lower()
        hit = sum(1 for k in kws if k in text)
        # نصّب pts على عدد الكلمات المفتاحية (كل كلمة مفتاحية = 1 نقطة) أو سقف pts
        return min(pts, hit) if kws else (pts if text.strip() else 0)
    return 0

def render_task_editor(section_key, idx=None):
    """
    يُظهر فورم لإنشاء/تعديل task:
    - type: radio / checkbox / text / tfn
    - q: نص السؤال
    - options (للـradio/checkbox) كل خيار في سطر
    - correct (radio: خيار واحد — checkbox: عدّة — tfn: T/F/NG — text: keywords)
    - points
    يرجّع dict جاهز للحفظ أو None لو أُلغي.
    """
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader(f"{section_key} — Add/Edit Task")

    if idx is None:
        itype = st.selectbox("Question type", ["radio","checkbox","text","tfn"], key=f"{section_key}_new_type")
        q = st.text_area("Question / Prompt", key=f"{section_key}_new_q")
        pts = st.number_input("Points", value=1, step=1, min_value=1, key=f"{section_key}_new_points")
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
        else:  # text
            options = []
            correct_raw = st.text_input("Keywords (comma-separated)", key=f"{section_key}_new_corr_txt", placeholder="payment, extension, invoice")
            correct = [k.strip() for k in correct_raw.split(",") if k.strip()]

        if st.button("➕ Add task", key=f"{section_key}_add_btn"):
            task = {"type": itype, "q": q.strip(), "options": options, "answer": correct, "points": int(pts)}
            st.success("Task added.")
            st.markdown("</div>", unsafe_allow_html=True)
            return task
    else:
        # Editing existing item idx
        data = st.session_state.exam[section_key]["tasks"][idx]
        itype = st.selectbox("Question type", ["radio","checkbox","text","tfn"], index=["radio","checkbox","text","tfn"].index(data.get("type","radio")), key=f"{section_key}_edit_type_{idx}")
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
        else:
            options = []
            correct_raw = st.text_input("Keywords (comma-separated)", value=", ".join(correct if isinstance(correct,list) else []), key=f"{section_key}_edit_corr_txt_{idx}")
            correct = [k.strip() for k in correct_raw.split(",") if k.strip()]

        left, right = st.columns(2)
        with left:
            if st.button("💾 Save changes", key=f"{section_key}_save_{idx}"):
                st.session_state.exam[section_key]["tasks"][idx] = {"type": itype, "q": q.strip(), "options": options, "answer": correct, "points": int(pts)}
                st.success("Task updated.")
        with right:
            if st.button("🗑️ Delete", key=f"{section_key}_del_{idx}"):
                st.session_state.exam[section_key]["tasks"].pop(idx)
                st.warning("Task deleted.")
                st.markdown("</div>", unsafe_allow_html=True)
                return "DELETED"

    st.markdown("</div>", unsafe_allow_html=True)
    return None

def render_section_tasks_admin(section_key):
    # Existing tasks list + add new
    tasks = st.session_state.exam[section_key]["tasks"]
    if tasks:
        st.write(f"**{len(tasks)} task(s) created**")
        for i, t in enumerate(tasks):
            with st.expander(f"Task {i+1} — {t.get('type','radio')}: {t.get('q','')[:60]}"):
                res = render_task_editor(section_key, idx=i)
                if res == "DELETED":
                    st.experimental_rerun()
    else:
        st.info("No tasks yet.")
    # Add new task
    new_task = render_task_editor(section_key)
    if new_task:
        st.session_state.exam[section_key]["tasks"].append(new_task)

def render_listening_admin():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Listening — Audio & Transcript")
    tr = st.text_area("Transcript", value=st.session_state.exam["listening"]["transcript"])
    st.session_state.exam["listening"]["transcript"] = tr

    up = st.file_uploader("Upload audio (MP3/WAV)", type=["mp3","wav"], key="listen_upload")
    if up:
        fname = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{clean_filename(up.name)}"
        fpath = os.path.join(MEDIA_DIR, fname)
        with open(fpath, "wb") as f:
            f.write(up.read())
        st.session_state.exam["listening"]["audio_path"] = fname
        st.success(f"Saved audio: media/{fname}")

    apath = st.session_state.exam["listening"]["audio_path"]
    if apath:
        st.caption(f"Current audio: media/{apath}")
        ab = audio_player_for(apath)
        if ab: st.audio(ab)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### Listening Tasks")
    render_section_tasks_admin("listening")

def render_reading_admin():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Reading — Passage")
    st.session_state.exam["reading"]["passage"] = st.text_area("Passage", value=st.session_state.exam["reading"]["passage"])
    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("### Reading Tasks")
    render_section_tasks_admin("reading")

def render_use_admin():
    st.markdown("### Use of English Tasks")
    render_section_tasks_admin("use")

def render_writing_admin():
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Writing — Prompt & Auto-criteria")
    w = st.session_state.exam["writing"]
    w["prompt"] = st.text_area("Prompt", value=w.get("prompt",""))
    cols = st.columns(2)
    w["min_words"] = cols[0].number_input("Min words", value=int(w.get("min_words",100)), min_value=0, step=5)
    w["max_words"] = cols[1].number_input("Max words", value=int(w.get("max_words",150)), min_value=0, step=5)
    kw_raw = st.text_input("Keywords (comma-separated)", value=", ".join(w.get("keywords",[])))
    w["keywords"] = [k.strip() for k in kw_raw.split(",") if k.strip()]
    st.session_state.exam["writing"] = w
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------- عنوان و ترويسة --------------------
ensure_exam_state()
c1,c2,c3 = st.columns([1,4,1])
with c1:
    lg = st.file_uploader("Logo (PNG/JPG)", type=["png","jpg","jpeg"], key="logo_up")
    if lg: st.session_state.logo_bytes = lg.read()
    if st.session_state.logo_bytes: st.image(st.session_state.logo_bytes, use_container_width=False)
with c2:
    st.markdown("<div class='title'>Mega Formation — Admin Exam Builder</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Build your exam → Save → Run candidate view</div>", unsafe_allow_html=True)
with c3:
    st.toggle("Admin mode", key="is_admin")

# -------------------- الشريط الجانبي --------------------
with st.sidebar:
    st.header("Exam Meta")
    meta = st.session_state.exam["meta"]
    meta["title"] = st.text_input("Exam Title", value=meta.get("title","Mega Formation English Exam"))
    meta["level"] = st.selectbox("Level", LEVELS, index=LEVELS.index(meta.get("level","B1")))
    meta["branch"] = st.selectbox("Branch", list(BRANCHES.keys()), index=list(BRANCHES.keys()).index(meta.get("branch","Menzel Bourguiba")))
    # مدة تلقائية حسب المستوى لكن تنجم تغيّرها
    default_min = DEFAULT_DUR.get(meta["level"], 60)
    meta["duration_min"] = st.number_input("Duration (minutes)", value=int(meta.get("duration_min", default_min)), min_value=10, step=5)
    meta["exam_id"] = st.text_input("Exam ID (for saving)", value=meta.get("exam_id",""))
    st.session_state.exam["meta"] = meta

    st.markdown("---")
    st.subheader("Save / Load Exam")
    fname_default = clean_filename(meta["exam_id"] or f"{meta['level']}_{BRANCHES[meta['branch']]}_{datetime.now().strftime('%Y%m%d')}.json")
    if st.button("💾 Save exam to exams/"):
        path = os.path.join(EXAMS_DIR, fname_default or f"exam_{int(time.time())}.json")
        save_json(path, st.session_state.exam)
        st.session_state.exam_loaded_path = path
        st.success(f"Saved: {path}")

    # Load existing
    existing = [f for f in os.listdir(EXAMS_DIR) if f.lower().endswith(".json")]
    sel = st.selectbox("Load exam", ["-- select --"]+existing)
    if sel != "-- select --":
        if st.button("📂 Load selected"):
            path = os.path.join(EXAMS_DIR, sel)
            data = load_json(path)
            if data:
                st.session_state.exam = data
                st.session_state.exam_loaded_path = path
                st.success(f"Loaded: {path}")

    st.markdown("---")
    # Candidate controls
    st.subheader("Candidate Controls")
    candidate_name = st.text_input("Candidate name")
    candidate_branch = st.selectbox("Candidate Branch", list(BRANCHES.keys()))
    if not st.session_state.candidate_started:
        if st.button("▶️ Start Candidate Exam"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = True
            dur = int(st.session_state.exam["meta"]["duration_min"] or default_min)
            st.session_state.deadline = datetime.utcnow() + timedelta(minutes=dur)
    else:
        if st.button("🔁 Restart Candidate"):
            st.session_state.answers = {s:{} for s in SECTIONS}
            st.session_state.candidate_started = False
            st.session_state.deadline = None

    # Admin results view
    st.markdown("---")
    st.subheader("Admin — Results by Branch")
    bcode = BRANCHES[candidate_branch]
    df = results_df(RESULT_PATHS[bcode])
    if df.empty:
        st.caption("No results yet.")
    else:
        lvl_filter = st.multiselect("Filter levels", LEVELS, default=LEVELS)
        view = df[df["level"].isin(lvl_filter)].copy()
        st.dataframe(view, use_container_width=True)
        st.download_button("⬇️ Download CSV", data=view.to_csv(index=False).encode(), file_name=f"results_{bcode}.csv", mime="text/csv")

# -------------------- واجهة الأدمين لبناء الامتحان --------------------
if st.session_state.is_admin:
    tabs_admin = st.tabs(["Listening","Reading","Use of English","Writing"])
    # Listening admin
    with tabs_admin[0]:
        render_listening_admin()
    # Reading admin
    with tabs_admin[1]:
        render_reading_admin()
    # Use admin
    with tabs_admin[2]:
        render_use_admin()
    # Writing admin
    with tabs_admin[3]:
        render_writing_admin()

# -------------------- واجهة المترشح --------------------
st.markdown("---")
st.subheader("🎓 Candidate View")
colA, colB = st.columns([1,1])
with colA:
    st.markdown(f"**Level**: <span class='badge'>{st.session_state.exam['meta']['level']}</span>", unsafe_allow_html=True)
with colB:
    # الوقت المتبقي
    if st.session_state.candidate_started and st.session_state.deadline:
        left = st.session_state.deadline - datetime.utcnow()
        mm, ss = max(0,int(left.total_seconds()))//60, max(0,int(left.total_seconds()))%60
        st.markdown(f"**Time Left**: <span class='kpi'>{mm:02d}:{ss:02d}</span>", unsafe_allow_html=True)

ctabs = st.tabs(SECTIONS)

# ========== Listening (Candidate) ==========
with ctabs[0]:
    L = st.session_state.exam["listening"]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Transcript (optional)**")
    if L.get("transcript"): st.info(L["transcript"])
    a_bytes = audio_player_for(L.get("audio_path",""))
    if a_bytes:
        st.audio(a_bytes)
    else:
        st.caption("No audio uploaded yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Render tasks
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Listening Tasks**")
    for i, t in enumerate(L.get("tasks", [])):
        tkey = f"L_{i}"
        if t["type"] == "radio":
            st.session_state.answers["Listening"][i] = st.radio(t["q"], t.get("options",[]), index=None, key=f"{tkey}")
        elif t["type"] == "checkbox":
            st.session_state.answers["Listening"][i] = st.multiselect(t["q"], t.get("options",[]), key=f"{tkey}")
        elif t["type"] == "tfn":
            st.session_state.answers["Listening"][i] = st.radio(t["q"], ["T","F","NG"], index=None, key=f"{tkey}")
        else:
            st.session_state.answers["Listening"][i] = st.text_input(t["q"], key=f"{tkey}")
    st.markdown("</div>", unsafe_allow_html=True)

# ========== Reading (Candidate) ==========
with ctabs[1]:
    R = st.session_state.exam["reading"]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Reading Passage**")
    if R.get("passage"): st.info(R["passage"])
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Reading Tasks**")
    for i, t in enumerate(R.get("tasks", [])):
        tkey = f"R_{i}"
        if t["type"] == "radio":
            st.session_state.answers["Reading"][i] = st.radio(t["q"], t.get("options",[]), index=None, key=f"{tkey}")
        elif t["type"] == "checkbox":
            st.session_state.answers["Reading"][i] = st.multiselect(t["q"], t.get("options",[]), key=f"{tkey}")
        elif t["type"] == "tfn":
            st.session_state.answers["Reading"][i] = st.radio(t["q"], ["T","F","NG"], index=None, key=f"{tkey}")
        else:
            st.session_state.answers["Reading"][i] = st.text_input(t["q"], key=f"{tkey}")
    st.markdown("</div>", unsafe_allow_html=True)

# ========== Use of English (Candidate) ==========
with ctabs[2]:
    U = st.session_state.exam["use"]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Use of English Tasks**")
    for i, t in enumerate(U.get("tasks", [])):
        tkey = f"U_{i}"
        if t["type"] == "radio":
            st.session_state.answers["Use of English"][i] = st.radio(t["q"], t.get("options",[]), index=None, key=f"{tkey}")
        elif t["type"] == "checkbox":
            st.session_state.answers["Use of English"][i] = st.multiselect(t["q"], t.get("options",[]), key=f"{tkey}")
        elif t["type"] == "tfn":
            st.session_state.answers["Use of English"][i] = st.radio(t["q"], ["T","F","NG"], index=None, key=f"{tkey}")
        else:
            st.session_state.answers["Use of English"][i] = st.text_input(t["q"], key=f"{tkey}")
    st.markdown("</div>", unsafe_allow_html=True)

# ========== Writing (Candidate) ==========
with ctabs[3]:
    W = st.session_state.exam["writing"]
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.write("**Writing Prompt**")
    if W.get("prompt"): st.write(W["prompt"])
    st.caption(f"Target words: {W.get('min_words',0)}–{W.get('max_words',0)} | Keywords: {', '.join(W.get('keywords',[])) or '—'}")
    st.session_state.answers["Writing"][0] = st.text_area("Your essay:", height=220, key="W_0")
    st.markdown("</div>", unsafe_allow_html=True)

# ========== Submit & Scoring ==========
def score_section(section_key, tasks, user_map):
    total_pts = sum(points_of(t) for t in tasks)
    got_pts = 0
    rows = []
    for i, t in enumerate(tasks):
        u = user_map.get(i)
        s = score_item(t, u)
        got_pts += s
        rows.append({
            "Q#": i+1,
            "type": t.get("type"),
            "q": t.get("q","")[:80],
            "user": u if not isinstance(u, list) else ", ".join(u),
            "correct": t.get("answer"),
            "points": points_of(t),
            "score": s
        })
    pct = round(100.0 * got_pts / total_pts, 1) if total_pts>0 else 0.0
    return pct, pd.DataFrame(rows)

def score_writing(text, min_w, max_w, keywords):
    wc = len((text or "").split())
    base = 40 if (min_w and max_w and min_w <= wc <= max_w) else (20 if wc>0 else 0)
    hits = sum(1 for k in (keywords or []) if k.lower() in (text or "").lower())
    kw_score = min(60, hits * 12)
    return min(100, base + kw_score), wc, hits

left, right = st.columns([2,1])
with left:
    can_submit = st.session_state.candidate_started
    if st.button("✅ Submit All", type="primary", disabled=not can_submit):
        # Time up guard (optional)
        st.session_state.candidate_started = False

        # Listening
        L_tasks = st.session_state.exam["listening"]["tasks"]
        L_pct, L_df = score_section("Listening", L_tasks, st.session_state.answers["Listening"])

        # Reading
        R_tasks = st.session_state.exam["reading"]["tasks"]
        R_pct, R_df = score_section("Reading", R_tasks, st.session_state.answers["Reading"])

        # Use
        U_tasks = st.session_state.exam["use"]["tasks"]
        U_pct, U_df = score_section("Use of English", U_tasks, st.session_state.answers["Use of English"])

        # Writing
        W = st.session_state.exam["writing"]
        W_text = st.session_state.answers["Writing"].get(0,"")
        W_pct, wc, hits = score_writing(W_text, W.get("min_words",0), W.get("max_words",0), W.get("keywords",[]))

        overall = round((L_pct + R_pct + U_pct + W_pct)/4, 1)
        st.success(f"**Overall: {overall}%**")
        st.write({"Listening":L_pct, "Reading":R_pct, "Use of English":U_pct, "Writing":W_pct})
        st.caption(f"Writing: words={wc}, keyword hits={hits}/{len(W.get('keywords',[]))}")

        # Save result by branch
        meta = st.session_state.exam["meta"]
        bcode = BRANCHES[meta["branch"]]
        exam_id = meta.get("exam_id") or os.path.basename(st.session_state.exam_loaded_path or "")
        row = {
            "timestamp": now_iso(),
            "name": st.session_state.get("candidate_name","") if st.session_state.get("candidate_name") else "",
            "branch": bcode,
            "level": meta["level"],
            "exam_id": exam_id,
            "overall": overall,
            "Listening": L_pct,
            "Reading": R_pct,
            "Use_of_English": U_pct,
            "Writing": W_pct,
        }
        save_result_row(bcode, row)

        # CSV downloads (اختياري)
        st.download_button("⬇️ Listening report", L_df.to_csv(index=False).encode(), "listening_report.csv", "text/csv")
        st.download_button("⬇️ Reading report",   R_df.to_csv(index=False).encode(), "reading_report.csv",   "text/csv")
        st.download_button("⬇️ Use report",       U_df.to_csv(index=False).encode(), "use_report.csv",       "text/csv")

with right:
    # عرض الوقت المتبقي
    if st.session_state.candidate_started and st.session_state.deadline:
        left = st.session_state.deadline - datetime.utcnow()
        if left.total_seconds() <= 0:
            st.warning("Time is up! Consider submitting.")
