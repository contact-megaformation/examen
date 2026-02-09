import uuid
from datetime import datetime
import pandas as pd
import streamlit as st
import gspread
from google.oauth2.service_account import Credentials


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Mega Portal (Staff + Students)", page_icon="🧩", layout="wide")

REQUIRED_SHEETS = {
    "Trainees": ["trainee_id", "full_name", "phone", "branch", "program", "group", "status", "created_at"],
    "Accounts": ["phone", "password", "trainee_id", "created_at", "last_login"],
    "Staff":    ["staff_id", "full_name", "phone", "password", "role", "branch_access", "program_access", "is_active", "created_at"],
    "Subjects": ["subject_id", "branch", "program", "group", "subject_name", "is_active", "created_at"],
    "Grades":   ["grade_id", "trainee_id", "subject_name", "exam_type", "score", "date", "staff_name", "note", "created_at"],
    "Timetable":["row_id", "branch", "program", "group", "day", "start", "end", "subject", "room", "teacher", "created_at"],
}

DAYS = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]


# =========================
# GSHEETS HELPERS
# =========================
@st.cache_resource
def gs_client():
    creds_dict = st.secrets["gcp_service_account"]
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    return gspread.authorize(creds)

def open_spreadsheet():
    sheet_id = st.secrets["GSHEET_ID"]
    return gs_client().open_by_key(sheet_id)

def ensure_worksheets_and_headers():
    sh = open_spreadsheet()
    existing = {ws.title: ws for ws in sh.worksheets()}

    for ws_name, headers in REQUIRED_SHEETS.items():
        if ws_name not in existing:
            sh.add_worksheet(title=ws_name, rows=2000, cols=max(10, len(headers)+2))
            existing[ws_name] = sh.worksheet(ws_name)

        ws = existing[ws_name]
        first_row = ws.row_values(1)
        if first_row != headers:
            ws.clear()
            ws.append_row(headers, value_input_option="RAW")

    return sh

@st.cache_data(ttl=8)
def read_df(ws_name: str) -> pd.DataFrame:
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    values = ws.get_all_values()
    if len(values) <= 1:
        return pd.DataFrame(columns=REQUIRED_SHEETS[ws_name])
    headers = values[0]
    rows = values[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df

def append_row(ws_name: str, row: dict):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    out = [str(row.get(h, "")).strip() for h in headers]
    ws.append_row(out, value_input_option="USER_ENTERED")
    st.cache_data.clear()

def update_cell(ws_name: str, row_index_1based: int, col_name: str, value):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    headers = REQUIRED_SHEETS[ws_name]
    col_index = headers.index(col_name) + 1
    ws.update_cell(row_index_1based, col_index, value)
    st.cache_data.clear()

def delete_row(ws_name: str, row_index_1based: int):
    sh = open_spreadsheet()
    ws = sh.worksheet(ws_name)
    ws.delete_rows(row_index_1based)
    st.cache_data.clear()

def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# =========================
# AUTH HELPERS
# =========================
def ensure_session():
    if "role" not in st.session_state:
        st.session_state.role = None   # "staff" or "student"
    if "user" not in st.session_state:
        st.session_state.user = {}
    if "page" not in st.session_state:
        st.session_state.page = "Login"

def logout():
    st.session_state.role = None
    st.session_state.user = {}
    st.session_state.page = "Login"

def staff_login(phone: str, password: str):
    df = read_df("Staff")
    if df.empty:
        return None
    df2 = df.copy()
    df2["phone"] = df2["phone"].astype(str).str.strip()
    df2["password"] = df2["password"].astype(str).str.strip()
    df2["is_active"] = df2["is_active"].astype(str).str.strip().str.lower()

    m = df2[(df2["phone"] == str(phone).strip()) & (df2["password"] == str(password).strip()) & (df2["is_active"] != "false")]
    if m.empty:
        return None
    return m.iloc[0].to_dict()

def student_login(phone: str, password: str):
    df = read_df("Accounts")
    if df.empty:
        return None
    df2 = df.copy()
    df2["phone"] = df2["phone"].astype(str).str.strip()
    df2["password"] = df2["password"].astype(str).str.strip()
    m = df2[(df2["phone"] == str(phone).strip()) & (df2["password"] == str(password).strip())]
    if m.empty:
        return None
    return m.iloc[0].to_dict()


# =========================
# UI HELPERS (MegaCRM style filters)
# =========================
def get_unique(df, col):
    if df.empty or col not in df.columns:
        return []
    return sorted([x for x in df[col].astype(str).str.strip().unique().tolist() if x])

def branch_program_group_filters(default_branch=None, default_program=None, default_group=None):
    tr = read_df("Trainees")
    branches = get_unique(tr, "branch")
    branch = st.selectbox("Branch", branches, index=branches.index(default_branch) if default_branch in branches else 0 if branches else None)
    tr_f = tr[tr["branch"].astype(str).str.strip() == str(branch).strip()] if branch else tr

    programs = get_unique(tr_f, "program")
    program = st.selectbox("Program", programs, index=programs.index(default_program) if default_program in programs else 0 if programs else None)
    tr_fp = tr_f[tr_f["program"].astype(str).str.strip() == str(program).strip()] if program else tr_f

    groups = get_unique(tr_fp, "group")
    group = st.selectbox("Group", groups, index=groups.index(default_group) if default_group in groups else 0 if groups else None)

    return branch, program, group


# =========================
# PAGES
# =========================
def page_login():
    st.title("🧩 Mega Portal — Login")

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("👩‍💼 Staff Login")
        phone = st.text_input("Staff Phone", key="staff_phone")
        pwd = st.text_input("Staff Password", type="password", key="staff_pwd")
        if st.button("Login as Staff", use_container_width=True):
            user = staff_login(phone, pwd)
            if user:
                st.session_state.role = "staff"
                st.session_state.user = user
                st.session_state.page = "Staff"
                st.success("Welcome!")
                st.rerun()
            else:
                st.error("Wrong staff phone/password, or account inactive.")

    with col2:
        st.subheader("🎓 Student Login")
        phone2 = st.text_input("Student Phone", key="stud_phone")
        pwd2 = st.text_input("Student Password", type="password", key="stud_pwd")
        if st.button("Login as Student", use_container_width=True):
            acc = student_login(phone2, pwd2)
            if acc:
                # update last_login
                df = read_df("Accounts")
                idx = df.index[df["phone"].astype(str).str.strip() == str(phone2).strip()].tolist()
                if idx:
                    update_cell("Accounts", idx[0] + 2, "last_login", now_str())

                st.session_state.role = "student"
                st.session_state.user = acc
                st.session_state.page = "Student"
                st.success("Welcome!")
                st.rerun()
            else:
                st.error("Wrong student phone/password.")

    st.divider()
    st.subheader("🆕 Student Registration")
    st.caption("مرة أولى: اختار الفرع/الاختصاص/المجموعة ثم اسمك، وحط رقم الهاتف ومودباس.")

    tr = read_df("Trainees")
    if tr.empty:
        st.warning("Trainees sheet فارغة. الموظّفين لازم يضيفو المتكوّنين أولاً.")
        return

    b, p, g = branch_program_group_filters()
    if not (b and p and g):
        st.info("اختار Branch/Program/Group باش تظهر قائمة الأسماء.")
        return

    trf = tr[
        (tr["branch"].astype(str).str.strip() == str(b).strip()) &
        (tr["program"].astype(str).str.strip() == str(p).strip()) &
        (tr["group"].astype(str).str.strip() == str(g).strip())
    ].copy()

    if trf.empty:
        st.warning("ما فماش متكوّنين في المجموعة هذي. كلم الإدارة.")
        return

    trf["label"] = trf["full_name"].astype(str).str.strip() + "  —  " + trf["trainee_id"].astype(str).str.strip()
    choice = st.selectbox("Choose your name", trf["label"].tolist())
    chosen_id = trf[trf["label"] == choice].iloc[0]["trainee_id"]

    phone = st.text_input("Phone (unique)", key="reg_phone")
    password = st.text_input("Password", type="password", key="reg_password")

    if st.button("Create Account", use_container_width=True):
        phone = str(phone).strip()
        password = str(password).strip()

        if not phone or not password:
            st.error("لازم Phone و Password.")
            return
        if len(password) < 4:
            st.error("المودباس قصير برشة. خليه 4 حروف/أرقام ولا أكثر.")
            return

        acc = read_df("Accounts")
        if not acc.empty:
            exists = acc["phone"].astype(str).str.strip().eq(phone).any()
            if exists:
                st.error("رقم الهاتف هذا مسجّل قبل. استعمل Login.")
                return

        append_row("Accounts", {
            "phone": phone,
            "password": password,
            "trainee_id": chosen_id,
            "created_at": now_str(),
            "last_login": ""
        })
        st.success("✅ Account created. تنجم توا تعمل Login كمتكوّن.")


def page_staff():
    st.title("👩‍💼 Staff Dashboard")
    staff = st.session_state.user
    st.caption(f"Logged in as: {staff.get('full_name','')} — Role: {staff.get('role','staff')}")

    tab1, tab2, tab3, tab4 = st.tabs(["👥 Trainees", "📚 Subjects", "📝 Grades", "🗓️ Timetable"])

    # ---------- Trainees ----------
    with tab1:
        st.subheader("Trainees Management")

        b, p, g = branch_program_group_filters()
        tr = read_df("Trainees")

        if b and p and g:
            trf = tr[
                (tr["branch"].astype(str).str.strip() == str(b).strip()) &
                (tr["program"].astype(str).str.strip() == str(p).strip()) &
                (tr["group"].astype(str).str.strip() == str(g).strip())
            ].copy()
        else:
            trf = tr.copy()

        left, right = st.columns([2, 1], gap="large")
        with left:
            st.write("Current list")
            if trf.empty:
                st.info("ما فماش متكوّنين في الفلترة هذي.")
            else:
                show = trf[["trainee_id","full_name","phone","branch","program","group","status"]].copy()
                st.dataframe(show, use_container_width=True, hide_index=True)

        with right:
            st.write("Add trainee")
            name = st.text_input("Full name", key="add_tr_name")
            phone = st.text_input("Phone (optional)", key="add_tr_phone")
            status = st.selectbox("Status", ["active","inactive"], index=0)

            if st.button("Add Trainee", use_container_width=True):
                if not (name and b and p and g):
                    st.error("لازم Full name + Branch/Program/Group.")
                else:
                    append_row("Trainees", {
                        "trainee_id": f"TR-{uuid.uuid4().hex[:8].upper()}",
                        "full_name": name.strip(),
                        "phone": str(phone).strip(),
                        "branch": str(b).strip(),
                        "program": str(p).strip(),
                        "group": str(g).strip(),
                        "status": status,
                        "created_at": now_str()
                    })
                    st.success("✅ Added.")
                    st.rerun()

    # ---------- Subjects ----------
    with tab2:
        st.subheader("Subjects (per Group)")

        b, p, g = branch_program_group_filters()
        sub = read_df("Subjects")

        if b and p and g:
            subf = sub[
                (sub["branch"].astype(str).str.strip() == str(b).strip()) &
                (sub["program"].astype(str).str.strip() == str(p).strip()) &
                (sub["group"].astype(str).str.strip() == str(g).strip())
            ].copy()
        else:
            subf = sub.copy()

        colA, colB = st.columns([2, 1], gap="large")
        with colA:
            if subf.empty:
                st.info("No subjects found for these filters.")
            else:
                st.dataframe(subf[["subject_id","subject_name","branch","program","group","is_active"]], use_container_width=True, hide_index=True)

        with colB:
            st.write("Add Subject")
            subject_name = st.text_input("Subject name", key="add_subject_name")
            if st.button("Add Subject", use_container_width=True):
                if not (b and p and g and subject_name.strip()):
                    st.error("لازم Branch/Program/Group + اسم مادة.")
                else:
                    append_row("Subjects", {
                        "subject_id": f"SB-{uuid.uuid4().hex[:8].upper()}",
                        "branch": str(b).strip(),
                        "program": str(p).strip(),
                        "group": str(g).strip(),
                        "subject_name": subject_name.strip(),
                        "is_active": "true",
                        "created_at": now_str()
                    })
                    st.success("✅ Added.")
                    st.rerun()

    # ---------- Grades ----------
    with tab3:
        st.subheader("Enter Grades")

        b, p, g = branch_program_group_filters()
        tr = read_df("Trainees")
        sub = read_df("Subjects")

        trf = tr[
            (tr["branch"].astype(str).str.strip() == str(b).strip()) &
            (tr["program"].astype(str).str.strip() == str(p).strip()) &
            (tr["group"].astype(str).str.strip() == str(g).strip())
        ].copy() if (b and p and g) else pd.DataFrame()

        subf = sub[
            (sub["branch"].astype(str).str.strip() == str(b).strip()) &
            (sub["program"].astype(str).str.strip() == str(p).strip()) &
            (sub["group"].astype(str).str.strip() == str(g).strip())
        ].copy() if (b and p and g) else pd.DataFrame()

        if trf.empty:
            st.warning("اختار Branch/Program/Group عندها متكوّنين.")
        elif subf.empty:
            st.warning("ما فماش مواد للمجموعة هذي. زيد مواد من تبويب Subjects.")
        else:
            trf["label"] = trf["full_name"].astype(str).str.strip() + " — " + trf["trainee_id"].astype(str).str.strip()
            trainee_choice = st.selectbox("Trainee", trf["label"].tolist())
            trainee_id = trf[trf["label"] == trainee_choice].iloc[0]["trainee_id"]

            subject = st.selectbox("Subject", sorted(subf["subject_name"].astype(str).str.strip().tolist()))
            exam_type = st.selectbox("Exam type", ["Exam", "Oral", "Final", "Surprise", "Other"])
            exam_type_custom = ""
            if exam_type == "Other":
                exam_type_custom = st.text_input("Custom exam type", placeholder="مثال: Devoir 1 / TP / ...")
            score = st.number_input("Score", min_value=0.0, max_value=20.0, value=10.0, step=0.25)
            date = st.date_input("Date")
            note = st.text_input("Note (optional)")

            if st.button("Save Grade", use_container_width=True):
                et = exam_type_custom.strip() if exam_type == "Other" else exam_type
                if not et:
                    st.error("اكتب نوع الإمتحان.")
                    return
                append_row("Grades", {
                    "grade_id": f"GR-{uuid.uuid4().hex[:10].upper()}",
                    "trainee_id": str(trainee_id).strip(),
                    "subject_name": str(subject).strip(),
                    "exam_type": et,
                    "score": str(score),
                    "date": str(date),
                    "staff_name": staff.get("full_name",""),
                    "note": note.strip(),
                    "created_at": now_str()
                })
                st.success("✅ Grade saved.")

            st.divider()
            st.write("Latest grades for this trainee")
            gr = read_df("Grades")
            grf = gr[gr["trainee_id"].astype(str).str.strip() == str(trainee_id).strip()].copy() if not gr.empty else pd.DataFrame()
            if grf.empty:
                st.info("No grades yet.")
            else:
                grf = grf.sort_values(by=["date","created_at"], ascending=False)
                st.dataframe(grf[["subject_name","exam_type","score","date","staff_name","note"]], use_container_width=True, hide_index=True)

    # ---------- Timetable ----------
    with tab4:
        st.subheader("Timetable (per Group)")

        b, p, g = branch_program_group_filters()
        tt = read_df("Timetable")

        ttf = tt[
            (tt["branch"].astype(str).str.strip() == str(b).strip()) &
            (tt["program"].astype(str).str.strip() == str(p).strip()) &
            (tt["group"].astype(str).str.strip() == str(g).strip())
        ].copy() if (b and p and g) else pd.DataFrame()

        st.caption("تنجم تزيد Rows جديدة وتعمل Save. إذا تحب delete، امسح الصف من الشيت مباشرة أو زيدنا زر delete من بعد.")

        if b and p and g:
            if ttf.empty:
                base = pd.DataFrame([{
                    "row_id": f"TT-{uuid.uuid4().hex[:8].upper()}",
                    "branch": b, "program": p, "group": g,
                    "day": "Monday", "start": "18:00", "end": "19:30",
                    "subject": "", "room": "", "teacher": "",
                    "created_at": now_str()
                }])
            else:
                base = ttf.copy()

            editable_cols = ["row_id","day","start","end","subject","room","teacher"]
            edited = st.data_editor(
                base[["row_id","branch","program","group","day","start","end","subject","room","teacher","created_at"]],
                use_container_width=True,
                num_rows="dynamic"
            )

            if st.button("Save Timetable", use_container_width=True):
                # Simplest approach: rewrite group rows -> delete old rows -> append all edited
                # We'll delete old rows by scanning sheet (careful with indices)
                sh = open_spreadsheet()
                ws = sh.worksheet("Timetable")
                all_vals = ws.get_all_values()
                headers = all_vals[0]
                rows = all_vals[1:]

                # find row indices to delete for this group
                to_delete = []
                for i, r in enumerate(rows, start=2):
                    rdict = dict(zip(headers, r + [""]*(len(headers)-len(r))))
                    if (rdict.get("branch","").strip() == str(b).strip()
                        and rdict.get("program","").strip() == str(p).strip()
                        and rdict.get("group","").strip() == str(g).strip()):
                        to_delete.append(i)

                # delete bottom-up to keep indices stable
                for ridx in sorted(to_delete, reverse=True):
                    ws.delete_rows(ridx)

                # append edited rows
                for _, row in edited.iterrows():
                    if str(row.get("day","")).strip() == "":
                        continue
                    append_row("Timetable", {
                        "row_id": str(row.get("row_id") or f"TT-{uuid.uuid4().hex[:8].upper()}").strip(),
                        "branch": str(b).strip(),
                        "program": str(p).strip(),
                        "group": str(g).strip(),
                        "day": str(row.get("day","")).strip(),
                        "start": str(row.get("start","")).strip(),
                        "end": str(row.get("end","")).strip(),
                        "subject": str(row.get("subject","")).strip(),
                        "room": str(row.get("room","")).strip(),
                        "teacher": str(row.get("teacher","")).strip(),
                        "created_at": str(row.get("created_at") or now_str()).strip(),
                    })

                st.success("✅ Timetable saved.")
                st.rerun()
        else:
            st.info("اختار Branch/Program/Group.")

    st.divider()
    if st.button("Logout"):
        logout()
        st.rerun()


def page_student():
    st.title("🎓 Student Portal")

    acc = st.session_state.user
    trainee_id = str(acc.get("trainee_id","")).strip()

    tr = read_df("Trainees")
    row = tr[tr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not tr.empty else pd.DataFrame()

    if row.empty:
        st.error("حسابك مربوط بtrainee_id غير موجود في Trainees. كلم الإدارة.")
        if st.button("Logout"):
            logout(); st.rerun()
        return

    info = row.iloc[0].to_dict()
    st.success(f"مرحباً {info.get('full_name','')} ✅")
    st.caption(f"Branch: {info.get('branch','')} | Program: {info.get('program','')} | Group: {info.get('group','')}")

    tab1, tab2, tab3 = st.tabs(["📝 Grades", "🗓️ Timetable", "⚙️ Account"])

    with tab1:
        gr = read_df("Grades")
        grf = gr[gr["trainee_id"].astype(str).str.strip() == trainee_id].copy() if not gr.empty else pd.DataFrame()
        if grf.empty:
            st.info("مازال ما تمش إدخال نوطات.")
        else:
            # compute simple averages per subject (optional)
            grf["score_num"] = pd.to_numeric(grf["score"], errors="coerce")
            st.dataframe(grf.sort_values(by=["date","created_at"], ascending=False)[
                ["subject_name","exam_type","score","date","staff_name","note"]
            ], use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("Averages (optional)")
            avg = grf.groupby("subject_name", as_index=False)["score_num"].mean().rename(columns={"score_num":"avg_score"})
            st.dataframe(avg, use_container_width=True, hide_index=True)

    with tab2:
        tt = read_df("Timetable")
        ttf = tt[
            (tt["branch"].astype(str).str.strip() == str(info.get("branch","")).strip()) &
            (tt["program"].astype(str).str.strip() == str(info.get("program","")).strip()) &
            (tt["group"].astype(str).str.strip() == str(info.get("group","")).strip())
        ].copy() if not tt.empty else pd.DataFrame()

        if ttf.empty:
            st.info("جدول الأوقات موش موجود توّا.")
        else:
            show = ttf[["day","start","end","subject","room","teacher"]].copy()
            st.dataframe(show, use_container_width=True, hide_index=True)

    with tab3:
        st.subheader("Change password")
        newp = st.text_input("New password", type="password")
        if st.button("Update password", use_container_width=True):
            newp = str(newp).strip()
            if len(newp) < 4:
                st.error("المودباس قصير.")
            else:
                df = read_df("Accounts")
                idx = df.index[df["phone"].astype(str).str.strip() == str(acc.get("phone","")).strip()].tolist()
                if not idx:
                    st.error("Account row not found.")
                else:
                    update_cell("Accounts", idx[0] + 2, "password", newp)
                    st.success("✅ Updated.")
                    # refresh session
                    st.session_state.user["password"] = newp

    st.divider()
    if st.button("Logout"):
        logout()
        st.rerun()


# =========================
# APP ROUTER
# =========================
def sidebar_nav():
    st.sidebar.title("Mega Portal")
    st.sidebar.caption("MegaCRM-style navigation")

    if st.session_state.role == "staff":
        st.sidebar.success("Role: Staff")
        choice = st.sidebar.radio("Pages", ["Staff", "Login"], index=0)
        if choice == "Login":
            st.session_state.page = "Login"
        else:
            st.session_state.page = "Staff"

    elif st.session_state.role == "student":
        st.sidebar.success("Role: Student")
        choice = st.sidebar.radio("Pages", ["Student", "Login"], index=0)
        if choice == "Login":
            st.session_state.page = "Login"
        else:
            st.session_state.page = "Student"

    else:
        st.sidebar.info("Not logged in")
        st.session_state.page = "Login"


def main():
    ensure_session()
    ensure_worksheets_and_headers()
    sidebar_nav()

    page = st.session_state.page

    if page == "Login":
        page_login()
    elif page == "Staff":
        if st.session_state.role != "staff":
            st.warning("You must login as staff.")
            st.session_state.page = "Login"
            st.rerun()
        page_staff()
    elif page == "Student":
        if st.session_state.role != "student":
            st.warning("You must login as student.")
            st.session_state.page = "Login"
            st.rerun()
        page_student()
    else:
        page_login()

if __name__ == "__main__":
    main()
