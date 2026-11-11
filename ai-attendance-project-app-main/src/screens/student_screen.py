

import streamlit as st

import time

from PIL import Image

import numpy as np



from src.ui.base_layout import (

    style_background_dashboard,

    style_base_layout,

)



from src.components.header import header_dashboard

from src.components.footer import footer_dashboard



from src.pipelines.face_pipeline import (

    predict_attendance,

    get_face_embeddings,

    train_classifier,

)



from src.pipelines.voice_pipeline import get_voice_embedding



from src.database.db import (

    get_all_students,

    create_student,

    get_student_subjects,

    get_student_attendance,

    unenroll_student_to_subject,

)



from src.components.dialog_enroll import enroll_dialog

from src.components.subject_card import subject_card





# =====================================================

# STUDENT DASHBOARD

# =====================================================



def student_dashboard():



    student_data = st.session_state.student_data

    student_id = student_data["student_id"]



    c1, c2 = st.columns(

        2,

        vertical_alignment="center",

        gap="xxlarge",

    )



    with c1:

        header_dashboard()



    with c2:

        st.subheader(f"Welcome, {student_data['name']}")



        if st.button(

            "Logout",

            type="secondary",

            key="loginbackbtn",

            shortcut="control+backspace",

        ):

            st.session_state["is_logged_in"] = False

            st.session_state.pop("student_data", None)

            st.session_state["student_portal_mode"] = None

            st.rerun()



    st.space()



    c1, c2 = st.columns(2)



    with c1:

        st.header("Your Enrolled Subjects")



    with c2:

        if st.button(

            "Enroll in Subject",

            type="primary",

            width="stretch",

            key="enroll_subject_btn",

        ):

            enroll_dialog()



    st.divider()



    with st.spinner("Loading your enrolled subjects..."):

        subjects = get_student_subjects(student_id) or []

        logs = get_student_attendance(student_id) or []



    stats_map = {}



    for log in logs:

        sid = log["subject_id"]



        if sid not in stats_map:

            stats_map[sid] = {

                "total": 0,

                "attended": 0,

            }



        stats_map[sid]["total"] += 1



        if log.get("is_present"):

            stats_map[sid]["attended"] += 1



    if not subjects:

        st.info(

            "You are not enrolled in any subjects yet. "

            "Click 'Enroll in Subject' to get started."

        )



    else:

        cols = st.columns(2)



        for i, sub_node in enumerate(subjects):



            sub = sub_node.get("subjects")



            if not sub:

                continue



            sid = sub["subject_id"]



            stats = stats_map.get(

                sid,

                {"total": 0, "attended": 0},

            )



            with cols[i % 2]:



                subject_card(

                    name=sub["name"],

                    code=sub["subject_code"],

                    section=sub["section"],

                    stats=[

                        ("📅", "Total", stats["total"]),

                        ("✅", "Attended", stats["attended"]),

                    ],

                    footer_callback=lambda subject_id=sid, subject_name=sub["name"]: (

                        unenroll_button(

                            student_id,

                            subject_id,

                            subject_name,

                        )

                    ),

                )



    footer_dashboard()





# =====================================================

# UNENROLL BUTTON

# =====================================================



def unenroll_button(student_id, subject_id, subject_name):



    if st.button(

        "Unenroll from this course",

        type="tertiary",

        width="stretch",

        icon=":material/delete_forever:",

        key=f"unenroll_{student_id}_{subject_id}",

    ):



        try:

            unenroll_student_to_subject(

                student_id,

                subject_id,

            )



            st.toast(

                f"Unenrolled from {subject_name} successfully!"

            )



            st.rerun()



        except Exception as e:

            st.error(

                f"Unable to unenroll from the subject: {e}"

            )





# =====================================================

# COMMON STUDENT PORTAL HEADER

# =====================================================



def student_portal_header():



    c1, c2 = st.columns(

        2,

        vertical_alignment="center",

        gap="xxlarge",

    )



    with c1:

        header_dashboard()



    with c2:

        if st.button(

            "Go back to Home",

            type="secondary",

            key="student_home_btn",

            shortcut="control+backspace",

        ):

            st.session_state["student_portal_mode"] = None

            st.session_state["login_type"] = None

            st.rerun()





# =====================================================

# STUDENT PORTAL HOME

# =====================================================



def student_portal_home():

    # Styling is intentionally scoped to the Student Portal selection screen.
    # Registration, Face ID login, database calls and session-state logic remain unchanged.
    st.markdown(
        """
        <style>
        .attendiq-portal-heading {
            text-align: center;
            margin: 2.4rem 0 0.35rem 0;
            color: #292b3b;
            font-size: clamp(2.2rem, 5vw, 3.6rem);
            line-height: 1.08;
            font-weight: 850;
            letter-spacing: -1.8px;
        }
        .attendiq-portal-subtitle {
            text-align: center;
            color: #55566b;
            font-size: 1.05rem;
            margin: 0 0 2.1rem 0;
        }

        /* Give both existing Streamlit containers a consistent card treatment. */
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-new_student_btn),
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-existing_student_btn) {
            background: rgba(255, 255, 255, 0.58);
            border: 1px solid rgba(91, 91, 130, 0.19);
            border-radius: 20px;
            padding: 1.15rem 1.15rem 1rem 1.15rem;
            min-height: 225px;
            box-shadow: 0 8px 24px rgba(48, 47, 91, 0.055);
            transition: transform 180ms ease, box-shadow 180ms ease,
                        border-color 180ms ease, background 180ms ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-new_student_btn):hover,
        div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-existing_student_btn):hover {
            transform: translateY(-4px);
            background: rgba(255, 255, 255, 0.78);
            border-color: rgba(91, 91, 235, 0.42);
            box-shadow: 0 14px 30px rgba(48, 47, 91, 0.11);
        }
        .st-key-new_student_btn button,
        .st-key-existing_student_btn button {
            min-height: 3rem;
            border-radius: 12px;
            font-weight: 650;
            letter-spacing: 0.1px;
            box-shadow: 0 4px 10px rgba(82, 91, 235, 0.16);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }
        .st-key-new_student_btn button:hover,
        .st-key-existing_student_btn button:hover {
            transform: translateY(-1px);
            box-shadow: 0 7px 16px rgba(82, 91, 235, 0.23);
        }
        .attendiq-card-kicker {
            color: #625cf0;
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 1.2px;
            text-transform: uppercase;
            margin: 0 0 0.45rem 0;
        }
        @media (max-width: 700px) {
            .attendiq-portal-heading {
                margin-top: 1.5rem;
                letter-spacing: -0.8px;
            }
            .attendiq-portal-subtitle {
                margin-bottom: 1.25rem;
            }
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-new_student_btn),
            div[data-testid="stVerticalBlockBorderWrapper"]:has(.st-key-existing_student_btn) {
                min-height: 0;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    student_portal_header()

    st.markdown(
        '<h1 class="attendiq-portal-heading">Welcome to Student Portal</h1>'
        '<p class="attendiq-portal-subtitle">Choose how you would like to continue.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2, gap="large")

    with col1:
        with st.container(border=True):
            st.markdown(
                '<div class="attendiq-card-kicker">Get started</div>',
                unsafe_allow_html=True,
            )
            st.subheader("New Student")
            st.write(
                "Create your student profile and register your face "
                "to get started with AttendIQ."
            )

            if st.button(
                "Register as New Student",
                type="primary",
                width="stretch",
                key="new_student_btn",
                icon=":material/person_add:",
            ):
                st.session_state["student_portal_mode"] = "register"
                st.rerun()

    with col2:
        with st.container(border=True):
            st.markdown(
                '<div class="attendiq-card-kicker">Welcome back</div>',
                unsafe_allow_html=True,
            )
            st.subheader("Existing Student")
            st.write(
                "Already registered? Sign in securely using "
                "your Face ID."
            )

            if st.button(
                "Login Using Face ID",
                type="primary",
                width="stretch",
                key="existing_student_btn",
                icon=":material/face:",
            ):
                st.session_state["student_portal_mode"] = "login"
                st.rerun()

    footer_dashboard()


# =====================================================

# EXISTING STUDENT LOGIN

# =====================================================



def student_login():



    student_portal_header()



    st.header(

        "Login using FaceID",

        text_alignment="center",

    )



    st.info(

        "Position your face clearly in the center of the camera."

    )



    photo_source = st.camera_input(

        "Capture your face",

        key="student_login_camera",

    )



    if photo_source:



        img = np.array(

            Image.open(photo_source).convert("RGB")

        )



        with st.spinner("AI is scanning..."):



            detected, all_ids, num_faces = predict_attendance(img)



            if num_faces == 0:



                st.warning("Face not found! Please try again.")



            elif num_faces > 1:



                st.warning(

                    "Multiple faces found. "

                    "Please capture only your face."

                )



            elif detected:



                student_id = list(detected.keys())[0]



                all_students = get_all_students() or []



                student = next(

                    (

                        s for s in all_students

                        if str(s["student_id"]) == str(student_id)

                    ),

                    None,

                )



                if student:



                    st.session_state.is_logged_in = True

                    st.session_state.user_role = "student"

                    st.session_state.student_data = student

                    st.session_state["student_portal_mode"] = None



                    st.toast(

                        f"Welcome back {student['name']}!"

                    )



                    time.sleep(1)

                    st.rerun()



                else:

                    st.error(

                        "Student record could not be found."

                    )



            else:



                st.error(

                    "Face not recognized. "

                    "Please try again or register if you are a new student."

                )



    st.divider()



    if st.button(

        "Back to Student Portal",

        key="login_mode_back",

    ):

        st.session_state["student_portal_mode"] = None

        st.rerun()



    footer_dashboard()





# =====================================================

# NEW STUDENT REGISTRATION

# =====================================================



def student_registration():

    student_portal_header()

    st.header(
        "Register New Student",
        text_alignment="center",
    )

    st.info(
        "Enter your name and capture one clear face photograph. "
        "Only one person should appear in the photograph."
    )

    new_name = st.text_input(
        "Enter your full name",
        placeholder="Enter your name",
        key="new_student_name",
    )

    st.subheader("Face Registration")

    photo_source = st.camera_input(
        "Capture your face",
        key="registration_camera",
    )

    st.caption(
        "Use a clear, front-facing photo with good lighting."
    )

    st.divider()

    st.subheader("Optional: Voice Enrollment")

    st.info(
        "You can optionally enroll your voice for "
        "voice-only attendance."
    )

    audio_data = None

    try:
        audio_data = st.audio_input(
            "Record a short phrase such as 'I am present'.",
            key="registration_audio",
        )
    except Exception:
        st.warning("Voice enrollment is unavailable.")

    if st.button(
        "Create Student Account",
        type="primary",
        width="stretch",
        key="create_student_account",
    ):

        if not new_name or not new_name.strip():
            st.warning("Please enter your name.")

        elif photo_source is None:
            st.warning("Please capture your face photo.")

        else:
            with st.spinner("Processing face photo..."):
                img = np.array(
                    Image.open(photo_source).convert("RGB")
                )

                encodings = get_face_embeddings(img)

            if len(encodings) != 1:
                st.error(
                    "Exactly one face must be visible. "
                    "Please capture another photo with only your face."
                )

            else:
                face_embedding = encodings[0].tolist()

                voice_emb = None

                if audio_data:
                    try:
                        voice_emb = get_voice_embedding(
                            audio_data.read()
                        )
                    except Exception as e:
                        st.warning(
                            f"Voice enrollment skipped: {e}"
                        )

                with st.spinner("Creating student profile..."):
                    try:
                        response_data = create_student(
                            new_name.strip(),
                            face_embedding=face_embedding,
                            voice_embedding=voice_emb,
                        )

                        if response_data:
                            train_classifier()

                            student = response_data[0]

                            st.session_state.is_logged_in = True
                            st.session_state.user_role = "student"
                            st.session_state.student_data = student
                            st.session_state["student_portal_mode"] = None

                            st.success(
                                "Student registration successful!"
                            )

                            time.sleep(1)
                            st.rerun()

                        else:
                            st.error(
                                "Registration failed. Please try again."
                            )

                    except Exception as e:
                        st.error(
                            f"Unable to create student profile: {e}"
                        )

    st.divider()

    if st.button(
        "Back to Student Portal",
        key="registration_mode_back",
    ):
        st.session_state["student_portal_mode"] = None
        st.rerun()

    footer_dashboard()


# =====================================================



def student_screen():



    style_background_dashboard()

    style_base_layout()



    # Existing logged-in students go directly to dashboard.

    if "student_data" in st.session_state:



        student_dashboard()

        return



    # Default portal selection.

    if "student_portal_mode" not in st.session_state:



        st.session_state["student_portal_mode"] = None



    mode = st.session_state["student_portal_mode"]



    if mode is None:



        student_portal_home()



    elif mode == "login":



        student_login()



    elif mode == "register":



        student_registration()



    else:



        st.session_state["student_portal_mode"] = None

        st.rerun()
