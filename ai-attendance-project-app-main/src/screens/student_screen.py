import streamlit as st
import time
from PIL import Image
import numpy as np

from src.ui.base_layout import style_background_dashboard, style_base_layout
from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from src.pipelines.face_pipeline import predict_attendance, get_face_embeddings, train_classifier
from src.pipelines.voice_pipeline import get_voice_embedding
from src.database.db import (
    get_all_students, create_student, get_student_subjects,
    get_student_attendance, unenroll_student_to_subject,
)
from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data["student_id"]

    c1, c2 = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        st.subheader(f"Welcome, {student_data['name']}")
        if st.button("Logout", type="secondary", key="loginbackbtn",
                     shortcut="control+backspace"):
            st.session_state["is_logged_in"] = False
            st.session_state.pop("student_data", None)
            st.session_state["student_portal_mode"] = None
            st.rerun()

    st.space()
    c1, c2 = st.columns(2)
    with c1:
        st.header("Your Enrolled Subjects")
    with c2:
        if st.button("Enroll in Subject", type="primary", width="stretch",
                     key="enroll_subject_btn"):
            enroll_dialog()

    st.divider()
    with st.spinner("Loading your enrolled subjects..."):
        subjects = get_student_subjects(student_id) or []
        logs = get_student_attendance(student_id) or []

    stats_map = {}
    for log in logs:
        sid = log["subject_id"]
        stats_map.setdefault(sid, {"total": 0, "attended": 0})
        stats_map[sid]["total"] += 1
        if log.get("is_present"):
            stats_map[sid]["attended"] += 1

    if not subjects:
        st.info("You are not enrolled in any subjects yet. Click 'Enroll in Subject' to get started.")
    else:
        cols = st.columns(2)
        for i, sub_node in enumerate(subjects):
            sub = sub_node.get("subjects")
            if not sub:
                continue
            sid = sub["subject_id"]
            stats = stats_map.get(sid, {"total": 0, "attended": 0})
            with cols[i % 2]:
                subject_card(
                    name=sub["name"], code=sub["subject_code"], section=sub["section"],
                    stats=[("📅", "Total", stats["total"]), ("✅", "Attended", stats["attended"])],
                    footer_callback=lambda subject_id=sid, subject_name=sub["name"]: (
                        unenroll_button(student_id, subject_id, subject_name)
                    ),
                )
    footer_dashboard()


def unenroll_button(student_id, subject_id, subject_name):
    if st.button("Unenroll from this course", type="tertiary", width="stretch",
                 icon=":material/delete_forever:",
                 key=f"unenroll_{student_id}_{subject_id}"):
        try:
            unenroll_student_to_subject(student_id, subject_id)
            st.toast(f"Unenrolled from {subject_name} successfully!")
            st.rerun()
        except Exception as e:
            st.error(f"Unable to unenroll from the subject: {e}")


def student_portal_header():
    c1, c2 = st.columns(2, vertical_alignment="center", gap="xxlarge")
    with c1:
        header_dashboard()
    with c2:
        if st.button("Go back to Home", type="secondary", key="student_home_btn",
                     shortcut="control+backspace"):
            st.session_state["student_portal_mode"] = None
            st.session_state["login_type"] = None
            st.rerun()


def student_portal_home():
    student_portal_header()
    st.title("Welcome to Student Portal", text_alignment="center")
    st.markdown("<p style='text-align: center;'>Choose an option to continue.</p>",
                unsafe_allow_html=True)
    st.space()
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.subheader("New Student")
            st.write("Create your student profile and register your facial samples.")
            if st.button("Register as New Student", type="primary", width="stretch",
                         key="new_student_btn"):
                st.session_state["student_portal_mode"] = "register"
                st.rerun()
    with col2:
        with st.container(border=True):
            st.subheader("Existing Student")
            st.write("Already registered? Login using Face ID.")
            if st.button("Login Using Face ID", type="primary", width="stretch",
                         key="existing_student_btn"):
                st.session_state["student_portal_mode"] = "login"
                st.rerun()
    footer_dashboard()


def student_login():
    student_portal_header()
    st.header("Login using FaceID", text_alignment="center")
    st.info("Position your face clearly in the center of the camera.")
    photo_source = st.camera_input("Capture your face", key="student_login_camera")

    if photo_source:
        img = np.asarray(Image.open(photo_source).convert("RGB"))
        with st.spinner("AI is scanning..."):
            detected, all_ids, num_faces = predict_attendance(img)

        if num_faces == 0:
            st.warning("Face not found! Face the camera and try again.")
        elif num_faces > 1:
            st.warning("Multiple faces found. Please capture only your face.")
        elif detected:
            student_id = list(detected.keys())[0]
            all_students = get_all_students() or []
            student = next((s for s in all_students
                            if str(s["student_id"]) == str(student_id)), None)
            if student:
                st.session_state.is_logged_in = True
                st.session_state.user_role = "student"
                st.session_state.student_data = student
                st.session_state["student_portal_mode"] = None
                st.toast(f"Welcome back {student['name']}!")
                time.sleep(1)
                st.rerun()
            else:
                st.error("Student record could not be found.")
        else:
            st.error("Face not recognized. Please try again or register if you are a new student.")

    st.divider()
    if st.button("Back to Student Portal", key="login_mode_back"):
        st.session_state["student_portal_mode"] = None
        st.rerun()
    footer_dashboard()


def student_registration():
    student_portal_header()
    st.header("Register New Student", text_alignment="center")
    st.info(
        "Enter your details and capture five clear face samples. "
        "Only one person should appear in each photograph. "
        "For reliable recognition, turn only slightly (about 15–30°), not a full profile."
    )

    new_name = st.text_input("Enter your full name", placeholder="Enter your name",
                             key="new_student_name")
    st.subheader("Face Registration")

    photo_sources = []
    for i in range(5):
        photo = st.camera_input(f"Face Sample {i + 1}",
                                key=f"registration_camera_{i + 1}")
        photo_sources.append(photo)

    st.caption(
        "Suggested samples: front, slight left, slight right, slight upward tilt, "
        "and slight downward tilt. Keep your face large, well-lit, and unobstructed."
    )
    st.divider()
    st.subheader("Optional: Voice Enrollment")
    st.info("You can optionally enroll your voice for voice-only attendance.")

    audio_data = None
    try:
        audio_data = st.audio_input("Record a short phrase such as 'I am present'.",
                                    key="registration_audio")
    except Exception:
        st.warning("Voice enrollment is unavailable.")

    if st.button("Create Student Account", type="primary", width="stretch",
                 key="create_student_account"):
        if not new_name or not new_name.strip():
            st.warning("Please enter your name.")
        elif not all(photo_sources):
            missing = [str(i + 1) for i, photo in enumerate(photo_sources) if photo is None]
            st.warning("Please capture all five face samples. Missing: " + ", ".join(missing))
        else:
            face_embeddings = []
            errors = []

            with st.spinner("Checking and processing all five face samples..."):
                for index, photo in enumerate(photo_sources, start=1):
                    try:
                        photo.seek(0)
                        img = np.asarray(Image.open(photo).convert("RGB"))
                        encodings = get_face_embeddings(img)

                        if len(encodings) != 1:
                            if len(encodings) == 0:
                                reason = (
                                    "No face could be detected. Face the camera more directly, "
                                    "improve lighting, and avoid a full side profile."
                                )
                            else:
                                reason = (
                                    f"{len(encodings)} faces were detected. Make sure only "
                                    "one person is visible."
                                )
                            errors.append(f"Sample {index}: {reason}")
                        else:
                            face_embeddings.append(encodings[0].tolist())
                    except Exception as exc:
                        errors.append(f"Sample {index}: image processing failed ({exc}).")

            if errors:
                for message in errors:
                    st.error(message)
                st.info("Retake only the failed sample(s), then click Create Student Account again.")
            elif len(face_embeddings) == 5:
                voice_emb = None
                if audio_data:
                    try:
                        audio_data.seek(0)
                        voice_emb = get_voice_embedding(audio_data.read())
                    except Exception as exc:
                        st.warning(f"Voice enrollment skipped: {exc}")

                try:
                    with st.spinner("Creating student profile..."):
                        response_data = create_student(
                            new_name.strip(),
                            face_embedding=face_embeddings,
                            voice_embedding=voice_emb,
                        )
                    if response_data:
                        train_classifier()
                        student = response_data[0]
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = "student"
                        st.session_state.student_data = student
                        st.session_state["student_portal_mode"] = None
                        st.success("Student registration successful! Five face samples were saved.")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error("Registration failed. Please try again.")
                except Exception as exc:
                    st.error(f"Unable to create student profile: {exc}")

    st.divider()
    if st.button("Back to Student Portal", key="registration_mode_back"):
        st.session_state["student_portal_mode"] = None
        st.rerun()
    footer_dashboard()


def student_screen():
    style_background_dashboard()
    style_base_layout()

    if "student_data" in st.session_state:
        student_dashboard()
        return

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
