
import streamlit as st

from src.ui.base_layout import (
    style_background_dashboard,
    style_base_layout,
)

from src.components.header import header_dashboard
from src.components.footer import footer_dashboard
from PIL import Image
import numpy as np

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

import time

from src.components.dialog_enroll import enroll_dialog
from src.components.subject_card import subject_card


def student_dashboard():
    student_data = st.session_state.student_data
    student_id = student_data["student_id"]

    # Header and logout
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
            st.rerun()

    st.space()

    # Enrolled subjects heading and enrollment button
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

    # Fetch enrolled subjects and attendance records
    with st.spinner("Loading your enrolled subjects..."):
        subjects = get_student_subjects(student_id) or []
        logs = get_student_attendance(student_id) or []

    # Calculate attendance statistics for each subject
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

    # Display enrolled subject cards
    if not subjects:
        st.info(
            "You are not enrolled in any subjects yet. "
            "Click 'Enroll in Subject' to get started."
        )
    else:
        cols = st.columns(2)

        for i, sub_node in enumerate(subjects):
            # Extract the subject information
            sub = sub_node.get("subjects")

            if not sub:
                continue

            sid = sub["subject_id"]

            stats = stats_map.get(
                sid,
                {"total": 0, "attended": 0},
            )

            # Display each subject inside its own column
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


def unenroll_button(student_id, subject_id, subject_name):
    """
    Display a uniquely identified unenroll button
    for the selected subject.
    """

    if st.button(
        "Unenroll from this course",
        type="tertiary",
        width="stretch",
        icon=":material/delete_forever:",
        key=f"unenroll_{student_id}_{subject_id}",
    ):
        try:
            unenroll_student_to_subject(student_id, subject_id)

            st.toast(
                f"Unenrolled from {subject_name} successfully!"
            )

            st.rerun()

        except Exception as e:
            st.error(f"Unable to unenroll from the subject: {e}")


def student_screen():

    style_background_dashboard()
    style_base_layout()

    # Show dashboard for logged-in students
    if "student_data" in st.session_state:
        student_dashboard()
        return

    # Login page header
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
            key="loginbackbtn",
            shortcut="control+backspace",
        ):
            st.session_state["login_type"] = None
            st.rerun()

    st.header(
        "Login using FaceID",
        text_alignment="center",
    )

    st.space()
    st.space()

    show_registration = False

    photo_source = st.camera_input(
        "Position your face in the center"
    )

    # Face recognition login
    if photo_source:
        img = np.array(Image.open(photo_source))

        with st.spinner("AI is scanning..."):
            detected, all_ids, num_faces = predict_attendance(img)

            if num_faces == 0:
                st.warning("Face not found!")

            elif num_faces > 1:
                st.warning("Multiple faces found")

            else:
                if detected:
                    student_id = list(detected.keys())[0]

                    all_students = get_all_students()

                    student = next(
                        (
                            s
                            for s in all_students
                            if s["student_id"] == student_id
                        ),
                        None,
                    )

                    if student:
                        st.session_state.is_logged_in = True
                        st.session_state.user_role = "student"
                        st.session_state.student_data = student

                        st.toast(
                            f"Welcome back {student['name']}"
                        )

                        time.sleep(1)
                        st.rerun()

                else:
                    st.info(
                        "Face not recognized! "
                        "You might be a new student."
                    )

                    show_registration = True

    # New student registration
    if show_registration:
        with st.container(border=True):

            st.header("Register new Profile")

            new_name = st.text_input(
                "Enter your name",
                placeholder="E.g. Hamza Rizvi",
            )

            st.subheader("Optional: Voice Enrollment")

            st.info(
                "Enroll your voice for voice-only attendance."
            )

            audio_data = None

            try:
                audio_data = st.audio_input(
                    "Record a short phrase like "
                    "'I am present' or 'My name is Akash'."
                )
            except Exception:
                st.error("Audio data failed!")

            if st.button(
                "Create Account",
                type="primary",
                key="create_student_account",
            ):

                if new_name and new_name.strip():

                    with st.spinner("Creating profile..."):

                        img = np.array(Image.open(photo_source))

                        encodings = get_face_embeddings(img)

                        if encodings:
                            face_emb = encodings[0].tolist()

                            voice_emb = None

                            if audio_data:
                                voice_emb = get_voice_embedding(
                                    audio_data.read()
                                )

                            response_data = create_student(
                                new_name.strip(),
                                face_embedding=face_emb,
                                voice_embedding=voice_emb,
                            )

                            if response_data:
                                train_classifier()

                                st.session_state.is_logged_in = True
                                st.session_state.user_role = "student"
                                st.session_state.student_data = (
                                    response_data[0]
                                )

                                st.toast(
                                    f"Profile created! Hi {new_name}!"
                                )

                                time.sleep(1)
                                st.rerun()

                        else:
                            st.error(
                                "Couldn't capture your facial "
                                "features for registration."
                            )

                else:
                    st.warning("Please enter your name!")

    footer_dashboard()