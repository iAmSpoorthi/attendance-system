
import dlib
import numpy as np
import face_recognition_models
import streamlit as st

from src.database.db import get_all_students


# Recognition settings
FACE_DISTANCE_THRESHOLD = 0.60
AMBIGUITY_MARGIN = 0.05
EMBEDDING_SIZE = 128


@st.cache_resource
def load_dlib_models():
    detector = dlib.get_frontal_face_detector()

    shape_predictor = dlib.shape_predictor(
        face_recognition_models.pose_predictor_model_location()
    )

    face_recognition_model = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )

    return detector, shape_predictor, face_recognition_model


def get_face_embeddings(image_np):
    detector, shape_predictor, face_recognition_model = load_dlib_models()

    image_np = np.asarray(image_np)

    faces = detector(image_np, 1)
    encodings = []

    for face in faces:
        shape = shape_predictor(image_np, face)

        descriptor = face_recognition_model.compute_face_descriptor(
            image_np,
            shape,
            1
        )

        encoding = np.asarray(descriptor, dtype=np.float64)

        if encoding.shape == (EMBEDDING_SIZE,):
            encodings.append(encoding)

    return encodings


def normalize_embedding(embedding):
    """Convert a stored JSON embedding into a valid 128-value array."""

    try:
        array = np.asarray(embedding, dtype=np.float64)

        # Support a single embedding stored as [[...]]
        if array.ndim == 2 and array.shape[0] >= 1:
            array = array[0]

        if array.shape != (EMBEDDING_SIZE,):
            return None

        if not np.all(np.isfinite(array)):
            return None

        return array

    except (TypeError, ValueError):
        return None


@st.cache_resource
def get_trained_model():
    students = get_all_students()

    if not students:
        return None

    embeddings = []
    student_ids = []

    for student in students:
        student_id = student.get("student_id")
        stored_embedding = student.get("face_embedding")

        if student_id is None or stored_embedding is None:
            continue

        embedding = normalize_embedding(stored_embedding)

        if embedding is None:
            continue

        embeddings.append(embedding)
        student_ids.append(student_id)

    if not embeddings:
        return None

    return {
        "X": embeddings,
        "y": student_ids
    }


def train_classifier():
    """Refresh the stored student embeddings after registration."""

    get_trained_model.clear()

    model_data = get_trained_model()

    return model_data is not None


def predict_attendance(class_image_np):
    encodings = get_face_embeddings(class_image_np)

    detected_student = {}

    model_data = get_trained_model()

    if model_data is None:
        return detected_student, [], len(encodings)

    known_embeddings = model_data["X"]
    known_student_ids = model_data["y"]

    all_students = list(dict.fromkeys(known_student_ids))

    for face_encoding in encodings:

        distances = []

        # Compare this face against every registered student
        for known_embedding, student_id in zip(
            known_embeddings,
            known_student_ids
        ):
            distance = np.linalg.norm(
                known_embedding - face_encoding
            )

            distances.append((distance, student_id))

        if not distances:
            continue

        # Sort by smallest face distance
        distances.sort(key=lambda item: item[0])

        best_distance, best_student_id = distances[0]

        # Reject faces that are too different from registered samples
        if best_distance > FACE_DISTANCE_THRESHOLD:
            continue

        # Reject ambiguous matches when two students look too similar
        if len(distances) > 1:
            second_best_distance = distances[1][0]

            if second_best_distance - best_distance < AMBIGUITY_MARGIN:
                continue

        detected_student[best_student_id] = True

    return detected_student, all_students, len(encodings)
