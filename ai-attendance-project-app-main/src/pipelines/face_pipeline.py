import dlib
import cv2
import numpy as np
import face_recognition_models
import streamlit as st
from PIL import Image

from src.database.db import get_all_students

FACE_DISTANCE_THRESHOLD = 0.60
AMBIGUITY_MARGIN = 0.05
EMBEDDING_SIZE = 128

# Dlib's HOG detector is strongest on near-frontal faces. OpenCV's profile
# cascades provide a second detection route for moderate left/right turns.
DETECTION_ROTATIONS = (0, -15, 15, -30, 30)
MIN_FACE_SIZE = (60, 60)


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


@st.cache_resource
def load_opencv_cascades():
    """Load OpenCV's bundled frontal and profile Haar cascades."""
    if not hasattr(cv2, "CascadeClassifier") or not hasattr(cv2, "data"):
        raise RuntimeError(
            "Your cv2 import is not a full OpenCV installation. "
            "In the active venv, uninstall all opencv-* packages and install "
            "opencv-python-headless==4.11.0.86."
        )

    base = cv2.data.haarcascades
    frontal = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
    profile = cv2.CascadeClassifier(base + "haarcascade_profileface.xml")
    if frontal.empty() or profile.empty():
        raise RuntimeError("OpenCV face-detection cascade files could not be loaded.")
    return frontal, profile


def _rect_iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def _deduplicate_dlib_rectangles(rectangles, overlap_threshold=0.55):
    """Remove highly overlapping detections from dlib's multi-scale scan."""
    boxes = []
    for rect in rectangles:
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        boxes.append((x1, y1, x2, y2))

    boxes.sort(key=lambda b: max(0, b[2] - b[0]) * max(0, b[3] - b[1]),
               reverse=True)
    kept = []
    for box in boxes:
        x1, y1, x2, y2 = box
        area = max(0, x2 - x1) * max(0, y2 - y1)
        duplicate = False
        for old in kept:
            ox1, oy1, ox2, oy2 = old
            ix1, iy1 = max(x1, ox1), max(y1, oy1)
            ix2, iy2 = min(x2, ox2), min(y2, oy2)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            old_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
            smaller_area = min(area, old_area)
            if smaller_area and intersection / smaller_area >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(box)

    return [dlib.rectangle(x1, y1, x2, y2) for x1, y1, x2, y2 in kept]


def _opencv_face_boxes(image_rgb):
    """
    Detect frontal/profile faces and suppress duplicate boxes from the
    different cascades. Haar cascades can report the same face at slightly
    different sizes/positions, so IoU alone is not sufficient for deduping.
    """
    frontal, profile = load_opencv_cascades()
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    detections = []
    detector_inputs = (
        (frontal, gray, False),
        (profile, gray, False),
        (profile, cv2.flip(gray, 1), True),
    )

    for cascade, source, flipped in detector_inputs:
        found = cascade.detectMultiScale(
            source,
            scaleFactor=1.10,
            minNeighbors=6,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        for x, y, w, h in found:
            if flipped:
                x = image_rgb.shape[1] - x - w

            # Ignore boxes outside the image or implausibly small boxes.
            x, y, w, h = map(int, (x, y, w, h))
            if w < 1 or h < 1:
                continue
            detections.append((x, y, w, h))

    # Non-maximum suppression. Treat boxes as duplicates when most of the
    # smaller box overlaps the larger one, even if their IoU is modest.
    detections.sort(key=lambda box: box[2] * box[3], reverse=True)
    kept = []

    for box in detections:
        x, y, w, h = box
        duplicate = False

        for old in kept:
            ox, oy, ow, oh = old
            ix1, iy1 = max(x, ox), max(y, oy)
            ix2, iy2 = min(x + w, ox + ow), min(y + h, oy + oh)
            intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)

            smaller_area = min(w * h, ow * oh)
            if smaller_area and intersection / smaller_area >= 0.55:
                duplicate = True
                break

            # Also merge near-identical centers when cascade box sizes vary.
            center_x, center_y = x + w / 2, y + h / 2
            old_center_x, old_center_y = ox + ow / 2, oy + oh / 2
            center_distance = np.hypot(
                center_x - old_center_x, center_y - old_center_y
            )
            smaller_side = min(w, h, ow, oh)
            if smaller_side and center_distance < 0.20 * smaller_side:
                size_ratio = min(w * h, ow * oh) / max(w * h, ow * oh)
                if size_ratio >= 0.55:
                    duplicate = True
                    break

        if not duplicate:
            kept.append(box)

    return kept


def _dlib_rect_from_box(box):
    x, y, w, h = box
    return dlib.rectangle(x, y, x + w, y + h)


def _detect_candidates(image_np):
    """Return one image coordinate system and face rectangles for that image."""
    image = np.asarray(image_np, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three color channels.")

    detector, _, _ = load_dlib_models()

    # Try the original first, then modest in-plane rotations for head tilt.
    pil_image = Image.fromarray(image).convert("RGB")
    for angle in DETECTION_ROTATIONS:
        candidate = np.asarray(
            pil_image if angle == 0 else pil_image.rotate(
                angle, resample=Image.Resampling.BICUBIC, expand=True
            )
        )
        for upsample in (1, 2):
            rectangles = detector(candidate, upsample)
            if rectangles:
                rectangles = _deduplicate_dlib_rectangles(rectangles)
                return candidate, rectangles

    # Fallback for profile / low-contrast faces. The returned boxes are used
    # with the same dlib landmark and descriptor model as the main pipeline.
    boxes = _opencv_face_boxes(image)
    if boxes:
        return image, [_dlib_rect_from_box(box) for box in boxes]

    return image, []


def get_face_embeddings(image_np):
    """Return one 128D descriptor for each detected face in an RGB image."""
    _, shape_predictor, face_recognition_model = load_dlib_models()
    image_for_detection, faces = _detect_candidates(image_np)
    encodings = []

    for face in faces:
        try:
            shape = shape_predictor(image_for_detection, face)
            descriptor = face_recognition_model.compute_face_descriptor(
                image_for_detection, shape, 1
            )
            encoding = np.asarray(descriptor, dtype=np.float64)
            if encoding.shape == (EMBEDDING_SIZE,) and np.all(np.isfinite(encoding)):
                encodings.append(encoding)
        except (RuntimeError, ValueError):
            # A weak profile box may not yield usable 68-point landmarks.
            continue

    return encodings


def normalize_embeddings(embedding_data):
    """Accept legacy [128] data and new [[128], ...] multi-sample data."""
    if embedding_data is None:
        return []
    try:
        array = np.asarray(embedding_data, dtype=np.float64)
        if array.ndim == 1:
            candidates = [array]
        elif array.ndim == 2:
            candidates = list(array)
        else:
            return []

        return [
            np.asarray(candidate, dtype=np.float64)
            for candidate in candidates
            if np.asarray(candidate).shape == (EMBEDDING_SIZE,)
            and np.all(np.isfinite(candidate))
        ]
    except (TypeError, ValueError):
        return []


def normalize_embedding(embedding):
    """Backward-compatible helper returning the first valid descriptor."""
    values = normalize_embeddings(embedding)
    return values[0] if values else None


@st.cache_resource
def get_trained_model():
    students = get_all_students() or []
    embeddings, student_ids = [], []

    for student in students:
        student_id = student.get("student_id")
        if student_id is None:
            continue
        for embedding in normalize_embeddings(student.get("face_embedding")):
            embeddings.append(embedding)
            student_ids.append(student_id)

    if not embeddings:
        return None
    return {"X": embeddings, "y": student_ids}


def train_classifier():
    get_trained_model.clear()
    return get_trained_model() is not None


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
        # Find the best sample distance for each distinct student.
        best_per_student = {}
        for known_embedding, student_id in zip(known_embeddings, known_student_ids):
            distance = float(np.linalg.norm(known_embedding - face_encoding))
            if student_id not in best_per_student or distance < best_per_student[student_id]:
                best_per_student[student_id] = distance

        ranked = sorted((distance, sid) for sid, distance in best_per_student.items())
        if not ranked:
            continue

        best_distance, best_student_id = ranked[0]
        if best_distance > FACE_DISTANCE_THRESHOLD:
            continue
        if len(ranked) > 1 and ranked[1][0] - best_distance < AMBIGUITY_MARGIN:
            continue

        detected_student[best_student_id] = True

    return detected_student, all_students, len(encodings)
