"""
face_landmarks.py
------------------
שכבת זיהוי הפנים של OptiFit AI.
משתמש בספריית MediaPipe (Google) לזיהוי 478 נקודות ציון (landmarks) על הפנים,
ומחשב מתוכן מדדים גיאומטריים בסיסיים הדרושים לסיווג צורת הפנים.

זהו בדיוק הרכיב "שימוש בספרייה קיימת לזיהוי פנים" שתואר בטופס הפרויקט.
"""

import math
import os
from dataclasses import dataclass

import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python import BaseOptions

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "assets", "face_landmarker.task")

# אינדקסים של נקודות ציון רלוונטיות מתוך 478 הנקודות של MediaPipe Face Mesh
# (המספרים האלה קבועים וידועים מתיעוד MediaPipe)
LANDMARK_INDICES = {
    "forehead_left": 54,
    "forehead_right": 284,
    "cheekbone_left": 234,
    "cheekbone_right": 454,
    "jaw_left": 172,
    "jaw_right": 397,
    "chin": 152,
    "top_forehead": 10,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
}


@dataclass
class FaceMeasurements:
    """מדדים גיאומטריים שחולצו מהפנים, ביחידות פיקסלים יחסיים לתמונה."""
    forehead_width: float
    cheekbone_width: float
    jaw_width: float
    face_length: float
    eye_distance: float

    def as_dict(self):
        return {
            "forehead_width": round(self.forehead_width, 4),
            "cheekbone_width": round(self.cheekbone_width, 4),
            "jaw_width": round(self.jaw_width, 4),
            "face_length": round(self.face_length, 4),
            "eye_distance": round(self.eye_distance, 4),
        }


def _dist(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


class FaceLandmarkDetector:
    """עטיפה נוחה סביב MediaPipe FaceLandmarker."""

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"לא נמצא קובץ המודל ב-{model_path}.\n"
                "יש להריץ קודם: bash download_model.sh"
            )
        base_options = BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        self.detector = mp_vision.FaceLandmarker.create_from_options(options)

    def detect(self, image_path: str):
        """
        מזהה פנים בתמונה ומחזיר את כל 478 נקודות הציון (landmarks).
        מחזיר None אם לא זוהו פנים בתמונה.
        """
        mp_image = mp.Image.create_from_file(image_path)
        result = self.detector.detect(mp_image)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]  # הפנים הראשונות שזוהו

    def extract_measurements(self, landmarks) -> FaceMeasurements:
        """ממיר רשימת landmarks גולמית למדדים גיאומטריים שימושיים."""
        li = LANDMARK_INDICES
        return FaceMeasurements(
            forehead_width=_dist(landmarks[li["forehead_left"]], landmarks[li["forehead_right"]]),
            cheekbone_width=_dist(landmarks[li["cheekbone_left"]], landmarks[li["cheekbone_right"]]),
            jaw_width=_dist(landmarks[li["jaw_left"]], landmarks[li["jaw_right"]]),
            face_length=_dist(landmarks[li["top_forehead"]], landmarks[li["chin"]]),
            eye_distance=_dist(landmarks[li["left_eye_outer"]], landmarks[li["right_eye_outer"]]),
        )


def draw_landmarks_debug(image_path: str, landmarks, output_path: str):
    """שומר תמונת דיבוג עם הנקודות המרכזיות מסומנות - שימושי לבדיקות (יום 1-2)."""
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    for name, idx in LANDMARK_INDICES.items():
        pt = landmarks[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        cv2.circle(img, (x, y), 4, (0, 215, 255), -1)
        cv2.putText(img, name, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 215, 255), 1)
    cv2.imwrite(output_path, img)


if __name__ == "__main__":
    # בדיקה ידנית מהירה
    detector = FaceLandmarkDetector()
    landmarks = detector.detect("data/test_face.jpg")
    if landmarks is None:
        print("לא זוהו פנים בתמונה")
    else:
        measurements = detector.extract_measurements(landmarks)
        print("מדדים שחולצו:", measurements.as_dict())
        draw_landmarks_debug("data/test_face.jpg", landmarks, "data/debug_output.jpg")
        print("תמונת דיבוג נשמרה ב-data/debug_output.jpg")
