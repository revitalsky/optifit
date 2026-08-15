"""
בדיקות יחידה למסווג צורת הפנים (face_shape_classifier).
מריצים מתיקיית השורש:  .venv\\Scripts\\python.exe -m unittest discover -s tests
"""

import os
import sys
import unittest

# מאפשר ייבוא של מודולי src בלי התקנה כחבילה
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from face_landmarks import FaceMeasurements          # noqa: E402
from face_shape_classifier import classify_face_shape, FACE_SHAPES  # noqa: E402


# מדדים סינתטיים אופייניים לכל צורה (יחסים, לא מ״מ אמיתיים)
SYNTHETIC = {
    "oval":    FaceMeasurements(forehead_width=0.9,  cheekbone_width=1.0, jaw_width=0.85, face_length=1.6, eye_distance=0.4),
    "round":   FaceMeasurements(forehead_width=0.95, cheekbone_width=1.0, jaw_width=0.95, face_length=1.1, eye_distance=0.4),
    "square":  FaceMeasurements(forehead_width=0.95, cheekbone_width=1.0, jaw_width=0.95, face_length=1.3, eye_distance=0.4),
    "heart":   FaceMeasurements(forehead_width=1.0,  cheekbone_width=0.95, jaw_width=0.7, face_length=1.3, eye_distance=0.4),
    "diamond": FaceMeasurements(forehead_width=0.8,  cheekbone_width=1.0, jaw_width=0.8, face_length=1.3, eye_distance=0.4),
}


class TestFaceShapeClassifier(unittest.TestCase):
    def test_each_synthetic_shape_classified_correctly(self):
        for expected, measurements in SYNTHETIC.items():
            with self.subTest(shape=expected):
                self.assertEqual(classify_face_shape(measurements), expected)

    def test_result_is_always_a_known_shape(self):
        # גם על קלט "ביניים" מוזר, התוצאה חייבת להיות אחת מהצורות המוגדרות
        odd = FaceMeasurements(forehead_width=0.9, cheekbone_width=1.0, jaw_width=0.9, face_length=1.3, eye_distance=0.4)
        self.assertIn(classify_face_shape(odd), FACE_SHAPES)

    def test_very_long_face_is_oval(self):
        long_face = FaceMeasurements(forehead_width=0.9, cheekbone_width=1.0, jaw_width=0.88, face_length=1.9, eye_distance=0.4)
        self.assertEqual(classify_face_shape(long_face), "oval")

    def test_wide_cheekbones_is_diamond(self):
        diamond = FaceMeasurements(forehead_width=0.75, cheekbone_width=1.0, jaw_width=0.75, face_length=1.3, eye_distance=0.4)
        self.assertEqual(classify_face_shape(diamond), "diamond")

    def test_narrow_jaw_wide_forehead_is_heart(self):
        heart = FaceMeasurements(forehead_width=1.0, cheekbone_width=0.95, jaw_width=0.65, face_length=1.3, eye_distance=0.4)
        self.assertEqual(classify_face_shape(heart), "heart")


if __name__ == "__main__":
    unittest.main()
