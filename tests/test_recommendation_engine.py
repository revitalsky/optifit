"""
בדיקות יחידה למנוע ההמלצות v2 (recommendation_engine_v2).
מריצים מתיקיית השורש:  .venv\\Scripts\\python.exe -m unittest discover -s tests
"""

import os
import sys
import unittest

SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, SRC)

from recommendation_engine_v2 import RecommendationEngineV2  # noqa: E402

FACE_SHAPES = ["oval", "round", "square", "heart", "diamond"]


class TestRecommendationEngineV2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationEngineV2()   # טוען rules_v2.json + catalog_v2.json האמיתיים

    def test_returns_at_most_max_items(self):
        results = self.engine.recommend("oval", undertone="warm", max_items=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)

    def test_results_sorted_by_overall_desc(self):
        results = self.engine.recommend("round", undertone="cool", max_items=5)
        scores = [r.overall_match_pct for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_all_percentages_in_valid_range(self):
        for r in self.engine.recommend("square", undertone="warm", max_items=5):
            for pct in (r.overall_match_pct, r.shape_match_pct, r.color_match_pct,
                        r.material_match_pct, r.thickness_match_pct):
                self.assertGreaterEqual(pct, 0)
                self.assertLessEqual(pct, 100)

    def test_category_filter_only_returns_that_category(self):
        for category in ("vision", "sun"):
            with self.subTest(category=category):
                results = self.engine.recommend("oval", category=category, max_items=10)
                self.assertTrue(results, f"אין תוצאות לקטגוריה {category}")
                for r in results:
                    self.assertEqual(r.frame.get("category"), category)

    def test_no_known_frame_means_no_size_match(self):
        for r in self.engine.recommend("heart", undertone="warm", max_items=5):
            self.assertIsNone(r.size_match_pct)

    def test_known_frame_produces_size_match(self):
        known = {"lens_width_mm": 49, "bridge_width_mm": 21, "temple_length_mm": 145}
        results = self.engine.recommend("round", undertone="cool", known_frame=known, max_items=5)
        self.assertTrue(any(r.size_match_pct is not None for r in results))
        for r in results:
            if r.size_match_pct is not None:
                self.assertGreaterEqual(r.size_match_pct, 0)
                self.assertLessEqual(r.size_match_pct, 100)

    def test_every_scored_frame_has_reasons(self):
        for r in self.engine.recommend("diamond", undertone="warm", max_items=3):
            self.assertTrue(r.reasons)      # רשימת סיבות לא ריקה
            self.assertIn("model", r.frame)

    def test_all_face_shapes_yield_recommendations(self):
        for shape in FACE_SHAPES:
            with self.subTest(shape=shape):
                self.assertTrue(self.engine.recommend(shape, undertone="warm", max_items=3))


if __name__ == "__main__":
    unittest.main()
