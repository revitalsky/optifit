"""
recommendation_engine_v2.py
-----------------------------
מנוע המלצות משוקלל (Weighted Multi-Factor Scoring), מבוסס על הסכמה
שהוגדרה ב-data/rules_v2.json (הועלה כ-optifit_rules_v0_1.json).

בשונה מהגרסה הראשונה (recommendation_engine.py, שהתבססה על good/bad/neutral),
כאן כל מסגרת מקבלת כמה ציוני משנה בנפרד (כל אחד 0-100%):

    Shape Match      - התאמת צורת המסגרת לצורת הפנים (מטריצת 1-5 ב-rules_v2.json)
    Color Match       - התאמת משפחת הצבע לאנדרטון שזוהה
    Material Match    - התאמת החומר להעדפת המשתמשת (אם הוצהרה)
    Thickness Match   - התאמת עובי המסגרת לצורת הפנים
    Size Match         - אופציונלי: רק אם המשתמשת הזינה מידה של משקפיים נוחים לה

ואז ציון Overall Match משוקלל:
    style_score = 0.4*Shape + 0.2*Color + 0.15*Thickness + 0.15*Material + 0.1*universal_score
    (המשקלים האלו מגיעים מ-rules_v2.json -> scoring.style_score_weights)

    אם יש Size Match:  Overall = 0.7*style_score + 0.3*Size Match
    אם אין:              Overall = style_score
    (משקל ה-70/30 הוא ברירת מחדל של המנוע - rules_v2.json לא מגדיר אותו,
    כי מידה מדויקת מוגדרת שם כ"hard filter" נפרד ולא כחלק מ-style_score_weights)

עקרונות אתיים שנשמרים (מתוך rules_v2.json / principles):
- צורת פנים היא רק "אות סגנוני", לא שיפוט של יופי
- לא מסיקים חומר מתאים לפי צורת פנים בלבד
- לא טוענים למידה מדויקת (מ"מ) מתמונה לא-מכוילת - Size Match מחושב רק אם המשתמשת
  הזינה בעצמה מידה של משקפיים נוחים לה, לא מנחשים אותה מהתמונה
- המלצות צבע הן must-be-fuzzy - ניתנות באמון (confidence) לא כוודאות
"""

import json
import os
from dataclasses import dataclass, field

from frame_shape_advice import shape_score as _shape_advice_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
RULES_PATH = os.path.join(DATA_DIR, "rules_v2.json")
CATALOG_PATH = os.path.join(DATA_DIR, "catalog_v2.json")

SIZE_MATCH_WEIGHT_IN_OVERALL = 0.3  # ברירת מחדל של המנוע - ראו הסבר בראש הקובץ


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ScoredFrame:
    frame: dict
    overall_match_pct: int
    shape_match_pct: int
    color_match_pct: int
    material_match_pct: int
    thickness_match_pct: int
    universal_score_pct: int
    size_match_pct: int = None  # None אם המשתמשת לא הזינה מידה ידועה
    reasons: list = field(default_factory=list)


class RecommendationEngineV2:
    def __init__(self, rules_path: str = RULES_PATH, catalog_path: str = CATALOG_PATH):
        self.rules = _load_json(rules_path)
        self.catalog = _load_json(catalog_path)
        self.weights = self.rules["scoring"]["style_score_weights"]
        self.face_shape_scores = self.rules["face_shape_frame_scores_1_to_5"]
        self.rim_prefs = self.rules["rim_thickness_preferences_by_face_shape"]
        self.color_rules = self.rules["color_rules"]
        self.size_tolerances = self.rules["size_rules"]["matching_from_known_comfortable_pair"]

    def _face_shape_score(self, face_shape: str, frame_shape: str) -> tuple:
        """
        מחזיר ציון 0-1 + סיבה להתאמת צורת המסגרת לצורת הפנים.

        המקור הראשי הוא היררכיית ההמלצות ב-frame_shape_advice (best/highly/…),
        כך שדירוג הצורות שמוצג למשתמש הוא בדיוק זה שמשפיע על הציון. אם צורת הפנים
        אינה מוכרת שם, נופלים בחזרה למטריצת ה-1-5 מ-rules_v2.json.
        """
        advice = _shape_advice_score(face_shape, frame_shape)
        if advice is not None:
            return advice, f"התאמת צורה (מומלץ): {round(advice * 100)}%"
        table = self.face_shape_scores.get(face_shape, {})
        raw = table.get(frame_shape, 3)  # ברירת מחדל: ניטרלי (3/5) אם השילוב לא מוגדר
        return raw / 5.0, f"התאמת צורה: {raw}/5"

    def _rim_thickness_score(self, face_shape: str, rim_thickness: str) -> tuple:
        """בודק אם עובי המסגרת נמצא ברשימת ההעדפות לצורת הפנים הזו."""
        preferred = self.rim_prefs.get(face_shape, [])
        if rim_thickness in preferred:
            idx = preferred.index(rim_thickness)
            # הראשון ברשימה מקבל ציון מלא, השני קצת פחות וכו'
            score = 1.0 - (idx * 0.15)
            reason = f"עובי מסגרת ({rim_thickness}) מועדף לצורה זו"
        else:
            score = 0.4
            reason = f"עובי מסגרת ({rim_thickness}) לא ברשימת ההעדפה, אך לא פסול"
        return max(0.0, score), reason

    def _color_score(self, undertone: str, color_family: str) -> tuple:
        """בודק אם משפחת הצבע של המסגרת מתאימה לאנדרטון שזוהה."""
        if undertone in (None, "unknown"):
            return 0.6, "לא זוהה גוון עור - ציון צבע ניטרלי"
        matching_families = self.color_rules["undertone"].get(undertone, [])
        if color_family in matching_families:
            return 1.0, f"גוון '{color_family}' מתאים לאנדרטון {undertone}"
        neutral_families = self.color_rules["undertone"].get("neutral", [])
        if color_family in neutral_families:
            return 0.7, f"גוון '{color_family}' ניטרלי - מתאים לרוב האנדרטונים"
        return 0.35, f"גוון '{color_family}' פחות טיפוסי לאנדרטון {undertone}"

    def _material_score(self, preferred_materials: list, material: str) -> tuple:
        """אם למשתמש אין העדפת חומר מוצהרת - ציון ניטרלי. אחרת בודק התאמה."""
        if not preferred_materials:
            return 0.7, "אין העדפת חומר - ציון ניטרלי"
        if material in preferred_materials:
            return 1.0, f"חומר ({material}) תואם להעדפה"
        return 0.4, f"חומר ({material}) לא בהעדפה המוצהרת"

    def _size_score(self, known_frame: dict, frame: dict):
        """
        משווה מידה ידועה (משקפיים נוחים) שהזינה המשתמשת למידות המסגרת בקטלוג.
        מחזיר (ציון 0-1, סיבה) או (None, None) אם חסר מידע לחישוב.
        known_frame: dict עם lens_width_mm / bridge_width_mm / temple_length_mm.
        """
        dims = [
            ("lens_width_mm", "lens_width_tolerance_mm", "רוחב עדשה"),
            ("bridge_width_mm", "bridge_width_tolerance_mm", "רוחב גשר"),
            ("temple_length_mm", "temple_length_tolerance_mm", "אורך זרוע"),
        ]
        dim_scores = []
        diffs = []
        for key, tolerance_key, label in dims:
            user_val = known_frame.get(key)
            frame_val = frame.get(key)
            if user_val is None or frame_val is None:
                continue
            tolerance = self.size_tolerances[tolerance_key]
            diff = abs(user_val - frame_val)
            dim_score = max(0.0, 1.0 - diff / (tolerance * 2))
            dim_scores.append(dim_score)
            diffs.append(f"{label} הפרש {diff:.0f}מ״מ")

        if not dim_scores:
            return None, None

        avg_score = sum(dim_scores) / len(dim_scores)
        reason = "התאמת מידה למשקפיים הנוחים שהוזנו: " + ", ".join(diffs)
        return avg_score, reason

    def recommend(
        self,
        face_shape: str,
        undertone: str = None,
        preferred_materials: list = None,
        known_frame: dict = None,
        category: str = None,
        max_items: int = 5,
    ) -> list:
        """
        מחזיר רשימת ScoredFrame ממוינת מהציון הגבוה לנמוך (Top max_items, מומלץ 3-5).

        known_frame: dict אופציונלי עם lens_width_mm/bridge_width_mm/temple_length_mm
        של זוג משקפיים שנוח למשתמשת. אם לא סופק (או None) - לא מחושב Size Match כלל,
        ולא מנסים להסיק מידה מהתמונה (בהתאם לעקרון size ב-rules_v2.json).

        category: "sun" | "vision" | None. אם סופק, מסננים את הקטלוג לקטגוריה הזו בלבד
        (משקפי שמש מול משקפי ראייה). None = כל הקטלוג.
        """
        preferred_materials = preferred_materials or []
        scored = []

        catalog = self.catalog
        if category:
            catalog = [f for f in catalog if f.get("category") == category]

        for frame in catalog:
            fs_score, fs_reason = self._face_shape_score(face_shape, frame["shape"])
            rim_score, rim_reason = self._rim_thickness_score(face_shape, frame["rim_thickness"])
            color_score, color_reason = self._color_score(undertone, frame["color_family"])
            mat_score, mat_reason = self._material_score(preferred_materials, frame["material"])
            uni_score = frame.get("universal_score", 0.7)

            style_score = (
                self.weights["face_shape_match"] * fs_score
                + self.weights["color_match"] * color_score
                + self.weights["rim_thickness_match"] * rim_score
                + self.weights["material_preference_match"] * mat_score
                + self.weights["universal_score"] * uni_score
            )

            reasons = [fs_reason, color_reason, rim_reason, mat_reason]

            size_score = None
            if known_frame:
                size_score, size_reason = self._size_score(known_frame, frame)
                if size_score is not None:
                    reasons.append(size_reason)

            if size_score is not None:
                overall = (1 - SIZE_MATCH_WEIGHT_IN_OVERALL) * style_score + SIZE_MATCH_WEIGHT_IN_OVERALL * size_score
            else:
                overall = style_score

            scored.append(
                ScoredFrame(
                    frame=frame,
                    overall_match_pct=round(overall * 100),
                    shape_match_pct=round(fs_score * 100),
                    color_match_pct=round(color_score * 100),
                    material_match_pct=round(mat_score * 100),
                    thickness_match_pct=round(rim_score * 100),
                    universal_score_pct=round(uni_score * 100),
                    size_match_pct=round(size_score * 100) if size_score is not None else None,
                    reasons=reasons,
                )
            )

        scored.sort(key=lambda s: s.overall_match_pct, reverse=True)
        return scored[:max_items]


if __name__ == "__main__":
    engine = RecommendationEngineV2()
    for shape in ["oval", "round", "square", "heart", "diamond"]:
        print(f"\n=== {shape} (undertone=warm) ===")
        results = engine.recommend(shape, undertone="warm", max_items=3)
        for r in results:
            print(f"  {r.frame['model']:30s} Overall: {r.overall_match_pct}%  |  {r.reasons[0]}")

    print("\n=== round (undertone=cool, known_frame=49/21/145) ===")
    results = engine.recommend(
        "round", undertone="cool",
        known_frame={"lens_width_mm": 49, "bridge_width_mm": 21, "temple_length_mm": 145},
        max_items=3,
    )
    for r in results:
        print(f"  {r.frame['model']:30s} Overall: {r.overall_match_pct}%  Size: {r.size_match_pct}%")
