"""
insights.py
-------------
שכבת "תרגום" מתוצאות המנוע הפנימי (עברית/מספרים) לתובנות אנושיות באנגלית
עבור ה-Front-End הפרימיום. לא משנה שום לוגיקה קיימת - רק ממפה פלט קיים
(ColorProfile, FaceMeasurements, ScoredFrame) לטקסט נקי ומוצג.
"""

FACE_SHAPE_EN = {
    "oval": "Oval",
    "round": "Round",
    "square": "Square",
    "heart": "Heart",
    "diamond": "Diamond",
}

SKIN_TONE_EN = {"warm": "Warm", "cool": "Cool", "neutral": "Neutral"}

# פלטות המלצה באנגלית לפי "עונת הצבעים" שכבר מחושבת ב-color_analysis (season_en)
# Warm seasons prioritise warm metals only (Gold / Rose Gold / Bronze / Antique
# Gold); Silver appears only for cool/neutral profiles, where it is actually
# supported — keeping metals consistent with the detected undertone.
SEASON_PALETTE_EN = {
    "Spring": {"colors": ["Amber", "Honey", "Warm Beige", "Peach"], "metals": ["Gold", "Rose Gold", "Bronze"]},
    "Autumn": {"colors": ["Tortoise", "Olive", "Cognac", "Warm Brown"], "metals": ["Gold", "Bronze", "Antique Gold"]},
    "Summer": {"colors": ["Smoke Gray", "Soft Blue", "Dusty Rose", "Silver"], "metals": ["Silver", "White Gold"]},
    "Winter": {"colors": ["Black", "Midnight Blue", "Cool Silver", "Burgundy"], "metals": ["Silver", "Platinum"]},
    "Neutral": {"colors": ["Tortoise", "Charcoal", "Natural Brown", "Black"], "metals": ["Gold", "Silver"]},
}


def _bin(value, excellent=90, great=80, good=68):
    if value >= excellent:
        return "Excellent"
    if value >= great:
        return "Great"
    if value >= good:
        return "Good"
    return "Fair"


def frame_width_label(top_scored):
    """תווית איכותית ל'רוחב מסגרת' לפי ציון התאמת הצורה של ההמלצה המובילה."""
    s = top_scored.shape_match_pct
    if s >= 90:
        return "Optimal"
    if s >= 78:
        return "Balanced"
    return "Good"


def bridge_fit_label(top_scored):
    """תווית איכותית ל'התאמת גשר' - מבוססת על ציון המידה אם הוזן, אחרת על העובי."""
    base = top_scored.size_match_pct if top_scored.size_match_pct is not None else top_scored.thickness_match_pct
    return _bin(base)


def english_reason(scored):
    """משפט הסבר קצר וברור באנגלית לכרטיס המלצה, נגזר מציוני המשנה האמיתיים."""
    reasons = []
    if scored.shape_match_pct >= 80:
        reasons.append("Recommended for your face shape")
    if scored.color_match_pct >= 90:
        reasons.append("Complements your skin tone")
    if scored.size_match_pct is not None and scored.size_match_pct >= 80:
        reasons.append("Excellent bridge fit")
    if scored.thickness_match_pct >= 90:
        reasons.append("Balanced proportions")
    if not reasons:
        reasons.append("Optimal frame width")
    return reasons[0]


def build_insights(color_profile, top_scored):
    """
    מחזיר dict תובנות באנגלית עבור ה-UI.
    color_profile: ColorProfile (מ-color_analysis)
    top_scored: ScoredFrame המוביל (מ-RecommendationEngineV2)
    """
    season = getattr(color_profile, "season_en", "Neutral")
    palette = SEASON_PALETTE_EN.get(season, SEASON_PALETTE_EN["Neutral"])
    low_conf = getattr(color_profile, "confidence", "high") == "low"
    return {
        "skin_tone": SKIN_TONE_EN.get(color_profile.undertone, "Neutral"),
        "recommended_colors": palette["colors"],
        "recommended_metals": palette["metals"],
        "frame_width": frame_width_label(top_scored),
        "bridge_fit": bridge_fit_label(top_scored),
        "lighting_note": "For best color accuracy, use even, natural lighting." if low_conf else None,
    }
