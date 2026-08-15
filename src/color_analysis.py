"""
color_analysis.py
-------------------
מודול ניתוח צבעים (ניתוח גוון עור) - "מה מתאים לי" מעבר לצורת המסגרת.
דוגם צבע עור מאזורי הלחיים והמצח (מתוך ה-landmarks שכבר זוהו), מסווג
אנדרטון (חם/קר/ניטרלי) ובהירות, וממפה ל"עונת צבעים" עם המלצות
לצבעי מסגרת ומתכת מתאימים.

זהו מנגנון נוסף מבוסס-חוקים (לא מודל AI מאומן) - עקבי עם אופי הפרויקט.
"""

import math
from dataclasses import dataclass

import cv2

# אינדקסים של נקודות דגימה על אזורי עור חלקים (לחי שמאל, לחי ימין, מצח)
SKIN_SAMPLE_INDICES = {
    "left_cheek": 234,
    "right_cheek": 454,
    "forehead": 10,
}


@dataclass
class ColorProfile:
    undertone: str          # "warm" | "cool" | "neutral"
    depth: str               # "light" | "medium" | "deep"
    season: str               # "אביב" | "קיץ" | "סתיו" | "חורף"
    season_en: str
    recommended_metals: list
    recommended_frame_colors: list
    palette_hex: list         # צבעי דוגמה להצגה ויזואלית
    explanation: str
    confidence: str            # "low" | "medium" | "high"
    confidence_reason: str


SEASON_DATA = {
    ("warm", "light"): dict(
        season="אביב", season_en="Spring",
        metals=["זהב", "זהב ורוד"],
        frame_colors=["חום דבש", "כתום צרוב", "זהב שקוף", "ירוק זית בהיר"],
        palette=["#E8A33D", "#C97B3E", "#8FB339", "#F2C879"],
        explanation="גוון עור חם ובהיר. מסגרות בגווני זהב וחום חמים ידגישו את הזוהר הטבעי של העור.",
    ),
    ("warm", "medium"): dict(
        season="סתיו", season_en="Autumn",
        metals=["זהב", "ברונזה"],
        frame_colors=["טרטוישל (צב)", "חום קוניאק", "זהב עתיק", "ירוק יער"],
        palette=["#8B5A2B", "#C1440E", "#6B8E23", "#B8860B"],
        explanation="גוון עור חם ועמוק. מסגרות בגווני אדמה עשירים (חום, קוניאק, זהב עתיק) יתאימו בצורה הרמונית.",
    ),
    ("cool", "light"): dict(
        season="קיץ", season_en="Summer",
        metals=["כסף", "זהב לבן"],
        frame_colors=["אפור עשן", "כחול פסטל", "ורוד אבק", "כסף מבריק"],
        palette=["#9DB4C0", "#C4A6C2", "#7C90A0", "#D6CFE1"],
        explanation="גוון עור קריר ובהיר. מסגרות בגווני כסף וטונים פסטליים קרירים יבליטו את העדינות של הגוון.",
    ),
    ("cool", "deep"): dict(
        season="חורף", season_en="Winter",
        metals=["כסף", "שחור מבריק", "פלטינה"],
        frame_colors=["שחור עמוק", "כחול חצות", "כסף קר", "סגול יין"],
        palette=["#1A1A2E", "#16213E", "#5C2A9D", "#0F3460"],
        explanation="גוון עור קריר ועמוק. מסגרות בניגודיות חדה - שחור, כחול חצות וכסף - יתאימו לעוצמה של הגוון.",
    ),
    ("neutral", "medium"): dict(
        season="ניטרלי מאוזן", season_en="Neutral",
        metals=["זהב", "כסף"],
        frame_colors=["חום טבעי", "אפור פחם", "טורטוישל קלאסי", "שחור"],
        palette=["#5C4A3A", "#4A4A4A", "#8B7355", "#2C2C2C"],
        explanation="גוון עור מאוזן שמתאים כמעט לכל גווני מתכת ומסגרת - יש כאן חופש בחירה רחב.",
    ),
}


def _bgr_to_hex(b, g, r):
    return f"#{r:02x}{g:02x}{b:02x}"


def _sample_patch_avg_bgr(image, x, y, radius=6):
    """ממוצע צבע (BGR) באזור קטן סביב נקודה, כדי להימנע מרעש של פיקסל בודד."""
    h, w = image.shape[:2]
    x0, x1 = max(0, x - radius), min(w, x + radius)
    y0, y1 = max(0, y - radius), min(h, y + radius)
    patch = image[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    b, g, r = patch[:, :, 0].mean(), patch[:, :, 1].mean(), patch[:, :, 2].mean()
    return b, g, r


def analyze_skin_tone(image_path: str, landmarks) -> ColorProfile:
    """
    מנתח את גוון העור מתוך התמונה + נקודות הציון שכבר זוהו,
    ומחזיר פרופיל צבעים עם המלצות.
    """
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    samples = []
    for idx in SKIN_SAMPLE_INDICES.values():
        pt = landmarks[idx]
        x, y = int(pt.x * w), int(pt.y * h)
        avg = _sample_patch_avg_bgr(img, x, y)
        if avg:
            samples.append(avg)

    if not samples:
        raise ValueError("לא ניתן היה לדגום את צבע העור מהתמונה")

    b = sum(s[0] for s in samples) / len(samples)
    g = sum(s[1] for s in samples) / len(samples)
    r = sum(s[2] for s in samples) / len(samples)

    # --- קביעת אנדרטון (חם/קר/ניטרלי) ---
    # יחס אדום-צהוב (R מול B) מעיד על חום; הפרש קטן -> ניטרלי
    warmth_score = (r - b) / max(1.0, (r + g + b) / 3)
    if warmth_score > 0.12:
        undertone = "warm"
    elif warmth_score < -0.03:
        undertone = "cool"
    else:
        undertone = "neutral"

    # --- קביעת בהירות (depth) לפי בהירות כללית (luminance) ---
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    if luminance > 175:
        depth = "light"
    elif luminance > 110:
        depth = "medium"
    else:
        depth = "deep"

    key = (undertone, depth)
    if key not in SEASON_DATA:
        # נפילה חכמה *בתוך אותו אנדרטון* - כדי לא לסתור את עצמנו: אנדרטון חם
        # לעולם לא ייפול לעונה ניטרלית (שממליצה גם על כסף) אלא לעונה חמה קרובה.
        same_undertone = [k for k in SEASON_DATA if k[0] == undertone]
        if same_undertone:
            key = (
                next((k for k in same_undertone if k[1] == depth), None)
                or next((k for k in same_undertone if k[1] == "medium"), None)
                or same_undertone[0]
            )
        else:
            key = ("neutral", "medium")

    data = SEASON_DATA[key]
    confidence, confidence_reason = _estimate_confidence(samples, luminance)

    return ColorProfile(
        undertone=undertone,
        depth=depth,
        season=data["season"],
        season_en=data["season_en"],
        recommended_metals=data["metals"],
        recommended_frame_colors=data["frame_colors"],
        palette_hex=data["palette"],
        explanation=data["explanation"],
        confidence=confidence,
        confidence_reason=confidence_reason,
    )


def _estimate_confidence(samples, avg_luminance):
    """
    מעריך כמה אפשר לסמוך על ניתוח הצבעים, בלי לפסול אותו - רק לתייג אותו.
    שלוש סיבות אפשריות ל-confidence נמוך:
      1. לא נדגמו כל שלוש נקודות הדגימה (למשל בגלל זווית פנים).
      2. תאורה קיצונית (חשוך מאוד או שרוף/בהיר מאוד) - הצבעים לא אמינים.
      3. הפיזור בין נקודות הדגימה (לחי/לחי/מצח) גדול מדי - סימן לתאורה
         לא אחידה על הפנים (למשל אור חד-צדדי) שמטה את קביעת האנדרטון.
    """
    if len(samples) < len(SKIN_SAMPLE_INDICES):
        return "low", f"נדגמו רק {len(samples)}/{len(SKIN_SAMPLE_INDICES)} אזורי עור (חלק לא זוהו בתמונה)"

    if avg_luminance < 55 or avg_luminance > 235:
        return "low", "תאורה קיצונית בתמונה (חשוכה או שרופה מדי) - הצבעים עלולים להיות לא מדויקים"

    luminances = [0.299 * r + 0.587 * g + 0.114 * b for b, g, r in samples]
    luminance_spread = max(luminances) - min(luminances)

    warmths = [(r - b) / max(1.0, (r + g + b) / 3) for b, g, r in samples]
    warmth_spread = max(warmths) - min(warmths)

    if luminance_spread > 45:
        return "low", "בהירות שונה מאוד בין אזורי הדגימה בפנים - כנראה תאורה לא אחידה (חד-צדדית)"
    if warmth_spread > 0.18:
        return "low", "גוון הצבע שונה מאוד בין אזורי הדגימה בפנים - כנראה תאורה צבעונית או לא אחידה"

    if luminance_spread > 25 or warmth_spread > 0.10:
        return "medium", "פיזור מתון בין אזורי הדגימה - התוצאה סבירה אך לא מושלמת"

    return "high", "דגימות עקביות ותאורה סבירה"


if __name__ == "__main__":
    print("מודול ניתוח צבעים - רשימת כל הפרופילים האפשריים:\n")
    for (undertone, depth), data in SEASON_DATA.items():
        print(f"{undertone}/{depth} -> {data['season']} ({data['season_en']})")
        print(f"   מתכות: {', '.join(data['metals'])}")
        print(f"   צבעי מסגרת: {', '.join(data['frame_colors'])}")
        print()
