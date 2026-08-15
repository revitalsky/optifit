"""
frame_shape_advice.py
---------------------
Reusable frame-shape guidance derived from the *detected face shape* (never
hardcoded for a single user). It provides three things the rest of the app
consumes:

1. recommended_frame_shapes(face_shape)  -> a ranked, tiered hierarchy of frame
   shapes with friendly names and a plain-language "why" (EN + HE), plus soft
   caution notes for shapes/traits that may provide less balance.

2. shape_score(face_shape, frame_shape)  -> a 0..1 compatibility score used by
   the recommendation engine so the hierarchy actually influences ranking
   (not just decorative text). Falls back gracefully for unknown shapes.

3. frame_why(scored, face_shape, undertone) -> the concrete, per-frame reasons
   ("Soft rectangular shape adds definition to your oval face", "Warm tortoise
   complements your undertone", ...) generated from that frame's real sub-scores.

The wording is intentionally soft ("Recommended", "Best suited", "May provide
less balance") — face shape is a style signal, not a rule.
"""

# Ranked hierarchy per detected face shape: (frame_shape, tier), best first.
# Tiers map to compatibility scores in TIER_SCORE below. Frame shapes match the
# catalog's frame_shape_classes; hexagon/geometric frames are catalogued as
# "soft_square", so that entry covers "Soft Square / Geometric".
HIERARCHY = {
    "oval": [
        ("rectangle", "best"), ("soft_square", "best"), ("cat_eye", "highly"),
        ("browline", "recommended"), ("panto", "recommended"), ("oval", "recommended"),
        ("aviator", "alternative"), ("round", "caution"),
    ],
    "round": [
        ("rectangle", "best"), ("soft_square", "best"), ("cat_eye", "highly"),
        ("browline", "recommended"), ("aviator", "recommended"), ("panto", "alternative"),
        ("oval", "caution"), ("round", "caution"),
    ],
    "square": [
        ("panto", "best"), ("round", "best"), ("oval", "highly"),
        ("cat_eye", "recommended"), ("aviator", "recommended"), ("browline", "recommended"),
        ("soft_square", "alternative"), ("rectangle", "caution"),
    ],
    "heart": [
        ("oval", "best"), ("panto", "best"), ("round", "highly"),
        ("cat_eye", "recommended"), ("aviator", "recommended"), ("soft_square", "recommended"),
        ("browline", "alternative"), ("rectangle", "alternative"),
    ],
    "oblong": [
        ("panto", "best"), ("round", "best"), ("soft_square", "highly"),
        ("cat_eye", "recommended"), ("browline", "recommended"), ("oval", "recommended"),
        ("aviator", "alternative"), ("rectangle", "alternative"),
    ],
    "diamond": [
        ("oval", "best"), ("cat_eye", "best"), ("browline", "highly"),
        ("panto", "recommended"), ("round", "recommended"), ("soft_square", "recommended"),
        ("aviator", "alternative"), ("rectangle", "alternative"),
    ],
}

TIER_SCORE = {"best": 1.0, "highly": 0.9, "recommended": 0.8, "alternative": 0.65, "caution": 0.4}
NEUTRAL_SCORE = 0.6  # frame shape not present in a face-shape's hierarchy

TIER_LABEL = {
    "best": {"en": "Best Match", "he": "התאמה מיטבית"},
    "highly": {"en": "Highly Recommended", "he": "מומלץ מאוד"},
    "recommended": {"en": "Recommended", "he": "מומלץ"},
    "alternative": {"en": "Good Alternative", "he": "חלופה טובה"},
    "caution": {"en": "May provide less balance", "he": "עשוי להעניק פחות איזון"},
}

FRIENDLY = {
    "rectangle": {"en": "Soft Rectangle", "he": "מלבני רך"},
    "soft_square": {"en": "Soft Square / Geometric", "he": "מרובע רך / גיאומטרי"},
    "cat_eye": {"en": "Subtle Cat-Eye", "he": "חתולי עדין"},
    "browline": {"en": "Browline", "he": "בראוליין"},
    "aviator": {"en": "Refined Aviator", "he": "טייסים מעודן"},
    "panto": {"en": "Panto / Soft Round", "he": "פנטו / עגול רך"},
    "oval": {"en": "Oval", "he": "אובלי"},
    "round": {"en": "Round", "he": "עגול"},
}

# Plain-language "why" per frame shape — balancing language, no measurements.
SHAPE_WHY = {
    "rectangle": {
        "en": "Adds gentle structure and definition while balancing softer facial contours.",
        "he": "מוסיף מבנה והגדרה עדינים תוך איזון קווי פנים רכים.",
    },
    "soft_square": {
        "en": "Adds visual definition without making the face appear too angular.",
        "he": "מוסיף הגדרה חזותית מבלי שהפנים ייראו זוויתיים מדי.",
    },
    "cat_eye": {
        "en": "Creates a slight lifting effect around the eyes and cheekbones.",
        "he": "יוצר אפקט הרמה עדין סביב העיניים ועצמות הלחיים.",
    },
    "browline": {
        "en": "Draws attention upward and frames the face with a confident line.",
        "he": "מושך את המבט כלפי מעלה וממסגר את הפנים בקו בטוח.",
    },
    "aviator": {
        "en": "A refined, narrow profile that keeps your proportions balanced.",
        "he": "פרופיל צר ומעודן ששומר על פרופורציות מאוזנות.",
    },
    "panto": {
        "en": "A classic rounded profile that softens and balances the features.",
        "he": "פרופיל עגול קלאסי שמרכך ומאזן את התווים.",
    },
    "oval": {
        "en": "Echoes your natural proportions for an easy, balanced look.",
        "he": "משתלב עם הפרופורציות הטבעיות למראה קליל ומאוזן.",
    },
    "round": {
        "en": "A soft, curved profile — best in moderate sizes to keep balance.",
        "he": "פרופיל רך ומעוגל — עדיף במידות מתונות לשמירה על איזון.",
    },
}

# Soft cautions (traits, not absolute rules). Shared across face shapes.
CAUTION_TRAITS = [
    {"en": "Very small, perfectly round frames — may provide less balance.",
     "he": "מסגרות עגולות קטנות ומדויקות — עשויות להעניק פחות איזון."},
    {"en": "Extremely oversized frames — may overwhelm your proportions.",
     "he": "מסגרות גדולות במיוחד — עלולות להכביד על הפרופורציות."},
    {"en": "Very thick, visually heavy frames — may provide less balance.",
     "he": "מסגרות עבות וכבדות חזותית — עשויות להעניק פחות איזון."},
]

DEPTH_DEFAULT = "oval"  # neutral fallback when face shape is unknown


def _hierarchy_for(face_shape):
    return HIERARCHY.get(face_shape) or HIERARCHY[DEPTH_DEFAULT]


def shape_score(face_shape, frame_shape):
    """0..1 compatibility of a frame shape for a detected face shape."""
    for fs, tier in _hierarchy_for(face_shape):
        if fs == frame_shape:
            return TIER_SCORE[tier]
    return NEUTRAL_SCORE


def recommended_frame_shapes(face_shape, max_items=5):
    """
    Ranked, tiered frame-shape guidance for the UI. Returns a dict with:
      recommended: [{shape, tier, friendly:{en,he}, tier_label:{en,he}, why:{en,he}}]
      cautions:    [{en, he}]   (soft "may provide less balance" notes)
    Only positive tiers (best..alternative) are listed under recommended;
    caution-tier shapes feed the cautions list instead.
    """
    hierarchy = _hierarchy_for(face_shape)
    recommended = []
    for fs, tier in hierarchy:
        if tier == "caution":
            continue
        recommended.append({
            "shape": fs,
            "tier": tier,
            "friendly": FRIENDLY.get(fs, {"en": fs, "he": fs}),
            "tier_label": TIER_LABEL[tier],
            "why": SHAPE_WHY.get(fs, {"en": "", "he": ""}),
        })
        if len(recommended) >= max_items:
            break
    return {"recommended": recommended, "cautions": list(CAUTION_TRAITS)}


def _color_sentence(color_name, undertone):
    tone = {"warm": "warm", "cool": "cool", "neutral": "neutral"}.get(undertone, "")
    tone_he = {"warm": "החם", "cool": "הקריר", "neutral": "הניטרלי"}.get(undertone, "")
    if not color_name:
        return None
    return {
        "en": f"{color_name} complements your {tone} undertone." if tone else f"{color_name} suits your coloring.",
        "he": f"גוון {color_name} מחמיא לגוון {tone_he} שלך." if tone_he else f"גוון {color_name} מתאים לגוון העור שלך.",
    }


_MATERIAL_SENTENCE = {
    "titanium": {"en": "Lightweight titanium sits comfortably for all-day wear.",
                 "he": "טיטניום קל מאפשר נשיאה נוחה לאורך היום."},
    "metal": {"en": "Light metal keeps the look clean and minimal.",
              "he": "מתכת קלה שומרת על מראה נקי ומינימלי."},
    "acetate": {"en": "Rich acetate adds depth and a classic finish.",
                "he": "אצטט עשיר מוסיף עומק וגימור קלאסי."},
    "combination": {"en": "A balanced mix of materials for a versatile look.",
                    "he": "שילוב מאוזן של חומרים למראה רב-גוני."},
}


def frame_why(scored, face_shape, undertone):
    """
    Per-frame explanation generated from the frame's real sub-scores.
    Returns {"en": [...], "he": [...]} — 2-3 short, customer-facing reasons.
    `scored` is a ScoredFrame (has shape_match_pct/color_match_pct/... and .frame).
    """
    frame = scored.frame
    face_en = face_shape or "face"
    face_he = {"oval": "אובלי", "round": "עגול", "square": "מרובע",
               "heart": "לב", "oblong": "מוארך", "diamond": "יהלום"}.get(face_shape, "שלך")
    friendly = FRIENDLY.get(frame.get("shape"), {"en": frame.get("shape", ""), "he": ""})
    en, he = [], []

    # 1) Shape compatibility (primary factor)
    if scored.shape_match_pct >= 78:
        en.append(f"{friendly['en']} shape adds definition to your {face_en} face.")
        he.append(f"צורת {friendly['he']} מוסיפה הגדרה לפנים ה{face_he}ות שלך.")

    # 2) Colour / undertone harmony
    if scored.color_match_pct >= 80:
        cs = _color_sentence(frame.get("color_name") or frame.get("color_family"), undertone)
        if cs:
            en.append(cs["en"])
            he.append(cs["he"])

    # 3) Frame width / fit balance
    fit_ok = (scored.size_match_pct is not None and scored.size_match_pct >= 75) \
        or scored.thickness_match_pct >= 78
    if fit_ok:
        en.append("Frame width is well balanced with your facial proportions.")
        he.append("רוחב המסגרת מאוזן היטב מול הפרופורציות של הפנים שלך.")

    # 4) Material note (only if we still have room — keeps it to ~3 lines)
    if len(en) < 3:
        ms = _MATERIAL_SENTENCE.get(frame.get("material"))
        if ms:
            en.append(ms["en"])
            he.append(ms["he"])

    if not en:  # always say something useful
        en.append("A well-rounded, versatile match for your profile.")
        he.append("התאמה מאוזנת ורב-שימושית לפרופיל שלך.")
    return {"en": en, "he": he}


if __name__ == "__main__":
    import json
    print(json.dumps(recommended_frame_shapes("oval"), ensure_ascii=False, indent=2))
    for fshape in ("rectangle", "soft_square", "cat_eye", "round", "aviator"):
        print(fshape, "->", shape_score("oval", fshape))
