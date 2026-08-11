"""
app.py
------
ממשק המשתמש של OptiFit AI.
מריצים עם: streamlit run app.py

עדכוני גרסה 2:
- אפשרות לצלם ישירות מהמצלמה (לא רק העלאת קובץ)
- ניתוח צבעים (גוון עור -> עונת צבעים -> המלצת מתכת/גוון מסגרת, עם ציון ביטחון)
- הדמיית landmarks על התמונה
- קטלוג מסגרות בלי מחירים

עדכוני גרסה 3 (RecommendationEngineV2):
- לכל מסגרת מומלצת: ציון Overall Match משוקלל (Shape/Color/Material/Thickness
  ואופציונלית Size Match, אם המשתמשת הזינה מידה ידועה) - במקום Fit Score כללי
  לצורת הפנים
- הדמיה ויזואלית (Try-On) של המסגרת המומלצת על תמונת הפנים

עדכוני גרסה 4 (עיצוב מחדש - Dark + Cyan):
- פריסה כהה בהשראת מוקאאפ "Advanced AI Eyewear Fitting": רקע כהה + דגשי תכלת (tech)
- תמונת המצולם גדולה במרכז, כרטיסי מסגרות מצד ימין (2 בכל פעם, כפתורי קודם/הבא)
- לחיצה על כרטיס מסגרת = הדמיה (Try-On) של אותה מסגרת בתמונה המרכזית
- שימוש ב-st.session_state כדי לשמור את תוצאות הניתוח בין ריצות (reruns)
"""

import json
import os
import sys

import cv2
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from face_landmarks import FaceLandmarkDetector, MODEL_PATH, LANDMARK_INDICES  # noqa: E402
from face_shape_classifier import classify_face_shape  # noqa: E402
from face_shape_display import FACE_SHAPE_DISPLAY  # noqa: E402
from recommendation_engine_v2 import RecommendationEngineV2, CATALOG_PATH  # noqa: E402
from color_analysis import analyze_skin_tone  # noqa: E402
from virtual_tryon import overlay_glasses  # noqa: E402
import frame_prep  # noqa: E402

CATEGORY_LABELS = {"vision": "👓 משקפי ראייה", "sun": "🕶️ משקפי שמש"}
# מידות ברירת-מחדל טיפוסיות לפי צורה (למסגרות שמועלות אוטומטית, בלי מידע יצרן)
SIZE_BY_SHAPE = {
    "round": (47, 21, 145), "oval": (53, 19, 145), "cat_eye": (52, 18, 140),
    "soft_square": (52, 20, 145), "rectangle": (53, 18, 145),
    "browline": (52, 18, 145), "aviator": (58, 14, 140), "panto": (49, 21, 145),
}


def _guess_material(color_family):
    return "metal" if color_family in ("silver", "gold", "gunmetal") else "acetate"


def _next_frame_id(catalog, prefix):
    nums = [int(c["id"][1:]) for c in catalog if c["id"].startswith(prefix) and c["id"][1:].isdigit()]
    return f"{prefix}{(max(nums) + 1) if nums else 1:02d}"

st.set_page_config(page_title="OptiFit AI", page_icon="👓", layout="wide")

# --- עיצוב: רקע כהה + דגשי תכלת (tech), בהשראת המוקאאפ ---
ACCENT_CYAN = "#22D3EE"
ACCENT_CYAN_DIM = "#0E7490"
BG_DARK = "#0B0F14"
PANEL_DARK = "#111820"
PANEL_BORDER = "#1E2A36"
TEXT_LIGHT = "#E6EDF3"
TEXT_MUTED = "#8B9AA8"
FRAMES_PER_PAGE = 2

st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG_DARK}; }}
    h1, h2, h3 {{ color: {TEXT_LIGHT}; font-family: 'Georgia', serif; letter-spacing: 0.5px; }}
    .stButton>button {{
        background-color: transparent; color: {ACCENT_CYAN}; font-weight: 600;
        border-radius: 6px; border: 1px solid {ACCENT_CYAN_DIM}; padding: 0.4em 1.1em;
        transition: all 0.15s ease-in-out;
    }}
    .stButton>button:hover {{
        background-color: {ACCENT_CYAN}; color: {BG_DARK}; border-color: {ACCENT_CYAN};
    }}
    .ofa-card {{
        background: linear-gradient(135deg, {PANEL_DARK} 0%, #0D141C 100%);
        color: {TEXT_LIGHT}; padding: 1.4em; border-radius: 14px; margin-bottom: 1em;
        border: 1px solid {PANEL_BORDER};
    }}
    .ofa-score-bar-bg {{ background: rgba(255,255,255,0.08); border-radius: 8px; height: 10px; width: 100%; }}
    .ofa-score-bar-fill {{
        background: linear-gradient(90deg, {ACCENT_CYAN_DIM} 0%, {ACCENT_CYAN} 100%);
        border-radius: 8px; height: 10px;
    }}
    .ofa-swatch {{
        display: inline-block; width: 34px; height: 34px; border-radius: 50%;
        border: 2px solid {PANEL_BORDER}; margin-left: 6px; box-shadow: 0 0 4px rgba(0,0,0,0.5);
    }}
    .ofa-tag {{
        display: inline-block; background: {PANEL_DARK}; color: {ACCENT_CYAN};
        border: 1px solid {PANEL_BORDER};
        border-radius: 20px; padding: 4px 14px; margin: 4px; font-size: 0.85em;
    }}
    .ofa-frame-card {{
        background: {PANEL_DARK}; border: 1px solid {PANEL_BORDER};
        border-radius: 12px; padding: 0.6em 0.7em; margin-bottom: 0.6em;
    }}
    .ofa-frame-card-selected {{
        border-color: {ACCENT_CYAN}; box-shadow: 0 0 10px rgba(34,211,238,0.35);
    }}
    .ofa-frame-name {{ color: {TEXT_LIGHT}; font-weight: 600; font-size: 0.95em; }}
    .ofa-frame-sub {{ color: {TEXT_MUTED}; font-size: 0.8em; margin-bottom: 4px; }}
    .ofa-center-caption {{
        color: {TEXT_LIGHT}; text-align: center; font-size: 1.05em; margin-top: 0.4em;
    }}
    .ofa-center-caption .pct {{ color: {ACCENT_CYAN}; font-weight: 700; }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"<h1 style='text-align:center;'>👁️ OptiFit AI</h1>"
    f"<p style='text-align:center; color:{ACCENT_CYAN}; letter-spacing:3px; font-size:0.9em;'>"
    f"ADVANCED AI EYEWEAR FITTING</p>",
    unsafe_allow_html=True,
)

if not os.path.exists(MODEL_PATH):
    st.warning(
        "⚠️ קובץ מודל זיהוי הפנים חסר. יש להריץ פעם אחת: `bash download_model.sh` "
        "(דורש חיבור לאינטרנט, הורדה חד-פעמית של כ-3.7MB)."
    )

# ---------------------------------------------------------------------------
# סרגל צד: בחירת קטגוריה (ראייה/שמש) + הוספת מסגרות אוטומטית לקטלוג
# ---------------------------------------------------------------------------
st.sidebar.markdown("### סוג משקפיים")
category = st.sidebar.radio(
    "בחרי קטגוריה להמלצות:",
    ["vision", "sun"],
    format_func=lambda c: CATEGORY_LABELS[c],
    key="category",
)

with st.sidebar.expander("➕ הוסף מסגרות לקטלוג"):
    st.caption(
        "העלי תמונות מסגרת (PNG שקוף). המערכת תזהה אוטומטית אם זו שמש או ראייה, "
        "תחתוך לחזית (בלי מוטות) ותוסיף לקטלוג המתאים."
    )
    admin_files = st.file_uploader(
        "תמונות מסגרת", type=["png"], accept_multiple_files=True, key="admin_up"
    )
    if st.button("עבד והוסף לקטלוג", key="admin_btn"):
        if not admin_files:
            st.warning("לא נבחרו קבצים.")
        else:
            catalog = json.load(open(CATALOG_PATH, encoding="utf-8"))
            added = []
            for f in admin_files:
                tmp = f"data/_incoming_{f.name}"
                with open(tmp, "wb") as w:
                    w.write(f.getbuffer())
                try:
                    cat, _ = frame_prep.classify_category(tmp)
                    prefix = "S" if cat == "sun" else "V"
                    nid = _next_frame_id(catalog, prefix)
                    out_png = f"assets/frames/{cat}/{nid}.png"
                    meta = frame_prep.prepare_frame(tmp, out_png)
                    lw, bw, tl = SIZE_BY_SHAPE.get(meta["shape"], (52, 19, 145))
                    entry = {
                        "id": nid, "brand": "OptiFit", "model": f"Uploaded {nid}",
                        "category": cat, "shape": meta["shape"],
                        "material": _guess_material(meta["color_family"]),
                        "rim_thickness": meta["rim_thickness"],
                        "color_name": meta["color_family"].title(),
                        "color_family": meta["color_family"],
                        "lens_width_mm": lw, "bridge_width_mm": bw, "temple_length_mm": tl,
                        "image_png": out_png, "style_tags": ["uploaded"], "universal_score": 0.75,
                    }
                    catalog.append(entry)
                    added.append((f.name, cat, nid, meta["shape"], meta["color_family"]))
                except ValueError as e:
                    st.warning(f"דילגתי על {f.name}: {e}")
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)
            if added:
                json.dump(catalog, open(CATALOG_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
                for name, cat, nid, shp, col in added:
                    st.success(f"✅ {name} → {CATEGORY_LABELS[cat]} ({nid}, {shp}/{col})")
                st.info("המסגרות יופיעו בניתוח הבא.")


def _score_bar_html(pct):
    return (
        f'<div class="ofa-score-bar-bg"><div class="ofa-score-bar-fill" '
        f'style="width:{pct}%;"></div></div>'
    )


# ---------------------------------------------------------------------------
# שלב 1: קלט (העלאה / מצלמה) + מידה אופציונלית + כפתור ניתוח
# ---------------------------------------------------------------------------
tab_upload, tab_camera = st.tabs(["📁 העלאת תמונה", "📷 צילום ישיר"])

image = None
with tab_upload:
    uploaded_file = st.file_uploader("העלו תמונת פנים ברורה (פנים ישרות מול המצלמה)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

with tab_camera:
    camera_file = st.camera_input("צלמו את הפנים ישירות (יש לאשר גישה למצלמה בדפדפן)")
    if camera_file is not None:
        image = Image.open(camera_file).convert("RGB")

if image is not None:
    temp_path = "data/_uploaded_temp.jpg"
    image.save(temp_path)

    with st.expander("📏 יש לך משקפיים שנוחים לך? (אופציונלי)"):
        st.caption(
            "אם תזיני את המידה שרשומה על הידית הפנימית של משקפיים שנוחים לך "
            "(למשל 49/21/145), נוסיף לכל מסגרת מומלצת גם ציון Size Match. "
            "בלי מידע כזה - לא מנחשים מידה מהתמונה, פשוט לא מוצג ציון גודל."
        )
        size_col1, size_col2, size_col3 = st.columns(3)
        with size_col1:
            lens_width_input = st.number_input("רוחב עדשה (מ״מ)", min_value=0, max_value=80, value=0, step=1)
        with size_col2:
            bridge_width_input = st.number_input("רוחב גשר (מ״מ)", min_value=0, max_value=40, value=0, step=1)
        with size_col3:
            temple_length_input = st.number_input("אורך זרוע (מ״מ)", min_value=0, max_value=200, value=0, step=1)

    known_frame = None
    if lens_width_input and bridge_width_input and temple_length_input:
        known_frame = {
            "lens_width_mm": lens_width_input,
            "bridge_width_mm": bridge_width_input,
            "temple_length_mm": temple_length_input,
        }

    if st.button("🔍 נתח פנים והצע מסגרות", use_container_width=True):
        with st.spinner("מנתח את מבנה הפנים והצבעים..."):
            try:
                detector = FaceLandmarkDetector()
                landmarks = detector.detect(temp_path)

                if landmarks is None:
                    st.error("לא זוהו פנים בתמונה. נסו תמונה ברורה יותר עם פנים ישרות מול המצלמה.")
                    st.session_state.pop("results", None)
                else:
                    # --- ניתוח צורת פנים ---
                    measurements = detector.extract_measurements(landmarks)
                    shape = classify_face_shape(measurements)
                    shape_info = FACE_SHAPE_DISPLAY[shape]

                    # --- ניתוח צבעים (לפני המנוע כי הוא צריך את האנדרטון) ---
                    color_profile = analyze_skin_tone(temp_path, landmarks)

                    # --- תמונה עם landmarks מסומנים ---
                    img_cv = cv2.imread(temp_path)
                    h, w = img_cv.shape[:2]
                    overlay = img_cv.copy()
                    for idx in LANDMARK_INDICES.values():
                        pt = landmarks[idx]
                        x, y = int(pt.x * w), int(pt.y * h)
                        cv2.circle(overlay, (x, y), 4, (238, 211, 34), -1)  # תכלת ב-BGR
                    marked_path = "data/_marked_temp.jpg"
                    cv2.imwrite(marked_path, overlay)

                    # --- המלצות + Try-On לשתי הקטגוריות (שמש/ראייה), כדי שמעבר קטגוריה יהיה מיידי ---
                    engine = RecommendationEngineV2()
                    by_cat = {}
                    for cat in ("vision", "sun"):
                        scored_frames = engine.recommend(
                            shape,
                            undertone=color_profile.undertone,
                            known_frame=known_frame,
                            category=cat,
                            max_items=5,
                        )
                        lst = []
                        for sf in scored_frames:
                            frame = sf.frame
                            tryon_path = None
                            if os.path.exists(frame["image_png"]):
                                tryon_img = overlay_glasses(img_cv, landmarks, frame["image_png"])
                                if tryon_img is not None:
                                    tryon_path = f"data/_tryon_{frame['id']}.jpg"
                                    cv2.imwrite(tryon_path, tryon_img)
                            lst.append({
                                "frame": frame,
                                "overall_match_pct": sf.overall_match_pct,
                                "shape_match_pct": sf.shape_match_pct,
                                "color_match_pct": sf.color_match_pct,
                                "material_match_pct": sf.material_match_pct,
                                "thickness_match_pct": sf.thickness_match_pct,
                                "size_match_pct": sf.size_match_pct,
                                "tryon_path": tryon_path,
                            })
                        by_cat[cat] = lst

                    # --- שמירה ל-session_state כדי לשרוד reruns ---
                    st.session_state.results = {
                        "shape": shape,
                        "shape_info": shape_info,
                        "color_profile": color_profile.__dict__,
                        "measurements": measurements.as_dict(),
                        "by_cat": by_cat,
                        "marked_path": marked_path,
                    }
                    st.session_state.frame_page = 0
                    st.session_state.selected_frame = 0
                    st.session_state.show_landmarks = False
                    st.session_state.active_category = category

            except FileNotFoundError as e:
                st.error(str(e))
else:
    st.info("⬆️ העלו תמונה או צלמו כדי להתחיל")


# ---------------------------------------------------------------------------
# שלב 2: רינדור התוצאות מ-session_state (רץ בכל rerun, לא רק בלחיצת הכפתור)
# ---------------------------------------------------------------------------
results = st.session_state.get("results")
if results:
    shape_info = results["shape_info"]
    color_profile = results["color_profile"]
    scored = results["by_cat"].get(category, [])

    # אתחול הגנתי (למקרה שהמפתחות התאפסו)
    st.session_state.setdefault("frame_page", 0)
    st.session_state.setdefault("selected_frame", 0)
    st.session_state.setdefault("show_landmarks", False)

    # החלפת קטגוריה (שמש<->ראייה) מאפסת את הבחירה והעמוד
    if st.session_state.get("active_category") != category:
        st.session_state.active_category = category
        st.session_state.selected_frame = 0
        st.session_state.frame_page = 0
    if st.session_state.selected_frame >= len(scored):
        st.session_state.selected_frame = 0

    if not scored:
        st.info(f"אין מסגרות בקטגוריה '{CATEGORY_LABELS[category]}' בקטלוג.")
        st.stop()

    st.divider()

    center_col, right_col = st.columns([2, 1])

    # --- עמודה מרכזית: תמונת ההדמיה / ה-landmarks בגדול ---
    with center_col:
        selected = scored[st.session_state.selected_frame]
        show_landmarks = st.session_state.show_landmarks

        if show_landmarks:
            st.image(results["marked_path"], use_container_width=True)
            st.markdown(
                f"<div class='ofa-center-caption'>ניתוח AI · צורת פנים: "
                f"{shape_info['icon']} {shape_info['label']}</div>",
                unsafe_allow_html=True,
            )
        else:
            center_image = selected["tryon_path"] or results["marked_path"]
            st.image(center_image, use_container_width=True)
            if selected["tryon_path"]:
                st.markdown(
                    f"<div class='ofa-center-caption'>{selected['frame']['brand']} "
                    f"{selected['frame']['model']} · "
                    f"<span class='pct'>{selected['overall_match_pct']}% Overall Match</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.info("אין תמונת מסגרת שקופה להדמיה עבור פריט זה - מוצג ניתוח ה-landmarks.")

        toggle_label = "👓 הצג הדמיה" if show_landmarks else "🔬 הצג ניתוח AI (landmarks)"
        if st.button(toggle_label, use_container_width=True, key="toggle_view"):
            st.session_state.show_landmarks = not show_landmarks
            st.rerun()

    # --- עמודה ימנית: כרטיסי מסגרות, 2 בכל פעם ---
    with right_col:
        st.markdown(
            f"<div style='color:{TEXT_LIGHT}; font-weight:700; font-size:1.1em; "
            f"margin-bottom:0.4em;'>✨ מסגרות מומלצות · {CATEGORY_LABELS[category]}</div>",
            unsafe_allow_html=True,
        )

        total_pages = max(1, (len(scored) + FRAMES_PER_PAGE - 1) // FRAMES_PER_PAGE)
        page = min(st.session_state.frame_page, total_pages - 1)
        start = page * FRAMES_PER_PAGE
        page_frames = scored[start:start + FRAMES_PER_PAGE]

        for offset, sf in enumerate(page_frames):
            idx = start + offset
            frame = sf["frame"]
            is_selected = idx == st.session_state.selected_frame
            card_class = "ofa-frame-card ofa-frame-card-selected" if is_selected else "ofa-frame-card"

            st.markdown(f"<div class='{card_class}'>", unsafe_allow_html=True)
            if os.path.exists(frame["image_png"]):
                st.image(frame["image_png"], use_container_width=True)
            st.markdown(
                f"<div class='ofa-frame-name'>{frame['brand']} {frame['model']}"
                f"{' ✓' if is_selected else ''}</div>"
                f"<div class='ofa-frame-sub'>{frame.get('color_name', '')} · "
                f"{frame.get('material', '')}</div>"
                f"{_score_bar_html(sf['overall_match_pct'])}"
                f"<div style='color:{TEXT_MUTED}; font-size:0.8em; margin-top:2px;'>"
                f"Overall {sf['overall_match_pct']}%</div>",
                unsafe_allow_html=True,
            )
            if st.button("נסי אותה 👓", key=f"try_{idx}", use_container_width=True):
                st.session_state.selected_frame = idx
                st.session_state.show_landmarks = False
                st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        # --- ניווט בין עמודים (קודם/הבא) ---
        nav_prev, nav_info, nav_next = st.columns([1, 1, 1])
        with nav_prev:
            if st.button("▲ קודם", use_container_width=True, disabled=(page <= 0), key="prev_page"):
                st.session_state.frame_page = page - 1
                st.rerun()
        with nav_info:
            st.markdown(
                f"<div style='text-align:center; color:{TEXT_MUTED}; padding-top:0.5em;'>"
                f"{page + 1}/{total_pages}</div>",
                unsafe_allow_html=True,
            )
        with nav_next:
            if st.button("הבא ▼", use_container_width=True, disabled=(page >= total_pages - 1), key="next_page"):
                st.session_state.frame_page = page + 1
                st.rerun()

    # --- ניתוח צבעים (רוחב מלא, מתחת ל-Hero) ---
    st.divider()
    st.subheader(f"🎨 ניתוח צבעים אישי: {color_profile['season']}")
    if color_profile.get("confidence") == "low":
        st.warning(f"⚠️ ביטחון נמוך בניתוח הצבעים - {color_profile.get('confidence_reason', '')}")
    elif color_profile.get("confidence") == "medium":
        st.caption(f"ℹ️ ביטחון בינוני בניתוח הצבעים - {color_profile.get('confidence_reason', '')}")
    st.write(color_profile["explanation"])

    swatches_html = "".join(
        f'<span class="ofa-swatch" style="background:{c};"></span>' for c in color_profile["palette_hex"]
    )
    st.markdown(swatches_html, unsafe_allow_html=True)

    st.write("")
    metals_html = "".join(f'<span class="ofa-tag">✨ {m}</span>' for m in color_profile["recommended_metals"])
    colors_html = "".join(f'<span class="ofa-tag">{c}</span>' for c in color_profile["recommended_frame_colors"])
    st.markdown(f"**מתכות מומלצות:** {metals_html}", unsafe_allow_html=True)
    st.markdown(f"**גווני מסגרת מומלצים:** {colors_html}", unsafe_allow_html=True)

    # --- פירוט הציונים של המסגרת הנבחרת + מדדים טכניים ---
    with st.expander("🔧 פירוט ציונים ומדדים טכניים (לצורך הצגה/דיבוג)"):
        sel = scored[st.session_state.selected_frame]
        breakdown = (
            f"צורה {sel['shape_match_pct']}% · צבע {sel['color_match_pct']}% · "
            f"חומר {sel['material_match_pct']}% · עובי {sel['thickness_match_pct']}%"
        )
        if sel["size_match_pct"] is not None:
            breakdown += f" · מידה {sel['size_match_pct']}%"
        st.write(f"**{sel['frame']['brand']} {sel['frame']['model']}** — {breakdown}")
        st.json(results["measurements"])
        st.json(color_profile)
