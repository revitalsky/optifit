"""
server.py
-----------
שרת קל-משקל (ספריית תקן בלבד, בלי התקנות) עבור חוויית ה-Front-End הפרימיום
של OptiFit. אחראי על:
  1. הגשת האתר הסטטי (web/) + נכסי המסגרות (assets/) + מודל ה-Face Landmarker.
  2. נקודת קצה POST /api/analyze - מריצה את *אותה* לוגיקת AI קיימת
     (FaceLandmarkDetector, classify_face_shape, analyze_skin_tone,
      RecommendationEngineV2) ומחזירה תובנות אנגלית + Top-3 המלצות.
  3. GET /api/frames?category=... - הקטלוג המלא לקטגוריה (לכל המלאי).

הרצה:  python server.py     ואז פותחים http://localhost:8000
כל לוגיקת ה-AI/הניקוד/הקטלוג נשארת ללא שינוי - זו רק שכבת הצגה חדשה.
"""

import base64
import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

from face_landmarks import FaceLandmarkDetector, LANDMARK_INDICES  # noqa: E402
from face_shape_classifier import classify_face_shape  # noqa: E402
from color_analysis import analyze_skin_tone  # noqa: E402
from recommendation_engine_v2 import RecommendationEngineV2, CATALOG_PATH  # noqa: E402
import insights as insights_mod  # noqa: E402
import frame_shape_advice as shape_advice  # noqa: E402

PORT = int(os.environ.get("OPTIFIT_PORT", "8000"))

# --- אתחול חד-פעמי של רכיבי ה-AI (יקר לטעון, שומרים בזיכרון) ---
_detector = None
_engine = None


def get_detector():
    global _detector
    if _detector is None:
        _detector = FaceLandmarkDetector()
    return _detector


def get_engine():
    global _engine
    if _engine is None:
        _engine = RecommendationEngineV2()
    return _engine


def _frame_to_public(frame):
    """ממיר רשומת קטלוג לפריט מלאי מוצג (נתיב תמונה ציבורי + מידות פיזיות)."""
    return {
        "id": frame["id"],
        "name": frame["model"],
        "model": frame["model"],
        "brand": frame.get("brand", "OptiFit"),
        "color": frame.get("color_name", ""),
        "color_name": frame.get("color_name", ""),
        "color_family": frame.get("color_family", ""),
        "material": frame.get("material", ""),
        "category": frame.get("category", "vision"),
        "shape": frame.get("shape", ""),
        "rim_thickness": frame.get("rim_thickness", ""),
        "lens_width_mm": frame.get("lens_width_mm"),
        "bridge_width_mm": frame.get("bridge_width_mm"),
        "temple_length_mm": frame.get("temple_length_mm"),
        "universal_score": frame.get("universal_score", 0.7),
        "style_tags": frame.get("style_tags", []),
        "image": "/" + frame["image_png"].replace("\\", "/"),
    }


def analyze_image(image_bytes, category, known_frame=None):
    """מריץ את כל הפייפליין הקיים ומחזיר dict תשובה ל-Front-End."""
    tmp = os.path.join(ROOT, "data", "_api_upload.jpg")
    with open(tmp, "wb") as f:
        f.write(image_bytes)

    detector = get_detector()
    landmarks = detector.detect(tmp)
    if landmarks is None:
        return {"ok": False, "error": "no_face"}

    measurements = detector.extract_measurements(landmarks)
    shape = classify_face_shape(measurements)
    color_profile = analyze_skin_tone(tmp, landmarks)

    engine = get_engine()
    scored = engine.recommend(
        shape,
        undertone=color_profile.undertone,
        known_frame=known_frame,
        category=category,
        max_items=3,
    )
    if not scored:
        return {"ok": False, "error": "no_frames"}

    ins = insights_mod.build_insights(color_profile, scored[0])
    top = []
    for rank, sf in enumerate(scored):
        pub = _frame_to_public(sf.frame)
        why = shape_advice.frame_why(sf, shape, color_profile.undertone)
        pub.update({
            "match": sf.overall_match_pct,
            "reason": insights_mod.english_reason(sf),
            # per-frame "Why this frame?" factors, generated from real sub-scores
            "why": why["en"],
            "why_he": why["he"],
            "scores": {
                "shape": sf.shape_match_pct,
                "color": sf.color_match_pct,
                "material": sf.material_match_pct,
                "fit": sf.size_match_pct if sf.size_match_pct is not None else sf.thickness_match_pct,
            },
            "availability": "In stock",
            "best_match": rank == 0,
        })
        top.append(pub)

    frame_shapes = shape_advice.recommended_frame_shapes(shape)

    return {
        "ok": True,
        "face_shape": insights_mod.FACE_SHAPE_EN.get(shape, shape.title()),
        "skin_tone": ins["skin_tone"],
        "recommended_colors": ins["recommended_colors"],
        "recommended_metals": ins["recommended_metals"],
        "frame_width": ins["frame_width"],
        "bridge_fit": ins["bridge_fit"],
        "lighting_note": ins["lighting_note"],
        # --- color analysis (for the web front-end swatches + deep view) ---
        "palette_hex": color_profile.palette_hex,
        "color_season": color_profile.season_en,
        "color_season_he": color_profile.season,
        "color_explanation": color_profile.explanation,
        "color_undertone": color_profile.undertone,
        "color_depth": color_profile.depth,
        "color_confidence": color_profile.confidence,
        "recommended_frame_colors": color_profile.recommended_frame_colors,
        # --- recommended frame SHAPES (ranked, tiered, with plain-language why) ---
        "face_shape_key": shape,
        "recommended_shapes": frame_shapes["recommended"],
        "shape_cautions": frame_shapes["cautions"],
        "top": top,
    }


# --- מיפוי סיומת -> Content-Type ---
CTYPES = {
    ".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8", ".json": "application/json; charset=utf-8",
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml", ".task": "application/octet-stream", ".ico": "image/x-icon",
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # שקט

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, obj, code=200):
        self._send(code, json.dumps(obj).encode("utf-8"))

    def _serve_file(self, rel_path):
        # אבטחה: מונע יציאה מחוץ לתיקיית הפרויקט
        full = os.path.normpath(os.path.join(ROOT, rel_path.lstrip("/")))
        if not full.startswith(ROOT) or not os.path.isfile(full):
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        ext = os.path.splitext(full)[1].lower()
        with open(full, "rb") as f:
            data = f.read()
        self._send(200, data, CTYPES.get(ext, "application/octet-stream"))

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "":
            self._serve_file("web/index.html")
        elif path == "/model":
            self._serve_file("assets/face_landmarker.task")
        elif path == "/api/frames":
            qs = parse_qs(parsed.query)
            cat = (qs.get("category", ["vision"])[0])
            catalog = json.load(open(CATALOG_PATH, encoding="utf-8"))
            if cat == "all":
                frames = [_frame_to_public(f) for f in catalog]
            else:
                frames = [_frame_to_public(f) for f in catalog if f.get("category") == cat]
            self._send_json({"ok": True, "frames": frames})
        elif path.startswith("/assets/") or path.startswith("/web/"):
            self._serve_file(path)
        elif path in ("/style.css", "/app.js"):
            self._serve_file("web" + path)
        else:
            self._send(404, b"Not found", "text/plain; charset=utf-8")

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/analyze":
            self._send(404, b"Not found", "text/plain; charset=utf-8")
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            payload = json.loads(raw.decode("utf-8"))
            data_url = payload.get("image", "")
            category = payload.get("category", "vision")
            known = payload.get("known_frame") or None
            if "," in data_url:
                data_url = data_url.split(",", 1)[1]
            image_bytes = base64.b64decode(data_url)
            result = analyze_image(image_bytes, category, known_frame=known)
            self._send_json(result)
        except Exception as e:  # תשובה נקייה ללקוח במקום קריסה
            self._send_json({"ok": False, "error": "server_error", "detail": str(e)}, code=500)


def main():
    print("OptiFit - warming up AI models...")
    get_detector()
    get_engine()
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"OptiFit is running ->  http://localhost:{PORT}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
