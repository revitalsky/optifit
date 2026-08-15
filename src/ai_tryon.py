"""
ai_tryon.py
-----------
הדמיית משקפיים ריאליסטית מבוססת AI, בעזרת מנוע עריכת התמונות של OpenAI
(gpt-image-1 - אותו מנוע שמאחורי יצירת התמונות ב-ChatGPT).

בשונה מ-virtual_tryon.py (הדבקה גיאומטרית מהירה של PNG על קו העיניים),
כאן שולחים ל-OpenAI את תמונת הפנים + תמונת המסגרת, ומבקשים ממנו "להלביש"
את המשקפיים על הפנים בצורה ריאליסטית (מיקום, פרספקטיבה, הצללות).

המודול עצמאי (לא תלוי ב-Streamlit) וניתן לבדיקה ישירה מה-CLI.
"""

import base64
import os

# ה-SDK של OpenAI מיובא בתוך הפונקציה כדי שהמודול (וה-import ב-app.py)
# לא ייכשל אם החבילה עדיין לא הותקנה - נחזיר שגיאה ידידותית רק בזמן שימוש.

MODEL = "gpt-image-1"

PROMPT = (
    "You are given two images. The first image is a photograph of a person's face. "
    "The second image is a product photo of eyeglasses on a transparent or plain "
    "background. Edit the FIRST image so that the person is realistically wearing "
    "these exact eyeglasses. Keep the person's face, identity, skin tone, hair, "
    "expression, pose, lighting and background completely unchanged. Match the "
    "glasses' size, position and perspective naturally to the eyes, and add subtle, "
    "realistic shadows. Only add the glasses - do not change anything else in the image."
)


class AITryOnError(Exception):
    """שגיאה ידידותית (טקסט בעברית) שניתן להציג ישירות למשתמשת בממשק."""


def get_api_key(explicit: str | None = None) -> str | None:
    """
    מחזיר את מפתח ה-OpenAI לפי סדר עדיפות:
    1. explicit - מה שהוזן בשדה בממשק (אם לא ריק)
    2. משתנה הסביבה OPENAI_API_KEY

    מחזיר None אם אין מפתח כלל (הקורא אחראי להציג הודעה מתאימה).
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env_key = os.environ.get("OPENAI_API_KEY")
    if env_key and env_key.strip():
        return env_key.strip()
    return None


def generate_ai_tryon(
    face_image_path: str,
    frame_image_path: str,
    api_key: str,
    *,
    frame_desc: str = "",
    quality: str = "medium",
    size: str = "auto",
) -> bytes:
    """
    מייצר תמונה ריאליסטית של הפנים שב-face_image_path כשהם מרכיבים את המשקפיים
    שב-frame_image_path, בעזרת OpenAI Images Edit (gpt-image-1).

    frame_desc - תיאור טקסטואלי אופציונלי של המסגרת (מותג/דגם/צבע) שמצורף כרמז.
    quality    - "low" / "medium" / "high" / "auto" (איזון עלות מול איכות).
    size       - "auto" / "1024x1024" / "1536x1024" / "1024x1536".

    מחזיר bytes של תמונת PNG.
    זורק AITryOnError עם הודעה בעברית בכל מקרה כשל (מפתח, תשלום, מדיניות, רשת).
    """
    if not os.path.exists(face_image_path):
        raise AITryOnError(f"לא נמצאה תמונת הפנים בנתיב: {face_image_path}")
    if not os.path.exists(frame_image_path):
        raise AITryOnError(f"לא נמצאה תמונת המסגרת בנתיב: {frame_image_path}")

    try:
        from openai import OpenAI
    except ImportError:
        raise AITryOnError(
            "ספריית openai לא מותקנת. הריצי: pip install -r requirements.txt"
        )

    prompt = PROMPT
    if frame_desc.strip():
        prompt += f" The eyeglasses in the second image are: {frame_desc.strip()}."

    client = OpenAI(api_key=api_key)

    face_fp = None
    frame_fp = None
    try:
        face_fp = open(face_image_path, "rb")
        frame_fp = open(frame_image_path, "rb")
        result = client.images.edit(
            model=MODEL,
            image=[face_fp, frame_fp],
            prompt=prompt,
            size=size,
            quality=quality,
            input_fidelity="high",  # שמירה גבוהה על זהות הפנים המקוריות
        )
    except Exception as e:  # noqa: BLE001 - ממפים כל שגיאת SDK/רשת להודעה ידידותית
        raise AITryOnError(_friendly_error(e)) from e
    finally:
        for fp in (face_fp, frame_fp):
            if fp is not None:
                fp.close()

    if not getattr(result, "data", None):
        raise AITryOnError("המנוע לא החזיר תמונה. נסי שוב או עם תמונה אחרת.")

    b64 = result.data[0].b64_json
    if not b64:
        raise AITryOnError("המנוע לא החזיר תמונה תקינה. נסי שוב.")

    return base64.b64decode(b64)


def _friendly_error(e: Exception) -> str:
    """ממפה שגיאת SDK/רשת של OpenAI להודעה קצרה וברורה בעברית."""
    msg = str(e).lower()
    status = getattr(e, "status_code", None)

    if status == 401 or "api key" in msg or "authentication" in msg or "invalid_api_key" in msg:
        return "מפתח ה-OpenAI לא תקין או חסר. בדקי את המפתח והזיני אותו מחדש."
    if status == 429 or "quota" in msg or "insufficient_quota" in msg or "rate limit" in msg:
        return (
            "חריגה מהמכסה או מגבלת הקצב של OpenAI (ייתכן שאין יתרת תשלום בחשבון). "
            "בדקי את החיוב/המכסה בחשבון OpenAI ונסי שוב."
        )
    if "content" in msg and ("policy" in msg or "moderation" in msg or "safety" in msg):
        return (
            "הבקשה נדחתה על-ידי מדיניות התוכן של OpenAI. נסי תמונת פנים אחרת "
            "(ברורה, אדם בוגר, פנים מול המצלמה)."
        )
    if "connection" in msg or "timeout" in msg or "network" in msg:
        return "בעיית תקשורת מול OpenAI. בדקי את חיבור האינטרנט ונסי שוב."
    return f"שגיאה ביצירת ההדמיה: {e}"


if __name__ == "__main__":
    # בדיקה ידנית מהירה מה-CLI:
    #   python src/ai_tryon.py <face.jpg> <frame.png>
    # דורש OPENAI_API_KEY במשתני הסביבה.
    import sys

    if len(sys.argv) < 3:
        print("שימוש: python src/ai_tryon.py <face_image> <frame_png>")
        sys.exit(1)

    key = get_api_key()
    if not key:
        print("חסר OPENAI_API_KEY במשתני הסביבה.")
        sys.exit(1)

    try:
        png_bytes = generate_ai_tryon(sys.argv[1], sys.argv[2], key)
    except AITryOnError as err:
        print("שגיאה:", err)
        sys.exit(1)

    out_path = "data/_ai_tryon_cli_test.png"
    with open(out_path, "wb") as f:
        f.write(png_bytes)
    print(f"נשמר: {out_path} ({len(png_bytes)} bytes)")
