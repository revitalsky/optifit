"""
frame_prep.py
---------------
עיבוד תמונות מסגרות עבור OptiFit AI (מבוסס-חוקים, ללא מודל AI מאומן):

- crop_front:        חיתוך אוטומטי לחזית המסגרת בלבד (השמטת מוטות/זרועות צד),
                     לפי ערוץ השקיפות - המוטות הן הרחבות "דקות" בקצוות והעדשות
                     הן עמודות "גבוהות".
- classify_category: זיהוי אוטומטי אם זו משקפי שמש (עדשות כהות/צבעוניות = אטומות)
                     או משקפי ראייה (עדשות שקופות), לפי השקיפות במרכז העדשה.
- guess_*:           ניחוש-מיטבי של צבע/צורה/עובי מסגרת לתמונות שמועלות אוטומטית.
- prepare_frame:     פייפליין מלא (חיתוך + הקטנה + שמירה) + החזרת מטא-דאטה.

השמות מיועדים לשימוש גם בבניית הקטלוג הראשוני וגם בהעלאת מסגרות חדשות מהממשק.
"""

import os

import numpy as np
from PIL import Image

ALPHA_THR = 20          # סף שקיפות לזיהוי "יש כאן פיקסל של מסגרת"
LENS_ALPHA_SUN = 0.5    # מעל זה במרכז העדשה => עדשה אטומה => שמש


def to_rgba_array(img):
    """מקבל נתיב / PIL.Image / np.ndarray ומחזיר מערך RGBA (H,W,4)."""
    if isinstance(img, str):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGBA"))
    return img


def _content_bbox(mask):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def is_empty(img):
    """True אם התמונה ריקה לגמרי (בלי שום פיקסל אטום) - קובץ פגום/ריק."""
    arr = to_rgba_array(img)
    return bool((arr[:, :, 3] > ALPHA_THR).sum() == 0)


def crop_front(img, height_ratio=0.45):
    """
    מחזיר PIL.Image (RGBA) חתוך לחזית המסגרת בלבד (בלי מוטות צד).
    height_ratio: עמודה נחשבת "חזית" אם גובה הפיקסלים האטומים בה >= היחס הזה
    מהמקסימום. המוטות (דקים) נופלים מתחת לסף ולכן נחתכים.
    """
    arr = to_rgba_array(img)
    mask = arr[:, :, 3] > ALPHA_THR
    bb = _content_bbox(mask)
    if bb is None:
        return Image.fromarray(arr)
    x0, y0, x1, y1 = bb
    sub = mask[y0:y1, x0:x1]
    col_height = sub.sum(axis=0)
    if col_height.max() == 0:
        return Image.fromarray(arr[y0:y1, x0:x1])
    thr = height_ratio * col_height.max()
    tall = np.where(col_height >= thr)[0]
    fx0 = x0 + int(tall.min())
    fx1 = x0 + int(tall.max()) + 1
    band = mask[:, fx0:fx1]
    ys = np.where(band.any(axis=1))[0]
    fy0, fy1 = int(ys.min()), int(ys.max()) + 1
    return Image.fromarray(arr[fy0:fy1, fx0:fx1])


def _lens_patches(arr):
    """מחזיר את פיקסלי מרכז שתי העדשות (לצורך זיהוי שמש/ראייה)."""
    mask = arr[:, :, 3] > ALPHA_THR
    bb = _content_bbox(mask)
    if bb is None:
        return None
    x0, y0, x1, y1 = bb
    w, h = x1 - x0, y1 - y0
    cy = y0 + h // 2
    patches = []
    for fx in (0.27, 0.73):
        cx = int(x0 + w * fx)
        p = arr[max(0, cy - h // 8):cy + h // 8, max(0, cx - w // 12):cx + w // 12]
        if p.size:
            patches.append(p.reshape(-1, 4))
    if not patches:
        return None
    return np.concatenate(patches, axis=0)


def classify_category(img):
    """
    מחזיר ('sun'|'vision', mean_lens_alpha).
    עדשות שמש אטומות (alpha גבוה במרכז), עדשות ראייה שקופות (alpha נמוך).
    """
    arr = to_rgba_array(img)
    patch = _lens_patches(arr)
    if patch is None:
        return "vision", 0.0
    mean_alpha = float(patch[:, 3].mean() / 255.0)
    return ("sun" if mean_alpha > LENS_ALPHA_SUN else "vision"), mean_alpha


def guess_shape(img):
    """ניחוש צורה גס לפי יחס רוחב/גובה של החזית (רק להעלאות אוטומטיות)."""
    front = crop_front(img)
    w, h = front.size
    ratio = w / max(1, h)
    if ratio > 3.4:
        return "rectangle"
    if ratio > 2.7:
        return "soft_square"
    if ratio > 2.2:
        return "oval"
    return "round"


def guess_color_family(img):
    """ניחוש משפחת צבע לפי פיקסלי המסגרת האטומים (רק להעלאות אוטומטיות)."""
    arr = to_rgba_array(img).astype(np.float32)
    opaque = arr[:, :, 3] > 200
    if opaque.sum() == 0:
        return "black"
    rgb = arr[:, :, :3][opaque]
    r, g, b = rgb[:, 0].mean(), rgb[:, 1].mean(), rgb[:, 2].mean()
    mx, mn = max(r, g, b), min(r, g, b)
    val = mx / 255.0
    sat = 0 if mx == 0 else (mx - mn) / mx
    if val < 0.28:
        return "black"
    if sat < 0.18:
        return "silver" if val > 0.55 else "gray"
    if r > g > b:
        return "gold" if val > 0.6 else ("tortoise" if val > 0.35 else "brown")
    if b >= r:
        return "navy"
    return "brown"


def guess_rim_thickness(img):
    """ניחוש עובי מסגרת לפי שטח החומר האטום ביחס לתיבת החזית."""
    front = to_rgba_array(crop_front(img))
    mask = front[:, :, 3] > 200
    fill = mask.mean()
    if fill > 0.42:
        return "thick"
    if fill > 0.24:
        return "medium"
    return "thin"


def prepare_frame(src, out_path, max_width=1000):
    """
    פייפליין מלא לתמונת מסגרת חדשה:
    חיתוך חזית (בלי מוטות) -> הקטנה למקסימום רוחב -> שמירת PNG שקוף ל-out_path.
    מחזיר dict עם המטא-דאטה שזוהתה (category/shape/color_family/rim_thickness).
    מרים ValueError אם התמונה ריקה.
    """
    if is_empty(src):
        raise ValueError("התמונה ריקה (בלי פיקסלים אטומים) - כנראה קובץ פגום")
    category, _ = classify_category(src)
    shape = guess_shape(src)
    color_family = guess_color_family(src)
    rim = guess_rim_thickness(src)

    front = crop_front(src)
    if front.width > max_width:
        r = max_width / front.width
        front = front.resize((max_width, int(front.height * r)), Image.LANCZOS)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    front.save(out_path)
    return {"category": category, "shape": shape, "color_family": color_family, "rim_thickness": rim}
