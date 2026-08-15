"""
virtual_tryon.py
-------------------
מודול "הדמיה ויזואלית" (Try-On): מדביק תמונת מסגרת (PNG, רצוי עם ערוץ שקיפות)
על גבי תמונת הפנים של המשתמש, במיקום ובזווית שמחושבים מתוך ה-landmarks
שכבר זוהו על ידי face_landmarks.py (קו העיניים).

לא מודל AI - הדבקה גיאומטרית פשוטה (מיקום, קנה מידה וסיבוב) עקבית עם
אופי הפרויקט המבוסס-חוקים.
"""

import math

import cv2
import numpy as np

from face_landmarks import LANDMARK_INDICES


def overlay_glasses(image_bgr, landmarks, frame_image_path, width_scale=1.6, vertical_offset_ratio=0.15):
    """
    מדביק את תמונת המסגרת שב-frame_image_path על image_bgr, ממורכז על קו העיניים
    (לפי left_eye_outer / right_eye_outer מתוך landmarks), עם קנה מידה וסיבוב
    שמותאמים למרחק ולזווית שבין העיניים.

    מחזיר תמונת cv2 (BGR) חדשה עם ההדבקה, או None אם קובץ המסגרת לא נמצא/לא קריא.
    """
    frame_img = cv2.imread(frame_image_path, cv2.IMREAD_UNCHANGED)
    if frame_img is None:
        return None

    # אם לתמונת המסגרת אין ערוץ אלפא (שקיפות), מוסיפים אחד מלא (הדבקה מלבנית מלאה)
    if frame_img.ndim == 2:
        frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGRA)
    elif frame_img.shape[2] == 3:
        alpha_channel = np.full(frame_img.shape[:2], 255, dtype=np.uint8)
        frame_img = np.dstack([frame_img, alpha_channel])

    h, w = image_bgr.shape[:2]
    li = LANDMARK_INDICES
    left_eye = landmarks[li["left_eye_outer"]]
    right_eye = landmarks[li["right_eye_outer"]]

    left_pt = np.array([left_eye.x * w, left_eye.y * h])
    right_pt = np.array([right_eye.x * w, right_eye.y * h])
    center = (left_pt + right_pt) / 2

    eye_distance_px = float(np.linalg.norm(right_pt - left_pt))
    if eye_distance_px <= 0:
        return None
    angle_deg = -math.degrees(math.atan2(right_pt[1] - left_pt[1], right_pt[0] - left_pt[0]))

    target_width = int(eye_distance_px * width_scale)
    scale = target_width / frame_img.shape[1]
    target_height = int(frame_img.shape[0] * scale)
    if target_width <= 0 or target_height <= 0:
        return None

    resized = cv2.resize(frame_img, (target_width, target_height), interpolation=cv2.INTER_AREA)

    rot_mat = cv2.getRotationMatrix2D((target_width / 2, target_height / 2), angle_deg, 1.0)
    rotated = cv2.warpAffine(
        resized, rot_mat, (target_width, target_height),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )

    result = image_bgr.copy()
    top_left_x = int(center[0] - target_width / 2)
    top_left_y = int(center[1] - target_height / 2 + eye_distance_px * vertical_offset_ratio)

    _alpha_blend(result, rotated, top_left_x, top_left_y)
    return result


def _alpha_blend(background, overlay_rgba, x, y):
    """מדביק overlay_rgba (BGRA) על background (BGR) במיקום (x, y), עם חיתוך לגבולות התמונה."""
    h, w = overlay_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]

    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, bg_w), min(y + h, bg_h)
    if x0 >= x1 or y0 >= y1:
        return

    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)

    overlay_crop = overlay_rgba[oy0:oy1, ox0:ox1]
    alpha = overlay_crop[:, :, 3:4].astype(float) / 255.0
    overlay_rgb = overlay_crop[:, :, :3].astype(float)
    bg_crop = background[y0:y1, x0:x1].astype(float)

    blended = overlay_rgb * alpha + bg_crop * (1 - alpha)
    background[y0:y1, x0:x1] = blended.astype(np.uint8)
