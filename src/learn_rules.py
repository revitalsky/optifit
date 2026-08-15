"""
learn_rules.py
----------------
קורא את טבלת הדוגמאות המתויגות (data/expert_labels.csv) שאת ממלאת,
ובונה ממנה "מטריצת ציונים" - לכל צירוף (צורת פנים, צורת מסגרת) ציון
התאמה מספרי, המבוסס על ריבוי הדוגמאות שסימנת ולא על חוקים קשיחים
שנכתבו מראש.

זהו מנגנון "הלמידה" של המערכת: ככל שתוסיפי דוגמאות רבות ועקביות יותר,
הציונים משתפרים ומדויקים יותר - בלי לאמן רשת נוירונים, אלא בעזרת
צבירה סטטיסטית פשוטה וברורה (Weighted Rule Learning).

איך מריצים:
    python3 src/learn_rules.py
זה קורא את data/expert_labels.csv (או expert_labels_template.csv אם
הראשון לא קיים) וכותב את data/learned_rules.json.
הרצה מחדש דורשת רק להריץ שוב את הסקריפט הזה אחרי כל עדכון של הטבלה.
"""

import csv
import json
import os
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CSV_PATH_MAIN = os.path.join(DATA_DIR, "expert_labels.csv")
CSV_PATH_TEMPLATE = os.path.join(DATA_DIR, "expert_labels_template.csv")
OUTPUT_PATH = os.path.join(DATA_DIR, "learned_rules.json")

VERDICT_SCORES = {"good": 1.0, "neutral": 0.0, "bad": -1.0, "excellent": 1.5}


def _resolve_csv_path():
    if os.path.exists(CSV_PATH_MAIN):
        return CSV_PATH_MAIN
    if os.path.exists(CSV_PATH_TEMPLATE):
        return CSV_PATH_TEMPLATE
    raise FileNotFoundError(
        "לא נמצא קובץ דוגמאות. יש ליצור את data/expert_labels.csv עם העמודות: "
        "face_shape, frame_shape, verdict, notes"
    )


def learn_rules(csv_path: str = None) -> dict:
    """קורא את הטבלה ובונה מטריצת ציונים + הערות שנצברו לכל צירוף."""
    csv_path = csv_path or _resolve_csv_path()

    # מבנה ביניים: {face_shape: {frame_shape: [scores...]}}
    raw_scores = defaultdict(lambda: defaultdict(list))
    raw_notes = defaultdict(lambda: defaultdict(list))

    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            face_shape = row["face_shape"].strip().lower()  # מפתח פנימי - תמיד באנגלית קטנה
            frame_shape = row["frame_shape"].strip()  # שומר את הכתיב המדויק כפי שמופיע בקטלוג (data/catalog.json)
            verdict = row["verdict"].strip().lower()
            notes = (row.get("notes") or "").strip()

            if verdict not in VERDICT_SCORES:
                continue  # מדלג על שורות עם ערך לא מוכר, לא מפיל את הריצה
            raw_scores[face_shape][frame_shape].append(VERDICT_SCORES[verdict])
            if notes:
                raw_notes[face_shape][frame_shape].append(notes)

    # ממוצע לכל צירוף + מספר הדוגמאות שתמכו בו (confidence)
    learned = {}
    for face_shape, frame_map in raw_scores.items():
        learned[face_shape] = {}
        for frame_shape, scores in frame_map.items():
            learned[face_shape][frame_shape] = {
                "score": round(sum(scores) / len(scores), 3),
                "sample_count": len(scores),
                "notes": raw_notes[face_shape][frame_shape],
            }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(learned, f, ensure_ascii=False, indent=2)

    return learned


if __name__ == "__main__":
    result = learn_rules()
    print(f"נלמדו חוקים עבור {len(result)} צורות פנים:\n")
    for face_shape, frame_map in result.items():
        ranked = sorted(frame_map.items(), key=lambda kv: kv[1]["score"], reverse=True)
        print(f"— {face_shape} —")
        for frame_shape, info in ranked:
            print(f"    {frame_shape:15s} ציון: {info['score']:+.2f}  (מבוסס על {info['sample_count']} דוגמאות)")
    print(f"\nנשמר ב-{OUTPUT_PATH}")
