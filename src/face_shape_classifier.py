"""
face_shape_classifier.py
-------------------------
מנגנון סיווג צורת פנים מבוסס חוקים (Rule-Based) - לא מודל AI מאומן.
מקבל מדדי פנים (FaceMeasurements) ומחזיר אחת מ-5 צורות פנים קלאסיות.

זהו בדיוק "מנגנון המלצה פשוט המבוסס על חוקי התאמה מוגדרים מראש" מהטופס.
"""

from face_landmarks import FaceMeasurements

FACE_SHAPES = ["oval", "round", "square", "heart", "diamond"]

FACE_SHAPES_HE = {
    "oval": "אובלי",
    "round": "עגול",
    "square": "מרובע",
    "heart": "לב",
    "diamond": "יהלום",
}


def classify_face_shape(m: FaceMeasurements) -> str:
    """
    מסווג צורת פנים לפי יחסים בין המדדים.
    היחסים מבוססים על כללי אצבע נפוצים בתחום האופטיקה/סטיילינג:

    - אורך פנים משמעותית גדול מרוחב -> אובלי
    - רוחב לחיים גדול משמעותית מרוחב מצח ומהלסת -> יהלום
    - מצח רחב משמעותית מהלסת -> לב
    - כל הרוחבים דומים + פנים ארוכות יחסית לרוחב -> מרובע
    - כל הרוחבים דומים + פנים לא ארוכות (יחס קרוב ל-1) -> עגול
    """
    length_to_width = m.face_length / m.cheekbone_width
    jaw_to_cheek = m.jaw_width / m.cheekbone_width
    forehead_to_cheek = m.forehead_width / m.cheekbone_width
    forehead_to_jaw = m.forehead_width / m.jaw_width

    # יהלום: עצמות הלחיים הן הרחבות ביותר בבירור — גם מהמצח וגם מהלסת.
    # סף מחמיר (< 0.82 לשניהם) כדי לא לתייג פנים רגילות (שבהן הלחיים ממילא
    # מעט רחבות מהמצח והלסת) כיהלום.
    if forehead_to_cheek < 0.82 and jaw_to_cheek < 0.82:
        return "diamond"

    # לב: מצח רחב משמעותית מהלסת, והלסת צרה (מתחדדת לכיוון הסנטר)
    if forehead_to_jaw > 1.18 and jaw_to_cheek < 0.85:
        return "heart"

    # אובלי: פנים ארוכות בבירור ביחס לרוחבן
    if length_to_width >= 1.42:
        return "oval"

    # מרובע: רוחבים דומים (מצח/לחיים/לסת קרובים), לסת חזקה, ואורך בינוני
    if forehead_to_cheek >= 0.86 and jaw_to_cheek >= 0.84 and length_to_width >= 1.15:
        return "square"

    # עגול: רוחבים דומים אך הפנים אינן ארוכות
    if length_to_width <= 1.12:
        return "round"

    # ברירת מחדל למקרי ביניים (אורך בינוני): אובלי
    return "oval"


if __name__ == "__main__":
    # בדיקות ידניות עם מדדים סינתטיים לכל אחת מ-5 הצורות
    test_cases = {
        "oval": FaceMeasurements(forehead_width=0.9, cheekbone_width=1.0, jaw_width=0.85, face_length=1.6, eye_distance=0.4),
        "round": FaceMeasurements(forehead_width=0.95, cheekbone_width=1.0, jaw_width=0.95, face_length=1.1, eye_distance=0.4),
        "square": FaceMeasurements(forehead_width=0.95, cheekbone_width=1.0, jaw_width=0.95, face_length=1.3, eye_distance=0.4),
        "heart": FaceMeasurements(forehead_width=1.0, cheekbone_width=0.95, jaw_width=0.7, face_length=1.3, eye_distance=0.4),
        "diamond": FaceMeasurements(forehead_width=0.8, cheekbone_width=1.0, jaw_width=0.8, face_length=1.3, eye_distance=0.4),
    }
    print("בדיקת סיווג צורות פנים על נתונים סינתטיים:\n")
    correct = 0
    for expected, measurements in test_cases.items():
        result = classify_face_shape(measurements)
        status = "✓" if result == expected else "✗"
        if result == expected:
            correct += 1
        print(f"{status} צפוי: {FACE_SHAPES_HE[expected]:8s} | התקבל: {FACE_SHAPES_HE[result]:8s} | {measurements.as_dict()}")
    print(f"\n{correct}/{len(test_cases)} עברו בהצלחה")
