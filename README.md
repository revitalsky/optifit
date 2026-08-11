# OptiFit AI — אבטיפוס

מערכת חכמה להתאמת מסגרות משקפיים באמצעות ניתוח פנים.
פרויקט גמר, מכללת אשקלון.

## מבנה הפרויקט

```
OPTI_FIT PROJECT/
├── app.py                         # ממשק Streamlit (עיצוב כהה + תכלת, Try-On, ציוני התאמה)
├── download_model.sh              # הורדה חד-פעמית של מודל MediaPipe
├── requirements.txt
├── .streamlit/
│   └── config.toml                # theme כהה גלובלי (רקע כהה + תכלת)
├── src/
│   ├── face_landmarks.py          # זיהוי פנים + חילוץ מדדים (MediaPipe)
│   ├── face_shape_classifier.py   # סיווג צורת פנים (מבוסס חוקים)
│   ├── face_shape_display.py      # מידע תצוגתי לכל צורת פנים (אייקון/תיאור)
│   ├── color_analysis.py          # ניתוח גוון עור -> עונת צבעים (עם ציון ביטחון)
│   ├── recommendation_engine_v2.py# מנוע המלצות משוקלל (Shape/Color/Material/Thickness/Size, מסונן לפי קטגוריה)
│   ├── recommendation_engine.py   # מנוע v1 מקורי (נשמר, לא בשימוש חי)
│   ├── virtual_tryon.py           # הדבקת מסגרת על תמונת הפנים (Try-On)
│   ├── frame_prep.py              # חיתוך חזית + השמטת מוטות + זיהוי שמש/ראייה אוטומטי
│   └── learn_rules.py             # למידת חוקים מטבלת מומחה (הרחבה עתידית)
├── data/
│   ├── catalog_v2.json            # קטלוג מסגרות (כולל שדה category: sun/vision + מידות/חומר/צבע)
│   ├── rules_v2.json              # סכמת החוקים והמשקלים למנוע v2
│   ├── catalog.json               # קטלוג v1 (נשמר)
│   ├── frame_rules.json           # חוקי v1 (נשמר)
│   └── expert_labels_template.csv # תבנית תיוג מומחה (להרחבה עתידית)
├── assets/
│   ├── face_landmarker.task       # מודל MediaPipe (מקומי)
│   └── frames/
│       ├── sun/                   # 8 תמונות משקפי שמש (PNG שקוף, חתוך לחזית)
│       └── vision/                # 10 תמונות משקפי ראייה (PNG שקוף, חתוך לחזית)
└── _backups/                      # גיבויים (גרסאות app.py קודמות + old_frames)
```

## פיצ'רים עיקריים בממשק

- **בחירת קטגוריה** (סרגל צד): המלצות נפרדות למשקפי ראייה או שמש.
- **הוספת מסגרות אוטומטית** (סרגל צד): מעלים תמונת PNG שקוף → המערכת מזהה לבד אם זו
  שמש או ראייה (לפי שקיפות העדשה), חותכת לחזית בלבד (משמיטה מוטות) ומוסיפה לקטלוג.
- **Try-On**: לחיצה על כרטיס מסגרת מדביקה אותה על הפנים במרכז, בהתאמה ל-landmarks.

## הרצה (על המחשב שלכם, עם אינטרנט)

```bash
python -m streamlit run app.py
```

> הערה: הרצה עם `python -m streamlit` (ולא `streamlit run`) — סביבת ה-`.venv` נוצרה במקור
> בתיקייה בשם אחר, ולכן קובצי ה-`.exe` של הסביבה (כמו `streamlit.exe`) מצביעים לנתיב הישן.
> ההרצה דרך `python -m streamlit` תמיד עובדת ללא תלות בכך. אם רוצים ליצור סביבה נקייה:
> ```bash
> python -m venv .venv
> .venv\Scripts\python -m pip install -r requirements.txt
> bash download_model.sh
> ```

## מה כבר עובד

- ✅ זיהוי פנים (MediaPipe) + חילוץ מדדים + סיווג ל-5 צורות פנים
- ✅ ניתוח צבעים עם ציון ביטחון (low/medium/high) — מזהה תאורה בעייתית
- ✅ מנוע המלצות v2: לכל מסגרת ציוני Shape/Color/Material/Thickness ו-Overall Match משוקלל
- ✅ Size Match אופציונלי — רק אם המשתמשת מזינה מידה של משקפיים נוחים לה
- ✅ הדמיה ויזואלית (Try-On) — לחיצה על כרטיס מסגרת מציגה אותה על הפנים במרכז
- ✅ ממשק כהה בהשראת מוקאאפ "Advanced AI Eyewear Fitting" (רקע כהה + דגשי תכלת)

## הערות

- תמונות מסגרות ל-Try-On עדיף שיהיו PNG עם רקע שקוף. חלק מהתמונות הנוכחיות על רקע לבן —
  אפשר להחליף אותן בתמונות שקופות ב-`assets/frames/` (בהתאם לשמות הקבצים ב-`catalog_v2.json`).
- שילוב `learn_rules.py` / `expert_labels` במנוע נשאר כהרחבה עתידית.
