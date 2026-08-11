#!/bin/bash
# מוריד פעם אחת את מודל זיהוי הפנים של MediaPipe (כ-3.7MB, קובץ .task)
# יש להריץ סקריפט זה פעם אחת, במחשב עם חיבור לאינטרנט, לפני הרצת האפליקציה
set -e
mkdir -p assets
curl -L -o assets/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
echo "המודל הורד בהצלחה ל-assets/face_landmarker.task"
