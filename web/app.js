// OptiFit — AI eyewear fitting front-end.
// Faithful implementation of the OptiFit design (multi-screen, bilingual EN/HE),
// wired to the real Python AI backend:
//   POST /api/analyze  -> real face shape / skin tone / insights / top-3 recommendations
//   GET  /api/frames    -> the measured catalog (per category, or ?category=all)
// Live try-on uses MediaPipe FaceLandmarker in the browser to place frames in real time.

import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";

const MP_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// MediaPipe canonical landmark indices (match the Python engine).
const L_EYE = 33, R_EYE = 263, NOSE_BRIDGE = 168;
const FACE_L = 234, FACE_R = 454;   // temple-to-temple (face width) — frames align to this
const FRAME_W = 1.0;                 // frame width as a fraction of face width
const VERT_NUDGE = 0.05;             // small downward nudge (fraction of frame height)

// ---------------------------------------------------------------------------
// Strings (EN / HE) — transcribed from the OptiFit design.
// ---------------------------------------------------------------------------
const STR = {
  en: {
    home:"Home", tryOnNav:"Try On", collection:"Collection", about:"About OptiFit",
    favorites:"Favorites", profile:"My Profile",
    homeBadge:"AI-powered frame matching",
    homeTitle1:"Find the frames", homeTitle2:"that fit your face.",
    homeSub:"OptiFit analyzes your facial proportions in real time and matches you with frames from our catalog — then lets you wear them live.",
    homeCta1:"Start Live Try-On", homeCta2:"Browse Collection",
    homeLiveTag:"Face analysis active",
    homeStats:[["18","Frames in catalog"],["6","Face shapes supported"],["2s","Analysis time"]],
    homeChips:[["Face shape","Oval"],["Skin tone","Warm"],["Fit score","98%"]],
    homeFeatures:[
      ["01","Real-time face mapping","468 facial landmarks tracked from your camera, mapped to lens width, bridge and temple measurements."],
      ["02","Matched, not guessed","Every frame is scored against your measurements and coloring — you see the score, and why."],
      ["03","Wear before you buy","Frames follow your head as you move, so you judge the fit the way you would in a store."]],
    permTitle:"Enable your camera",
    permSub:"OptiFit needs your camera to map your face and place frames in real time.",
    permPoints:["Video is processed on your device only","Nothing is uploaded or stored","You can revoke access at any time"],
    permAllow:"Allow camera access", permUpload:"Use a photo instead",
    permPrivacy:"You can switch to a still photo at any point during the session.",
    anaTitle:"Analyzing your face", anaSub:"Mapping landmarks and measuring proportions",
    anaSteps:["Detecting facial landmarks","Measuring face width and proportions","Classifying face shape","Scoring catalog frames"],
    tryTitle:"Live Virtual Try-On", trySub:"AI-selected frames matched to your face.",
    tryAnalysisDone:"AI Analysis Complete", flipCamera:"Flip Camera", snapshot:"Snapshot",
    photoPromptTitle:"Add your photo", photoPromptSub:"Take a photo or upload one to see the frames on your face.",
    takePhoto:"Take a photo", uploadPhoto:"Upload a photo", changePhoto:"Change photo",
    detecting:"Detecting face…", shareBtn:"Share",
    adjustBtn:"Adjust", adjustDone:"Done", adjustReset:"Auto-fit", adjustHint:"Drag the frame to reposition",
    compareBtn:"Compare top 3", compareTitle:"Compare your top matches", compareSub:"Your 3 best-scoring frames, side by side on your photo.", tryThis:"Try this one",
    navLabel:"Main navigation", skipToContent:"Skip to main content",
    errNoFace:"No face detected in the photo. Try a clear, front-facing photo.",
    errCamDenied:"Camera access was blocked. You can upload a photo instead.",
    tryingOn:"You're trying on:", fitScore:"Fit Score", excellentMatch:"Excellent Match", viewDetails:"View Details",
    insightsTitle:"Your AI Insights",
    insights:[
      ["Face Shape","Oval","Ideal for versatile frame styles"],
      ["Skin Tone","Warm","Golden undertones detected"],
      ["Best Colors","Amber, Olive, Tortoise","Earthy tones enhance your natural glow"],
      ["Best Metals","Gold, Rose Gold","Warm metals complement you"],
      ["Frame Width","Excellent","Current frames fit your face perfectly"],
      ["Bridge Fit","Optimal","Comfort and stability guaranteed"]],
    recTitle:"Top 3 AI Recommendations", whyThese:"Why these?", match:"Match", tryOn:"Try On",
    shapesTitle:"Recommended Frame Shapes", shapesBestFor:"Best for you", shapesWhy:"Why",
    shapesLessTitle:"May be less flattering", whyThisFrame:"Why this frame?",
    colorTitle:"Personal Color Analysis", colorDeep:"Deep analysis", colorHide:"Hide analysis",
    colorSeason:"Color season", colorUndertone:"Undertone", colorMetals:"Best metals", colorFrames:"Recommended frame colours", colorPalette:"Your palette",
    moreTitle:"More matches from the catalog", prevBtn:"Previous", nextBtn:"Next",
    catAll:"All", catVision:"Eyeglasses", catSun:"Sunglasses", noneInCat:"No frames in this category.",
    exploreTitle:"Explore Full Collection", exploreSub:"Discover more frames from our collection",
    footNote:"AI technology analyzes your facial features to find your perfect match.",
    footPrivacy:"Your data is private and secure.",
    bestMatch:"BEST MATCH",
    colTitle:"The Collection", colSub:"18 frames, each scored against your face profile.",
    filters:[["all","All frames"],["vision","Eyeglasses"],["sun","Sunglasses"],["acetate","Acetate"],["metal","Metal"]],
    matchForYou:"Match for your face", tryOnLive:"Try On Live", back:"Back",
    addFav:"Add to Favorites", inFav:"Saved",
    specKeys:["Shape","Material","Rim","Colour","Lens width","Bridge","Temple","Category"],
    favTitle:"Favorites", favSub:"frames saved for later.",
    favEmptyTitle:"No favorites yet", favEmptySub:"Save frames while you try them on and they will appear here.",
    profTitle:"My Profile", profName:"Guest Session", profSince:"Face profile created today",
    profRescan:"Re-scan my face", profFaceTitle:"Face profile", profPrefTitle:"Style preferences",
    profileFacts:[["Face shape","Oval"],["Skin tone","Warm"],["Face width","139 mm"],["Pupillary distance","62 mm"],["Bridge","Medium"],["Temple length","145 mm"],["Best lens width","50–53 mm"],["Frames tried","6"]],
    prefTags:["Classic","Minimal","Tortoise","Gold metal","Thin rims","Round shapes"],
    aboutTitle:"Frames should be measured, not guessed.",
    aboutBody:"OptiFit is a final-year project from Ashkelon Academic College. It combines real-time facial landmark detection with a measured frame catalog: every recommendation comes from the relationship between your face measurements and the physical dimensions of a frame — lens width, bridge width, temple length. Nothing is decided by taste alone.",
    aboutHow:"How it works",
    aboutSteps:[
      ["STEP 01","Capture","Your camera streams to the browser. 468 landmarks are detected per frame, on your device."],
      ["STEP 02","Measure","Face width, pupillary distance, jaw and cheekbone ratios are computed and your face shape is classified."],
      ["STEP 03","Match","Each catalog frame is scored on fit, proportion and colour harmony. The top matches are placed on your face live."]],
    aboutTech:"Built with",
    techTags:["MediaPipe Face Landmarker","Python / Flask","Real-time canvas compositing","Measured frame catalog","On-device processing"],
    photo:"PHOTO", noFace:"No face detected — center your face and look straight ahead.",
    langLabel:"עברית", dir:"ltr"
  },
  he: {
    home:"בית", tryOnNav:"מדידה", collection:"קולקציה", about:"אודות OptiFit",
    favorites:"מועדפים", profile:"הפרופיל שלי",
    homeBadge:"התאמת מסגרות מבוססת בינה מלאכותית",
    homeTitle1:"למצוא את המסגרת", homeTitle2:"שמתאימה לפנים שלך.",
    homeSub:"OptiFit מנתח את פרופורציות הפנים שלך בזמן אמת, מתאים מסגרות מתוך הקטלוג — ומאפשר למדוד אותן חי מול המצלמה.",
    homeCta1:"התחלת מדידה חיה", homeCta2:"לצפייה בקולקציה",
    homeLiveTag:"ניתוח פנים פעיל",
    homeStats:[["18","מסגרות בקטלוג"],["6","צורות פנים נתמכות"],["2 שנ׳","זמן ניתוח"]],
    homeChips:[["צורת פנים","אובלי"],["גוון עור","חם"],["ציון התאמה","98%"]],
    homeFeatures:[
      ["01","מיפוי פנים בזמן אמת","468 נקודות ציון בפנים נקראות מהמצלמה ומתורגמות למידות עדשה, גשר וזרוע."],
      ["02","התאמה מדודה, לא ניחוש","כל מסגרת מקבלת ציון מול המידות וגווני העור שלך — הציון מוצג, וגם הסיבה לו."],
      ["03","למדוד לפני שקונים","המסגרת עוקבת אחרי תנועת הראש, כך שאפשר לשפוט את ההתאמה כמו בחנות."]],
    permTitle:"הפעלת המצלמה",
    permSub:"כדי למפות את הפנים ולהניח מסגרות בזמן אמת, OptiFit זקוק לגישה למצלמה.",
    permPoints:["הווידאו מעובד על המכשיר שלך בלבד","שום תמונה אינה נשלחת או נשמרת","אפשר לבטל את ההרשאה בכל רגע"],
    permAllow:"אישור גישה למצלמה", permUpload:"שימוש בתמונה במקום",
    permPrivacy:"אפשר לעבור לתמונה סטטית בכל שלב במהלך המדידה.",
    anaTitle:"מנתח את הפנים שלך", anaSub:"מיפוי נקודות ציון ומדידת פרופורציות",
    anaSteps:["זיהוי נקודות ציון בפנים","מדידת רוחב הפנים והפרופורציות","סיווג צורת הפנים","דירוג המסגרות בקטלוג"],
    tryTitle:"מדידה וירטואלית חיה", trySub:"מסגרות שנבחרו עבורך בהתאמה לפנים.",
    tryAnalysisDone:"הניתוח הושלם", flipCamera:"החלפת מצלמה", snapshot:"צילום",
    photoPromptTitle:"הוספת תמונה", photoPromptSub:"צלמו תמונה או העלו תמונה כדי לראות את המסגרות על הפנים שלכם.",
    takePhoto:"צילום תמונה", uploadPhoto:"העלאת תמונה", changePhoto:"החלפת תמונה",
    detecting:"מזהה פנים…", shareBtn:"שיתוף",
    adjustBtn:"כוונון", adjustDone:"סיום", adjustReset:"התאמה אוטומטית", adjustHint:"גררו את המסגרת כדי למקם מחדש",
    compareBtn:"השוואת 3 המובילות", compareTitle:"השוואת ההתאמות המובילות", compareSub:"שלוש המסגרות עם ההתאמה הגבוהה ביותר, זו לצד זו על התמונה שלך.", tryThis:"למדוד את זו",
    navLabel:"ניווט ראשי", skipToContent:"דילוג לתוכן הראשי",
    errNoFace:"לא זוהו פנים בתמונה. נסו תמונה ברורה עם מבט ישר למצלמה.",
    errCamDenied:"הגישה למצלמה נחסמה. אפשר להעלות תמונה במקום.",
    tryingOn:"מודדים כרגע:", fitScore:"ציון התאמה", excellentMatch:"התאמה מצוינת", viewDetails:"לפרטי המסגרת",
    insightsTitle:"התובנות שלך",
    insights:[
      ["צורת פנים","אובלי","מתאים למגוון רחב של סגנונות"],
      ["גוון עור","חם","זוהו גוונים זהובים"],
      ["צבעים מומלצים","ענבר, זית, מנומר","גוונים חמים מחמיאים לגוון העור"],
      ["מתכות מומלצות","זהב, זהב ורוד","מתכות חמות משתלבות היטב"],
      ["רוחב מסגרת","מצוין","המסגרת הנוכחית יושבת מדויק"],
      ["התאמת גשר","אופטימלית","נוחות ויציבות לאורך זמן"]],
    recTitle:"3 ההמלצות המובילות", whyThese:"למה אלה?", match:"התאמה", tryOn:"למדידה",
    shapesTitle:"צורות מסגרת מומלצות", shapesBestFor:"הכי מתאים לך", shapesWhy:"למה",
    shapesLessTitle:"עשוי להחמיא פחות", whyThisFrame:"למה המסגרת הזו?",
    colorTitle:"ניתוח צבעים אישי", colorDeep:"ניתוח מעמיק", colorHide:"סגירת הניתוח",
    colorSeason:"עונת צבע", colorUndertone:"גוון בסיס", colorMetals:"מתכות מומלצות", colorFrames:"גווני מסגרת מומלצים", colorPalette:"הפלטה שלך",
    moreTitle:"עוד התאמות מהקטלוג", prevBtn:"הקודם", nextBtn:"הבא",
    catAll:"הכול", catVision:"ראייה", catSun:"שמש", noneInCat:"אין מסגרות בקטגוריה זו.",
    exploreTitle:"לקולקציה המלאה", exploreSub:"עוד מסגרות מתוך הקטלוג שלנו",
    footNote:"הבינה המלאכותית מנתחת את תווי הפנים כדי למצוא את ההתאמה המדויקת.",
    footPrivacy:"המידע שלך פרטי ומאובטח.",
    bestMatch:"ההתאמה הטובה ביותר",
    colTitle:"הקולקציה", colSub:"18 מסגרות, כל אחת מדורגת מול פרופיל הפנים שלך.",
    filters:[["all","הכול"],["vision","משקפי ראייה"],["sun","משקפי שמש"],["acetate","אצטט"],["metal","מתכת"]],
    matchForYou:"התאמה לפנים שלך", tryOnLive:"מדידה חיה", back:"חזרה",
    addFav:"הוספה למועדפים", inFav:"נשמר",
    specKeys:["צורה","חומר","עובי מסגרת","צבע","רוחב עדשה","גשר","זרוע","קטגוריה"],
    favTitle:"מועדפים", favSub:"מסגרות שנשמרו להמשך.",
    favEmptyTitle:"אין עדיין מועדפים", favEmptySub:"אפשר לשמור מסגרות תוך כדי מדידה והן יופיעו כאן.",
    profTitle:"הפרופיל שלי", profName:"משתמש אורח", profSince:"פרופיל הפנים נוצר היום",
    profRescan:"סריקת פנים מחדש", profFaceTitle:"פרופיל הפנים", profPrefTitle:"העדפות סגנון",
    profileFacts:[["צורת פנים","אובלי"],["גוון עור","חם"],["רוחב פנים","139 מ״מ"],["מרחק אישונים","62 מ״מ"],["גשר","בינוני"],["אורך זרוע","145 מ״מ"],["רוחב עדשה מומלץ","50–53 מ״מ"],["מסגרות שנמדדו","6"]],
    prefTags:["קלאסי","מינימלי","מנומר","מתכת זהב","מסגרת דקה","צורות עגולות"],
    aboutTitle:"מסגרת צריך למדוד, לא לנחש.",
    aboutBody:"OptiFit הוא פרויקט גמר במכללה האקדמית אשקלון. המערכת משלבת זיהוי נקודות ציון בפנים בזמן אמת עם קטלוג מסגרות מדוד: כל המלצה נגזרת מהיחס בין מידות הפנים שלך למידות הפיזיות של המסגרת — רוחב עדשה, רוחב גשר ואורך זרוע. שום החלטה אינה מבוססת על טעם בלבד.",
    aboutHow:"איך זה עובד",
    aboutSteps:[
      ["שלב 01","צילום","המצלמה משדרת לדפדפן. 468 נקודות ציון מזוהות בכל פריים, על המכשיר עצמו."],
      ["שלב 02","מדידה","רוחב הפנים, מרחק האישונים ויחסי הלסת ועצמות הלחיים מחושבים, וצורת הפנים מסווגת."],
      ["שלב 03","התאמה","כל מסגרת בקטלוג מקבלת ציון על התאמה, פרופורציה והרמוניית צבע. המובילות מונחות על הפנים בזמן אמת."]],
    aboutTech:"נבנה עם",
    techTags:["MediaPipe Face Landmarker","Python / Flask","הרכבת תמונה בזמן אמת","קטלוג מסגרות מדוד","עיבוד על המכשיר"],
    photo:"תמונה", noFace:"לא זוהו פנים — מקמי את הפנים במרכז והביטי ישר.",
    langLabel:"English", dir:"rtl"
  }
};

const SHAPE_EN = {round:"Round",cat_eye:"Cat-Eye",rectangle:"Rectangle",soft_square:"Square",oval:"Oval",browline:"Browline"};
const SHAPE_HE = {round:"עגול",cat_eye:"חתולי",rectangle:"מלבני",soft_square:"מרובע",oval:"אובלי",browline:"בראוליין"};
const MAT_EN = {acetate:"Acetate",metal:"Metal",titanium:"Titanium",combination:"Combination"};
const MAT_HE = {acetate:"אצטט",metal:"מתכת",titanium:"טיטניום",combination:"משולב"};
const RIM_EN = {thin:"Thin",medium:"Medium",thick:"Thick"};
const RIM_HE = {thin:"דקה",medium:"בינונית",thick:"עבה"};
const HOME_HERO = "/assets/hero-home.jpg";   // dedicated home-page hero image
const REC_IDS = ["V04","V06","V03"];
const MORE_PER = 6;   // frames per page in the "More matches" list
const WHY = {
  en:{V04:"Optimal proportions for your oval face shape",V06:"Balanced width and excellent bridge fit",V03:"Strong jawline balance and face harmony"},
  he:{V04:"פרופורציות אופטימליות לצורת פנים אובלית",V06:"רוחב מאוזן והתאמת גשר מצוינת",V03:"איזון קו הלסת והרמוניה עם הפנים"}
};
// Color-analysis fallbacks (shown before/without a real backend analysis).
const DEFAULT_PALETTE = ["#E8A33D", "#C97B3E", "#8FB339", "#B8860B"];
const COLOR_EXPL_EN = {
  Spring:"Warm, light skin tone. Gold and warm-brown frames enhance your natural glow.",
  Autumn:"Warm, deep skin tone. Rich earthy frames — brown, cognac, antique gold — sit in harmony with your coloring.",
  Summer:"Cool, light skin tone. Silver frames and cool pastels highlight the softness of your complexion.",
  Winter:"Cool, deep skin tone. High-contrast frames — black, midnight blue, silver — match the intensity of your coloring.",
  Neutral:"Balanced skin tone that suits almost any metal and frame colour — you have wide freedom of choice.",
};
const UNDERTONE = {
  en:{warm:"Warm",cool:"Cool",neutral:"Neutral"},
  he:{warm:"חם",cool:"קריר",neutral:"ניטרלי"}
};
// Visual weight per recommendation tier (badge colours).
const TIER_STYLE = {
  best:        { bg:"#35dff2", fg:"#03181e" },
  highly:      { bg:"rgba(53,223,242,.16)", fg:"#7fe9f7", bc:"#2c6f7e" },
  recommended: { bg:"#132131", fg:"#9dafc0", bc:"#1e2c3a" },
  alternative: { bg:"#0f1720", fg:"#7d8fa2", bc:"#1b2836" },
};
// Offline fallback for the shape guidance (Oval = the design default face).
// Real detections replace this with the backend's per-face-shape hierarchy.
const DEFAULT_SHAPE_ADVICE = {
  en: {
    recommended: [
      { tier:"best", tierLabel:"Best Match", friendly:"Soft Rectangle", why:"Adds gentle structure and definition while balancing softer facial contours." },
      { tier:"best", tierLabel:"Best Match", friendly:"Soft Square / Geometric", why:"Adds visual definition without making the face appear too angular." },
      { tier:"highly", tierLabel:"Highly Recommended", friendly:"Subtle Cat-Eye", why:"Creates a slight lifting effect around the eyes and cheekbones." },
      { tier:"recommended", tierLabel:"Recommended", friendly:"Browline", why:"Draws attention upward and frames the face with a confident line." },
      { tier:"recommended", tierLabel:"Recommended", friendly:"Panto / Soft Round", why:"A classic rounded profile that softens and balances the features." },
    ],
    cautions: [
      "Very small, perfectly round frames — may provide less balance.",
      "Extremely oversized frames — may overwhelm your proportions.",
      "Very thick, visually heavy frames — may provide less balance.",
    ],
  },
  he: {
    recommended: [
      { tier:"best", tierLabel:"התאמה מיטבית", friendly:"מלבני רך", why:"מוסיף מבנה והגדרה עדינים תוך איזון קווי פנים רכים." },
      { tier:"best", tierLabel:"התאמה מיטבית", friendly:"מרובע רך / גיאומטרי", why:"מוסיף הגדרה חזותית מבלי שהפנים ייראו זוויתיים מדי." },
      { tier:"highly", tierLabel:"מומלץ מאוד", friendly:"חתולי עדין", why:"יוצר אפקט הרמה עדין סביב העיניים ועצמות הלחיים." },
      { tier:"recommended", tierLabel:"מומלץ", friendly:"בראוליין", why:"מושך את המבט כלפי מעלה וממסגר את הפנים בקו בטוח." },
      { tier:"recommended", tierLabel:"מומלץ", friendly:"פנטו / עגול רך", why:"פרופיל עגול קלאסי שמרכך ומאזן את התווים." },
    ],
    cautions: [
      "מסגרות עגולות קטנות ומדויקות — עשויות להעניק פחות איזון.",
      "מסגרות גדולות במיוחד — עלולות להכביד על הפרופורציות.",
      "מסגרות עבות וכבדות חזותית — עשויות להעניק פחות איזון.",
    ],
  },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const state = {
  screen: "home", lang: "en",
  currentId: "V04", detailId: "V04",
  favs: [], filter: "all", catalog: [], morePage: 0, moreCat: "all", colorOpen: false,
  camera: "idle",        // idle | live | photo
  facing: "user",
  progress: 0,
  analysis: null,        // real /api/analyze result
  realScore: {},         // id -> real match %
  photoSrc: "/assets/hero-home.jpg",
  hasPhoto: false,       // true only once the user captured/uploaded a real photo
  promptError: null,     // null | "noFace" | "camDenied" — shown on the photo prompt
  adjust: { on: false, committed: false, x: 50, y: 40, w: 42, rot: 0 },  // manual frame placement (%)
  autoFit: null,         // last auto-computed fit, used to seed manual mode
};

// ---------------------------------------------------------------------------
// Persistence — favourites + language survive a page reload
// ---------------------------------------------------------------------------
const LS = { favs: "optifit.favs", lang: "optifit.lang" };
function loadPrefs() {
  try {
    const favs = JSON.parse(localStorage.getItem(LS.favs) || "[]");
    if (Array.isArray(favs)) state.favs = favs;
    const lang = localStorage.getItem(LS.lang);
    if (lang === "he" || lang === "en") state.lang = lang;
  } catch (e) { /* storage blocked (private mode) — run with defaults */ }
}
function saveFavs() { try { localStorage.setItem(LS.favs, JSON.stringify(state.favs)); } catch (e) {} }
function saveLang() { try { localStorage.setItem(LS.lang, state.lang); } catch (e) {} }
loadPrefs();

let landmarker = null, trackingReady = false, rafId = null;
let stream = null;
const app = document.getElementById("app");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const T   = () => STR[state.lang];
const he  = () => state.lang === "he";
const esc = (s) => String(s == null ? "" : s).replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

const find  = (id) => state.catalog.find(f => f.id === id) || null;
const imgOf = (f) => f ? f.image : "/assets/frames/vision/vision_04.png";
const nameOf = (f) => f ? f.name : "—";
function metaOf(f) {
  if (!f) return "";
  const s = (he() ? SHAPE_HE : SHAPE_EN)[f.shape] || f.shape;
  const m = (he() ? MAT_HE : MAT_EN)[f.material] || f.material;
  return s + " • " + m;
}
function score(f) {
  if (!f) return 98;
  if (state.realScore[f.id] != null) return state.realScore[f.id];
  const i = REC_IDS.indexOf(f.id);
  if (i === 0) return 98; if (i === 1) return 94; if (i === 2) return 91;
  return Math.round(60 + (f.universal_score || 0.7) * 33);
}

// ---------------------------------------------------------------------------
// SVG snippets
// ---------------------------------------------------------------------------
const svgSpark = (s=14, fill="currentColor") => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="${fill}"><path d="M12 2l2.2 6.3L20.5 10l-6.3 2.2L12 18.5 9.8 12.2 3.5 10l6.3-1.7z"/></svg>`;
// OptiFit brand mark — eye/lens outline with a centered "A".
const svgLogo = (h=31, c="#35dff2") => `<svg height="${h}" viewBox="0 0 120 80" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M8 40 Q60 4 112 40 Q60 76 8 40 Z" stroke="${c}" stroke-width="7" stroke-linejoin="round"/><path d="M51 52 L60 29 L69 52" stroke="${c}" stroke-width="6" stroke-linecap="round" stroke-linejoin="round"/><path d="M54.5 45 L65.5 45" stroke="${c}" stroke-width="6" stroke-linecap="round"/></svg>`;
const svgHeart = (s=19, fill="none") => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="${fill}" stroke="currentColor" stroke-width="1.7"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1-1.1a5.5 5.5 0 0 0-7.8 7.8l1.1 1L12 21.2l7.7-7.8 1.1-1a5.5 5.5 0 0 0 0-7.8z"/></svg>`;
const svgCheck = (s=17, stroke="#35dff2") => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="${stroke}" stroke-width="2.2"><path d="M20 6L9 17l-5-5"/></svg>`;
const svgArrow = (s=16) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 5l7 7-7 7"/></svg>`;
const svgCam   = (s=19) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 8a2 2 0 0 1 2-2h2l1.5-2h7L17 6h2a2 2 0 0 1 2 2v9a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><circle cx="12" cy="12.5" r="3.5"/></svg>`;

// ---------------------------------------------------------------------------
// Header
// ---------------------------------------------------------------------------
function headerHTML() {
  const t = T();
  const active = (k) => isNavActive(k);
  const nav = [["home",t.home],["tryon",t.tryOnNav],["collection",t.collection],["about",t.about]]
    .map(([k,label]) => `<div class="hv-white" data-act="go" data-arg="${k}" style="display:flex;align-items:center;padding:0 26px;font-size:15.5px;font-weight:500;cursor:pointer;color:${active(k)?"#ffffff":"#93a5b8"};border-bottom:2px solid ${active(k)?"#35dff2":"transparent"}">${esc(label)}</div>`)
    .join("");
  return `
  <header style="height:66px;flex:none;display:flex;align-items:center;justify-content:space-between;padding:0 34px;background:#070b11;border-bottom:1px solid #141f2b;position:sticky;top:0;z-index:40">
    <div dir="ltr" data-act="go" data-arg="home" style="display:flex;align-items:center;gap:11px;cursor:pointer">
      <img src="/assets/optifit-mark.png" alt="OptiFit" style="height:34px;width:auto;object-fit:contain;flex:none" />
      <div style="display:flex;align-items:baseline;gap:1px;font-size:27px;font-weight:800;letter-spacing:-.6px">
        <span style="color:#ffffff">Opti</span><span style="color:#35dff2">Fit</span>
      </div>
    </div>
    <nav aria-label="${esc(t.navLabel)}" style="display:flex;align-items:stretch;gap:6px;height:66px">${nav}</nav>
    <div style="display:flex;align-items:center;gap:22px">
      <div class="hv-white" data-act="go" data-arg="favorites" style="display:flex;align-items:center;gap:9px;font-size:15px;color:${state.screen==="favorites"?"#ffffff":"#93a5b8"};cursor:pointer">
        ${svgHeart(19)}<span>${esc(t.favorites)}</span>
        <span style="min-width:20px;height:20px;padding:0 6px;border-radius:10px;background:#13212e;color:#35dff2;font-size:12px;font-weight:700;display:flex;align-items:center;justify-content:center">${state.favs.length}</span>
      </div>
      <div class="hv-white" data-act="go" data-arg="profile" style="display:flex;align-items:center;gap:9px;font-size:15px;color:${state.screen==="profile"?"#ffffff":"#93a5b8"};cursor:pointer">
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="8" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/></svg>
        <span>${esc(t.profile)}</span>
      </div>
      <div class="hv-cyanbc" data-act="toggleLang" style="display:flex;align-items:center;gap:7px;padding:6px 13px;border:1px solid #1e2c3b;border-radius:20px;font-size:13px;font-weight:600;color:#9fb0c2;cursor:pointer;letter-spacing:.3px">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7"><circle cx="12" cy="12" r="9"/><path d="M3 12h18M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18"/></svg>
        <span>${esc(t.langLabel)}</span>
      </div>
    </div>
  </header>`;
}

// ---------------------------------------------------------------------------
// Screen: Home
// ---------------------------------------------------------------------------
function homeHTML() {
  const t = T();
  const stats = t.homeStats.map(([n,l]) => `<div><div style="font-size:30px;font-weight:800;color:#ffffff;letter-spacing:-1px">${esc(n)}</div><div style="font-size:14px;color:#7d8fa2;margin-top:2px">${esc(l)}</div></div>`).join("");
  const chips = t.homeChips.map(([k,v]) => `<div style="flex:1;padding:11px 14px;border-radius:10px;background:rgba(9,15,21,.82);border:1px solid #1c2a37;backdrop-filter:blur(8px)"><div style="font-size:11.5px;color:#7d8fa2;letter-spacing:.4px;text-transform:uppercase">${esc(k)}</div><div style="font-size:15px;font-weight:700;margin-top:3px">${esc(v)}</div></div>`).join("");
  const feats = t.homeFeatures.map(([n,tt,d]) => `<div class="of-card hv-b26" style="padding:30px 28px;border-radius:16px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f"><div style="width:44px;height:44px;border-radius:11px;background:rgba(53,223,242,.10);color:#35dff2;display:flex;align-items:center;justify-content:center;font-size:20px;font-weight:800;margin-bottom:18px">${esc(n)}</div><h3 style="margin:0 0 10px;font-size:20px;font-weight:700">${esc(tt)}</h3><p style="margin:0;font-size:15.5px;line-height:1.65;color:#8496a9">${esc(d)}</p></div>`).join("");
  const heroImg = HOME_HERO;
  return `
  <div style="flex:1;animation:ofRise .4s ease both">
    <section data-responsive="grid2" style="padding:96px 60px 70px;display:grid;grid-template-columns:1.05fr .95fr;gap:56px;align-items:center;background:radial-gradient(900px 500px at 78% 20%,rgba(53,223,242,.10),transparent 70%)">
      <div>
        <div style="display:inline-flex;align-items:center;gap:9px;padding:7px 15px;border:1px solid #1d3a45;border-radius:20px;background:rgba(53,223,242,.06);font-size:13.5px;color:#35dff2;font-weight:600;margin-bottom:26px">${svgSpark(14)}<span>${esc(t.homeBadge)}</span></div>
        <h1 style="margin:0 0 20px;font-size:66px;line-height:1.04;font-weight:800;letter-spacing:-2px">${esc(t.homeTitle1)}<br /><span style="color:#35dff2">${esc(t.homeTitle2)}</span></h1>
        <p style="margin:0 0 36px;font-size:19.5px;line-height:1.6;color:#93a5b8;max-width:520px">${esc(t.homeSub)}</p>
        <div style="display:flex;gap:14px;margin-bottom:44px;flex-wrap:wrap">
          <div class="hv-bright" data-act="startFlow" style="display:flex;align-items:center;gap:11px;padding:16px 30px;border-radius:11px;background:linear-gradient(100deg,#35dff2,#12a9cd);color:#03181e;font-size:16.5px;font-weight:700;cursor:pointer;box-shadow:0 10px 30px -12px rgba(53,223,242,.7)">${svgCam(19)}<span>${esc(t.homeCta1)}</span></div>
          <div class="hv-cyanbc" data-act="go" data-arg="collection" style="display:flex;align-items:center;padding:16px 30px;border-radius:11px;border:1px solid #22323f;color:#dbe6f0;font-size:16.5px;font-weight:600;cursor:pointer">${esc(t.homeCta2)}</div>
        </div>
        <div style="display:flex;gap:44px;flex-wrap:wrap">${stats}</div>
      </div>
      <div style="position:relative;border-radius:20px;overflow:hidden;border:1px solid #1a2735;background:#0a1017;aspect-ratio:4/3.4">
        <img src="${heroImg}" alt="" style="width:100%;height:100%;object-fit:cover" onerror="this.style.opacity=0" />
        <div style="position:absolute;inset:0;background:linear-gradient(to top,rgba(5,8,12,.85),transparent 45%);pointer-events:none"></div>
        <div style="position:absolute;left:0;right:0;height:2px;background:linear-gradient(90deg,transparent,#35dff2,transparent);opacity:.55;animation:ofScan 3.4s linear infinite"></div>
        <div style="position:absolute;top:18px;left:18px;display:flex;align-items:center;gap:8px;padding:7px 14px;border-radius:8px;background:rgba(5,10,15,.75);backdrop-filter:blur(8px);font-size:13px;font-weight:600"><span style="width:8px;height:8px;border-radius:50%;background:#35dff2;animation:ofPulse 1.6s infinite"></span><span>${esc(t.homeLiveTag)}</span></div>
        <div style="position:absolute;bottom:20px;left:20px;right:20px;display:flex;gap:10px">${chips}</div>
      </div>
    </section>
    <section data-responsive="grid3" style="padding:24px 60px 80px;display:grid;grid-template-columns:repeat(3,1fr);gap:20px">${feats}</section>
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Permission
// ---------------------------------------------------------------------------
function permissionHTML() {
  const t = T();
  const points = t.permPoints.map(p => `<div style="display:flex;align-items:center;gap:12px;padding:13px 16px;border-radius:10px;background:#0a1119;border:1px solid #16222f;font-size:15px;color:#c3d1de">${svgCheck(17)}<span>${esc(p)}</span></div>`).join("");
  return `
  <div style="flex:1;display:flex;align-items:center;justify-content:center;padding:60px;animation:ofRise .35s ease both">
    <div style="width:560px;max-width:100%;padding:48px 46px;border-radius:20px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #17232f;text-align:center">
      <div style="width:82px;height:82px;margin:0 auto 26px;border-radius:22px;background:rgba(53,223,242,.09);border:1px solid #1d3a45;color:#35dff2;display:flex;align-items:center;justify-content:center">${svgCam(38)}</div>
      <h2 style="margin:0 0 12px;font-size:30px;font-weight:800;letter-spacing:-.6px">${esc(t.permTitle)}</h2>
      <p style="margin:0 0 30px;font-size:16.5px;line-height:1.65;color:#8b9dae">${esc(t.permSub)}</p>
      <div style="display:flex;flex-direction:column;gap:11px;margin-bottom:26px;text-align:${he()?"right":"left"}">${points}</div>
      <div class="hv-bright" data-act="allow" style="padding:16px;border-radius:11px;background:linear-gradient(100deg,#35dff2,#12a9cd);color:#03181e;font-size:16.5px;font-weight:700;cursor:pointer;margin-bottom:12px">${esc(t.permAllow)}</div>
      <div class="hv-cyanbc" data-act="usePhoto" style="padding:15px;border-radius:11px;border:1px solid #22323f;color:#cad7e3;font-size:15.5px;font-weight:600;cursor:pointer">${esc(t.permUpload)}</div>
      <p style="margin:22px 0 0;font-size:13px;color:#63768a">${esc(t.permPrivacy)}</p>
    </div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Analyzing
// ---------------------------------------------------------------------------
function analyzingHTML() {
  const t = T();
  const steps = t.anaSteps.map((label,i) => {
    const done = state.progress >= (i+1)*25, cur = state.progress === i*25;
    const color = done ? "#e9f0f7" : (cur ? "#c3d1de" : "#5c6e81");
    const dot   = done ? "#35dff2" : (cur ? "#2b6d7c" : "#233240");
    const border= (done||cur) ? "#1d3a45" : "#16222f";
    return `<div style="display:flex;align-items:center;gap:13px;padding:13px 17px;border-radius:10px;background:#0a1119;border:1px solid ${border};font-size:15.5px;color:${color}"><span style="width:9px;height:9px;border-radius:50%;background:${dot}"></span><span>${esc(label)}</span></div>`;
  }).join("");
  return `
  <div style="flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;padding:60px;animation:ofRise .35s ease both">
    <div style="position:relative;width:260px;height:260px;border-radius:50%;overflow:hidden;border:1px solid #1d3a45;margin-bottom:34px">
      <img src="${state.photoSrc}" alt="" style="width:100%;height:100%;object-fit:cover" onerror="this.style.opacity=0" />
      <div style="position:absolute;inset:0;background:rgba(5,10,15,.45);pointer-events:none"></div>
      <div style="position:absolute;left:0;right:0;height:2px;background:#35dff2;box-shadow:0 0 18px 3px rgba(53,223,242,.8);animation:ofScan 2s linear infinite;pointer-events:none"></div>
      <div style="position:absolute;inset:-1px;border-radius:50%;border:2px solid transparent;border-top-color:#35dff2;animation:ofSpin 1.4s linear infinite;pointer-events:none"></div>
    </div>
    <h2 style="margin:0 0 10px;font-size:32px;font-weight:800;letter-spacing:-.7px">${esc(t.anaTitle)}</h2>
    <p style="margin:0 0 30px;font-size:16.5px;color:#8b9dae">${esc(t.anaSub)}</p>
    <div style="width:520px;max-width:100%;height:6px;border-radius:4px;background:#111b25;overflow:hidden;margin-bottom:28px"><div style="height:100%;width:${state.progress}%;background:linear-gradient(90deg,#12a9cd,#35dff2);transition:width .4s ease"></div></div>
    <div style="width:520px;max-width:100%;display:flex;flex-direction:column;gap:11px">${steps}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Build recommendations + insights (real data when available)
// ---------------------------------------------------------------------------
function buildRecs() {
  const t = T();
  if (state.analysis && state.analysis.top && state.analysis.top.length) {
    return state.analysis.top.map((f, i) => {
      const whyList = (he() ? f.why_he : f.why);
      const why = (whyList && whyList.length) ? whyList[0] : ((WHY[state.lang][f.id]) || f.reason);
      return {
        id: f.id, img: f.image, name: f.name, meta: metaOf(find(f.id)) || (f.color || ""),
        score: f.match, why,
        whyList: (whyList && whyList.length) ? whyList : [why],
        first: i === 0,
      };
    });
  }
  return REC_IDS.map((id, i) => {
    const f = find(id);
    const why = WHY[state.lang][id];
    return { id, img: imgOf(f), name: nameOf(f), meta: metaOf(f), score: [98,94,91][i], why, whyList: [why], first: i === 0 };
  });
}
function buildInsights() {
  const t = T(), a = state.analysis;
  const icons = ["◯","◐","◍","✦","▭","◡"];
  const vals = a ? [
    a.face_shape,
    a.skin_tone,
    (a.recommended_colors || []).join(", "),
    (a.recommended_metals || []).join(", "),
    a.frame_width,
    a.bridge_fit,
  ] : t.insights.map(x => x[1]);
  // Keep the "Best Metals" description consistent with the detected undertone
  // (a warm profile shouldn't read "warm metals" while also listing silver).
  const undKey = a ? (a.color_undertone || "") : "warm";
  const metalDesc = {
    en:{warm:"Warm metals complement you",cool:"Cool metals complement you",neutral:"Warm and cool metals both suit you"},
    he:{warm:"מתכות חמות משתלבות היטב",cool:"מתכות קרירות משתלבות היטב",neutral:"מתכות חמות וקרירות מתאימות"}
  }[he()?"he":"en"][undKey];
  return t.insights.map(([k,v,d],i) => ({ k, v: vals[i] || v, d: (i === 3 && metalDesc) ? metalDesc : d, icon: icons[i] }));
}

// Resolve the color-analysis data (real backend result, or design fallback).
function colorData() {
  const a = state.analysis, ins = buildInsights();
  const splitTags = (s) => String(s || "").split(/[,،]/).map(x => x.trim()).filter(Boolean);
  const hex = (a && Array.isArray(a.palette_hex) && a.palette_hex.length) ? a.palette_hex : DEFAULT_PALETTE;
  const seasonEn = (a && a.color_season) ? a.color_season : "Autumn";
  const seasonHe = (a && a.color_season_he) ? a.color_season_he : "סתיו";
  const undKey = (a && a.color_undertone) ? a.color_undertone : "warm";
  return {
    hex,
    season: he() ? seasonHe : seasonEn,
    undertone: (he() ? UNDERTONE.he : UNDERTONE.en)[undKey] || undKey,
    explanation: he()
      ? ((a && a.color_explanation) || "גוון עור חם. מסגרות בגווני אדמה חמים ידגישו את הזוהר הטבעי של העור.")
      : (COLOR_EXPL_EN[seasonEn] || COLOR_EXPL_EN.Autumn),
    metals: splitTags(ins[3].v),          // "Best Metals" insight value
    frameColors: he()
      ? ((a && a.recommended_frame_colors) || splitTags(ins[2].v))
      : splitTags(ins[2].v),              // "Best Colors" insight value
  };
}

// Color-analysis box: color circles always visible + a button that expands a
// deeper analysis (season, undertone, metals, recommended frame colours).
function colorAnalysisHTML() {
  const t = T(), c = colorData();
  const circles = c.hex.map(hx => `<span title="${esc(hx)}" style="width:34px;height:34px;border-radius:50%;background:${esc(hx)};border:2px solid rgba(255,255,255,.14);box-shadow:0 2px 8px rgba(0,0,0,.4);flex:none"></span>`).join("");
  const tag = (x) => `<span style="padding:7px 14px;border-radius:20px;border:1px solid #1d3a45;background:rgba(53,223,242,.06);color:#9fe9f6;font-size:13.5px;font-weight:600">${esc(x)}</span>`;
  const fact = (k, v) => `<div style="padding:12px 14px;border-radius:10px;background:#090f16;border:1px solid #16222f"><div style="font-size:12px;color:#7d8fa2;letter-spacing:.3px;text-transform:uppercase">${esc(k)}</div><div style="font-size:16px;font-weight:700;margin-top:3px">${esc(v)}</div></div>`;
  const deep = state.colorOpen ? `
    <div style="margin-top:16px;padding-top:16px;border-top:1px solid #16222f;animation:ofRise .3s ease both">
      <p style="margin:0 0 14px;font-size:14.5px;line-height:1.65;color:#8fa2b4">${esc(c.explanation)}</p>
      <div data-responsive="grid2" style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:16px">
        ${fact(t.colorSeason, c.season)}
        ${fact(t.colorUndertone, c.undertone)}
      </div>
      <div style="font-size:13px;color:#7d8fa2;margin-bottom:8px">${esc(t.colorMetals)}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px;margin-bottom:16px">${c.metals.map(tag).join("")}</div>
      <div style="font-size:13px;color:#7d8fa2;margin-bottom:8px">${esc(t.colorFrames)}</div>
      <div style="display:flex;flex-wrap:wrap;gap:8px">${c.frameColors.map(tag).join("")}</div>
    </div>` : "";
  return `
  <div style="margin-top:16px;padding:22px 24px;border-radius:14px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap">
      <div style="display:flex;align-items:center;gap:12px;flex-wrap:wrap">
        <h3 style="margin:0;font-size:19px;font-weight:700">${esc(t.colorTitle)}</h3>
        <div style="display:flex;align-items:center;gap:9px;padding-inline-start:6px">${circles}</div>
      </div>
      <div class="hv-cyanbc" data-act="colorToggle" style="display:flex;align-items:center;gap:8px;padding:10px 18px;border-radius:9px;border:1px solid #22323f;font-size:14.5px;font-weight:600;cursor:pointer;color:#dbe6f0">
        <span>${esc(state.colorOpen ? t.colorHide : t.colorDeep)}</span>
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="transform:rotate(${state.colorOpen?"180":"0"}deg);transition:transform .2s"><path d="M6 9l6 6 6-6"/></svg>
      </div>
    </div>
    ${deep}
  </div>`;
}

// The rest of the catalog (everything not in the Top-3), scored and ranked so
// each frame shows its own match percentage.
function otherMatches() {
  const recIds = new Set(buildRecs().map(r => r.id));
  const cat = state.moreCat;
  return state.catalog
    .filter(f => !recIds.has(f.id))
    .filter(f => cat === "all" ? true : f.category === cat)
    .map(f => ({ f, s: score(f) }))
    .sort((a, b) => b.s - a.s);
}

// Paginated, scrollable list of additional matches shown under the Top-3.
function moreMatchesHTML() {
  const t = T();
  // Hide the whole section only if there is nothing beyond the Top-3 at all.
  const recIds = new Set(buildRecs().map(r => r.id));
  if (!state.catalog.some(f => !recIds.has(f.id))) return "";
  const list = otherMatches();
  const pages = Math.max(1, Math.ceil(list.length / MORE_PER));
  const page = Math.min(Math.max(state.morePage, 0), pages - 1);
  const start = page * MORE_PER;
  const slice = list.slice(start, start + MORE_PER);

  // Category chooser: All / Eyeglasses / Sunglasses.
  const cats = [["all", t.catAll], ["vision", t.catVision], ["sun", t.catSun]];
  const catTabs = cats.map(([k, label]) => {
    const on = state.moreCat === k;
    return `<div class="hv-cyanb" data-act="moreCat" data-arg="${k}" style="padding:7px 14px;border-radius:8px;border:1px solid ${on?"#2c93ad":"#1b2836"};background:${on?"rgba(53,223,242,.10)":"#0a1017"};color:${on?"#35dff2":"#a9bacb"};font-size:13px;font-weight:600;cursor:pointer">${esc(label)}</div>`;
  }).join("");
  const rows = slice.map(({ f, s }) => {
    const isFav = state.favs.includes(f.id);
    return `
    <div class="hv-b26" style="display:flex;align-items:center;gap:12px;padding:11px 12px;border-radius:11px;background:#0a1017;border:1px solid #16222f">
      <img src="${f.image}" alt="" style="width:64px;height:40px;object-fit:contain;flex:none" />
      <div style="flex:1;min-width:0">
        <div style="font-size:15px;font-weight:700;letter-spacing:-.2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(f.name)}</div>
        <div style="font-size:12.5px;color:#8496a9;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">${esc(metaOf(f))}</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:6px;flex:none">
        <div style="display:flex;align-items:baseline;gap:4px"><span style="font-size:18px;font-weight:800;color:#35dff2;letter-spacing:-.5px">${s}%</span><span style="font-size:11.5px;color:#7d8fa2">${esc(t.match)}</span></div>
        <div style="display:flex;align-items:center;gap:8px">
          <div data-act="fav" data-arg="${f.id}" style="cursor:pointer;color:${isFav?"#35dff2":"#65788c"}">${svgHeart(15, isFav?"#35dff2":"none")}</div>
          <div class="hv-cyanbc" data-act="tryOn" data-arg="${f.id}" style="padding:6px 12px;border-radius:7px;border:1px solid #22323f;font-size:12.5px;font-weight:600;cursor:pointer">${esc(t.tryOn)}</div>
        </div>
      </div>
    </div>`;
  }).join("");
  const chev = (d) => `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="${d==="left"?"M15 5l-7 7 7 7":"M9 5l7 7-7 7"}"/></svg>`;
  const prevOff = page <= 0, nextOff = page >= pages - 1;
  const navBtn = (act, off, label, chevD, before) => `
    <div class="hv-cyanbc" data-act="${off?"":act}" style="display:flex;align-items:center;gap:8px;padding:10px 16px;border-radius:9px;border:1px solid #22323f;font-size:14px;font-weight:600;cursor:${off?"default":"pointer"};opacity:${off?".4":"1"};pointer-events:${off?"none":"auto"}">${before?chev(chevD):""}<span>${esc(label)}</span>${before?"":chev(chevD)}</div>`;

  const body = list.length
    ? `<div style="max-height:326px;overflow-y:auto;display:flex;flex-direction:column;gap:10px;padding-inline-end:4px">${rows}</div>
       <div style="display:flex;align-items:center;justify-content:space-between;margin-top:14px;gap:10px">
         ${navBtn("morePrev", prevOff, t.prevBtn, "left", true)}
         <span style="font-size:12.5px;color:#7d8fa2" dir="ltr">${page + 1} / ${pages}</span>
         ${navBtn("moreNext", nextOff, t.nextBtn, "right", false)}
       </div>`
    : `<div style="padding:26px 10px;text-align:center;font-size:14px;color:#7d8fa2">${esc(t.noneInCat)}</div>`;

  return `
  <div style="margin-top:16px;padding:18px 18px 16px;border-radius:14px;background:linear-gradient(180deg,#0b1118,#080d13);border:1px solid #16222f">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;gap:10px">
      <h3 style="margin:0;font-size:17px;font-weight:700;letter-spacing:-.2px">${esc(t.moreTitle)}</h3>
      <span style="font-size:12.5px;color:#7d8fa2;flex:none" dir="ltr">${list.length ? `${start + 1}–${start + slice.length} / ${list.length}` : "0"}</span>
    </div>
    <div style="display:flex;gap:8px;margin-bottom:14px;flex-wrap:wrap">${catTabs}</div>
    ${body}
  </div>`;
}

// Resolve the recommended frame-shape guidance (backend result, or Oval fallback).
function shapeAdvice() {
  const a = state.analysis;
  if (a && Array.isArray(a.recommended_shapes) && a.recommended_shapes.length) {
    return {
      recommended: a.recommended_shapes.map(x => ({
        tier: x.tier,
        tierLabel: he() ? x.tier_label.he : x.tier_label.en,
        friendly: he() ? x.friendly.he : x.friendly.en,
        why: he() ? x.why.he : x.why.en,
      })),
      cautions: (a.shape_cautions || []).map(c => he() ? c.he : c.en),
    };
  }
  return DEFAULT_SHAPE_ADVICE[he() ? "he" : "en"];
}

// "Recommended Frame Shapes" card: ranked, tiered guidance + soft cautions, so
// the customer sees WHAT to choose and WHY — not just the detected face shape.
function shapeAdviceHTML() {
  const t = T(), adv = shapeAdvice();
  if (!adv || !adv.recommended.length) return "";
  const list = adv.recommended.slice(0, 5);
  const best = list[0];
  const chips = list.slice(0, 3).map(r => esc(r.friendly)).join("  ·  ");
  const badge = (r) => {
    const s = TIER_STYLE[r.tier] || TIER_STYLE.recommended;
    return `<span style="display:inline-flex;align-items:center;padding:4px 10px;border-radius:6px;background:${s.bg};color:${s.fg};border:1px solid ${s.bc||"transparent"};font-size:11px;font-weight:800;letter-spacing:.4px;white-space:nowrap">${esc(r.tierLabel)}</span>`;
  };
  const rows = list.map(r => `
    <div style="padding:12px 14px;border-radius:11px;background:#0a1017;border:1px solid #16222f">
      <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;flex-wrap:wrap">
        <span style="font-size:15.5px;font-weight:700;letter-spacing:-.2px">${esc(r.friendly)}</span>
        ${badge(r)}
      </div>
      <div style="font-size:13px;line-height:1.5;color:#8496a9;margin-top:6px">${esc(r.why)}</div>
    </div>`).join("");
  const cautions = adv.cautions.length ? `
    <div style="margin-top:14px;padding:13px 15px;border-radius:11px;background:#0a0f16;border:1px dashed #22323f">
      <div style="font-size:12.5px;color:#7d8fa2;letter-spacing:.3px;text-transform:uppercase;margin-bottom:8px">${esc(t.shapesLessTitle)}</div>
      ${adv.cautions.map(c => `<div style="display:flex;gap:8px;font-size:13px;line-height:1.5;color:#77899c;margin-top:4px"><span style="color:#4a5b6e;flex:none">–</span><span>${esc(c)}</span></div>`).join("")}
    </div>` : "";
  return `
  <div style="margin-top:16px;padding:22px 24px;border-radius:14px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f">
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px"><span style="color:#35dff2;display:flex">${svgSpark(18,"#35dff2")}</span><h3 style="margin:0;font-size:19px;font-weight:700">${esc(t.shapesTitle)}</h3></div>
    <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-bottom:14px;padding:12px 15px;border-radius:11px;background:linear-gradient(100deg,rgba(53,223,242,.10),transparent);border:1px solid #1d3a45">
      <span style="font-size:12.5px;color:#7d8fa2;letter-spacing:.3px;text-transform:uppercase;flex:none">${esc(t.shapesBestFor)}</span>
      <span style="font-size:17px;font-weight:800;color:#35dff2;letter-spacing:-.3px">${esc(best.friendly)}</span>
      <span style="font-size:13.5px;color:#8fa2b4;flex-basis:100%;line-height:1.5">${esc(best.why)}</span>
    </div>
    <div style="display:flex;flex-direction:column;gap:10px">${rows}</div>
    ${cautions}
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Try-On
// ---------------------------------------------------------------------------
function tryOnHTML() {
  const t = T();
  const cur = find(state.currentId);
  const camLive = state.camera === "live";
  const liveDot = camLive ? "#35dff2" : "#f0a935";
  const liveLabel = camLive ? "LIVE" : t.photo;
  const insights = buildInsights().map(i => `
    <div class="of-card hv-b26" style="padding:15px 15px 17px;border-radius:11px;background:#090f16;border:1px solid #16222f">
      <div style="display:flex;align-items:center;gap:9px;margin-bottom:9px"><span style="color:#35dff2;font-size:17px">${i.icon}</span><span style="font-size:13px;color:#7d8fa2">${esc(i.k)}</span></div>
      <div style="font-size:16px;font-weight:700;line-height:1.3;margin-bottom:8px">${esc(i.v)}</div>
      <div style="font-size:13px;line-height:1.5;color:#77899c">${esc(i.d)}</div>
    </div>`).join("");
  const recs = buildRecs().map((r, i) => {
    const isFav = state.favs.includes(r.id);
    const badge = r.first ? t.bestMatch : "#" + (i+1);
    const badgeBg = r.first ? "#35dff2" : "#16222f", badgeColor = r.first ? "#03181e" : "#9dafc0";
    const bg = r.first ? "linear-gradient(140deg,rgba(53,223,242,.09),#0a1017)" : "#0a1017";
    const border = r.first ? "#2c93ad" : "#16222f";
    const btnBg = r.first ? "linear-gradient(100deg,#35dff2,#12a9cd)" : "#0f1720";
    const btnBorder = r.first ? "transparent" : "#22323f", btnColor = r.first ? "#03181e" : "#dbe6f0";
    return `
    <div style="padding:16px;border-radius:13px;background:${bg};border:1px solid ${border};display:grid;grid-template-columns:200px 1fr;gap:14px;align-items:start">
      <div>
        <div style="display:inline-flex;align-items:center;padding:5px 11px;border-radius:7px;background:${badgeBg};color:${badgeColor};font-size:11.5px;font-weight:800;letter-spacing:.7px">${esc(badge)}</div>
        <img src="${r.img}" alt="" style="width:100%;height:96px;object-fit:contain;margin-top:10px" />
      </div>
      <div>
        <div style="display:flex;align-items:flex-start;justify-content:space-between;gap:10px">
          <div style="font-size:19px;font-weight:700;letter-spacing:-.3px;line-height:1.2">${esc(r.name)}</div>
          <div data-act="fav" data-arg="${r.id}" style="cursor:pointer;color:${isFav?"#35dff2":"#65788c"};flex:none">${svgHeart(19, isFav?"#35dff2":"none")}</div>
        </div>
        <div style="font-size:14.5px;color:#8496a9;margin-top:3px">${esc(r.meta)}</div>
        <div style="display:flex;align-items:baseline;gap:7px;margin-top:9px"><span style="font-size:25px;font-weight:800;color:#35dff2;letter-spacing:-.8px">${r.score}%</span><span style="font-size:14.5px;color:#9dafc0">${esc(t.match)}</span></div>
        <div style="margin-top:9px">
          <div style="font-size:12px;color:#7d8fa2;letter-spacing:.3px;text-transform:uppercase;margin-bottom:6px">${esc(t.whyThisFrame)}</div>
          ${(r.whyList || [r.why]).slice(0,3).map(w => `<div style="display:flex;gap:7px;font-size:13px;line-height:1.45;color:#8496a9;margin-top:4px"><span style="color:#35dff2;flex:none;margin-top:2px">${svgCheck(12)}</span><span>${esc(w)}</span></div>`).join("")}
        </div>
        <div class="hv-bright" data-act="tryOn" data-arg="${r.id}" style="margin-top:13px;display:flex;align-items:center;justify-content:space-between;padding:12px 18px;border-radius:9px;background:${btnBg};border:1px solid ${btnBorder};color:${btnColor};font-size:15.5px;font-weight:700;cursor:pointer"><span>${esc(t.tryOn)}</span>${svgArrow(16)}</div>
      </div>
    </div>`;
  }).join("");

  // Show the try-on stage only when we have real input: a live camera, or a
  // photo the user actually took/uploaded. Otherwise prompt them to add one.
  const hasStage = camLive || state.hasPhoto;
  const promptMsg = state.promptError === "noFace" ? t.errNoFace
                  : state.promptError === "camDenied" ? t.errCamDenied : "";
  const promptErrHTML = promptMsg ? `
      <div role="alert" style="display:flex;align-items:center;gap:9px;margin-bottom:18px;padding:11px 16px;border-radius:10px;background:rgba(240,120,90,.09);border:1px solid #5a3330;color:#f0a58f;font-size:14px;max-width:400px"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="flex:none" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 8v5M12 16.5v.5"/></svg><span>${esc(promptMsg)}</span></div>` : "";
  const photoPrompt = `
    <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:32px;gap:6px">
      <div style="width:78px;height:78px;margin-bottom:16px;border-radius:22px;background:rgba(53,223,242,.09);border:1px solid #1d3a45;color:#35dff2;display:flex;align-items:center;justify-content:center">${svgCam(36)}</div>
      <h3 style="margin:0;font-size:24px;font-weight:800;letter-spacing:-.4px">${esc(t.photoPromptTitle)}</h3>
      <p style="margin:4px 0 22px;font-size:15.5px;line-height:1.55;color:#8b9dae;max-width:360px">${esc(t.photoPromptSub)}</p>
      ${promptErrHTML}
      <div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center">
        <div class="hv-bright" data-act="allow" style="display:flex;align-items:center;gap:9px;padding:14px 22px;border-radius:11px;background:linear-gradient(100deg,#35dff2,#12a9cd);color:#03181e;font-size:15.5px;font-weight:700;cursor:pointer">${svgCam(18)}<span>${esc(t.takePhoto)}</span></div>
        <div class="hv-cyanbc" data-act="usePhoto" style="display:flex;align-items:center;gap:9px;padding:14px 22px;border-radius:11px;border:1px solid #22323f;color:#cad7e3;font-size:15.5px;font-weight:600;cursor:pointer"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M4 16l4.6-4.6a2 2 0 0 1 2.8 0L16 16"/><path d="M14 14l1.6-1.6a2 2 0 0 1 2.8 0L20 14"/><rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/></svg><span>${esc(t.uploadPhoto)}</span></div>
      </div>
    </div>`;
  const stageInner = camLive
    ? `<video id="ofVideo" playsinline muted style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)"></video>`
    : state.hasPhoto
      ? `<img id="ofPhoto" src="${state.photoSrc}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover" onerror="this.style.opacity=0" />`
      : photoPrompt;
  const overlayStatic = camLive ? "" : `left:50%;top:40%;width:42%;`;

  // Analysis details render full-width BELOW the grid, so the try-on photo can
  // stay pinned (sticky) while the frame list on the side scrolls.
  const analysisSection = `
    <div style="padding:0 34px 8px">
      <div style="padding:22px 24px;border-radius:14px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:18px"><span style="color:#35dff2;display:flex">${svgSpark(18,"#35dff2")}</span><h3 style="margin:0;font-size:19px;font-weight:700">${esc(t.insightsTitle)}</h3></div>
        <div data-responsive="grid3" style="display:grid;grid-template-columns:repeat(6,1fr);gap:12px">${insights}</div>
      </div>
      ${shapeAdviceHTML()}
      ${colorAnalysisHTML()}
      <div style="display:flex;align-items:center;justify-content:center;gap:18px;padding:20px 0 8px;font-size:13.5px;color:#63768a;flex-wrap:wrap">
        <span>${esc(t.footNote)}</span><span style="color:#243343">|</span>
        <span style="display:flex;align-items:center;gap:7px"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><rect x="4" y="10" width="16" height="10" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3"/></svg><span>${esc(t.footPrivacy)}</span></span>
      </div>
    </div>`;

  return `
  <div style="flex:1">
    <div data-responsive="grid2" style="display:grid;grid-template-columns:1fr 520px;gap:22px;padding:26px 34px 18px;align-items:stretch">
    <div>
      <div style="display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:18px;gap:16px;flex-wrap:wrap">
        <div>
          <h1 style="margin:0 0 6px;font-size:38px;font-weight:800;letter-spacing:-1.1px">${esc(t.tryTitle)}</h1>
          <p style="margin:0;font-size:16px;color:#8b9dae">${esc(t.trySub)}</p>
        </div>
        <div style="display:flex;align-items:center;gap:10px;padding:11px 20px;border-radius:10px;border:1px solid #1d3a45;background:rgba(53,223,242,.05);color:#35dff2;font-size:15px;font-weight:600"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.2l2.4 2.4 4.6-4.8"/></svg><span>${esc(t.tryAnalysisDone)}</span></div>
      </div>

      <div class="of-tryon-sticky" style="position:sticky;top:88px">
      <div style="position:relative;border-radius:16px;overflow:hidden;border:1px solid #1a2735;background:#080d13;aspect-ratio:16/10.6">
        ${stageInner}
        ${hasStage ? `<img id="ofOverlay" src="${imgOf(cur)}" alt="${esc(nameOf(cur))}" style="position:absolute;${overlayStatic}transform:translate(-50%,-50%);pointer-events:${state.adjust.on?"auto":"none"};cursor:${state.adjust.on?"grab":"default"};touch-action:none;filter:drop-shadow(0 6px 14px rgba(0,0,0,.55))" />
        <div id="ofDetecting" role="status" aria-live="polite" style="position:absolute;top:16px;right:16px;display:flex;align-items:center;gap:9px;padding:8px 15px;border-radius:9px;background:rgba(6,11,16,.78);backdrop-filter:blur(8px);font-size:13px;font-weight:600;color:#cbd8e5"><span style="width:14px;height:14px;border-radius:50%;border:2px solid #35dff2;border-top-color:transparent;animation:ofSpin 1s linear infinite"></span><span>${esc(t.detecting)}</span></div>
        <div style="position:absolute;top:16px;left:16px;display:flex;align-items:center;gap:9px;padding:8px 15px;border-radius:9px;background:rgba(6,11,16,.78);backdrop-filter:blur(8px);font-size:13px;font-weight:700;letter-spacing:.6px"><span style="width:8px;height:8px;border-radius:50%;background:${liveDot};animation:ofPulse 1.6s infinite"></span><span>${esc(liveLabel)}</span></div>
        <div style="position:absolute;bottom:16px;left:16px;display:flex;gap:10px">
          <div class="hv-cyanb" data-act="retake" style="display:flex;align-items:center;gap:10px;padding:11px 17px;border-radius:9px;background:rgba(9,15,21,.8);border:1px solid #1e2c3a;backdrop-filter:blur(8px);font-size:14.5px;font-weight:600;cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M21 8a9 9 0 0 0-15-3.7L3 7"/><path d="M3 3v4h4"/><path d="M3 16a9 9 0 0 0 15 3.7L21 17"/><path d="M21 21v-4h-4"/></svg><span>${esc(t.changePhoto)}</span></div>
          <div class="hv-cyanb" data-act="adjust" style="display:flex;align-items:center;gap:9px;padding:11px 17px;border-radius:9px;background:${state.adjust.on?"rgba(53,223,242,.14)":"rgba(9,15,21,.8)"};border:1px solid ${state.adjust.on?"#35dff2":"#1e2c3a"};color:${state.adjust.on?"#35dff2":"inherit"};backdrop-filter:blur(8px);font-size:14.5px;font-weight:600;cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M4 21v-7M4 10V3M12 21v-9M12 8V3M20 21v-5M20 12V3M1 14h6M9 8h6M17 16h6"/></svg><span>${esc(state.adjust.on?t.adjustDone:t.adjustBtn)}</span></div>
        </div>
        ${state.adjust.on ? `<div style="position:absolute;bottom:74px;left:50%;transform:translateX(-50%);display:flex;align-items:center;gap:8px;padding:9px 12px;border-radius:11px;background:rgba(6,11,16,.9);border:1px solid #223444;backdrop-filter:blur(8px)">
          <span style="font-size:12.5px;color:#9db0c2;padding:0 6px">${esc(t.adjustHint)}</span>
          <div class="hv-cyanb" data-act="adjSize" data-arg="-3" title="−" style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:8px;border:1px solid #223444;cursor:pointer"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M5 12h14"/></svg></div>
          <div class="hv-cyanb" data-act="adjSize" data-arg="3" title="+" style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:8px;border:1px solid #223444;cursor:pointer"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M12 5v14M5 12h14"/></svg></div>
          <div class="hv-cyanb" data-act="adjRot" data-arg="-3" title="⟲" style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:8px;border:1px solid #223444;cursor:pointer"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M3 8a9 9 0 0 1 15-3.7L21 7"/><path d="M21 3v4h-4"/></svg></div>
          <div class="hv-cyanb" data-act="adjRot" data-arg="3" title="⟳" style="width:36px;height:36px;display:flex;align-items:center;justify-content:center;border-radius:8px;border:1px solid #223444;cursor:pointer"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" style="transform:scaleX(-1)"><path d="M3 8a9 9 0 0 1 15-3.7L21 7"/><path d="M21 3v4h-4"/></svg></div>
          <div class="hv-cyanbc" data-act="adjReset" style="padding:0 13px;height:36px;display:flex;align-items:center;border-radius:8px;border:1px solid #223444;font-size:13px;font-weight:600;cursor:pointer">${esc(t.adjustReset)}</div>
        </div>` : ""}
        <div style="position:absolute;bottom:16px;right:16px;display:flex;gap:10px">
          ${camLive ? `<div class="hv-cyanb" data-act="flip" title="${esc(t.flipCamera)}" style="display:flex;align-items:center;justify-content:center;width:46px;border-radius:9px;background:rgba(9,15,21,.8);border:1px solid #1e2c3a;backdrop-filter:blur(8px);cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M3 12a9 9 0 0 1 15-6.7L21 8"/><path d="M21 4v4h-4"/><path d="M21 12a9 9 0 0 1-15 6.7L3 16"/><path d="M3 20v-4h4"/></svg></div>` : ""}
          <div class="hv-cyanb" data-act="snapshot" style="display:flex;align-items:center;gap:10px;padding:11px 17px;border-radius:9px;background:rgba(9,15,21,.8);border:1px solid #1e2c3a;backdrop-filter:blur(8px);font-size:14.5px;font-weight:600;cursor:pointer">${svgCam(17)}<span>${esc(t.snapshot)}</span></div>
          <div class="hv-cyanb" data-act="share" title="${esc(t.shareBtn)}" style="display:flex;align-items:center;justify-content:center;width:46px;border-radius:9px;background:rgba(9,15,21,.8);border:1px solid #1e2c3a;backdrop-filter:blur(8px);cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><path d="M8.6 13.5l6.8 4M15.4 6.5l-6.8 4"/></svg></div>
          <div class="hv-cyanb" data-act="full" style="display:flex;align-items:center;justify-content:center;width:46px;border-radius:9px;background:rgba(9,15,21,.8);border:1px solid #1e2c3a;backdrop-filter:blur(8px);cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M9 3H3v6M15 3h6v6M9 21H3v-6M15 21h6v-6"/></svg></div>
        </div>` : ""}
      </div>

      <div style="margin-top:16px;display:flex;align-items:center;gap:30px;padding:20px 24px;border-radius:14px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f;flex-wrap:wrap">
        <img src="${imgOf(cur)}" alt="" style="width:112px;height:56px;object-fit:contain" />
        <div style="flex:1;min-width:160px">
          <div style="font-size:13.5px;color:#7d8fa2;margin-bottom:3px">${esc(t.tryingOn)}</div>
          <div style="font-size:22px;font-weight:700;letter-spacing:-.4px">${esc(nameOf(cur))}</div>
          <div style="font-size:14.5px;color:#8496a9;margin-top:2px">${esc(metaOf(cur))}</div>
        </div>
        <div style="padding-inline-end:30px;border-inline-end:1px solid #1b2836">
          <div style="font-size:13.5px;color:#7d8fa2;margin-bottom:2px">${esc(t.fitScore)}</div>
          <div style="font-size:34px;font-weight:800;color:#35dff2;line-height:1.05;letter-spacing:-1px">${score(cur)}%</div>
          <div style="display:flex;align-items:center;gap:7px;font-size:13.5px;color:#35dff2;margin-top:4px"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><circle cx="12" cy="12" r="9"/><path d="M8.5 12.2l2.4 2.4 4.6-4.8"/></svg><span>${esc(t.excellentMatch)}</span></div>
        </div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          ${state.hasPhoto ? `<div class="hv-cyanbc" data-act="compare" style="display:flex;align-items:center;gap:9px;padding:14px 22px;border-radius:10px;border:1px solid #22323f;font-size:15.5px;font-weight:600;cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><rect x="3" y="4" width="7" height="16" rx="1.5"/><rect x="14" y="4" width="7" height="16" rx="1.5"/></svg><span>${esc(t.compareBtn)}</span></div>` : ""}
          <div class="hv-cyanbc" data-act="viewDetails" style="display:flex;align-items:center;gap:12px;padding:14px 26px;border-radius:10px;border:1px solid #22323f;font-size:15.5px;font-weight:600;cursor:pointer"><span>${esc(t.viewDetails)}</span>${svgArrow(16)}</div>
        </div>
      </div>
      </div>
    </div>

    <aside style="padding:22px 20px;border-radius:16px;background:linear-gradient(180deg,#0b1118,#080d13);border:1px solid #16222f;position:sticky;top:88px">
      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:18px">
        <div style="display:flex;align-items:center;gap:11px"><span style="color:#35dff2;display:flex">${svgSpark(19,"#35dff2")}</span><h3 style="margin:0;font-size:20px;font-weight:700;letter-spacing:-.3px">${esc(t.recTitle)}</h3></div>
        <div class="hv-cyan" style="display:flex;align-items:center;gap:7px;font-size:13.5px;color:#8b9dae;cursor:pointer"><span>${esc(t.whyThese)}</span><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></svg></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:13px">${recs}</div>
      ${moreMatchesHTML()}
      <div class="hv-cyanb" data-act="go" data-arg="collection" style="margin-top:14px;display:flex;align-items:center;gap:16px;padding:18px 20px;border-radius:13px;border:1px solid #1a2735;background:#090f16;cursor:pointer">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" stroke="#35dff2" stroke-width="1.6"><circle cx="6.5" cy="13" r="3.6"/><circle cx="17.5" cy="13" r="3.6"/><path d="M10.1 12.6c.8-.7 3-.7 3.8 0M1.6 9.6c1-.6 2-.9 2.9-1M22.4 9.6c-1-.6-2-.9-2.9-1"/></svg>
        <div style="flex:1"><div style="font-size:17.5px;font-weight:700">${esc(t.exploreTitle)}</div><div style="font-size:14px;color:#8496a9;margin-top:2px">${esc(t.exploreSub)}</div></div>
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#8496a9" stroke-width="2"><path d="M9 5l7 7-7 7"/></svg>
      </div>
    </aside>
    </div>
    ${analysisSection}
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Collection
// ---------------------------------------------------------------------------
function frameCard(f) {
  const t = T();
  const isFav = state.favs.includes(f.id);
  return `
  <div class="of-card hv-b2b" style="border-radius:14px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f;overflow:hidden">
    <div data-act="open" data-arg="${f.id}" style="position:relative;height:150px;display:flex;align-items:center;justify-content:center;background:radial-gradient(closest-side,rgba(53,223,242,.07),transparent);cursor:pointer">
      <img src="${f.image}" alt="" style="width:78%;object-fit:contain" />
      <div data-act="fav" data-arg="${f.id}" style="position:absolute;top:12px;inset-inline-end:12px;width:32px;height:32px;border-radius:50%;background:rgba(6,11,16,.75);display:flex;align-items:center;justify-content:center;color:${isFav?"#35dff2":"#8fa2b4"};cursor:pointer">${svgHeart(16, isFav?"#35dff2":"none")}</div>
    </div>
    <div style="padding:16px 17px 18px;border-top:1px solid #131e29">
      <div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px">
        <div style="font-size:16.5px;font-weight:700;letter-spacing:-.2px">${esc(f.name)}</div>
        <div style="font-size:15px;font-weight:800;color:#35dff2">${score(f)}%</div>
      </div>
      <div style="font-size:13.5px;color:#7d8fa2;margin-top:3px">${esc(metaOf(f))}</div>
      <div class="hv-cyanbc" data-act="tryOn" data-arg="${f.id}" style="margin-top:13px;padding:10px;border-radius:8px;border:1px solid #22323f;text-align:center;font-size:14.5px;font-weight:600;cursor:pointer">${esc(t.tryOn)}</div>
    </div>
  </div>`;
}
function collectionHTML() {
  const t = T();
  const filters = t.filters.map(([k,label]) => {
    const on = state.filter === k;
    return `<div class="hv-cyanb" data-act="filter" data-arg="${k}" style="padding:10px 20px;border-radius:9px;border:1px solid ${on?"#2c93ad":"#1b2836"};background:${on?"rgba(53,223,242,.10)":"#090f16"};color:${on?"#35dff2":"#a9bacb"};font-size:14.5px;font-weight:600;cursor:pointer">${esc(label)}</div>`;
  }).join("");
  const list = state.catalog.filter(f => state.filter === "all" ? true : (f.category === state.filter || f.material === state.filter));
  const grid = list.map(frameCard).join("") || `<div style="color:#7d8fa2;padding:40px">…</div>`;
  return `
  <div style="flex:1;padding:44px 60px 70px;animation:ofRise .35s ease both">
    <h1 style="margin:0 0 8px;font-size:42px;font-weight:800;letter-spacing:-1.3px">${esc(t.colTitle)}</h1>
    <p style="margin:0 0 26px;font-size:17px;color:#8b9dae">${esc(t.colSub)}</p>
    <div style="display:flex;gap:10px;margin-bottom:26px;flex-wrap:wrap">${filters}</div>
    <div data-responsive="grid4" style="display:grid;grid-template-columns:repeat(4,1fr);gap:18px">${grid}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Details
// ---------------------------------------------------------------------------
function detailsHTML() {
  const t = T();
  const d = find(state.detailId);
  if (!d) return `<div style="flex:1;padding:60px">…</div>`;
  const shapeMap = he()?SHAPE_HE:SHAPE_EN, matMap = he()?MAT_HE:MAT_EN, rimMap = he()?RIM_HE:RIM_EN;
  const specVals = [
    shapeMap[d.shape]||d.shape, matMap[d.material]||d.material, rimMap[d.rim_thickness]||d.rim_thickness,
    d.color_name || d.color, (d.lens_width_mm!=null?d.lens_width_mm+" mm":"—"),
    (d.bridge_width_mm!=null?d.bridge_width_mm+" mm":"—"), (d.temple_length_mm!=null?d.temple_length_mm+" mm":"—"),
    d.category==="sun"?(he()?"משקפי שמש":"Sunglasses"):(he()?"משקפי ראייה":"Eyeglasses"),
  ];
  const specs = t.specKeys.map((k,i) => `<div style="padding:15px 17px;border-radius:11px;background:#090f16;border:1px solid #16222f"><div style="font-size:12.5px;color:#7d8fa2;letter-spacing:.4px;text-transform:uppercase">${esc(k)}</div><div style="font-size:16.5px;font-weight:700;margin-top:4px">${esc(specVals[i])}</div></div>`).join("");
  const isFav = state.favs.includes(d.id);
  const why = WHY[state.lang][d.id] || (he()?"מידות שיושבות טוב על מבנה הפנים שלך":"Dimensions that sit well on your face structure");
  return `
  <div style="flex:1;padding:34px 60px 70px;animation:ofRise .35s ease both">
    <div class="hv-cyan" data-act="back" style="display:inline-flex;align-items:center;gap:9px;font-size:15px;color:#8b9dae;cursor:pointer;margin-bottom:24px"><svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M15 5l-7 7 7 7"/></svg><span>${esc(t.back)}</span></div>
    <div data-responsive="grid2" style="display:grid;grid-template-columns:1.15fr 1fr;gap:44px;align-items:start">
      <div style="border-radius:18px;border:1px solid #16222f;background:radial-gradient(closest-side,rgba(53,223,242,.09),#080e14);height:420px;display:flex;align-items:center;justify-content:center"><img src="${d.image}" alt="" style="width:72%;object-fit:contain;filter:drop-shadow(0 20px 40px rgba(0,0,0,.6))" /></div>
      <div>
        <div style="font-size:13.5px;letter-spacing:1.4px;text-transform:uppercase;color:#35dff2;font-weight:700;margin-bottom:10px">${esc(d.brand)}</div>
        <h1 style="margin:0 0 8px;font-size:44px;font-weight:800;letter-spacing:-1.4px;line-height:1.1">${esc(nameOf(d))}</h1>
        <p style="margin:0 0 22px;font-size:17px;color:#8b9dae">${esc(metaOf(d))}</p>
        <div style="display:flex;align-items:center;gap:16px;padding:20px 22px;border-radius:13px;background:linear-gradient(100deg,rgba(53,223,242,.10),transparent);border:1px solid #1d3a45;margin-bottom:26px">
          <div style="font-size:40px;font-weight:800;color:#35dff2;letter-spacing:-1.4px">${score(d)}%</div>
          <div><div style="font-size:16px;font-weight:700">${esc(t.matchForYou)}</div><div style="font-size:14px;color:#8b9dae;margin-top:2px">${esc(why)}</div></div>
        </div>
        <div data-responsive="grid2" style="display:grid;grid-template-columns:repeat(2,1fr);gap:12px;margin-bottom:28px">${specs}</div>
        <div style="display:flex;gap:12px;flex-wrap:wrap">
          <div class="hv-bright" data-act="tryOn" data-arg="${d.id}" style="flex:1;min-width:180px;padding:16px;border-radius:11px;background:linear-gradient(100deg,#35dff2,#12a9cd);color:#03181e;font-size:16.5px;font-weight:700;text-align:center;cursor:pointer">${esc(t.tryOnLive)}</div>
          <div class="hv-cyanb" data-act="fav" data-arg="${d.id}" style="display:flex;align-items:center;gap:10px;padding:16px 24px;border-radius:11px;border:1px solid #22323f;font-size:16px;font-weight:600;cursor:pointer;color:${isFav?"#35dff2":"#dbe6f0"}">${svgHeart(18, isFav?"#35dff2":"none")}<span>${esc(isFav?t.inFav:t.addFav)}</span></div>
        </div>
      </div>
    </div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Favorites
// ---------------------------------------------------------------------------
function favoritesHTML() {
  const t = T();
  const favFrames = state.catalog.filter(f => state.favs.includes(f.id));
  let body;
  if (favFrames.length) {
    const cards = favFrames.map(g => {
      return `
      <div style="border-radius:14px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f;overflow:hidden">
        <div data-act="open" data-arg="${g.id}" style="height:150px;display:flex;align-items:center;justify-content:center;background:radial-gradient(closest-side,rgba(53,223,242,.07),transparent);cursor:pointer"><img src="${g.image}" alt="" style="width:78%;object-fit:contain" /></div>
        <div style="padding:16px 17px 18px;border-top:1px solid #131e29">
          <div style="font-size:16.5px;font-weight:700">${esc(g.name)}</div>
          <div style="font-size:13.5px;color:#7d8fa2;margin-top:3px">${esc(metaOf(g))}</div>
          <div style="display:flex;gap:9px;margin-top:13px">
            <div class="hv-cyanbc" data-act="tryOn" data-arg="${g.id}" style="flex:1;padding:10px;border-radius:8px;border:1px solid #22323f;text-align:center;font-size:14.5px;font-weight:600;cursor:pointer">${esc(t.tryOn)}</div>
            <div class="hv-cyanb" data-act="fav" data-arg="${g.id}" style="padding:10px 13px;border-radius:8px;border:1px solid #22323f;color:#35dff2;cursor:pointer">${svgHeart(16,"#35dff2")}</div>
          </div>
        </div>
      </div>`;
    }).join("");
    body = `<div data-responsive="grid4" style="display:grid;grid-template-columns:repeat(4,1fr);gap:18px">${cards}</div>`;
  } else {
    body = `
    <div style="padding:80px;border-radius:16px;border:1px dashed #1e2c3a;text-align:center">
      <div style="font-size:20px;font-weight:700;margin-bottom:8px">${esc(t.favEmptyTitle)}</div>
      <div style="font-size:15.5px;color:#8b9dae;margin-bottom:22px">${esc(t.favEmptySub)}</div>
      <div class="hv-bright" data-act="go" data-arg="collection" style="display:inline-block;padding:14px 28px;border-radius:10px;background:linear-gradient(100deg,#35dff2,#12a9cd);color:#03181e;font-weight:700;cursor:pointer">${esc(t.homeCta2)}</div>
    </div>`;
  }
  return `
  <div style="flex:1;padding:44px 60px 70px;animation:ofRise .35s ease both">
    <h1 style="margin:0 0 8px;font-size:42px;font-weight:800;letter-spacing:-1.3px">${esc(t.favTitle)}</h1>
    <p style="margin:0 0 28px;font-size:17px;color:#8b9dae">${state.favs.length} ${esc(t.favSub)}</p>
    ${body}
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Profile
// ---------------------------------------------------------------------------
function profileHTML() {
  const t = T();
  const facts = t.profileFacts.map(([k,v]) => `<div style="padding:15px 16px;border-radius:11px;background:#090f16;border:1px solid #16222f"><div style="font-size:12.5px;color:#7d8fa2;letter-spacing:.4px;text-transform:uppercase">${esc(k)}</div><div style="font-size:17px;font-weight:700;margin-top:4px">${esc(v)}</div></div>`).join("");
  const tags = t.prefTags.map(tag => `<div style="padding:9px 17px;border-radius:20px;border:1px solid #1d3a45;background:rgba(53,223,242,.06);color:#9fe9f6;font-size:14.5px;font-weight:600">${esc(tag)}</div>`).join("");
  return `
  <div style="flex:1;padding:44px 60px 70px;animation:ofRise .35s ease both">
    <h1 style="margin:0 0 28px;font-size:42px;font-weight:800;letter-spacing:-1.3px">${esc(t.profTitle)}</h1>
    <div data-responsive="grid2" style="display:grid;grid-template-columns:340px 1fr;gap:22px;align-items:start">
      <div style="padding:28px;border-radius:16px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f;text-align:center">
        <div style="width:120px;height:120px;margin:0 auto 18px;border-radius:50%;overflow:hidden;border:2px solid #1d3a45"><img src="${state.photoSrc}" alt="" style="width:100%;height:100%;object-fit:cover" onerror="this.style.opacity=0" /></div>
        <div style="font-size:22px;font-weight:700">${esc(t.profName)}</div>
        <div style="font-size:14.5px;color:#8b9dae;margin-top:4px">${esc(t.profSince)}</div>
        <div class="hv-cyanbc" data-act="startFlow" style="margin-top:20px;padding:13px;border-radius:10px;border:1px solid #22323f;font-size:15px;font-weight:600;cursor:pointer">${esc(t.profRescan)}</div>
      </div>
      <div style="display:flex;flex-direction:column;gap:18px">
        <div style="padding:24px 26px;border-radius:16px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f">
          <h3 style="margin:0 0 16px;font-size:19px;font-weight:700">${esc(t.profFaceTitle)}</h3>
          <div data-responsive="grid4" style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px">${facts}</div>
        </div>
        <div style="padding:24px 26px;border-radius:16px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f">
          <h3 style="margin:0 0 16px;font-size:19px;font-weight:700">${esc(t.profPrefTitle)}</h3>
          <div style="display:flex;flex-wrap:wrap;gap:10px">${tags}</div>
        </div>
      </div>
    </div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: About
// ---------------------------------------------------------------------------
function aboutHTML() {
  const t = T();
  const steps = t.aboutSteps.map(([n,tt,d]) => `<div style="padding:28px;border-radius:16px;background:linear-gradient(180deg,#0c131b,#080e14);border:1px solid #16222f"><div style="font-size:13px;letter-spacing:1.4px;color:#35dff2;font-weight:700;margin-bottom:12px">${esc(n)}</div><h4 style="margin:0 0 9px;font-size:19px;font-weight:700">${esc(tt)}</h4><p style="margin:0;font-size:15.5px;line-height:1.65;color:#8496a9">${esc(d)}</p></div>`).join("");
  const tags = t.techTags.map(tag => `<div style="padding:11px 20px;border-radius:9px;border:1px solid #1b2836;background:#090f16;font-size:14.5px;color:#c3d1de;font-weight:500">${esc(tag)}</div>`).join("");
  return `
  <div style="flex:1;padding:60px 60px 80px;animation:ofRise .35s ease both">
    <div style="max-width:760px;margin-bottom:52px">
      <h1 style="margin:0 0 18px;font-size:52px;font-weight:800;letter-spacing:-1.8px;line-height:1.1">${esc(t.aboutTitle)}</h1>
      <p style="margin:0;font-size:19px;line-height:1.7;color:#93a5b8">${esc(t.aboutBody)}</p>
    </div>
    <h3 style="margin:0 0 20px;font-size:22px;font-weight:700">${esc(t.aboutHow)}</h3>
    <div data-responsive="grid3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;margin-bottom:52px">${steps}</div>
    <h3 style="margin:0 0 20px;font-size:22px;font-weight:700">${esc(t.aboutTech)}</h3>
    <div style="display:flex;flex-wrap:wrap;gap:11px">${tags}</div>
  </div>`;
}

// ---------------------------------------------------------------------------
// Screen: Compare — the top-3 frames placed on the user's photo, side by side
// ---------------------------------------------------------------------------
function compareHTML() {
  const t = T();
  const recs = buildRecs().slice(0, 3);
  const cols = recs.map((r, i) => {
    const badge = r.first ? t.bestMatch : "#" + (i + 1);
    const badgeBg = r.first ? "#35dff2" : "#16222f", badgeColor = r.first ? "#03181e" : "#9dafc0";
    const border = r.first ? "#2c93ad" : "#1a2735";
    return `
    <div style="display:flex;flex-direction:column;gap:12px">
      <div class="of-compare-stage" style="position:relative;border-radius:14px;overflow:hidden;border:1px solid ${border};background:#080d13;aspect-ratio:3/4">
        <img class="of-compare-photo" src="${state.photoSrc}" alt="" style="position:absolute;inset:0;width:100%;height:100%;object-fit:cover" onerror="this.style.opacity=0" />
        <img class="of-compare-frame" data-frame="${r.id}" src="${imgOf(find(r.id))}" alt="${esc(r.name)}" style="position:absolute;left:50%;top:40%;width:42%;transform:translate(-50%,-50%);pointer-events:none;filter:drop-shadow(0 6px 14px rgba(0,0,0,.55))" />
        <div style="position:absolute;top:12px;inset-inline-start:12px;padding:5px 11px;border-radius:7px;background:${badgeBg};color:${badgeColor};font-size:11.5px;font-weight:800;letter-spacing:.6px">${esc(badge)}</div>
      </div>
      <div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px">
        <div style="font-size:17px;font-weight:700;letter-spacing:-.3px">${esc(r.name)}</div>
        <div style="font-size:19px;font-weight:800;color:#35dff2">${r.score}%</div>
      </div>
      <div style="font-size:13.5px;color:#8496a9;margin-top:-4px">${esc(r.meta)}</div>
      <div class="hv-bright" data-act="tryOn" data-arg="${r.id}" style="margin-top:4px;padding:12px;border-radius:9px;background:${r.first?"linear-gradient(100deg,#35dff2,#12a9cd)":"#0f1720"};border:1px solid ${r.first?"transparent":"#22323f"};color:${r.first?"#03181e":"#dbe6f0"};font-size:14.5px;font-weight:700;text-align:center;cursor:pointer">${esc(t.tryThis)}</div>
    </div>`;
  }).join("");
  return `
  <div style="flex:1;padding:26px 34px 30px;animation:ofRise .35s ease both">
    <div style="display:flex;align-items:flex-end;justify-content:space-between;gap:16px;flex-wrap:wrap;margin-bottom:22px">
      <div>
        <h1 style="margin:0 0 6px;font-size:34px;font-weight:800;letter-spacing:-1px">${esc(t.compareTitle)}</h1>
        <p style="margin:0;font-size:15.5px;color:#8b9dae">${esc(t.compareSub)}</p>
      </div>
      <div class="hv-cyanbc" data-act="compareBack" style="display:flex;align-items:center;gap:9px;padding:12px 20px;border-radius:10px;border:1px solid #22323f;font-size:15px;font-weight:600;cursor:pointer"><svg width="17" height="17" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9"><path d="M19 12H5"/><path d="M12 19l-7-7 7-7"/></svg><span>${esc(t.back)}</span></div>
    </div>
    <div data-responsive="grid3" style="display:grid;grid-template-columns:repeat(3,1fr);gap:20px;max-width:1000px">${cols}</div>
  </div>`;
}

// Detect the face in the photo once, then fit each column's frame to it.
function placeCompareOverlays() {
  let tries = 0;
  const run = () => {
    if (state.screen !== "compare") return;
    const stages = [...document.querySelectorAll(".of-compare-stage")];
    if (!landmarker || !stages.length) { if (tries++ < 60) setTimeout(run, 150); return; }
    const probe = new Image();
    probe.onload = () => {
      let res; try { res = landmarker.detectForVideo(probe, performance.now()); } catch (e) { return; }
      const lm = res && res.faceLandmarks && res.faceLandmarks[0];
      if (!lm) return;
      stages.forEach(st => {
        const o = st.querySelector(".of-compare-frame");
        if (!o) return;
        const place = () => placeOverlay(o, st, lm, probe.naturalWidth, probe.naturalHeight, false, true);
        if (o.complete && o.naturalWidth) place(); else o.onload = place;
      });
    };
    probe.src = state.photoSrc;
  };
  run();
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------
function screenHTML() {
  switch (state.screen) {
    case "home":       return homeHTML();
    case "permission": return permissionHTML();
    case "analyzing":  return analyzingHTML();
    case "tryon":      return tryOnHTML();
    case "compare":    return compareHTML();
    case "collection": return collectionHTML();
    case "details":    return detailsHTML();
    case "favorites":  return favoritesHTML();
    case "profile":    return profileHTML();
    case "about":      return aboutHTML();
    default:           return homeHTML();
  }
}
function render() {
  app.dir = T().dir;
  document.documentElement.lang = state.lang;   // a11y: expose language to AT
  document.documentElement.dir = T().dir;
  app.innerHTML = `<a href="#ofMain" class="of-skip">${esc(T().skipToContent)}</a>`
    + headerHTML() + `<main id="ofMain" tabindex="-1" style="flex:1;display:flex;flex-direction:column">` + screenHTML() + `</main>`;
  enhanceA11y();
  if (state.screen === "tryon" && state.camera === "live") attachStream();
  else if (state.screen === "tryon" && state.hasPhoto) placePhotoOverlay();
  if (state.screen === "tryon" && state.adjust.on) wireAdjustDrag();
  if (state.screen === "compare") placeCompareOverlays();
}
// Shared "is this nav item active?" so the header highlight and aria-current agree.
function isNavActive(k) {
  return state.screen === k
    || (k === "tryon" && (state.screen === "permission" || state.screen === "analyzing" || state.screen === "compare"))
    || (k === "collection" && state.screen === "details");
}
// Accessible names for icon-only controls that carry no visible text.
const A11Y = {
  en: { fav:"Add to favourites", full:"Fullscreen", share:"Share", flip:"Flip camera",
        adjSize:"Resize frame", adjRot:"Rotate frame", toggleLang:"Switch language" },
  he: { fav:"הוספה למועדפים", full:"מסך מלא", share:"שיתוף", flip:"החלפת מצלמה",
        adjSize:"שינוי גודל מסגרת", adjRot:"סיבוב מסגרת", toggleLang:"החלפת שפה" },
};
// After each render, make every [data-act] control keyboard-operable and named.
function enhanceA11y() {
  const labels = A11Y[state.lang] || A11Y.en;
  // Every inline icon here is decorative; hide them so AT reads the label/text.
  app.querySelectorAll("svg:not([aria-hidden])").forEach(s => s.setAttribute("aria-hidden", "true"));
  app.querySelectorAll("[data-act]").forEach(el => {
    if (!el.hasAttribute("role")) el.setAttribute("role", "button");
    if (!el.hasAttribute("tabindex")) el.setAttribute("tabindex", "0");
    const act = el.dataset.act;
    const hasText = el.textContent.trim().length > 0;
    if (!el.getAttribute("aria-label")) {
      if (el.title) el.setAttribute("aria-label", el.title);
      else if (!hasText && labels[act]) el.setAttribute("aria-label", labels[act]);
    }
    if (act === "fav") {
      el.setAttribute("aria-pressed", state.favs.includes(el.dataset.arg) ? "true" : "false");
      if (!el.getAttribute("aria-label")) el.setAttribute("aria-label", labels.fav);
    }
    if (act === "adjust") el.setAttribute("aria-pressed", state.adjust.on ? "true" : "false");
    if (act === "colorToggle") el.setAttribute("aria-expanded", state.colorOpen ? "true" : "false");
    if (act === "go") {
      if (isNavActive(el.dataset.arg)) el.setAttribute("aria-current", "page");
      else el.removeAttribute("aria-current");
    }
  });
}

// ---------------------------------------------------------------------------
// Camera + MediaPipe tracking
// ---------------------------------------------------------------------------
async function initTracking() {
  try {
    const fileset = await FilesetResolver.forVisionTasks(MP_WASM);
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "/model", delegate: "GPU" },
      runningMode: "VIDEO", numFaces: 1,
    });
    trackingReady = true;
  } catch (e) {
    console.warn("Face tracking unavailable:", e);
    trackingReady = false;
  }
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: state.facing, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
    return true;
  } catch (e) {
    stream = null;
    return false;
  }
}
function stopCamera() {
  if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
}
function attachStream() {
  const v = document.getElementById("ofVideo");
  if (v && stream && v.srcObject !== stream) {
    v.srcObject = stream;
    v.play().catch(() => {});
  }
  startTrackingLoop();
}
// Position the frame overlay onto a face, accounting for the stage's
// object-fit: cover cropping so the frame lands on the real eyes — for both the
// live (mirrored) video and a still uploaded photo (not mirrored).
function placeOverlay(o, stage, lm, srcW, srcH, mirror, forceAuto) {
  // Manual placement (being edited, or committed via "Done") overrides auto-tracking.
  // forceAuto is used by the compare view, which always fits each frame automatically.
  if (!forceAuto && (state.adjust.on || state.adjust.committed)) { applyManual(o); hideDetecting(); return; }
  const CW = stage.clientWidth, CH = stage.clientHeight;
  if (!CW || !CH || !srcW || !srcH) return;
  const scale = Math.max(CW / srcW, CH / srcH);   // object-fit: cover
  const dispW = srcW * scale, dispH = srcH * scale;
  const offX = (dispW - CW) / 2, offY = (dispH - CH) / 2;
  const toPx = (nx, ny) => {
    let px = nx * dispW; if (mirror) px = dispW - px;
    return [px - offX, ny * dispH - offY];
  };
  const [lx, ly] = toPx(lm[L_EYE].x, lm[L_EYE].y);
  const [rx, ry] = toPx(lm[R_EYE].x, lm[R_EYE].y);
  const [flx, fly] = toPx(lm[FACE_L].x, lm[FACE_L].y);
  const [frx, fry] = toPx(lm[FACE_R].x, lm[FACE_R].y);
  const [bx, by] = toPx(lm[NOSE_BRIDGE].x, lm[NOSE_BRIDGE].y);
  const faceW = Math.hypot(frx - flx, fry - fly);   // temple-to-temple width
  // Eye-line tilt. Mirroring the selfie view swaps the eye order and would make
  // atan2 read ~180° (frame upside-down), so normalise into (-90°, 90°] — a
  // symmetric frame never needs to flip past that.
  let roll = Math.atan2(ry - ly, rx - lx) * 180 / Math.PI;
  if (roll > 90) roll -= 180;
  else if (roll < -90) roll += 180;
  const frameWpx = faceW * FRAME_W;
  const aspect = (o.naturalWidth && o.naturalHeight) ? (o.naturalHeight / o.naturalWidth) : 0.33;
  const frameHpx = frameWpx * aspect;
  o.style.left = (bx / CW * 100) + "%";
  o.style.top = ((by + VERT_NUDGE * frameHpx) / CH * 100) + "%";
  o.style.width = (frameWpx / CW * 100) + "%";
  o.style.transform = "translate(-50%,-50%) rotate(" + roll + "deg)";
  // remember the auto fit so manual mode can start from a sensible baseline
  state.autoFit = { x: bx / CW * 100, y: (by + VERT_NUDGE * frameHpx) / CH * 100, w: frameWpx / CW * 100, rot: roll };
  hideDetecting();
}
function hideDetecting() {
  const b = document.getElementById("ofDetecting");
  if (b) b.style.display = "none";
}
// Apply the user's manual placement (percentages of the stage) to the frame.
function applyManual(o) {
  if (!o) return;
  const a = state.adjust;
  o.style.left = a.x + "%";
  o.style.top = a.y + "%";
  o.style.width = a.w + "%";
  o.style.transform = "translate(-50%,-50%) rotate(" + a.rot + "deg)";
}
// Toggle manual placement mode, seeding it from the current auto fit.
function toggleAdjust() {
  if (!state.adjust.on) {
    // Enter edit mode. Keep committed values if any; else seed from the auto fit.
    if (!state.adjust.committed) {
      const o = document.getElementById("ofOverlay");
      const seed = state.autoFit || {
        x: o ? (parseFloat(o.style.left) || 50) : 50,
        y: o ? (parseFloat(o.style.top) || 40) : 40,
        w: o ? (parseFloat(o.style.width) || 42) : 42,
        rot: 0,
      };
      state.adjust.x = seed.x; state.adjust.y = seed.y; state.adjust.w = seed.w; state.adjust.rot = seed.rot;
    }
    state.adjust.on = true;
  } else {
    // "Done" — lock the manual placement in so it survives re-renders.
    state.adjust.on = false;
    state.adjust.committed = true;
  }
  render();
}
// While in manual mode, let the user drag the frame and wheel to resize.
function wireAdjustDrag() {
  const o = document.getElementById("ofOverlay");
  if (!o) return;
  const stage = o.parentElement;
  let dragging = false, sx = 0, sy = 0, bx = 0, by = 0;
  o.addEventListener("pointerdown", (e) => {
    if (!state.adjust.on) return;
    dragging = true; try { o.setPointerCapture(e.pointerId); } catch (err) {}
    o.style.cursor = "grabbing";
    sx = e.clientX; sy = e.clientY; bx = state.adjust.x; by = state.adjust.y;
    e.preventDefault();
  });
  o.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    const r = stage.getBoundingClientRect();
    state.adjust.x = clamp(bx + (e.clientX - sx) / r.width * 100, 0, 100);
    state.adjust.y = clamp(by + (e.clientY - sy) / r.height * 100, 0, 100);
    applyManual(o);
  });
  const end = () => { if (dragging) { dragging = false; o.style.cursor = "grab"; } };
  o.addEventListener("pointerup", end);
  o.addEventListener("pointercancel", end);
  o.addEventListener("wheel", (e) => {
    if (!state.adjust.on) return;
    e.preventDefault();
    state.adjust.w = clamp(state.adjust.w + (e.deltaY < 0 ? 1.5 : -1.5), 8, 90);
    applyManual(o);
  }, { passive: false });
}

function startTrackingLoop() {
  if (rafId) cancelAnimationFrame(rafId);
  const loop = () => {
    rafId = requestAnimationFrame(loop);
    const v = document.getElementById("ofVideo"), o = document.getElementById("ofOverlay");
    if (!v || !o || !landmarker || v.readyState < 2) return;
    let res; try { res = landmarker.detectForVideo(v, performance.now()); } catch (e) { return; }
    const lm = res && res.faceLandmarks && res.faceLandmarks[0];
    if (!lm) return;
    placeOverlay(o, o.parentElement, lm, v.videoWidth || 1280, v.videoHeight || 720, true);
  };
  loop();
}

// Detect the face in the still photo and fit the frame to it (photo mode).
function placePhotoOverlay() {
  const o = document.getElementById("ofOverlay");
  if (!o) return;
  let tries = 0;
  const run = () => {
    if (state.screen !== "tryon" || state.camera === "live" || !state.hasPhoto) return;
    const img = document.getElementById("ofPhoto");
    if (!img) return;
    if (!landmarker || !img.complete || !img.naturalWidth) {
      if (tries++ < 60) setTimeout(run, 150);
      return;
    }
    let res; try { res = landmarker.detectForVideo(img, performance.now()); } catch (e) { return; }
    const lm = res && res.faceLandmarks && res.faceLandmarks[0];
    if (lm) placeOverlay(o, o.parentElement, lm, img.naturalWidth, img.naturalHeight, false);
  };
  run();
}

async function flipCamera() {
  state.facing = state.facing === "user" ? "environment" : "user";
  if (state.camera === "live") {
    stopCamera();
    await startCamera();
    render();
  }
}

// grab a still (un-mirrored) from an offscreen video bound to the live stream
async function captureFromStream() {
  if (!stream) return null;
  const v = document.createElement("video");
  v.playsInline = true; v.muted = true; v.srcObject = stream;
  await v.play().catch(() => {});
  await new Promise(r => (v.readyState >= 2 ? r() : v.addEventListener("loadeddata", r, { once: true })));
  const w = v.videoWidth || 1280, h = v.videoHeight || 720;
  const c = document.createElement("canvas"); c.width = w; c.height = h;
  c.getContext("2d").drawImage(v, 0, 0, w, h);
  v.pause(); v.srcObject = null;
  return c.toDataURL("image/jpeg", 0.9);
}
function captureFromImage(src) {
  return new Promise((resolve) => {
    const im = new Image();
    im.crossOrigin = "anonymous";
    im.onload = () => {
      const c = document.createElement("canvas"); c.width = im.naturalWidth || 640; c.height = im.naturalHeight || 640;
      try { c.getContext("2d").drawImage(im, 0, 0); resolve(c.toDataURL("image/jpeg", 0.9)); }
      catch (e) { resolve(null); }
    };
    im.onerror = () => resolve(null);
    im.src = src;
  });
}

// ---------------------------------------------------------------------------
// Analysis flow (real backend)
// ---------------------------------------------------------------------------
async function runAnalysis(withCamera) {
  // Capture the still up front so the analyzing screen shows the real image
  // (not the old sample), and so we have one source for both analysis + display.
  let dataURL = null;
  if (withCamera) dataURL = await captureFromStream();
  else dataURL = await captureFromImage(state.photoSrc);
  if (dataURL) state.photoSrc = dataURL;

  state.screen = "analyzing"; state.progress = 0; state.analysis = null; render();

  // animate the four-step progress bar
  let p = 0;
  const timer = setInterval(() => {
    p = Math.min(p + 25, 100);
    state.progress = p;
    if (state.screen === "analyzing") render();
    if (p >= 100) clearInterval(timer);
  }, 520);

  // call the real Python engine
  let noFace = false;
  try {
    if (dataURL) {
      const res = await fetch("/api/analyze", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: dataURL, category: "vision" }),
      });
      const data = await res.json();
      if (data.ok) applyAnalysis(data);
      else if (data.error === "no_face") noFace = true;
    }
  } catch (e) { /* fall back to design defaults */ }

  // wait for the progress animation to finish
  await new Promise(r => setTimeout(r, Math.max(0, 2200 - 0)));
  clearInterval(timer); state.progress = 100;

  // No face found: return to the prompt with a clear message instead of
  // showing frames floating on a face that was never detected.
  if (noFace) {
    stopCamera();
    state.hasPhoto = false; state.camera = "photo";
    state.photoSrc = "/assets/hero-home.jpg";
    state.promptError = "noFace";
    state.screen = "tryon";
    render();
    return;
  }

  // reveal try-on (a fresh photo starts from the automatic fit)
  state.promptError = null;
  state.adjust.on = false; state.adjust.committed = false;
  state.screen = "tryon";
  state.camera = withCamera && stream ? "live" : "photo";
  // We now have real input (live camera or a captured/uploaded still).
  if (state.camera === "live" || dataURL) state.hasPhoto = true;
  render();
}

function applyAnalysis(data) {
  state.analysis = data;
  state.realScore = {};
  (data.top || []).forEach(f => { state.realScore[f.id] = f.match; });
  if (data.top && data.top[0]) state.currentId = data.top[0].id;
}

// ---------------------------------------------------------------------------
// Snapshot / fullscreen
// ---------------------------------------------------------------------------
// Composite the current try-on (photo/video + placed frame) onto a canvas and
// hand it back via cb. Shared by both download (snapshot) and Web Share.
function buildSnapshotCanvas(cb) {
  const overlay = document.getElementById("ofOverlay");
  const stage = overlay && overlay.parentElement;
  if (!stage) { cb(null); return; }
  // Match the canvas to the on-screen stage so object-fit:cover maths and the
  // overlay's percentage position/size map straight across.
  const ar = (stage.clientHeight / stage.clientWidth) || (10.6 / 16);
  const c = document.createElement("canvas");
  c.width = 1280; c.height = Math.round(1280 * ar);
  const g = c.getContext("2d");
  g.fillStyle = "#05080c"; g.fillRect(0, 0, c.width, c.height);

  // draw a source image/video using the same object-fit:cover cropping as the DOM
  const cover = (img, sw, sh, mirror) => {
    if (!sw || !sh) return;
    const scale = Math.max(c.width / sw, c.height / sh);
    const w = sw * scale, h = sh * scale;
    const x = (c.width - w) / 2, y = (c.height - h) / 2;
    g.save();
    if (mirror) { g.translate(c.width, 0); g.scale(-1, 1); }
    g.drawImage(img, x, y, w, h);
    g.restore();
  };

  // place the frame exactly where it sits on screen (percentages + rotation)
  const drawOverlayAndDone = () => {
    try {
      if (overlay && overlay.naturalWidth) {
        const lp = (parseFloat(overlay.style.left) / 100) || 0.5;
        const tp = (parseFloat(overlay.style.top) / 100) || 0.4;
        const wp = (parseFloat(overlay.style.width) / 100) || 0.42;
        const w = c.width * wp, h = w * overlay.naturalHeight / overlay.naturalWidth;
        const m = /rotate\(([-\d.]+)deg\)/.exec(overlay.style.transform || "");
        const ang = m ? parseFloat(m[1]) * Math.PI / 180 : 0;
        g.save();
        g.translate(c.width * lp, c.height * tp);
        g.rotate(ang);
        g.drawImage(overlay, -w / 2, -h / 2, w, h);
        g.restore();
      }
    } catch (e) {}
    cb(c);
  };

  const v = document.getElementById("ofVideo");
  const photo = document.getElementById("ofPhoto");
  if (state.camera === "live" && v && v.videoWidth) {
    try { cover(v, v.videoWidth, v.videoHeight, true); } catch (e) {}
    drawOverlayAndDone();
  } else if (photo) {
    const im = new Image();                       // reload to be sure it's decoded
    im.onload = () => { try { cover(im, im.naturalWidth, im.naturalHeight, false); } catch (e) {} drawOverlayAndDone(); };
    im.onerror = drawOverlayAndDone;
    im.src = state.photoSrc;
  } else {
    drawOverlayAndDone();
  }
}
function snapshot() {
  buildSnapshotCanvas((c) => {
    if (!c) return;
    const a = document.createElement("a");
    a.download = "optifit-tryon.png";
    a.href = c.toDataURL("image/png");
    a.click();
  });
}
// Share the try-on via the native share sheet (mobile/modern browsers), with a
// graceful fallback to downloading the image when sharing isn't available.
function shareSnapshot() {
  buildSnapshotCanvas((c) => {
    if (!c) return;
    const cur = find(state.currentId);
    const caption = "OptiFit — " + (cur ? nameOf(cur) : "Virtual Try-On");
    c.toBlob(async (blob) => {
      if (!blob) { snapshot(); return; }
      const file = new File([blob], "optifit-tryon.png", { type: "image/png" });
      if (navigator.canShare && navigator.canShare({ files: [file] })) {
        try { await navigator.share({ files: [file], title: "OptiFit", text: caption }); return; }
        catch (e) { if (e && e.name === "AbortError") return; }  // user cancelled
      }
      snapshot();   // no share support → download instead
    }, "image/png");
  });
}
function toggleFull() {
  const el = document.documentElement;
  if (document.fullscreenElement) document.exitFullscreen();
  else if (el.requestFullscreen) el.requestFullscreen();
}

// ---------------------------------------------------------------------------
// Navigation
// ---------------------------------------------------------------------------
function go(screen) {
  if (screen !== "tryon") stopCamera();
  if (screen !== "tryon" && screen !== "analyzing") { state.camera = "idle"; }
  state.screen = screen;
  window.scrollTo(0, 0);
  render();
}
function tryFrame(id) {
  state.currentId = id;
  state.screen = "tryon";
  if (state.camera === "idle") state.camera = "photo";
  window.scrollTo(0, 0);
  render();
  if (state.camera === "live") attachStream();
}
function openFrame(id) { state.detailId = id; state.screen = "details"; window.scrollTo(0, 0); render(); }
function toggleFav(id) {
  state.favs = state.favs.includes(id) ? state.favs.filter(x => x !== id) : state.favs.concat(id);
  saveFavs();
  render();
}

// ---------------------------------------------------------------------------
// Event delegation
// ---------------------------------------------------------------------------
function handleAct(act, arg, el) {
  switch (act) {
    case "go":          go(arg); break;
    case "toggleLang":  state.lang = he() ? "en" : "he"; saveLang(); render(); break;
    case "startFlow":   state.screen = "permission"; window.scrollTo(0,0); render(); break;
    case "allow":       (async () => {
                          const ok = await startCamera();
                          if (!ok) {   // permission denied / no camera → prompt with a message
                            state.promptError = "camDenied"; state.screen = "tryon"; state.camera = "photo"; state.hasPhoto = false;
                            window.scrollTo(0,0); render(); return;
                          }
                          runAnalysis(true);
                        })(); break;
    case "usePhoto":    state.promptError = null; document.getElementById("ofUpload").click(); break;
    case "fav":         toggleFav(arg); break;
    case "tryOn":       tryFrame(arg); break;
    case "open":        openFrame(arg); break;
    case "filter":      state.filter = arg; render(); break;
    case "back":        go("collection"); break;
    case "viewDetails": state.detailId = state.currentId; state.screen = "details"; window.scrollTo(0,0); render(); break;
    case "colorToggle": state.colorOpen = !state.colorOpen; render(); break;
    case "moreCat":     if (state.moreCat !== arg) { state.moreCat = arg; state.morePage = 0; render(); } break;
    case "morePrev":    if (state.morePage > 0) { state.morePage--; render(); } break;
    case "moreNext":    { const pages = Math.max(1, Math.ceil(otherMatches().length / MORE_PER)); if (state.morePage < pages - 1) { state.morePage++; render(); } } break;
    case "flip":        flipCamera(); break;
    case "retake":      stopCamera(); state.hasPhoto = false; state.camera = "photo"; state.photoSrc = "/assets/hero-home.jpg"; state.promptError = null; state.adjust.on = false; state.adjust.committed = false; window.scrollTo(0,0); render(); break;
    case "snapshot":    snapshot(); break;
    case "share":       shareSnapshot(); break;
    case "adjust":      toggleAdjust(); break;
    case "adjSize":     state.adjust.w = clamp(state.adjust.w + parseFloat(arg), 8, 90); applyManual(document.getElementById("ofOverlay")); break;
    case "adjRot":      state.adjust.rot += parseFloat(arg); applyManual(document.getElementById("ofOverlay")); break;
    case "adjReset":    state.adjust.on = false; state.adjust.committed = false; render(); break;
    case "compare":     state.screen = "compare"; window.scrollTo(0,0); render(); break;
    case "compareBack": state.screen = "tryon"; window.scrollTo(0,0); render(); break;
    case "full":        toggleFull(); break;
  }
}

// Pointer + keyboard both drive the same actions (a11y: controls are operable
// with mouse, touch, Enter and Space).
app.addEventListener("click", (e) => {
  const el = e.target.closest("[data-act]");
  if (!el) return;
  handleAct(el.dataset.act, el.dataset.arg, el);
});
app.addEventListener("keydown", (e) => {
  if (e.key !== "Enter" && e.key !== " " && e.key !== "Spacebar") return;
  const el = e.target.closest("[data-act]");
  if (!el || el.tagName === "INPUT" || el.tagName === "TEXTAREA" || el.tagName === "SELECT") return;
  e.preventDefault();   // Space would otherwise scroll the page
  handleAct(el.dataset.act, el.dataset.arg, el);
});

// photo upload -> use as the analysis source
document.getElementById("ofUpload").addEventListener("change", (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) { e.target.value = ""; return; }   // dialog cancelled: keep the prompt
  const reader = new FileReader();
  reader.onload = () => { state.photoSrc = reader.result; runAnalysis(false); };
  reader.readAsDataURL(file);
  e.target.value = "";
});

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
async function boot() {
  render();
  initTracking();
  try {
    const res = await fetch("/api/frames?category=all");
    const data = await res.json();
    if (data.ok) { state.catalog = data.frames; render(); }
  } catch (e) { console.warn("catalog load failed", e); }
}
boot();
