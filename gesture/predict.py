import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import os
import pickle
import time
import subprocess
from collections import deque, Counter
from PIL import Image, ImageDraw, ImageFont

MODEL_DIR = "models"
IMG_SIZE = 64
CONF_THRESHOLD   = 0.60
FINAL_THRESHOLD  = 0.60
SMOOTHING_WINDOW = 15          # wider window -> smoother votes
PRED_COOLDOWN    = 2
HOLD_DURATION    = 5.0

# ── Smoothing knobs ────────────────────────────────────────────────────────────
LM_EMA_ALPHA      = 0.40   # landmark EMA  (lower = smoother, higher = snappier)
PROB_EMA_ALPHA    = 0.35   # probability EMA blend
MAJORITY_RATIO    = 0.55   # fraction of window that must agree for a stable pred
NO_PRED_TOLERANCE = 10     # frames of low-conf before we clear the display label
# ───────────────────────────────────────────────────────────────────────────────

ARABIC_FONT_PATHS = [
    "/usr/share/fonts/truetype/arabic/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf",
    "/usr/share/fonts/truetype/fonts-arabeyes/ae_AlArabiya.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]

with open(os.path.join(MODEL_DIR, "model_config.json"), encoding="utf-8") as f:
    config = json.load(f)

CLASSES     = config["classes"]
NUM_CLASSES = config["num_classes"]
LM_DIM      = config["lm_dim"]
BEST_MODEL  = config.get("best_model", "mlp_model")
USE_TFLITE  = config.get("use_tflite", True)
USE_CNN     = "cnn" in BEST_MODEL

with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb") as f:
    le = pickle.load(f)

LABEL_MAP = {
    "hello": "مرحبا",
    "marhaba": "مرحبا",
    "thank you": "شكراً",
    "shukran": "شكراً",
    "no": "لا",
    "la": "لا",
    "yes": "نعم",
    "naam": "نعم",
    "please": "من فضلك",
    "min fadlak": "من فضلك",
    "min_fadlak": "من فضلك",
}

def to_arabic_label(label):
    return LABEL_MAP.get(label.lower().strip(), label)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def find_arabic_font(size):
    for path in ARABIC_FONT_PATHS:
        if os.path.exists(path):
            print(f"Font: {path}")
            return ImageFont.truetype(path, size)
    print("WARNING: No Arabic font found. Install with: sudo apt install fonts-noto-core")
    return ImageFont.load_default()


FONT_LARGE  = find_arabic_font(64)
FONT_MEDIUM = find_arabic_font(32)
FONT_SMALL  = find_arabic_font(22)

WG_LOWER       = np.array([0,   0, 170])
WG_UPPER       = np.array([180, 60, 255])
SKIN_COLOR_BGR = (90, 130, 180)


def preprocess_for_white_glove(frame):
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, WG_LOWER, WG_UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,  kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
    skin = frame.copy()
    skin[mask > 0] = SKIN_COLOR_BGR
    lab     = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    skin    = cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)
    return skin


def fix_arabic(text):
    try:
        from arabic_reshaper import reshape
        from bidi.algorithm import get_display
        return get_display(reshape(text))
    except ImportError:
        return text[::-1]


def draw_arabic_text(frame, text, position, font,
                     color=(255, 255, 255), bg_color=None, padding=8):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    shaped  = fix_arabic(text)
    bbox    = draw.textbbox((0, 0), shaped, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x, y   = position
    if bg_color is not None:
        draw.rounded_rectangle(
            [x - padding, y - padding, x + tw + padding, y + th + padding],
            radius=10, fill=bg_color)
    draw.text((x, y), shaped, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def draw_arabic_centered(frame, text, cx, y, font,
                         color=(255, 255, 255), bg_color=None, padding=10):
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw    = ImageDraw.Draw(pil_img)
    shaped  = fix_arabic(text)
    bbox    = draw.textbbox((0, 0), shaped, font=font)
    tw, th  = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x       = cx - tw // 2
    if bg_color is not None:
        draw.rounded_rectangle(
            [x - padding, y - padding, x + tw + padding, y + th + padding],
            radius=12, fill=bg_color)
    draw.text((x, y), shaped, font=font, fill=color)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def load_tflite(name):
    path   = os.path.join(MODEL_DIR, f"{name}.tflite")
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp, interp.get_input_details(), interp.get_output_details()


def load_keras(name):
    path = os.path.join(MODEL_DIR, f"{name}.h5")
    if not os.path.exists(path):
        path = os.path.join(MODEL_DIR, f"{name}_best.h5")
    return tf.keras.models.load_model(path)


def load_model():
    for ext in [".h5", "_best.h5"]:
        path = os.path.join(MODEL_DIR, f"{BEST_MODEL}{ext}")
        if os.path.exists(path):
            model = tf.keras.models.load_model(path)
            print(f"Keras loaded: {path}  | CNN={USE_CNN}")
            return "keras", model, None, None
    raise FileNotFoundError(f"No model found for {BEST_MODEL} in {MODEL_DIR}/")


# ── Landmark feature extraction ────────────────────────────────────────────────

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
               (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
               (15,16),(0,17),(17,18),(18,19),(19,20)]


def extract_landmarks_from_pts(pts):
    """pts: (21, 3) float32 array in normalised MediaPipe space."""
    coords_norm = (pts - pts[0]).flatten()
    scale       = np.max(np.abs(coords_norm)) + 1e-6
    coords_norm = coords_norm / scale
    angles = []
    for a, b in CONNECTIONS:
        v = pts[b] - pts[a]
        angles.extend(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate([coords_norm, angles]).astype(np.float32)


def extract_landmarks(hand_landmarks):
    """Compatibility wrapper."""
    pts = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                   dtype=np.float32)
    return extract_landmarks_from_pts(pts)


def crop_hand_roi(frame, hand_landmarks, padding=30):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x1 = max(0, int(min(xs)) - padding)
    y1 = max(0, int(min(ys)) - padding)
    x2 = min(w, int(max(xs)) + padding)
    y2 = min(h, int(max(ys)) + padding)
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return roi, (x1, y1, x2, y2)


def run_inference(model_type, model, inp_details, out_details, img_arr, lm_arr):
    if model_type == "tflite":
        for det in inp_details:
            shape = det['shape']
            model.set_tensor(det['index'], img_arr if len(shape) == 4 else lm_arr)
        model.invoke()
        return model.get_tensor(out_details[0]['index'])[0]
    if USE_CNN:
        return model.predict([img_arr, lm_arr], verbose=0)[0]
    return model.predict(lm_arr, verbose=0)[0]


def speak(text):
    try:
        from gtts import gTTS
        import tempfile
        tts = gTTS(text=text, lang='ar', slow=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
        tts.save(tmp.name)
        subprocess.Popen(['mpg123', '-q', tmp.name],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


# ── UI helpers ─────────────────────────────────────────────────────────────────

def draw_hold_timer(frame, cx, cy, progress, spoken):
    radius, thickness = 34, 5
    cv2.circle(frame, (cx, cy), radius, (50, 50, 50), thickness)
    if progress <= 0:
        return frame
    angle = int(360 * min(progress, 1.0))
    color = (0, 255, 80) if spoken else (0, 220, 255)
    cv2.ellipse(frame, (cx, cy), (radius, radius),
                -90, 0, angle, color, thickness, cv2.LINE_AA)
    label = "OK" if spoken else f"{max(0.0, HOLD_DURATION*(1.0-progress)):.1f}s"
    cv2.putText(frame, label, (cx - 18, cy + 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_topk_arabic(frame, probs, k=5):
    top_idx = np.argsort(probs)[::-1][:k]
    panel_x, panel_y = frame.shape[1] - 230, 95
    gap, bar_max = 34, 180

    ov = frame.copy()
    cv2.rectangle(ov, (panel_x - 10, panel_y - 25),
                  (frame.shape[1] - 3, panel_y + k * gap + 12), (15, 15, 28), -1)
    cv2.addWeighted(ov, 0.82, frame, 0.18, 0, frame)

    frame = draw_arabic_text(frame, "أعلى التوقعات",
                              (panel_x - 5, panel_y - 22), FONT_SMALL,
                              color=(200, 200, 200))
    for i, idx in enumerate(top_idx):
        p     = float(probs[idx])
        label = to_arabic_label(le.classes_[idx])
        y_bar = panel_y + i * gap + gap - 8
        bar_w = int(p * bar_max)
        color = (0, 210, 100) if i == 0 else (0, 100, 190)
        cv2.rectangle(frame, (panel_x, y_bar - 14), (panel_x + bar_w, y_bar), color, -1)
        cv2.putText(frame, f"{p*100:.0f}%", (panel_x + bar_w + 4, y_bar - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1)
        frame = draw_arabic_text(frame, label, (panel_x, y_bar - 14), FONT_SMALL,
                                  color=(255, 255, 255))
    return frame


def draw_ui(frame, pred, conf, stable, fps, last_spoken, model_name,
            hold_progress=0.0, hold_spoken=False):
    h, w = frame.shape[:2]

    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 90), (10, 10, 22), -1)
    cv2.addWeighted(ov, 0.85, frame, 0.15, 0, frame)

    cv2.putText(frame, "Arabic Sign Language", (12, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200, 50), 2)
    cv2.putText(frame, f"Model: {model_name}  FPS: {fps:.1f}", (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 255, 160), 1)

    if pred:
        bar_w     = int(conf * (w - 40))
        bar_color = (0, 200, 80) if conf >= CONF_THRESHOLD else (0, 90, 200)
        cv2.rectangle(frame, (20, 70), (20 + bar_w, 82), bar_color, -1)
        cv2.putText(frame, f"{conf*100:.1f}%", (w - 80, 82),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

        bg = (0, 100, 40) if stable else (100, 80, 0)
        frame = draw_arabic_centered(frame, pred, cx=w // 2, y=h - 130,
                                     font=FONT_LARGE, color=(255, 255, 255),
                                     bg_color=bg, padding=14)

        stab_lbl   = "مستقر" if stable else "غير مستقر"
        stab_color = (0, 255, 120) if stable else (255, 200, 0)
        frame = draw_arabic_centered(frame, stab_lbl, cx=w // 2, y=h - 55,
                                     font=FONT_SMALL, color=stab_color)

        if hold_progress > 0 or hold_spoken:
            frame = draw_hold_timer(frame, 50, h - 90, hold_progress, hold_spoken)

    if last_spoken:
        frame = draw_arabic_text(frame, f"آخر كلمة: {last_spoken}",
                                  (12, h - 40), FONT_SMALL, color=(180, 180, 255))

    cv2.putText(frame, "Q=Quit  S=Speak  C=Clear", (12, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1)
    return frame


def draw_hand_label(frame, pred, conf, stable, bbox):
    if pred is None or bbox is None:
        return frame
    x1, y1, x2, y2 = bbox
    border_color = (0, 220, 80) if stable else (0, 140, 240)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)
    label_bg = (0, 80, 30) if stable else (0, 50, 120)
    frame = draw_arabic_text(frame, pred, (x1, max(0, y1 - 50)), FONT_MEDIUM,
                              color=(255, 255, 255), bg_color=label_bg, padding=6)
    return frame


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("=== Arabic Sign Language Predictor ===")
    print("Install tip: pip install arabic-reshaper python-bidi")
    model_type, model, inp_details, out_details = load_model()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,   # was 0.3 — tighter initial lock-on
        min_tracking_confidence=0.6,    # was 0.3 — keeps skeleton stable
    )

    pred_buf = deque(maxlen=SMOOTHING_WINDOW)
    conf_buf = deque(maxlen=SMOOTHING_WINDOW)

    lm_ema   = None   # (21, 3) EMA of landmark positions
    prob_ema = None   # (NUM_CLASSES,) EMA of class probabilities

    disp_pred      = None   # last confident label for drop-tolerance display
    disp_conf      = 0.0
    no_pred_frames = 0      # consecutive frames without a confident raw pred

    gesture_start_t = None
    current_gesture = None
    hold_spoken     = False

    last_spoken = ""
    fps_t = time.time()
    fps_n = 0
    fps   = 0.0

    print("Running. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        fps_n += 1
        dt = time.time() - fps_t
        if dt >= 0.5:
            fps   = fps_n / dt
            fps_n = 0
            fps_t = time.time()

        enhanced     = preprocess_for_white_glove(frame)
        rgb_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        result       = hands.process(rgb_enhanced)

        pred          = None
        conf          = 0.0
        probs_display = prob_ema   # keep last smooth probs visible during brief drops
        stable        = False
        bbox          = None
        hold_progress = 0.0

        if result.multi_hand_landmarks:
            hand_lm = result.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(
                frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

            # 1. EMA-smooth raw landmark positions before feature extraction
            raw_pts = np.array([[lm.x, lm.y, lm.z]
                                 for lm in hand_lm.landmark], dtype=np.float32)
            if lm_ema is None:
                lm_ema = raw_pts.copy()
            else:
                lm_ema = LM_EMA_ALPHA * raw_pts + (1.0 - LM_EMA_ALPHA) * lm_ema

            lm_vec   = extract_landmarks_from_pts(lm_ema)
            lm_input = np.expand_dims(lm_vec, 0)

            img_input = None
            if USE_CNN:
                roi, bbox = crop_hand_roi(frame, hand_lm)
                if roi is not None:
                    img_input = np.expand_dims(roi, 0)

            if not USE_CNN or img_input is not None:
                # 2. Raw inference on smooth features
                raw_probs = run_inference(model_type, model, inp_details,
                                          out_details, img_input, lm_input)

                # 3. EMA-smooth probabilities across frames
                if prob_ema is None:
                    prob_ema = raw_probs.copy()
                else:
                    prob_ema = (PROB_EMA_ALPHA * raw_probs
                                + (1.0 - PROB_EMA_ALPHA) * prob_ema)

                probs_display = prob_ema

                smooth_idx  = int(np.argmax(prob_ema))
                smooth_conf = float(prob_ema[smooth_idx])

                # 4. Accumulate votes using smoothed index
                if smooth_conf >= FINAL_THRESHOLD:
                    mapped = to_arabic_label(le.classes_[smooth_idx])
                    if not (mapped == "لا" and smooth_conf < 0.90):
                        pred_buf.append(smooth_idx)
                        conf_buf.append(smooth_conf)

                # 5. Majority vote — require MAJORITY_RATIO agreement in buffer
                min_votes = max(3, int(SMOOTHING_WINDOW * 0.4))
                if len(pred_buf) >= min_votes:
                    top_idx, top_count = Counter(pred_buf).most_common(1)[0]
                    if top_count >= len(pred_buf) * MAJORITY_RATIO:
                        pred   = to_arabic_label(le.classes_[top_idx])
                        conf   = float(np.mean(conf_buf))
                        stable = True

                if probs_display is not None:
                    frame = draw_topk_arabic(frame, probs_display)

                # 6. Drop-tolerance: hold last label for NO_PRED_TOLERANCE frames
                if pred:
                    disp_pred      = pred
                    disp_conf      = conf
                    no_pred_frames = 0
                else:
                    no_pred_frames += 1
                    if no_pred_frames < NO_PRED_TOLERANCE:
                        pred  = disp_pred
                        conf  = disp_conf
                    else:
                        disp_pred = None
                        disp_conf = 0.0

                frame = draw_hand_label(frame, pred, conf, stable, bbox)

                # 7. Hold-timer
                now = time.time()
                if pred:
                    if pred != current_gesture:
                        current_gesture = pred
                        gesture_start_t = now
                        hold_spoken     = False

                    elapsed       = now - gesture_start_t
                    hold_progress = min(elapsed / HOLD_DURATION, 1.0)

                    if (elapsed >= HOLD_DURATION
                            and not hold_spoken
                            and pred != last_spoken):
                        speak(pred)
                        last_spoken = pred
                        hold_spoken = True
                else:
                    hold_progress = 0.0

        else:
            # Hand lost — reset all state
            lm_ema          = None
            prob_ema        = None
            pred_buf.clear()
            conf_buf.clear()
            no_pred_frames  = 0
            disp_pred       = None
            disp_conf       = 0.0
            current_gesture = None
            gesture_start_t = None
            hold_spoken     = False
            hold_progress   = 0.0

        frame = draw_ui(frame, pred, conf, stable, fps, last_spoken, BEST_MODEL,
                        hold_progress=hold_progress, hold_spoken=hold_spoken)
        cv2.imshow("Arabic Sign Language", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and pred:
            speak(pred)
            last_spoken = pred
            hold_spoken = True
        elif key == ord('c'):
            pred_buf.clear()
            conf_buf.clear()
            lm_ema          = None
            prob_ema        = None
            disp_pred       = None
            disp_conf       = 0.0
            last_spoken     = ""
            current_gesture = None
            gesture_start_t = None
            hold_spoken     = False

    hands.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()