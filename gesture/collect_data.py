import cv2
import mediapipe as mp
import numpy as np
import os
import json
import time
from datetime import datetime

CLASSES = [
    'أ', 'ب', 'ت', 'ث', 'ج', 'ح', 'خ', 'د', 'ذ', 'ر',
    'ز', 'س', 'ش', 'ص', 'ض', 'ط', 'ظ', 'ع', 'غ', 'ف',
    'ق', 'ك', 'ل', 'م', 'ن', 'ه', 'و', 'ي',
    'مرحبا', 'شكراً', 'لا', 'نعم', 'من_فضلك'
]

DATASET_DIR       = "arabic_sign_dataset"
LANDMARKS_DIR     = os.path.join(DATASET_DIR, "landmarks")
IMAGES_DIR        = os.path.join(DATASET_DIR, "images")
SAMPLES_PER_CLASS = 200
IMG_SIZE          = 64
COUNTDOWN         = 3

mp_hands          = mp.solutions.hands
mp_drawing        = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def setup_dirs():
    for cls in CLASSES:
        os.makedirs(os.path.join(LANDMARKS_DIR, cls), exist_ok=True)
        os.makedirs(os.path.join(IMAGES_DIR,    cls), exist_ok=True)


def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    coords = np.array(coords)
    wrist  = coords[:3]
    pts    = coords.reshape(21, 3)
    norm   = (pts - wrist).flatten()
    scale  = np.max(np.abs(norm)) + 1e-6
    norm   = norm / scale
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),
        (15,16),(0,17),(17,18),(18,19),(19,20)
    ]
    angles = []
    for a, b in connections:
        v = pts[b] - pts[a]
        angles.extend(v / (np.linalg.norm(v) + 1e-6))
    return np.concatenate([norm, angles]).astype(np.float32)


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
    return roi, (x1, y1, x2, y2)


def draw_ui(frame, cls_name, cls_idx, count, total, phase, countdown=None):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 80), (15, 15, 30), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    progress = int((count / total) * (w - 40))
    cv2.rectangle(frame, (20, 60), (w - 20, 72), (40, 40, 60), -1)
    cv2.rectangle(frame, (20, 60), (20 + progress, 72), (0, 200, 100), -1)

    cv2.putText(frame, f"Class [{cls_idx+1}/{len(CLASSES)}]: {cls_name}",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 220, 80), 2)
    cv2.putText(frame, f"Samples: {count}/{total}",
                (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 255, 180), 1)

    if phase == "countdown" and countdown is not None:
        cv2.putText(frame, f"GET READY: {countdown}",
                    (w//2 - 100, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 120, 255), 3)
    elif phase == "collecting":
        cv2.circle(frame, (w - 25, 25), 10, (0, 0, 255), -1)
        cv2.putText(frame, "REC", (w - 70, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    elif phase == "done":
        cv2.putText(frame, "DONE! Press SPACE for next",
                    (50, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2)
    elif phase == "no_hand":
        cv2.putText(frame, "No hand detected",
                    (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 80, 255), 2)
    elif phase == "waiting":
        cv2.putText(frame, "SPACE=start | N=skip | Q=quit",
                    (15, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)


def collect_class(cap, hands, cls_name, cls_idx):
    lm_dir  = os.path.join(LANDMARKS_DIR, cls_name)
    img_dir = os.path.join(IMAGES_DIR,    cls_name)

    existing = len([f for f in os.listdir(lm_dir) if f.endswith('.npy')])
    count    = existing
    phase    = "waiting"
    collecting       = False
    countdown_start  = None

    print(f"\n[{cls_idx+1}/{len(CLASSES)}] Class: {cls_name}  Existing: {existing}")
    print("Press SPACE to start | N to skip | Q to quit")

    while count < SAMPLES_PER_CLASS:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
        hand_detected = result.multi_hand_landmarks is not None

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return False
        if key == ord('n'):
            return True
        if key == ord(' ') and not collecting:
            collecting      = True
            phase           = "countdown"
            countdown_start = time.time()

        # always draw landmarks + bbox
        if hand_detected:
            for hand_lm in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )
                h_f, w_f = frame.shape[:2]
                xs = [lm.x * w_f for lm in hand_lm.landmark]
                ys = [lm.y * h_f for lm in hand_lm.landmark]
                x1 = max(0, int(min(xs)) - 30)
                y1 = max(0, int(min(ys)) - 30)
                x2 = min(w_f, int(max(xs)) + 30)
                y2 = min(h_f, int(max(ys)) + 30)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 2)
                break

        # countdown
        if phase == "countdown":
            elapsed   = time.time() - countdown_start
            remaining = COUNTDOWN - int(elapsed)
            if elapsed >= COUNTDOWN:
                phase = "collecting"
            else:
                draw_ui(frame, cls_name, cls_idx, count, SAMPLES_PER_CLASS,
                        "countdown", remaining)
                cv2.imshow("Arabic Sign Language Collector", frame)
                continue

        # collecting
        if phase == "collecting":
            if hand_detected:
                for hand_lm in result.multi_hand_landmarks:
                    lm_vec    = extract_landmarks(hand_lm)
                    roi, bbox = crop_hand_roi(frame, hand_lm)
                    if roi is not None:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        np.save(os.path.join(lm_dir, f"{ts}.npy"), lm_vec)
                        cv2.imwrite(os.path.join(img_dir, f"{ts}.jpg"), roi)
                        count += 1
                    break
                draw_ui(frame, cls_name, cls_idx, count, SAMPLES_PER_CLASS, "collecting")
            else:
                draw_ui(frame, cls_name, cls_idx, count, SAMPLES_PER_CLASS, "no_hand")

            if count >= SAMPLES_PER_CLASS:
                phase = "done"
                break

        elif phase == "waiting":
            draw_ui(frame, cls_name, cls_idx, count, SAMPLES_PER_CLASS, "waiting")

        cv2.imshow("Arabic Sign Language Collector", frame)

    # wait for SPACE to go to next class
    if count >= SAMPLES_PER_CLASS:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            draw_ui(frame, cls_name, cls_idx, count, SAMPLES_PER_CLASS, "done")
            cv2.imshow("Arabic Sign Language Collector", frame)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord(' '), 13):
                break
            if k == ord('q'):
                return False
    return True


def save_metadata():
    meta = {
        "classes":           CLASSES,
        "num_classes":       len(CLASSES),
        "samples_per_class": SAMPLES_PER_CLASS,
        "img_size":          IMG_SIZE,
        "landmark_features": 123,
        "created":           datetime.now().isoformat()
    }
    with open(os.path.join(DATASET_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main():
    setup_dirs()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.6
    )

    print("=== Arabic Sign Language Data Collector ===")
    print(f"Classes: {len(CLASSES)}  |  Samples/class: {SAMPLES_PER_CLASS}")
    print("Controls: SPACE=start/next | N=skip | Q=quit\n")

    for i, cls in enumerate(CLASSES):
        lm_dir   = os.path.join(LANDMARKS_DIR, cls)
        existing = len([f for f in os.listdir(lm_dir) if f.endswith('.npy')])
        if existing >= SAMPLES_PER_CLASS:
            print(f"[SKIP] {cls}")
            continue
        ok = collect_class(cap, hands, cls, i)
        if not ok:
            print("Stopped.")
            break

    save_metadata()
    hands.close()
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nDataset saved: {os.path.abspath(DATASET_DIR)}/")


if __name__ == "__main__":
    main()