import os
import json
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import gc

DATASET_DIR = "arabic_sign_dataset"
IMAGES_DIR = os.path.join(DATASET_DIR, "images")
LANDMARKS_DIR = os.path.join(DATASET_DIR, "landmarks")
MODEL_DIR = "models"
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-3

os.makedirs(MODEL_DIR, exist_ok=True)

tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)

with open(os.path.join(DATASET_DIR, "metadata.json"), encoding="utf-8") as f:
    metadata = json.load(f)
CLASSES = metadata["classes"]
NUM_CLASSES = metadata["num_classes"]


def load_landmarks_only():
    landmarks, labels = [], []
    for cls in CLASSES:
        lm_dir = os.path.join(LANDMARKS_DIR, cls)
        if not os.path.exists(lm_dir):
            continue
        files = [f for f in os.listdir(lm_dir) if f.endswith('.npy')]
        for fname in files:
            lm = np.load(os.path.join(lm_dir, fname))
            landmarks.append(lm)
            labels.append(cls)
    landmarks = np.array(landmarks, dtype=np.float32)
    le = LabelEncoder()
    labels_enc = le.fit_transform(labels)
    with open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "wb") as f:
        pickle.dump(le, f)
    print(f"Landmarks loaded: {len(landmarks)} samples | {NUM_CLASSES} classes")
    return landmarks, labels_enc, le


def load_images_chunked(le, chunk_size=150):
    all_img, all_lm, all_lbl = [], [], []
    for cls in CLASSES:
        img_dir = os.path.join(IMAGES_DIR, cls)
        lm_dir = os.path.join(LANDMARKS_DIR, cls)
        if not os.path.exists(img_dir):
            continue
        files = [f[:-4] for f in os.listdir(img_dir) if f.endswith('.jpg')]
        count = 0
        for fname in files:
            if count >= chunk_size:
                break
            img_path = os.path.join(img_dir, fname + '.jpg')
            lm_path = os.path.join(lm_dir, fname + '.npy')
            img = cv2.imread(img_path)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            all_img.append(img)
            all_lm.append(np.load(lm_path) if os.path.exists(lm_path) else np.zeros(123))
            all_lbl.append(le.transform([cls])[0])
            count += 1
        gc.collect()

    images = np.array(all_img, dtype=np.float32) / 255.0
    landmarks = np.array(all_lm, dtype=np.float32)
    labels = np.array(all_lbl)
    print(f"Images loaded: {len(images)} samples (max {chunk_size}/class)")
    return images, landmarks, labels


def build_landmark_mlp(input_dim, num_classes):
    inp = layers.Input(shape=(input_dim,), name="lm_input")
    x = layers.Dense(512)(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inp, out, name="LandmarkMLP")


def build_tiny_cnn(img_shape, lm_dim, num_classes):
    img_inp = layers.Input(shape=img_shape, name="image_input")
    lm_inp = layers.Input(shape=(lm_dim,), name="lm_input")

    x = layers.Conv2D(16, 3, padding='same', activation='relu')(img_inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(32, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)

    lm = layers.Dense(64, activation='relu')(lm_inp)
    lm = layers.Dropout(0.3)(lm)

    fused = layers.Concatenate()([x, lm])
    fused = layers.Dense(128, activation='relu')(fused)
    fused = layers.BatchNormalization()(fused)
    fused = layers.Dropout(0.35)(fused)
    out = layers.Dense(num_classes, activation='softmax')(fused)

    return models.Model(inputs=[img_inp, lm_inp], outputs=out, name="TinyCNN")


def get_callbacks(prefix=''):
    return [
        callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.5,
                                    patience=8, min_lr=1e-6, verbose=1),
        callbacks.EarlyStopping(monitor='val_accuracy', patience=20,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, f"{prefix}best.h5"),
            monitor='val_accuracy', save_best_only=True, verbose=1
        )
    ]


def save_tflite(model, name):
    # TFLite conversion is broken on this TF version (flatbuffers mismatch)
    # Skipping — using .h5 directly for inference instead
    print(f"[INFO] TFLite skipped — using .h5 model for inference")


def plot_history(history, name):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history['accuracy'], label='Train')
    axes[0].plot(history.history['val_accuracy'], label='Val')
    axes[0].set_title(f'{name} Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    axes[1].plot(history.history['loss'], label='Train')
    axes[1].plot(history.history['val_loss'], label='Val')
    axes[1].set_title(f'{name} Loss')
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{name}_history.png"), dpi=100)
    plt.close()
    gc.collect()


def plot_confusion(y_true, y_pred, class_names, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                annot_kws={"size": 6})
    plt.title(f'{name} Confusion Matrix')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45, ha='right', fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f"{name}_cm.png"), dpi=100)
    plt.close()
    gc.collect()


def train_mlp(landmarks, labels, le):
    print("\n" + "="*50)
    print("PHASE 1: Landmark MLP")
    print("="*50)

    labels_cat = to_categorical(labels, NUM_CLASSES)
    X_tr, X_te, y_tr, y_te = train_test_split(
        landmarks, labels_cat, test_size=0.15, random_state=42, stratify=labels)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tr, y_tr, test_size=0.15, random_state=42)

    print(f"Train: {len(X_tr)} | Val: {len(X_val)} | Test: {len(X_te)}")

    model = build_landmark_mlp(landmarks.shape[1], NUM_CLASSES)
    model.summary()
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(X_tr, y_tr, validation_data=(X_val, y_val),
                        epochs=EPOCHS, batch_size=BATCH_SIZE,
                        callbacks=get_callbacks('mlp_'), verbose=1)

    plot_history(history, "MLP")

    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nMLP Test Accuracy: {acc*100:.2f}%")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    plot_confusion(y_true, y_pred, le.classes_, "MLP")

    model.save(os.path.join(MODEL_DIR, "mlp_model.h5"))
    save_tflite(model, "mlp_model")
    gc.collect()
    return acc


def train_cnn(le):
    print("\n" + "="*50)
    print("PHASE 2: Tiny CNN + Landmark Fusion")
    print("="*50)

    images, landmarks, labels = load_images_chunked(le)
    labels_cat = to_categorical(labels, NUM_CLASSES)

    X_img_tr, X_img_te, X_lm_tr, X_lm_te, y_tr, y_te = train_test_split(
        images, landmarks, labels_cat,
        test_size=0.15, random_state=42, stratify=labels)
    X_img_tr, X_img_val, X_lm_tr, X_lm_val, y_tr, y_val = train_test_split(
        X_img_tr, X_lm_tr, y_tr, test_size=0.15, random_state=42)

    print(f"Train: {len(X_img_tr)} | Val: {len(X_img_val)} | Test: {len(X_img_te)}")

    model = build_tiny_cnn((IMG_SIZE, IMG_SIZE, 3), landmarks.shape[1], NUM_CLASSES)
    model.summary()
    model.compile(optimizer=optimizers.Adam(LEARNING_RATE),
                  loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(
        [X_img_tr, X_lm_tr], y_tr,
        validation_data=([X_img_val, X_lm_val], y_val),
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        callbacks=get_callbacks('cnn_'), verbose=1)

    plot_history(history, "CNN")

    loss, acc = model.evaluate([X_img_te, X_lm_te], y_te, verbose=0)
    print(f"\nCNN Test Accuracy: {acc*100:.2f}%")

    y_pred = np.argmax(model.predict([X_img_te, X_lm_te], verbose=0), axis=1)
    y_true = np.argmax(y_te, axis=1)
    print(classification_report(y_true, y_pred, target_names=le.classes_))
    plot_confusion(y_true, y_pred, le.classes_, "CNN")

    model.save(os.path.join(MODEL_DIR, "cnn_model.h5"))
    save_tflite(model, "cnn_model")

    del images, landmarks
    gc.collect()
    return acc


def main():
    print("=== Arabic Sign Language Trainer (Pi-Optimized) ===")
    print(f"TF: {tf.__version__} | Classes: {NUM_CLASSES} | Img: {IMG_SIZE}px | Batch: {BATCH_SIZE}\n")

    landmarks, labels, le = load_landmarks_only()
    lm_dim = landmarks.shape[1]

    mlp_acc = train_mlp(landmarks, labels, le)
    del landmarks
    gc.collect()

    cnn_acc = train_cnn(le)

    best = "cnn_model" if cnn_acc >= mlp_acc else "mlp_model"

    cfg = {
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "classes": CLASSES,
        "lm_dim": int(lm_dim),
        "mlp_accuracy": float(mlp_acc),
        "cnn_accuracy": float(cnn_acc),
        "best_model": best,
        "use_tflite": False
    }
    with open(os.path.join(MODEL_DIR, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    print("\n" + "="*50)
    print(f"MLP Accuracy : {mlp_acc*100:.2f}%")
    print(f"CNN Accuracy : {cnn_acc*100:.2f}%")
    print(f"Best Model   : {best}")
    print(f"Saved to     : {os.path.abspath(MODEL_DIR)}/")
    print("="*50)


if __name__ == "__main__":
    main()