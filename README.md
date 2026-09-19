# Arabic Sign Recognition Sensor Glove

An assistive-computing project for recognizing Arabic hand gestures using computer vision, hand landmarks, and deep learning.

## Overview

The project captures hand-gesture data, extracts hand landmarks, trains classification models, and performs real-time prediction. The inference pipeline includes Arabic label rendering and optional Arabic text-to-speech output.

## Main Features

- Real-time hand detection using MediaPipe
- Landmark-based gesture representation
- CNN + landmark feature fusion
- MLP landmark classifier
- Arabic text rendering
- Prediction smoothing for stable real-time output
- Optional Arabic speech output
- Dataset collection and training scripts

## Technology Stack

- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- scikit-learn
- Pillow
- Matplotlib

## Project Structure

```text
arabic-sensor-Glove/
└── gesture/
    ├── collect_data.py
    ├── train.py
    ├── predict.py
    ├── arabic_sign_dataset/
    ├── models/
    └── Arabic_SensorGlove/
```

## Model Pipeline

The project supports two main model approaches:

1. **Landmark MLP** — classifies numerical hand-landmark features.
2. **Tiny CNN + Landmark Fusion** — combines cropped hand images with landmark features.

The prediction stage applies smoothing over recent frames to reduce unstable gesture changes.

## Run the Project

Install the required Python packages for TensorFlow, OpenCV, MediaPipe, NumPy, scikit-learn, and Pillow.

Train:

```bash
cd gesture
python train.py
```

Run real-time recognition:

```bash
python predict.py
```

## Accessibility Goal

The project explores how AI-based gesture recognition can support communication by converting recognized signs into readable Arabic text and, where configured, spoken Arabic output.

## Future Improvements

- Expand the Arabic gesture vocabulary
- Improve dataset balance and lighting robustness
- Add embedded sensor fusion from a physical glove
- Benchmark latency on Raspberry Pi / edge devices
- Add a mobile or web interface

## Author

**Sadik Shaik**

Computer Engineering / AI & Embedded Systems Projects
