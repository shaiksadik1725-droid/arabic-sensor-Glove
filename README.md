# Arabic Sign Recognition Sensor Glove

<p align="center">
  <strong>Assistive AI for real-time Arabic hand-gesture recognition</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/MediaPipe-Hand_Landmarks-00BFA5" />
  <a href="https://github.com/shaiksadik1725-droid/arabic-sensor-Glove/actions/workflows/python-syntax.yml"><img src="https://github.com/shaiksadik1725-droid/arabic-sensor-Glove/actions/workflows/python-syntax.yml/badge.svg" alt="Python syntax check" /></a>
</p>

## Project at a Glance

| Item | Details |
|---|---|
| Domain | Assistive computing / computer vision |
| Input | Camera-based hand gestures |
| Feature representation | MediaPipe landmarks + image features |
| Models | Landmark MLP and CNN + landmark fusion |
| Output | Arabic label with optional speech |
| Status | Academic prototype |

## Overview

This project captures hand gestures, extracts hand landmarks, trains classification models, and performs real-time prediction. It also includes Arabic text rendering, prediction smoothing, and optional Arabic text-to-speech.

## Training Evidence

<p align="center">
  <img src="gesture/models/CNN_history.png" width="48%" alt="CNN training history" />
  <img src="gesture/models/CNN_cm.png" width="48%" alt="CNN confusion matrix" />
</p>

<p align="center">
  <img src="gesture/models/MLP_history.png" width="48%" alt="MLP training history" />
  <img src="gesture/models/MLP_cm.png" width="48%" alt="MLP confusion matrix" />
</p>

## Recognition Pipeline

```mermaid
flowchart LR
    A[Camera Frame] --> B[MediaPipe Hand Detection]
    B --> C[Landmark Extraction]
    B --> D[Hand ROI Image]
    C --> E[Landmark MLP]
    C --> F[CNN + Landmark Fusion]
    D --> F
    E --> G[Smoothed Prediction]
    F --> G
    G --> H[Arabic Text]
    H --> I[Optional Arabic Speech]
```

## Main Features

- Real-time hand detection
- Landmark-based gesture representation
- CNN + landmark feature fusion
- MLP landmark classifier
- Arabic text rendering
- Temporal smoothing for stable predictions
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
- Matplotlib / Seaborn

## Repository Structure

```text
arabic-sensor-Glove/
├── README.md
├── requirements.txt
└── gesture/
    ├── collect_data.py
    ├── train.py
    ├── predict.py
    ├── arabic_sign_dataset/
    ├── models/
    └── Arabic_SensorGlove/
```

## Setup

```bash
git clone https://github.com/shaiksadik1725-droid/arabic-sensor-Glove.git
cd arabic-sensor-Glove
pip install -r requirements.txt
cd gesture
python train.py
```

Run real-time recognition:

```bash
python predict.py
```

## Accessibility Goal

The project explores how computer vision can translate recognized hand gestures into readable Arabic text and, when configured, spoken Arabic output.

## Future Work

- Expand the Arabic sign vocabulary
- Increase dataset diversity
- Improve lighting and background robustness
- Add physical glove-sensor fusion
- Benchmark Raspberry Pi / edge performance
- Add a mobile or web interface

## Author

**Sadik Shaik**

Computer Engineering · Artificial Intelligence · Embedded Systems
