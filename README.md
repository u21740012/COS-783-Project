# AI Image Forensics Tool

COS 783 - Digital Forensics Final Assignment  
Option 6: Image and Video Forensics

---

## Overview

An AI-powered tool for detecting tampered/manipulated images using a ResNet50V2 CNN trained on the CG-1050 dataset, with Error Level Analysis (ELA) for visual forensic inspection.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

Open `Image_Analysis.ipynb` in Google Colab, run all cells. The last cell saves the model to `models/tampering_detector.keras`. Download it and place it in the `models/` folder.

### 3. Run the App

```bash
streamlit run app.py
```

---

## Project Structure

```
COS-783-Project/
├── app.py                  # Streamlit frontend
├── requirements.txt        # Dependencies
├── Image_Analysis.ipynb    # Colab training notebook
├── src/
│   └── preprocess.py       # Preprocessing utilities
├── data/
│   ├── TRAINING_CG-1050/   # Training images (ORIGINAL + TAMPERED)
│   └── VALIDATION_CG-1050/ # Validation images
└── models/                 # Saved trained model
```

---

## Model

- Architecture: ResNet50V2 (pretrained on ImageNet)
- Input: 224x224 RGB images
- Output: Binary classification (Original / Tampered)
- Dataset: CG-1050 (1460 training, 628 validation)
- Validation Accuracy: ~65%
- Validation Precision: ~67%

---

## Technologies

- Python 3.13
- TensorFlow / Keras
- Streamlit
- OpenCV
- Pillow
- NumPy / Matplotlib
