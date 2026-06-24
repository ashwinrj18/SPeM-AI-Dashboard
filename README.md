# 🛏️ SPeM: Smart Pressure e-Mat (AI Prototype)

A non-invasive, privacy-preserving AI system designed to classify human sleep postures in real-time using piezoresistive sensors and a state-of-the-art Convolutional Recurrent Neural Network (CRNN).

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-%23FE4B4B.svg?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## 📖 Project Overview
Prolonged immobility in bedridden patients frequently leads to severe pressure ulcers (bedsores). Current monitoring methods rely heavily on overburdened nursing staff or privacy-violating camera systems. 

**SPeM** solves this by using a $27 \times 27$ grid of Velostat material to capture continuous "pressure images." This data is fed into a custom PyTorch Deep Learning model to automatically track patient sleeping postures and alert healthcare providers to localized pressure risks.

## ✨ Key Features
*   **Privacy-Preserving:** 100% camera-free monitoring. The AI only interprets analog resistance pressure data.
*   **Spatiotemporal AI:** Utilizes a CRNN. The **ResNet-PI** layers extract the spatial shape of the body, while the **LSTM** memory network analyzes how the posture changes over a 10-frame sequence.
*   **Live Clinical Dashboard:** Built with Streamlit to render real-time *Inferno* heatmaps and provide automated "Bedsore Risk" alerts if a patient is immobile for > 2 hours.
*   **High Accuracy:** The current prototype achieves a **99.81%** confidence score on synthetic validation sets.

## 🧠 System Architecture
1. **Data Acquisition:** Physical compression alters the piezoresistive state of the Velostat Mat.
2. **Signal Processing:** An Arduino Uno (ADC) converts voltage drops to digital matrices.
3. **Serial Stream:** Data is streamed via USB at 9600 baud.
4. **Inference Engine:** PyTorch processes the accumulated 5D tensor.
5. **UI Rendering:** Streamlit updates the web portal with predicted postures (`Supine`, `Prone`, `Left Lateral`, `Right Lateral`).

## 📁 Repository Structure
```text
├── app.py               # Streamlit web dashboard and live inference engine
├── model.py             # PyTorch CRNN architecture (ResNet + LSTM classes)
├── train.py             # Training loop, loss calculation, and backpropagation
├── index.html           # Project landing page and architectural documentation
├── spem_weights.pth     # Trained PyTorch model weights (The "Brain")
└── requirements.txt     # Python package dependencies
