# 🎤 Emotion Recognition from Speech using Deep Learning

This repository contains a deep learning pipeline for recognizing human emotions (e.g., happy, sad, angry) from raw audio speech. The model is trained on publicly available datasets and uses a combination of CNN and LSTM architectures to capture both local and sequential features of speech.

## 🚀 Project Overview

- **Goal:** Build a robust deep learning model that can classify emotions from speech audio using features like MFCCs.
- **Approach:** Deep learning with CNN + LSTM using extracted MFCCs, Delta & Delta-Delta features.
- **Datasets Used:** RAVDESS, TESS, EMO-DB (flexible to integrate)

## 📁 Project Structure

├── emotion_recognizer_dl_final.py # Training script

├── inference_script_re_provide.py # Inference script for new audio prediction
├── speech_emotion_dl_model.h5 # Saved Keras model
├── label_encoder_dl.pkl # Saved label encoder for mapping predictions
├── /RAVDESS # Dataset folder (not included)
└── README.md # Project documentation

## 🧠 Features

- 📌 **MFCCs + Deltas**: Extract Mel-Frequency Cepstral Coefficients (MFCC), Delta & Delta-Delta for richer representations
- 🧱 **Model Architecture**: Combines CNN (Conv1D, MaxPooling) and LSTM to learn temporal dependencies
- 💾 **Model Saving**: Model is saved using `.h5`, and encoder via `joblib`
- 🎯 **Inference Ready**: Inference script to test on unseen `.wav` audio files
- 📊 **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score via classification report

## 🛠️ Tech Stack

- Python 3.9+
- TensorFlow / Keras
- Librosa
- Scikit-learn
- Pandas, NumPy, Matplotlib, Seaborn
- Joblib (for saving encoder)

## 📦 How to Run

### 🔹 1. Train the Model

```bash
python emotion_recognizer_dl_final.py


Make sure the dataset path in the script points to your local RAVDESS folder
Model and label encoder will be saved upon completion

🔹 2. Run Inference
bash
Copy
Edit
python inference_script_re_provide.py
Modify the script to test with your own .wav file

You will get a predicted emotion as output

🔍 Sample Output

🎧 Input Audio: 03-01-05-01-01-01-01.wav
🎯 Predicted Emotion: Angry
📊 Probabilities: {'Angry': 0.94, 'Happy': 0.03, 'Neutral': 0.02, 'Sad': 0.01}

📚 Dataset Reference

RAVDESS
TESS
EMO-DB

🙌 Acknowledgements
This project leverages powerful speech processing libraries like Librosa and the capabilities of deep neural networks for audio understanding. Huge thanks to open-source contributors and dataset curators for enabling projects like this.


💡 Future Enhancements
Add support for noise filtering & audio augmentation
Extend to real-time emotion recognition using mic input
Build web app interface using Streamlit or Flask





