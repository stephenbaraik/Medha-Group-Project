# Facial Emotion Recognition (FER) using Deep Learning

A CNN-based facial emotion recognition system trained on the FER2013 dataset with a Streamlit web app for inference and analysis.

**7 Emotion Classes:** Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral

## Architecture

Custom CNN inspired by Mellouk & Handouzi (2020) with:
- Inception-style multi-scale convolution modules (Mollahosseini et al.)
- Residual connections (Deepak et al.)
- Batch Normalization, Dropout, L2 regularization
- Global Average Pooling
- Mixed precision training (float16) for T4 GPU

**Expected accuracy:** ~65-68% on FER2013 test set (human agreement is ~65-72%).

## Project Structure

```
fer-emotion-recognition/
├── notebooks/FER_Training.ipynb     # Colab training notebook (T4 GPU)
├── app/
│   ├── streamlit_app.py             # Streamlit web application
│   └── utils/                       # Preprocessing, prediction, visualization
├── models/                          # Trained model weights (after training)
├── config.py                        # Central configuration
├── requirements.txt                 # Streamlit app dependencies
└── requirements-colab.txt           # Colab notebook dependencies
```

## Setup

### Step 1: Train the Model (Google Colab)

1. Open `notebooks/FER_Training.ipynb` in [Google Colab](https://colab.research.google.com/)
2. Set runtime to **GPU (T4)**
3. Run all cells — training takes ~30-60 minutes
4. Download the generated artifacts when prompted:
   - `fer_model.h5`
   - `training_history.json`
   - `classification_report.json`
   - `confusion_matrix.npy`

5. Place all downloaded files in the `models/` directory

### Step 2: Run the Streamlit App (Local)

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/streamlit_app.py
```

The app opens at `http://localhost:8501` with three pages:
- **Live Prediction** — Upload images or use webcam for real-time emotion detection
- **Model Analysis** — Training curves, confusion matrix, per-class metrics
- **About** — Project details and technical information

## Features

- **Multi-face detection** using OpenCV Haar Cascade
- **Grad-CAM visualization** showing what the model attends to
- **Interactive charts** with Plotly
- **Class imbalance handling** with computed class weights
- **Mixed precision training** for faster T4 GPU performance

## References

Mellouk, W., & Handouzi, W. (2020). Facial emotion recognition using deep learning: review and insights. *Procedia Computer Science*, 175, 689-694.
