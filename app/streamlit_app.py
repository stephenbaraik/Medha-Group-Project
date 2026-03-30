import sys
import os

# Add project root to path so config and app modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import streamlit as st
import numpy as np
import cv2
import json
from PIL import Image

from config import (
    IMG_SIZE, NUM_CLASSES, EMOTIONS, EMOTION_EMOJIS,
    MODEL_PATH, HISTORY_PATH, REPORT_PATH, CM_PATH, CASCADE_PATH
)
from app.utils.preprocessor import FacePreprocessor
from app.utils.predictor import EmotionPredictor
from app.utils.visualization import (
    plot_prediction_bars, plot_confusion_matrix, plot_training_history,
    plot_class_distribution, plot_per_class_metrics, overlay_gradcam,
    draw_bounding_boxes
)

st.set_page_config(
    page_title="FER - Emotion Recognition",
    page_icon="\U0001f3ad",
    layout="wide"
)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("\U0001f3ad FER Emotion Recognition")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["\U0001f3af Live Prediction", "\U0001f4ca Model Analysis", "\u2139\ufe0f About"]
)


# ── Cached resource loaders ─────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    return EmotionPredictor(MODEL_PATH)


@st.cache_resource
def load_preprocessor():
    return FacePreprocessor(CASCADE_PATH, IMG_SIZE)


@st.cache_data
def load_training_history():
    with open(HISTORY_PATH, 'r') as f:
        return json.load(f)


@st.cache_data
def load_classification_report():
    with open(REPORT_PATH, 'r') as f:
        return json.load(f)


@st.cache_data
def load_confusion_matrix_data():
    return np.load(CM_PATH)


# ── Helper: run prediction pipeline on an image ─────────────────────────────
def predict_on_image(image_array):
    """Full pipeline: detect faces, predict emotions, display results."""
    preprocessor = load_preprocessor()
    predictor = load_predictor()

    faces = preprocessor.detect_faces(image_array)

    if len(faces) == 0:
        st.warning(
            "No face detected. Using the full image as fallback. "
            "For best results, upload a clear frontal face photo."
        )
        preprocessed = preprocessor.preprocess_full_image(image_array)
        pred_idx, probs = predictor.predict(preprocessed)
        heatmap = predictor.get_gradcam(preprocessed)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_array, channels="BGR", caption="Input Image", use_container_width=True)
        with col2:
            emotion = EMOTIONS[pred_idx]
            emoji = EMOTION_EMOJIS[emotion]
            confidence = probs[pred_idx] * 100
            st.markdown(f"### {emoji} {emotion}")
            st.markdown(f"**Confidence:** {confidence:.1f}%")
            st.plotly_chart(
                plot_prediction_bars(probs, EMOTIONS, EMOTION_EMOJIS),
                use_container_width=True
            )

        # Grad-CAM
        st.subheader("Grad-CAM Visualization")
        if len(image_array.shape) == 3:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_array
        face_resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        gradcam_overlay = overlay_gradcam(face_resized, heatmap)
        st.image(gradcam_overlay, channels="BGR", caption="Grad-CAM Heatmap", width=300)
        return

    # Faces detected — annotate and predict each
    predictions = []
    for bbox in faces:
        preprocessed = preprocessor.preprocess_face(image_array, bbox)
        pred_idx, probs = predictor.predict(preprocessed)
        predictions.append((pred_idx, probs))

    annotated = draw_bounding_boxes(image_array, faces, predictions, EMOTIONS, EMOTION_EMOJIS)
    st.image(annotated, channels="BGR", caption="Detected Faces", use_container_width=True)

    st.markdown("---")

    for i, ((x, y, w, h), (pred_idx, probs)) in enumerate(zip(faces, predictions)):
        emotion = EMOTIONS[pred_idx]
        emoji = EMOTION_EMOJIS[emotion]
        confidence = probs[pred_idx] * 100

        st.subheader(f"Face {i + 1}: {emoji} {emotion} ({confidence:.1f}%)")

        col1, col2, col3 = st.columns([1, 1, 2])

        with col1:
            face_crop = image_array[y:y + h, x:x + w]
            st.image(face_crop, channels="BGR", caption="Face Crop", use_container_width=True)

        with col2:
            preprocessed = preprocessor.preprocess_face(image_array, (x, y, w, h))
            heatmap = predictor.get_gradcam(preprocessed)
            if len(face_crop.shape) == 2:
                gray_crop = face_crop
            else:
                gray_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            gradcam_overlay = overlay_gradcam(
                cv2.resize(gray_crop, (IMG_SIZE, IMG_SIZE)), heatmap
            )
            st.image(
                gradcam_overlay, channels="BGR",
                caption="Grad-CAM", use_container_width=True
            )

        with col3:
            st.plotly_chart(
                plot_prediction_bars(probs, EMOTIONS, EMOTION_EMOJIS),
                use_container_width=True
            )

        st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: LIVE PREDICTION
# ═══════════════════════════════════════════════════════════════════════════════
if page == "\U0001f3af Live Prediction":
    st.title("\U0001f3af Live Prediction")
    st.markdown("Upload an image or use your webcam to predict facial emotions.")

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at `{MODEL_PATH}`. "
            "Please train the model using the Colab notebook and place `fer_model.h5` "
            "in the `models/` directory."
        )
        st.stop()

    tab_upload, tab_webcam = st.tabs(["\U0001f4f7 Image Upload", "\U0001f4f9 Webcam Capture"])

    with tab_upload:
        uploaded_file = st.file_uploader(
            "Upload a face image",
            type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
            help="Upload an image containing one or more faces."
        )

        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is None:
                st.error("Could not read the uploaded image. Please try another file.")
            else:
                with st.spinner("Detecting faces and predicting emotions..."):
                    predict_on_image(image)

    with tab_webcam:
        camera_image = st.camera_input("Take a photo")

        if camera_image is not None:
            file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            if image is not None:
                with st.spinner("Detecting faces and predicting emotions..."):
                    predict_on_image(image)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: MODEL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "\U0001f4ca Model Analysis":
    st.title("\U0001f4ca Model Analysis")

    # ── Training Curves ──────────────────────────────────────────────────────
    st.header("Training Curves")
    if os.path.exists(HISTORY_PATH):
        history = load_training_history()
        acc_fig, loss_fig = plot_training_history(history)
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(acc_fig, use_container_width=True)
        with col2:
            st.plotly_chart(loss_fig, use_container_width=True)

        total_epochs = len(history['accuracy'])
        best_val_acc = max(history['val_accuracy'])
        best_epoch = history['val_accuracy'].index(best_val_acc) + 1
        st.info(
            f"Trained for **{total_epochs} epochs**. "
            f"Best validation accuracy: **{best_val_acc:.4f}** at epoch **{best_epoch}**."
        )
    else:
        st.warning(f"Training history not found at `{HISTORY_PATH}`.")

    st.markdown("---")

    # ── Confusion Matrix ─────────────────────────────────────────────────────
    st.header("Confusion Matrix")
    if os.path.exists(CM_PATH):
        cm = load_confusion_matrix_data()
        st.plotly_chart(
            plot_confusion_matrix(cm, EMOTIONS),
            use_container_width=True
        )
    else:
        st.warning(f"Confusion matrix not found at `{CM_PATH}`.")

    st.markdown("---")

    # ── Per-Class Metrics ────────────────────────────────────────────────────
    st.header("Per-Class Metrics")
    if os.path.exists(REPORT_PATH):
        report = load_classification_report()

        st.plotly_chart(
            plot_per_class_metrics(report, EMOTIONS),
            use_container_width=True
        )

        # Table view
        st.subheader("Detailed Metrics Table")
        import pandas as pd
        metrics_data = []
        for emotion in EMOTIONS:
            if emotion in report:
                metrics_data.append({
                    'Emotion': f"{EMOTION_EMOJIS[emotion]} {emotion}",
                    'Precision': f"{report[emotion]['precision']:.3f}",
                    'Recall': f"{report[emotion]['recall']:.3f}",
                    'F1-Score': f"{report[emotion]['f1-score']:.3f}",
                    'Support': int(report[emotion]['support'])
                })
        df = pd.DataFrame(metrics_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        if 'accuracy' in report:
            st.metric("Overall Test Accuracy", f"{report['accuracy']:.4f}")
    else:
        st.warning(f"Classification report not found at `{REPORT_PATH}`.")

    st.markdown("---")

    # ── Dataset Distribution ─────────────────────────────────────────────────
    st.header("FER2013 Dataset Distribution")
    st.plotly_chart(plot_class_distribution(), use_container_width=True)
    st.caption(
        "The FER2013 dataset has severe class imbalance. **Disgust** has only ~547 training "
        "samples compared to **Happy** with ~8,989. This imbalance is addressed during "
        "training using class weights."
    )

    st.markdown("---")

    # ── Model Architecture ───────────────────────────────────────────────────
    st.header("Model Architecture")
    if os.path.exists(MODEL_PATH):
        with st.expander("View Model Summary", expanded=False):
            predictor = load_predictor()
            summary_text = predictor.get_model_summary()
            st.code(summary_text, language=None)
    else:
        st.warning("Model not loaded. Cannot display architecture.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3: ABOUT
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "\u2139\ufe0f About":
    st.title("\u2139\ufe0f About This Project")

    st.markdown("""
    ## Facial Emotion Recognition (FER) using Deep Learning

    This project implements an end-to-end facial emotion recognition system using a
    Convolutional Neural Network (CNN) trained on the **FER2013** dataset.

    ### Research Foundation

    The architecture is inspired by the survey paper:

    > **Mellouk, W., & Handouzi, W. (2020).** *Facial emotion recognition using deep
    > learning: review and insights.* Procedia Computer Science, 175, 689-694.

    Key architectural elements drawn from the reviewed literature:
    - **Inception-style modules** (Mollahosseini et al.) for multi-scale feature extraction
    - **Residual connections** (Deepak et al.) for better gradient flow
    - **Batch normalization** and **dropout** for regularization
    - **Global Average Pooling** instead of fully connected layers to reduce overfitting

    ---

    ### Emotion Categories

    | Emotion | Description |
    |---------|-------------|
    | \U0001f620 Angry | Expressions of anger, frustration, or irritation |
    | \U0001f922 Disgust | Expressions of disgust or distaste |
    | \U0001f628 Fear | Expressions of fear, anxiety, or apprehension |
    | \U0001f60a Happy | Expressions of happiness, joy, or amusement |
    | \U0001f622 Sad | Expressions of sadness, grief, or disappointment |
    | \U0001f632 Surprise | Expressions of surprise or astonishment |
    | \U0001f610 Neutral | Neutral or baseline facial expressions |

    ---

    ### Technical Details

    | Parameter | Value |
    |-----------|-------|
    | **Dataset** | FER2013 (35,887 grayscale 48x48 images) |
    | **Input Size** | 48 x 48 x 1 (grayscale) |
    | **Number of Classes** | 7 |
    | **Architecture** | Custom CNN with Inception + Residual blocks |
    | **Optimizer** | Adam (initial LR: 1e-3) |
    | **LR Schedule** | ReduceLROnPlateau (factor=0.5, patience=5) |
    | **Regularization** | L2 (1e-4), Dropout (0.25-0.5), Batch Normalization |
    | **Training** | Mixed precision (float16) on Google Colab T4 GPU |
    | **Expected Accuracy** | ~65-68% on FER2013 test set |

    ---

    ### Why ~65-68% Accuracy?

    FER2013 is a notoriously challenging benchmark. The images are:
    - **Low resolution** (48x48 pixels)
    - **Collected in the wild** (not lab-controlled)
    - **Noisy labels** (many images are ambiguous even to humans)
    - **Severely imbalanced** (Disgust has 10x fewer samples than Happy)

    Human-level agreement on FER2013 is estimated at ~65-72%. Achieving higher accuracy
    often requires ensemble methods, external data, or face alignment preprocessing.

    ---

    ### How It Works

    1. **Face Detection**: OpenCV Haar Cascade detects face regions in the input image
    2. **Preprocessing**: Faces are cropped, converted to grayscale, resized to 48x48,
       and normalized to [0, 1]
    3. **Prediction**: The CNN outputs probability scores for all 7 emotion classes
    4. **Grad-CAM**: Gradient-weighted Class Activation Maps highlight the facial regions
       most important to the model's decision

    ---

    ### Project Structure

    ```
    fer-emotion-recognition/
    \u251c\u2500\u2500 notebooks/FER_Training.ipynb   # Colab training notebook
    \u251c\u2500\u2500 app/streamlit_app.py           # This Streamlit application
    \u251c\u2500\u2500 app/utils/                     # Preprocessing, prediction, visualization
    \u251c\u2500\u2500 models/                        # Trained model weights & artifacts
    \u2514\u2500\u2500 config.py                      # Central configuration
    ```
    """)

    st.markdown("---")
    st.caption(
        "Built with TensorFlow, Streamlit, OpenCV, and Plotly. "
        "Trained on Google Colab with T4 GPU."
    )
