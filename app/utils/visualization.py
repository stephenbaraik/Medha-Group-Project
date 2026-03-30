import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2


def plot_prediction_bars(probabilities, emotions, emotion_emojis):
    """Horizontal bar chart of prediction probabilities."""
    labels = [f"{emotion_emojis[e]} {e}" for e in emotions]
    fig = px.bar(
        x=probabilities * 100,
        y=labels,
        orientation='h',
        labels={'x': 'Confidence (%)', 'y': 'Emotion'},
        color=probabilities * 100,
        color_continuous_scale='Viridis'
    )
    fig.update_layout(
        height=350,
        showlegend=False,
        coloraxis_showscale=False,
        yaxis=dict(autorange='reversed')
    )
    return fig


def plot_confusion_matrix(cm, emotions):
    """Interactive Plotly confusion matrix heatmap."""
    fig = px.imshow(
        cm,
        text_auto=True,
        x=emotions,
        y=emotions,
        labels=dict(x='Predicted', y='Actual', color='Count'),
        color_continuous_scale='Blues'
    )
    fig.update_layout(height=500, title='Confusion Matrix')
    return fig


def plot_training_history(history):
    """Accuracy and loss curves as two separate figures."""
    # Accuracy plot
    acc_fig = go.Figure()
    acc_fig.add_trace(go.Scatter(
        y=history['accuracy'], name='Train Accuracy', mode='lines',
        line=dict(color='#636EFA')
    ))
    acc_fig.add_trace(go.Scatter(
        y=history['val_accuracy'], name='Val Accuracy', mode='lines',
        line=dict(color='#EF553B')
    ))
    acc_fig.update_layout(
        title='Training & Validation Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        height=400
    )

    # Loss plot
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(
        y=history['loss'], name='Train Loss', mode='lines',
        line=dict(color='#636EFA')
    ))
    loss_fig.add_trace(go.Scatter(
        y=history['val_loss'], name='Val Loss', mode='lines',
        line=dict(color='#EF553B')
    ))
    loss_fig.update_layout(
        title='Training & Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        height=400
    )

    return acc_fig, loss_fig


def plot_class_distribution():
    """Bar chart of FER2013 class distribution."""
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    # Approximate FER2013 training set distribution
    counts = [4953, 547, 5121, 8989, 6077, 4002, 6198]

    fig = px.bar(
        x=emotions, y=counts,
        labels={'x': 'Emotion', 'y': 'Number of Samples'},
        color=emotions,
        title='FER2013 Training Set Class Distribution'
    )
    fig.update_layout(height=400, showlegend=False)
    return fig


def plot_per_class_metrics(report, emotions):
    """Grouped bar chart for precision, recall, F1-score per class."""
    precision = [report.get(e, report.get(e.lower(), {}))['precision'] for e in emotions]
    recall = [report.get(e, report.get(e.lower(), {}))['recall'] for e in emotions]
    f1 = [report.get(e, report.get(e.lower(), {}))['f1-score'] for e in emotions]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Precision', x=emotions, y=precision))
    fig.add_trace(go.Bar(name='Recall', x=emotions, y=recall))
    fig.add_trace(go.Bar(name='F1-Score', x=emotions, y=f1))
    fig.update_layout(
        barmode='group',
        title='Per-Class Metrics',
        yaxis_title='Score',
        height=400
    )
    return fig


def overlay_gradcam(image, heatmap, alpha=0.4):
    """Overlay Grad-CAM heatmap on image."""
    heatmap = np.float32(heatmap)
    image = np.uint8(image)
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    superimposed = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)
    return superimposed


def draw_bounding_boxes(image, faces, predictions, emotions, emotion_emojis):
    """Draw bounding boxes and labels on image."""
    annotated = image.copy()
    for (x, y, w, h), (pred_idx, probs) in zip(faces, predictions):
        emotion = emotions[pred_idx]
        confidence = probs[pred_idx] * 100
        label = f"{emotion} ({confidence:.1f}%)"

        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # Background for text
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        cv2.rectangle(
            annotated,
            (x, y - text_size[1] - 10),
            (x + text_size[0], y),
            (0, 255, 0), -1
        )
        cv2.putText(
            annotated, label, (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
    return annotated
