import tensorflow as tf
import numpy as np


class EmotionPredictor:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, preprocessed_face):
        """Return (predicted_class_index, probabilities_array)."""
        preds = self.model.predict(preprocessed_face, verbose=0)
        return int(np.argmax(preds[0])), preds[0]

    def get_gradcam(self, preprocessed_face, last_conv_layer_name=None):
        """Generate Grad-CAM heatmap for the predicted class."""
        if last_conv_layer_name is None:
            for layer in reversed(self.model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv_layer_name = layer.name
                    break

        grad_model = tf.keras.models.Model(
            [self.model.inputs],
            [self.model.get_layer(last_conv_layer_name).output, self.model.output]
        )

        preprocessed_face = tf.cast(preprocessed_face, tf.float32)

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(preprocessed_face)
            pred_index = tf.argmax(predictions[0])
            class_channel = predictions[:, pred_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy().astype(np.float32)

    def get_model_summary(self):
        """Return model summary as a string."""
        lines = []
        self.model.summary(print_fn=lambda x: lines.append(x))
        return '\n'.join(lines)
