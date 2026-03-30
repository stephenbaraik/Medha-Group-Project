import cv2
import numpy as np


class FacePreprocessor:
    def __init__(self, cascade_path, img_size=48):
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.img_size = img_size

    def detect_faces(self, image):
        """Detect faces and return list of (x, y, w, h) bounding boxes."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        return faces

    def preprocess_face(self, image, bbox):
        """Crop, resize, normalize a single face for model input."""
        x, y, w, h = bbox
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        face = gray[y:y + h, x:x + w]
        face = cv2.resize(face, (self.img_size, self.img_size))
        face = face.astype('float32') / 255.0
        face = np.expand_dims(face, axis=(0, -1))  # (1, 48, 48, 1)
        return face

    def preprocess_full_image(self, image):
        """Fallback: resize full image to model input when no face is detected."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        resized = cv2.resize(gray, (self.img_size, self.img_size))
        resized = resized.astype('float32') / 255.0
        resized = np.expand_dims(resized, axis=(0, -1))
        return resized
