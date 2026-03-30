import os

IMG_SIZE = 48
NUM_CLASSES = 7
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_EMOJIS = {
    'Angry': '\U0001f620', 'Disgust': '\U0001f922', 'Fear': '\U0001f628',
    'Happy': '\U0001f60a', 'Sad': '\U0001f622', 'Surprise': '\U0001f632',
    'Neutral': '\U0001f610'
}

MODEL_PATH = os.path.join('models', 'fer_model.h5')
HISTORY_PATH = os.path.join('models', 'training_history.json')
REPORT_PATH = os.path.join('models', 'classification_report.json')
CM_PATH = os.path.join('models', 'confusion_matrix.npy')
CASCADE_PATH = os.path.join('app', 'assets', 'haarcascade_frontalface_default.xml')
